#!/usr/bin/env python3
"""
Decision Policy — Política de decisión genérica y escalable para identificación SKU.

Define cómo se toman decisiones finales por bbox, incluyendo:
- Thresholds configurables
- Reglas de split condicional
- Criterios de aceptación de split
- Métricas de calidad de bbox
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DecisionPolicyConfig:
    """
    Configuración de la política de decisión.
    
    Puede tener diferentes perfiles según el entorno:
    - catalog-only: imágenes limpias
    - shelf/video: ruido real de góndola
    - low-light: condiciones de iluminación baja
    """
    # Thresholds de identificación
    match_threshold: float = 0.28
    unknown_threshold: float = 0.20
    ambiguity_margin: float = 0.02
    
    # Reglas de split
    split_delta_min: float = 0.05  # Mejora mínima para aceptar split
    split_min_confianza: float = 0.25  # Confianza mínima del full para considerar split
    split_max_confianza: float = 0.45  # Confianza máxima del full para considerar split
    
    # Calidad de bbox (para decidir split)
    bbox_quality_aspect_weight: float = 0.4
    bbox_quality_area_weight: float = 0.2
    bbox_quality_yolo_conf_weight: float = 0.3
    bbox_quality_edge_weight: float = 0.1
    bbox_quality_threshold: float = 0.6  # Si score > threshold, considerar split
    
    # Inner crop ratio
    inner_crop_ratio: float = 0.75
    
    @classmethod
    def catalog_only(cls) -> "DecisionPolicyConfig":
        """Perfil para imágenes limpias de catálogo."""
        return cls(
            match_threshold=0.30,
            unknown_threshold=0.22,
            ambiguity_margin=0.015,
            split_delta_min=0.03,
        )
    
    @classmethod
    def shelf_video(cls) -> "DecisionPolicyConfig":
        """Perfil para videos de góndola real (ruido, oclusiones)."""
        return cls(
            match_threshold=0.28,
            unknown_threshold=0.20,
            ambiguity_margin=0.02,
            split_delta_min=0.05,
            split_min_confianza=0.25,
            split_max_confianza=0.45,
        )
    
    @classmethod
    def low_light(cls) -> "DecisionPolicyConfig":
        """Perfil para condiciones de baja iluminación."""
        return cls(
            match_threshold=0.25,
            unknown_threshold=0.18,
            ambiguity_margin=0.025,
            split_delta_min=0.06,
        )


class DecisionPolicy:
    """
    Política de decisión genérica para identificación SKU.
    
    Implementa:
    - Decisión final por bbox (1 bbox = 1 resultado)
    - Reglas de split condicional
    - Criterios de aceptación de split
    """
    
    def __init__(self, config: Optional[DecisionPolicyConfig] = None):
        self.config = config or DecisionPolicyConfig.shelf_video()
    
    def decidir_resultado_final(
        self,
        resultado_full: Dict[str, Any],
        resultado_left: Optional[Dict[str, Any]] = None,
        resultado_right: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Decide el resultado final para un bbox.
        
        Regla: 1 bbox = 1 decisión final.
        
        Si hay split, solo se acepta si mejora significativamente al resultado full.
        
        Args:
            resultado_full: Resultado del crop completo.
            resultado_left: Resultado del crop izquierdo (si hubo split).
            resultado_right: Resultado del crop derecho (si hubo split).
        
        Returns:
            Un único resultado final (el mejor).
        """
        # Si no hay split, retornar el resultado full directamente
        if resultado_left is None or resultado_right is None:
            return resultado_full
        
        # Si el resultado full es matched y con buena confianza, no usar split
        status_full = str(resultado_full.get("status", "unknown"))
        conf_full = float(resultado_full.get("confianza", 0.0))
        
        if status_full == "matched" and conf_full >= self.config.match_threshold:
            return resultado_full
        
        # Evaluar si el split mejora significativamente
        best_split = self._elegir_mejor_split(resultado_left, resultado_right)
        
        if self._split_mejora(resultado_full, best_split):
            return best_split
        
        # Si el split no mejora, quedarse con el full (aunque sea UNKNOWN/ambiguous)
        return resultado_full
    
    def _elegir_mejor_split(
        self,
        resultado_left: Dict[str, Any],
        resultado_right: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Elige el mejor resultado entre left y right."""
        conf_left = float(resultado_left.get("confianza", 0.0))
        conf_right = float(resultado_right.get("confianza", 0.0))
        
        # Preferir matched sobre unknown/ambiguous
        status_left = str(resultado_left.get("status", "unknown"))
        status_right = str(resultado_right.get("status", "unknown"))
        
        if status_left == "matched" and status_right != "matched":
            return resultado_left
        if status_right == "matched" and status_left != "matched":
            return resultado_right
        
        # Si ambos tienen el mismo status, elegir por confianza
        if conf_left >= conf_right:
            return resultado_left
        return resultado_right
    
    def _split_mejora(
        self,
        resultado_full: Dict[str, Any],
        best_split: Dict[str, Any]
    ) -> bool:
        """
        Determina si el split mejora significativamente al resultado full.
        
        Criterios:
        - El split debe tener mejor confianza (delta >= split_delta_min)
        - O el split debe ser "matched" cuando el full no lo es
        """
        conf_full = float(resultado_full.get("confianza", 0.0))
        conf_split = float(best_split.get("confianza", 0.0))
        
        status_full = str(resultado_full.get("status", "unknown"))
        status_split = str(best_split.get("status", "unknown"))
        
        # Si el split es matched y el full no, aceptar split
        if status_split == "matched" and status_full != "matched":
            return True
        
        # Si ambos son matched, verificar mejora de confianza
        if status_split == "matched" and status_full == "matched":
            return conf_split >= (conf_full + self.config.split_delta_min)
        
        # Si el split no es matched pero tiene mejor confianza significativa
        if conf_split >= (conf_full + self.config.split_delta_min):
            return True
        
        return False
    
    def deberia_intentar_split(
        self,
        resultado_full: Dict[str, Any],
        bbox_quality_score: float
    ) -> bool:
        """
        Determina si se debe intentar split basado en:
        - Calidad del bbox (probabilidad de mezcla)
        - Estado del resultado full (solo si es dudoso)
        """
        status_full = str(resultado_full.get("status", "unknown"))
        conf_full = float(resultado_full.get("confianza", 0.0))
        
        # Solo intentar split si el resultado full es dudoso
        if status_full == "matched" and conf_full >= self.config.match_threshold:
            return False
        
        # El resultado debe estar en rango dudoso para considerar split
        if conf_full < self.config.split_min_confianza or conf_full > self.config.split_max_confianza:
            return False
        
        # El bbox debe tener calidad suficiente (probablemente mezclado)
        return bbox_quality_score >= self.config.bbox_quality_threshold
