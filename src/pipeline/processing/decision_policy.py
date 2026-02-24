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
    
    def decide_final_result(
        self,
        full_result: Dict[str, Any],
        left_result: Optional[Dict[str, Any]] = None,
        right_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Decide el resultado final para un bbox.
        
        Regla: 1 bbox = 1 decisión final.
        
        Si hay split, solo se acepta si mejora significativamente al resultado full.
        
        Args:
            full_result: Resultado del crop completo.
            left_result: Resultado del crop izquierdo (si hubo split).
            right_result: Resultado del crop derecho (si hubo split).
        
        Returns:
            Un único resultado final (el mejor).
        """
        # Si no hay split, retornar el resultado full directamente
        if left_result is None or right_result is None:
            return full_result
        
        # Si el resultado full es matched y con buena confianza, no usar split
        status_full = str(full_result.get("status", "unknown"))
        conf_full = float(full_result.get("confianza", 0.0))
        
        if status_full == "matched" and conf_full >= self.config.match_threshold:
            return full_result
        
        # Evaluar si el split mejora significativamente
        best_split = self._choose_best_split(left_result, right_result)
        
        if self._split_improves(full_result, best_split):
            return best_split
        
        # Si el split no mejora, quedarse con el full (aunque sea UNKNOWN/ambiguous)
        return full_result
    
    def _choose_best_split(
        self,
        left_result: Dict[str, Any],
        right_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Elige el mejor resultado entre left y right."""
        conf_left = float(left_result.get("confianza", 0.0))
        conf_right = float(right_result.get("confianza", 0.0))
        
        # Preferir matched sobre unknown/ambiguous
        status_left = str(left_result.get("status", "unknown"))
        status_right = str(right_result.get("status", "unknown"))
        
        if status_left == "matched" and status_right != "matched":
            return left_result
        if status_right == "matched" and status_left != "matched":
            return right_result
        
        # Si ambos tienen el mismo status, elegir por confianza
        if conf_left >= conf_right:
            return left_result
        return right_result
    
    def _split_improves(
        self,
        full_result: Dict[str, Any],
        best_split: Dict[str, Any]
    ) -> bool:
        """
        Determina si el split mejora significativamente al resultado full.
        
        Criterios:
        - El split debe tener mejor confianza (delta >= split_delta_min)
        - O el split debe ser "matched" cuando el full no lo es
        """
        conf_full = float(full_result.get("confianza", 0.0))
        conf_split = float(best_split.get("confianza", 0.0))
        
        status_full = str(full_result.get("status", "unknown"))
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
    
    def should_attempt_split(
        self,
        full_result: Dict[str, Any],
        bbox_quality_score: float
    ) -> bool:
        """
        Determina si se debe intentar split basado en:
        - Calidad del bbox (probabilidad de mezcla)
        - Estado del resultado full (solo si es dudoso)
        """
        status_full = str(full_result.get("status", "unknown"))
        conf_full = float(full_result.get("confianza", 0.0))
        
        # Solo intentar split si el resultado full es dudoso
        if status_full == "matched" and conf_full >= self.config.match_threshold:
            return False
        
        # El resultado debe estar en rango dudoso para considerar split
        if conf_full < self.config.split_min_confianza or conf_full > self.config.split_max_confianza:
            return False
        
        # El bbox debe tener calidad suficiente (probablemente mezclado)
        return bbox_quality_score >= self.config.bbox_quality_threshold
