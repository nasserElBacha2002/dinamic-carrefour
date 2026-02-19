#!/usr/bin/env python3
"""
Learning Manager — Gestiona el dataset evolutivo por ejecución.

Cada ejecución genera:
  - output/<video_timestamp>/learning/
      unknown/          # Crops no identificados
      ambiguous/         # Crops con top1-top2 muy cercanos
      metadata/          # JSONL con toda la información
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class LearningManager:
    """
    Gestiona la captura y estructuración de datos de aprendizaje por ejecución.
    
    Cada crop dudoso se guarda con metadata completa para revisión humana rápida.
    """

    def __init__(
        self,
        output_dir: Path,
        execution_id: str,
        video_path: str,
        execution_params: Optional[Dict] = None
    ):
        """
        Args:
            output_dir: Directorio base de output (ej: output/IMG_2199_20260219_171424/)
            execution_id: ID único de la ejecución
            video_path: Ruta al video procesado
            execution_params: Parámetros de la ejecución (thresholds, etc.)
        """
        self.output_dir = Path(output_dir)
        self.execution_id = execution_id
        self.video_path = str(video_path)
        self.execution_params = execution_params or {}
        
        # Directorios de aprendizaje
        self.learning_dir = self.output_dir / "learning"
        self.unknown_dir = self.learning_dir / "unknown"
        self.ambiguous_dir = self.learning_dir / "ambiguous"
        self.metadata_dir = self.learning_dir / "metadata"
        
        # Crear directorios
        for d in [self.unknown_dir, self.ambiguous_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Archivo JSONL de metadata
        self.metadata_file = self.metadata_dir / "crops_index.jsonl"
        
        # Contadores
        self.unknown_count = 0
        self.ambiguous_count = 0
        self.total_crops_saved = 0
        
        # Guardar metadata de ejecución
        self._guardar_execution_meta()

    def _guardar_execution_meta(self):
        """Guarda metadata general de la ejecución."""
        meta = {
            "execution_id": self.execution_id,
            "video_path": self.video_path,
            "video_name": Path(self.video_path).name,
            "timestamp": datetime.now().isoformat(),
            "execution_params": self.execution_params,
        }
        
        meta_file = self.metadata_dir / "execution_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def guardar_crop_dudoso(
        self,
        crop: np.ndarray,
        crop_id: str,
        frame_path: Optional[str] = None,
        frame_idx: Optional[int] = None,
        detection_info: Optional[Dict] = None,
        packaging_info: Optional[Dict] = None,
        sku_info: Optional[Dict] = None,
        decision: str = "unknown"  # "unknown" | "ambiguous"
    ) -> Optional[str]:
        """
        Guarda un crop dudoso con toda su metadata.
        
        Args:
            crop: Imagen del crop (numpy array BGR)
            crop_id: ID único del crop (ej: "frame_003_crop_001")
            frame_path: Ruta al frame original (opcional)
            frame_idx: Índice del frame (opcional)
            detection_info: Info de detección YOLO
            packaging_info: Info de categorización packaging
            sku_info: Info de identificación SKU (top-K candidatos)
            decision: "unknown" o "ambiguous"
        
        Returns:
            Ruta al crop guardado o None si falla
        """
        if decision not in ["unknown", "ambiguous"]:
            return None
        
        # Determinar directorio y nombre de archivo
        target_dir = self.unknown_dir if decision == "unknown" else self.ambiguous_dir
        crop_filename = f"{crop_id}.jpg"
        crop_path = target_dir / crop_filename
        
        # Guardar imagen
        try:
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            print(f"  ⚠️ Error guardando crop {crop_id}: {e}")
            return None
        
        # Construir metadata completa
        metadata = {
            "crop_id": crop_id,
            "timestamp": datetime.now().isoformat(),
            "execution_id": self.execution_id,
            
            "detection": detection_info or {},
            "packaging": packaging_info or {},
            "sku_identification": sku_info or {},
            
            "decision": decision,
            "frame_path": frame_path,
            "frame_idx": frame_idx,
            
            "paths": {
                "crop": str(crop_path.relative_to(self.output_dir)),
                "frame": frame_path,
            },
            
            "review": {
                "status": "pending",
                "assigned_ean": None,
                "reviewed_at": None,
                "reviewer": None,
                "notes": None,
            }
        }
        
        # Guardar en JSONL
        try:
            with open(self.metadata_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  ⚠️ Error guardando metadata {crop_id}: {e}")
        
        # Actualizar contadores
        if decision == "unknown":
            self.unknown_count += 1
        else:
            self.ambiguous_count += 1
        self.total_crops_saved += 1
        
        return str(crop_path)

    def resumen(self) -> Dict[str, Any]:
        """Retorna un resumen de los crops guardados."""
        return {
            "execution_id": self.execution_id,
            "total_crops_saved": self.total_crops_saved,
            "unknown_count": self.unknown_count,
            "ambiguous_count": self.ambiguous_count,
            "learning_dir": str(self.learning_dir.relative_to(self.output_dir.parent)),
            "metadata_file": str(self.metadata_file.relative_to(self.output_dir.parent)),
        }

    def __repr__(self) -> str:
        return f"LearningManager(execution_id={self.execution_id}, crops={self.total_crops_saved})"
