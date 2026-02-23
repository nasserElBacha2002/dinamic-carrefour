#!/usr/bin/env python3
"""
Detection Processor — Procesamiento de detecciones por frame.

Responsabilidades:
- Para cada detección: padding capado (A), inner-crop para identificación (B)
- Split vertical condicional basado en calidad de bbox y política de decisión
- 1 bbox = 1 decisión final (no doble conteo)
- Conteos por frame (Counter) + etiquetas para anotación
- Preparar lista de crops dudosos para Learning Manager
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from src.pipeline.crop_processor import CropProcessor
from src.pipeline.decision_policy import DecisionPolicy, DecisionPolicyConfig
from src.pipeline.bbox_quality import BBoxQualityScorer


class DetectionProcessor:
    def __init__(
        self,
        identificador: Any,  # SKUIdentifier
        crop_processor: CropProcessor,
        guardar_crops: bool = False,
        crops_dir: Optional[Path] = None,
        decision_policy: Optional[DecisionPolicy] = None,
        bbox_quality_scorer: Optional[BBoxQualityScorer] = None,
    ):
        self.identificador = identificador
        self.crop_processor = crop_processor
        self.guardar_crops = bool(guardar_crops)
        self.crops_dir = crops_dir
        
        # Política de decisión (genérica y escalable)
        self.decision_policy = decision_policy or DecisionPolicy()
        
        # Scorer de calidad de bbox (genérico)
        config = self.decision_policy.config
        self.bbox_scorer = bbox_quality_scorer or BBoxQualityScorer(
            aspect_weight=config.bbox_quality_aspect_weight,
            area_weight=config.bbox_quality_area_weight,
            yolo_conf_weight=config.bbox_quality_yolo_conf_weight,
            edge_weight=config.bbox_quality_edge_weight,
        )

    def procesar_detecciones_en_frame(
        self,
        frame: np.ndarray,
        detecciones: List[Dict[str, Any]],
        out_idx: int,
        t_sec: float,
        detector: Any,  # YOLODetector
    ) -> Tuple[Counter, Dict[int, str], List[Dict[str, Any]]]:
        """
        Procesa todas las detecciones de un frame.

        Returns:
            - conteo_frame: Counter(EAN -> cantidad) en este frame
            - etiquetas_sku: {idx_det -> label} para anotación
            - crops_para_learning: lista con dicts para guardar en learning (solo dudosos)
        """
        h, w = frame.shape[:2]

        conteo_frame: Counter = Counter()
        etiquetas_sku: Dict[int, str] = {}
        crops_para_learning: List[Dict[str, Any]] = []

        for i, det in enumerate(detecciones):
            if "bbox" not in det:
                continue

            x1, y1, x2, y2 = det["bbox"]
            box_w = int(x2 - x1)
            box_h = int(y2 - y1)
            if box_w <= 2 or box_h <= 2:
                continue

            # Padding dinámico del detector pero capado (Solución A)
            pad_raw = int(detector._padding_dinamico(x1, y1, x2, y2))
            x1p, y1p, x2p, y2p = self.crop_processor.calcular_bbox_padded(
                x1=x1, y1=y1, x2=x2, y2=y2,
                pad_raw=pad_raw, frame_w=w, frame_h=h
            )
            if x2p <= x1p or y2p <= y1p:
                continue

            det["bbox_padded"] = [x1p, y1p, x2p, y2p]

            # Crop padded (visual) -> sirve para guardar y revisar
            crop_padded = frame[y1p:y2p, x1p:x2p].copy()

            # Inner crop para identificación (Solución B)
            ix1, iy1, ix2, iy2 = self.crop_processor.inner_crop_rect(
                x1p, y1p, x2p, y2p, ratio=0.75
            )
            ix1 = max(0, ix1)
            iy1 = max(0, iy1)
            ix2 = min(w, ix2)
            iy2 = min(h, iy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            crop_for_id = frame[iy1:iy2, ix1:ix2].copy()
            crop_id = f"frame_{out_idx:05d}_crop_{i:03d}"

            # Guardado debug del crop (guardamos el PADDED)
            if self.guardar_crops and self.crops_dir:
                crop_name = f"{crop_id}.jpg"
                cv2.imwrite(
                    str(self.crops_dir / crop_name),
                    crop_padded,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

            # Identificación con política de decisión (1 bbox = 1 resultado final)
            resultado_final = self._identificar_crop_con_politica(
                frame=frame,
                bbox_padded=(x1p, y1p, x2p, y2p),
                crop_for_id=crop_for_id,
                crop_id=crop_id,
                det=det,
                w=w,
                h=h,
            )

            # Conteo: 1 bbox = 1 EAN (no doble conteo)
            ean_final = str(resultado_final.get("ean", "UNKNOWN"))
            status_final = str(resultado_final.get("status", "matched"))
            
            conteo_frame[ean_final] += 1

            # Learning: solo si es dudoso
            if status_final in ("unknown", "ambiguous"):
                crops_para_learning.append({
                    "crop_padded": crop_padded,
                    "crop_id": crop_id,
                    "det": det,
                    "resultado_sku": resultado_final,
                    "resultados_identificacion": [resultado_final],  # Solo el final
                    "out_idx": int(out_idx),
                    "t_sec": float(t_sec),
                    "j": 0,  # Siempre 0 porque es un solo resultado
                })

            # Etiqueta para anotación (1 por bbox)
            etiquetas_sku[i] = self.generar_etiqueta_sku([resultado_final])

        return conteo_frame, etiquetas_sku, crops_para_learning

    def _identificar_crop_con_politica(
        self,
        frame: np.ndarray,
        bbox_padded: Tuple[int, int, int, int],
        crop_for_id: np.ndarray,
        crop_id: str,
        det: Dict[str, Any],
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """
        Identifica un crop con política de decisión genérica.
        
        Flujo:
        1. Calcular embedding y packaging (una sola vez)
        2. Identificar crop completo
        3. Si es dudoso Y bbox tiene calidad baja → intentar split
        4. Si split mejora significativamente → usar split
        5. Retornar 1 resultado final
        
        Returns:
            Un único resultado final (no lista).
        """
        x1p, y1p, x2p, y2p = bbox_padded
        
        # 1. Identificar crop completo (siempre)
        resultado_full = self.identificador.identificar_crop_numpy(
            crop_for_id, crop_id=crop_id
        )
        
        # 2. Calcular calidad del bbox (genérico)
        bbox_quality = self.bbox_scorer.calcular_score_desde_deteccion(
            deteccion=det,
            frame_shape=(h, w),
            roi=None,  # TODO: pasar ROI si está disponible
        )
        
        # 3. Decidir si intentar split (basado en política)
        if self.decision_policy.deberia_intentar_split(resultado_full, bbox_quality):
            # Split vertical
            left, right = self.crop_processor.split_vertical(crop_for_id)
            
            # IMPORTANTE: Reutilizar la categoría del full (no recalcular packaging)
            categoria_full = resultado_full.get("categoria", "")
            
            # Identificar splits usando la misma categoría del full (evita recalcular packaging)
            res_left = self.identificador.identificar_crop_numpy(
                left, crop_id=f"{crop_id}_L", categoria_forzada=categoria_full
            )
            res_right = self.identificador.identificar_crop_numpy(
                right, crop_id=f"{crop_id}_R", categoria_forzada=categoria_full
            )
            
            # 4. Decidir resultado final (política de decisión)
            resultado_final = self.decision_policy.decidir_resultado_final(
                resultado_full=resultado_full,
                resultado_left=res_left,
                resultado_right=res_right,
            )
            
            return resultado_final
        
        # Sin split: retornar resultado full
        return resultado_full

    @staticmethod
    def generar_etiqueta_sku(resultados_identificacion: List[Dict[str, Any]]) -> str:
        """Genera etiqueta para anotación a partir del/los resultados."""
        if not resultados_identificacion:
            return "UNKNOWN"

        if len(resultados_identificacion) == 2:
            def _fmt(res: Dict[str, Any]) -> str:
                e = str(res.get("ean", "UNKNOWN"))
                if e != "UNKNOWN":
                    d = res.get("descripcion") or ""
                    return f"{e} {str(d)[:18]}".strip()
                return "UNKNOWN"

            l1 = _fmt(resultados_identificacion[0])
            l2 = _fmt(resultados_identificacion[1])
            if l1 == l2:
                return l1
            return f"L:{l1} | R:{l2}"

        best = resultados_identificacion[0]
        e = str(best.get("ean", "UNKNOWN"))
        if e != "UNKNOWN":
            d = best.get("descripcion") or ""
            return f"{e} {str(d)[:30]}".strip()

        conf = float(best.get("confianza") or 0.0)
        return f"UNKNOWN ({conf:.2f})"