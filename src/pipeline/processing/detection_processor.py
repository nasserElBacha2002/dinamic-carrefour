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

from src.pipeline.processing.crop_processor import CropProcessor
from src.pipeline.processing.decision_policy import DecisionPolicy, DecisionPolicyConfig
from src.pipeline.processing.bbox_quality import BBoxQualityScorer


class DetectionProcessor:
    def __init__(
        self,
        identifier: Any,  # SKUIdentifier
        crop_processor: CropProcessor,
        save_crops: bool = False,
        crops_dir: Optional[Path] = None,
        decision_policy: Optional[DecisionPolicy] = None,
        bbox_quality_scorer: Optional[BBoxQualityScorer] = None,
    ):
        self.identifier = identifier
        self.crop_processor = crop_processor
        self.save_crops = bool(save_crops)
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

    def process_detections_in_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        out_idx: int,
        t_sec: float,
        detector: Any,  # YOLODetector
        sku_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Tuple[Counter, Dict[int, str], List[Dict[str, Any]]]:
        """
        Procesa todas las detecciones de un frame.

        Returns:
            - frame_count: Counter(EAN -> cantidad) en este frame
            - sku_labels: {idx_det -> label} para anotación
            - crops_for_learning: lista con dicts para guardar en learning (solo dudosos)
        """
        h, w = frame.shape[:2]

        frame_count: Counter = Counter()
        sku_labels: Dict[int, str] = {}
        crops_for_learning: List[Dict[str, Any]] = []

        for i, det in enumerate(detections):
            if "bbox" not in det:
                continue

            x1, y1, x2, y2 = det["bbox"]
            box_w = int(x2 - x1)
            box_h = int(y2 - y1)
            if box_w <= 2 or box_h <= 2:
                continue

            # Padding dinámico del detector pero capado (Solución A)
            pad_raw = int(detector._padding_dinamico(x1, y1, x2, y2))
            x1p, y1p, x2p, y2p = self.crop_processor.calculate_padded_bbox(
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
            
            # Mejorar crop_id para trazabilidad (CORRECCIÓN P1)
            # Si hay cache, significa que hay tracking activo
            track_id = None
            if sku_cache and i in sku_cache:
                # Intentar obtener track_id del cache si está disponible
                cache_meta = sku_cache[i].get("_meta", {})
                track_id = cache_meta.get("track_id")
            
            if track_id is not None:
                crop_id = f"frame_{out_idx:05d}_track_{track_id}_det_{i:03d}"
            else:
                crop_id = f"frame_{out_idx:05d}_det_{i:03d}"

            # Guardado debug del crop (guardamos el PADDED)
            if self.save_crops and self.crops_dir:
                crop_name = f"{crop_id}.jpg"
                cv2.imwrite(
                    str(self.crops_dir / crop_name),
                    crop_padded,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

            # Identificación: usar cache si está disponible (CORRECCIÓN P0: evitar doble identificación)
            if sku_cache and i in sku_cache:
                # Reutilizar resultado del cache (ya fue identificado para votos)
                final_result = dict(sku_cache[i])  # Copia para no modificar el original
                final_result.pop("_meta", None)  # Limpiar metadata interna
            else:
                # Identificación con política de decisión (1 bbox = 1 resultado final)
                final_result = self._identify_crop_with_policy(
                    frame=frame,
                    bbox_padded=(x1p, y1p, x2p, y2p),
                    crop_for_id=crop_for_id,
                    crop_id=crop_id,
                    det=det,
                    w=w,
                    h=h,
                )

            # Conteo: 1 bbox = 1 EAN (no doble conteo)
            ean_final = str(final_result.get("ean", "UNKNOWN"))
            status_final = str(final_result.get("status", "matched"))
            
            frame_count[ean_final] += 1

            # Learning: solo si es dudoso
            if status_final in ("unknown", "ambiguous"):
                crops_for_learning.append({
                    "crop_padded": crop_padded,
                    "crop_id": crop_id,
                    "det": det,
                    "sku_result": final_result,
                    "identification_results": [final_result],  # Solo el final
                    "out_idx": int(out_idx),
                    "t_sec": float(t_sec),
                    "j": 0,  # Siempre 0 porque es un solo resultado
                })

            # Etiqueta para anotación (1 por bbox)
            sku_labels[i] = self.generate_sku_label([final_result])

        return frame_count, sku_labels, crops_for_learning

    def _identify_crop_with_policy(
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
        full_result = self.identifier.identificar_crop_numpy(
            crop_for_id, crop_id=crop_id
        )
        
        # 2. Calcular calidad del bbox (genérico)
        bbox_quality = self.bbox_scorer.calculate_score_from_detection(
            detection=det,
            frame_shape=(h, w),
            roi=None,  # TODO: pasar ROI si está disponible
        )
        
        # 3. Decidir si intentar split (basado en política)
        if self.decision_policy.should_attempt_split(full_result, bbox_quality):
            # Split vertical
            left, right = self.crop_processor.split_vertical(crop_for_id)
            
            # IMPORTANTE: Reutilizar la categoría del full (no recalcular packaging)
            categoria_full = full_result.get("categoria", "")
            
            # Identificar splits usando la misma categoría del full (evita recalcular packaging)
            left_result = self.identifier.identificar_crop_numpy(
                left, crop_id=f"{crop_id}_L", categoria_forzada=categoria_full
            )
            right_result = self.identifier.identificar_crop_numpy(
                right, crop_id=f"{crop_id}_R", categoria_forzada=categoria_full
            )
            
            # 4. Decidir resultado final (política de decisión)
            final_result = self.decision_policy.decide_final_result(
                full_result=full_result,
                left_result=left_result,
                right_result=right_result,
            )
            
            return final_result
        
        # Sin split: retornar resultado full
        return full_result

    def process_individual_detection(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        out_idx: int,
        t_sec: float,
        detector: Any,  # YOLODetector
    ) -> Optional[Dict[str, Any]]:
        """
        Procesa una detección individual y retorna el resultado SKU.
        
        Útil para tracking: permite obtener resultado SKU por detección
        para acumular votos por track.
        
        Returns:
            Dict con resultado SKU (ean, status, confianza, top_matches) o None si falla
        """
        if "bbox" not in detection:
            return None
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = detection["bbox"]
        box_w = int(x2 - x1)
        box_h = int(y2 - y1)
        if box_w <= 2 or box_h <= 2:
            return None
        
        # Padding dinámico del detector pero capado
        pad_raw = int(detector._padding_dinamico(x1, y1, x2, y2))
        x1p, y1p, x2p, y2p = self.crop_processor.calcular_bbox_padded(
            x1=x1, y1=y1, x2=x2, y2=y2,
            pad_raw=pad_raw, frame_w=w, frame_h=h
        )
        if x2p <= x1p or y2p <= y1p:
            return None
        
        # Inner crop para identificación
        ix1, iy1, ix2, iy2 = self.crop_processor.inner_crop_rect(
            x1p, y1p, x2p, y2p, ratio=0.75
        )
        ix1 = max(0, ix1)
        iy1 = max(0, iy1)
        ix2 = min(w, ix2)
        iy2 = min(h, iy2)
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        
        crop_for_id = frame[iy1:iy2, ix1:ix2].copy()
        crop_id = f"frame_{out_idx:05d}_crop_track"
        
        # Identificación con política de decisión
        final_result = self._identify_crop_with_policy(
            frame=frame,
            bbox_padded=(x1p, y1p, x2p, y2p),
            crop_for_id=crop_for_id,
            crop_id=crop_id,
            det=detection,
            w=w,
            h=h,
        )
        
        return final_result

    @staticmethod
    def generate_sku_label(identification_results: List[Dict[str, Any]]) -> str:
        """Genera etiqueta para anotación a partir del/los resultados."""
        if not identification_results:
            return "UNKNOWN"

        if len(identification_results) == 2:
            def _fmt(res: Dict[str, Any]) -> str:
                e = str(res.get("ean", "UNKNOWN"))
                if e != "UNKNOWN":
                    d = res.get("descripcion") or ""
                    return f"{e} {str(d)[:18]}".strip()
                return "UNKNOWN"

            l1 = _fmt(identification_results[0])
            l2 = _fmt(identification_results[1])
            if l1 == l2:
                return l1
            return f"L:{l1} | R:{l2}"

        best = identification_results[0]
        e = str(best.get("ean", "UNKNOWN"))
        if e != "UNKNOWN":
            d = best.get("descripcion") or ""
            return f"{e} {str(d)[:30]}".strip()

        conf = float(best.get("confianza") or 0.0)
        return f"UNKNOWN ({conf:.2f})"