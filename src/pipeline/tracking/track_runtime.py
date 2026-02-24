#!/usr/bin/env python3
"""
Track Runtime — Maneja la lógica de tracking durante el procesamiento de frames.

Responsabilidades:
- Procesar frame con tracking (filtrado, actualización, votos)
- Mantener cache de identificación
- Finalizar tracks que terminan
- Retornar detecciones procesadas y cache
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

from src.tracking.sort_like_tracker import SortLikeTracker
from src.tracking.track_vote_accumulator import TrackVoteAccumulator, TrackDecision
from src.pipeline.processing.detection_processor import DetectionProcessor
from src.detector.yolo_detector import YOLODetector
from src.pipeline.tracking.track_integration import (
    convert_yolo_detections_to_detections,
    export_tracks_debug,
    filter_detections_by_area,
)


class TrackRuntime:
    """
    Maneja la lógica de tracking durante el procesamiento de frames.
    
    Encapsula toda la complejidad de tracking para que engine.py sea más limpio.
    """
    
    def __init__(
        self,
        tracker: SortLikeTracker,
        vote_accumulator: TrackVoteAccumulator,
        detection_processor: DetectionProcessor,
        detector: YOLODetector,
        min_rel_area: float,
        max_rel_area: float,
        output_dir: Any,  # Path
    ):
        self.tracker = tracker
        self.vote_accumulator = vote_accumulator
        self.detection_processor = detection_processor
        self.detector = detector
        self.min_rel_area = min_rel_area
        self.max_rel_area = max_rel_area
        self.output_dir = output_dir
        
        # Funciones helper
        self._convert_detections = convert_yolo_detections_to_detections
        self._export_tracks_debug = export_tracks_debug
        self._filter_by_area = filter_detections_by_area
        
        # Estado interno
        self._track_decisions: Dict[int, TrackDecision] = {}
    
    def process_frame(
        self,
        frame: np.ndarray,
        dets: List[Dict[str, Any]],
        out_idx: int,
        t_sec: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Any]]:
        """
        Procesa un frame con tracking.
        
        Args:
            frame: Frame actual
            dets: Detecciones YOLO del frame
            out_idx: Índice del frame
            t_sec: Tiempo en segundos
        
        Returns:
            Tuple de:
            - dets_to_use: Detecciones filtradas para procesar
            - sku_cache_by_det: Cache de identificación (det_idx -> resultado)
            - frame_info: Info del frame para tqdm (tracks, valid, decisions)
        """
        # 1) Filtrar detecciones por área relativa + bordes + aspecto
        frame_h, frame_w = frame.shape[:2]
        filtered_dets = self._filter_by_area(
            dets if dets else [],
            frame_w=frame_w,
            frame_h=frame_h,
            min_rel_area=self.min_rel_area,
            max_rel_area=self.max_rel_area,
        ) if dets else []
        
        # 2) Convertir y actualizar tracker (SIEMPRE, aunque esté vacío)
        detections = self._convert_detections(filtered_dets)
        tracks_active = self.tracker.update(detections)
        
        # 3) Obtener asignaciones del tracker
        assignments = self.tracker.get_last_assignments()
        
        # 4) Exportar debug de tracks
        if filtered_dets or tracks_active:
            self._export_tracks_debug(
                output_dir=self.output_dir,
                frame_idx=int(out_idx),
                detections=filtered_dets if filtered_dets else [],
                tracks=tracks_active,
                assignments=assignments,
                min_hits=self.tracker.min_hits,
            )
        
        # 5) Identificación 1 vez + votos (con cache)
        sku_cache_by_det: Dict[int, Dict[str, Any]] = {}
        
        if self.vote_accumulator and filtered_dets:
            for det_idx, det in enumerate(filtered_dets):
                track_id = assignments.get(det_idx)
                if track_id is None:
                    continue  # Detección sin track asignado
                
                # Procesar detección individual para obtener resultado SKU (1 sola vez)
                sku_result = self.detection_processor.process_individual_detection(
                    frame=frame,
                    detection=det,
                    out_idx=int(out_idx),
                    t_sec=float(t_sec),
                    detector=self.detector,
                )
                
                if sku_result:
                    # Guardar en cache para reutilizar después (con metadata de track_id)
                    sku_result["_meta"] = {"track_id": track_id}
                    sku_cache_by_det[det_idx] = sku_result
                    
                    ean = str(sku_result.get("ean", "UNKNOWN"))
                    # Obtener similitud del top1 match
                    top_matches = sku_result.get("top_matches", [])
                    sim = float(top_matches[0].get("similitud", 0.0)) if top_matches else 0.0
                    
                    # Acumular voto
                    self.vote_accumulator.add(
                        track_id=track_id,
                        sku_pred=ean,
                        sim=sim,
                        frame_idx=int(out_idx),
                        meta={"status": sku_result.get("status", "unknown")},
                    )
        
        # 6) Finalizar tracks que terminaron en este update
        ended_track_ids = [
            tid for tid in self.tracker.get_ended_track_ids()
            if tid not in self._track_decisions
        ]
        for tid in ended_track_ids:
            decision = self.vote_accumulator.finalize(tid, ended_reason="ended_in_update")
            if decision:
                self._track_decisions[tid] = decision
        
        # 7) Info para tqdm
        frame_info = {
            "tracks": len(tracks_active),
            "valid": sum(1 for t in tracks_active if self.tracker.is_valid_track(t)),
            "decisions": len(self._track_decisions),
        }
        
        return filtered_dets, sku_cache_by_det, frame_info
    
    def finalize_remaining_tracks(self) -> None:
        """
        Finaliza todos los tracks activos al final del video.
        """
        remaining_track_ids = self.tracker.flush_active_track_ids()
        if remaining_track_ids:
            for tid in remaining_track_ids:
                if tid not in self._track_decisions:
                    decision = self.vote_accumulator.finalize(tid, ended_reason="video_end")
                    if decision:
                        self._track_decisions[tid] = decision
    
    def get_track_decisions(self) -> Dict[int, TrackDecision]:
        """Retorna las decisiones finales de todos los tracks."""
        return self._track_decisions
