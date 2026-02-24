#!/usr/bin/env python3
"""
Track Setup — Configuración inicial de tracking.

Responsabilidades:
- Inicializar tracker y vote_accumulator
- Configurar filtros y parámetros
- Retornar componentes listos para usar
"""

from __future__ import annotations

from typing import Tuple, Optional, Any, Dict

from src.tracking.sort_like_tracker import SortLikeTracker
from src.tracking.track_vote_accumulator import TrackVoteAccumulator, DecisionProfile
from src.pipeline.tracking.track_integration import (
    convert_yolo_detections_to_detections,
    export_tracks_debug,
    filter_detections_by_area,
)


class TrackSetup:
    """
    Configuración inicial de tracking.
    """
    
    @staticmethod
    def create(
        track_iou: float = 0.4,
        track_min_hits: int = 3,
        track_max_age: int = 15,
        decision_profile: DecisionProfile = DecisionProfile.WAREHOUSE_BALANCED,
        min_conf_create: float = 0.55,
        min_rel_area: float = 0.002,
        max_rel_area: float = 0.25,
    ) -> Tuple[
        SortLikeTracker,
        TrackVoteAccumulator,
        Dict[str, Any],
    ]:
        """
        Crea y configura los componentes de tracking.
        
        Args:
            track_iou: IoU threshold para tracking
            track_min_hits: Mínimo hits para track válido
            track_max_age: Máximo frames sin update antes de terminar track
            decision_profile: Perfil de decisión (SHELF_STRICT, WAREHOUSE_LENIENT, etc.)
            min_conf_create: Confianza mínima para crear tracks nuevos
            min_rel_area: Área relativa mínima para filtrar detecciones
            max_rel_area: Área relativa máxima para filtrar detecciones
        
        Returns:
            Tuple de (tracker, vote_accumulator, config_dict)
        """
        # Crear tracker
        tracker = SortLikeTracker(
            iou_threshold=float(track_iou),
            min_hits=int(track_min_hits),
            max_age=int(track_max_age),
            min_conf_create=min_conf_create,
        )
        
        # Crear vote accumulator con perfil
        vote_accumulator = TrackVoteAccumulator(
            profile=decision_profile,
        )
        
        # Configuración para uso en runtime
        config = {
            "min_rel_area": min_rel_area,
            "max_rel_area": max_rel_area,
            "convert_detections": convert_yolo_detections_to_detections,
            "export_tracks_debug": export_tracks_debug,
            "filter_by_area": filter_detections_by_area,
        }
        
        print("   🎯 Tracking temporal activado (Sprint 3)")
        print(f"      Filtros: min_conf_create={min_conf_create} (solo para crear tracks), area_rel=[{min_rel_area}, {max_rel_area}]")
        print("   📊 Acumulador de votos activado (Sprint 3.2)")
        
        return tracker, vote_accumulator, config
