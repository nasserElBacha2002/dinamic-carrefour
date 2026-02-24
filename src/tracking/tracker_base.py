#!/usr/bin/env python3
"""
Tracker Base — Interfaz base para trackers.

Define la interfaz que deben implementar todos los trackers.
"""

from typing import List, Set
from .track_types import Detection, Track


class TrackerBase:
    """
    Interfaz base para trackers.

    Un tracker asigna identidad persistente (track_id) a detecciones
    a lo largo de múltiples frames.
    """

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Actualiza el tracker con las detecciones del frame actual.

        Args:
            detections: Lista de detecciones del frame actual

        Returns:
            Lista de tracks activos (state == "ACTIVE")
        """
        raise NotImplementedError

    def get_ended_track_ids(self) -> Set[int]:
        """
        Obtiene los IDs de tracks que terminaron desde la última actualización.

        Returns:
            Set de track_ids que terminaron (state == "ENDED")
        """
        raise NotImplementedError
