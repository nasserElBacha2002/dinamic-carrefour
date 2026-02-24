#!/usr/bin/env python3
"""
Track Types — Tipos base para tracking.

Define los dataclasses y tipos utilizados por el sistema de tracking.
"""

from dataclasses import dataclass
from typing import Tuple

# Tipo para bounding box: (x1, y1, x2, y2) en píxeles
BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    """
    Detección de un producto en un frame.

    Attributes:
        bbox: Bounding box (x1, y1, x2, y2) en píxeles
        conf: Confianza de la detección (0.0-1.0)
        class_id: ID de clase del detector
        raw_label: Label original del detector
    """
    bbox: BBox
    conf: float
    class_id: int
    raw_label: str


@dataclass
class Track:
    """
    Track (rastreo) de un producto a lo largo del tiempo.

    Un track representa un producto físico real visible en múltiples frames.

    Attributes:
        track_id: ID único del track
        bbox: Bounding box actual (x1, y1, x2, y2)
        conf: Confianza actual de la detección
        age: Frames desde que se creó el track
        hits: Cantidad de frames donde se hizo match
        time_since_update: Frames desde la última actualización
        state: Estado del track ("ACTIVE" | "LOST" | "ENDED")
    """
    track_id: int
    bbox: BBox
    conf: float
    age: int
    hits: int
    time_since_update: int
    state: str  # "ACTIVE" | "LOST" | "ENDED"
