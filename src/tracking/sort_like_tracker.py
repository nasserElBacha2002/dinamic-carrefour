#!/usr/bin/env python3
"""
SORT-like Tracker — Implementación MVP de tracking basada en IoU.

Esta es una implementación simple y sin dependencias externas que permite
avanzar con Sprint 3.1. En el futuro puede reemplazarse por ByteTrack
sin tocar el resto del pipeline.
"""

from typing import List, Dict, Set, Tuple
from .track_types import Detection, Track, BBox
from .tracker_base import TrackerBase


def iou(a: BBox, b: BBox) -> float:
    """
    Calcula Intersection over Union (IoU) entre dos bounding boxes.

    Args:
        a: Bounding box (x1, y1, x2, y2)
        b: Bounding box (x1, y1, x2, y2)

    Returns:
        IoU entre 0.0 y 1.0
    """
    x1_a, y1_a, x2_a, y2_a = a
    x1_b, y1_b, x2_b, y2_b = b

    # Calcular intersección
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    area_i = (x2_i - x1_i) * (y2_i - y1_i)

    # Calcular áreas
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    area_u = area_a + area_b - area_i

    if area_u <= 0:
        return 0.0

    return area_i / area_u


class SortLikeTracker(TrackerBase):
    """
    Tracker simple basado en matching por IoU (SORT-like).

    Algoritmo:
    1. Para cada frame, intenta matchear detecciones con tracks existentes por IoU
    2. Si no hay match, crea un nuevo track
    3. Si un track no se actualiza por max_age frames, se marca como ENDED

    Parámetros:
        iou_threshold: IoU mínimo para considerar match (default: 0.4)
        min_hits: Mínimo de hits para considerar track válido (default: 3)
        max_age: Máximo de frames sin actualizar antes de terminar track (default: 15)
    """

    def __init__(
        self,
        iou_threshold: float = 0.4,
        min_hits: int = 3,
        max_age: int = 15,
        min_conf_create: float = 0.55,
    ):
        self.iou_threshold = float(iou_threshold)
        self.min_hits = int(min_hits)
        self.max_age = int(max_age)
        self.min_conf_create = float(min_conf_create)

        self._next_id = 1
        self._tracks: Dict[int, Track] = {}
        self._ended: Set[int] = set()
        
        # Sprint 3.1+ : Asignaciones del último frame (det_idx -> track_id)
        self._last_assignments: Dict[int, int] = {}

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Actualiza el tracker con las detecciones del frame actual.

        Args:
            detections: Lista de detecciones del frame actual (puede estar vacía)

        Returns:
            Lista de tracks activos
        """
        # P1.1: No limpiar _ended aquí - se consume explícitamente en get_ended_track_ids()
        # self._ended.clear()  # ← Removido: ahora se limpia en get_ended_track_ids() (edge-trigger)
        self._last_assignments.clear()  # Reset asignaciones del frame anterior

        # 1) Marcar todos los tracks como no actualizados
        for tid, trk in self._tracks.items():
            trk.age += 1
            trk.time_since_update += 1
            if trk.time_since_update > 0:
                trk.state = "LOST"
            else:
                trk.state = "ACTIVE"

        # 2) Matching greedy por IoU (solo si hay detecciones)
        if detections:
            unmatched_det = set(range(len(detections)))
            unmatched_tracks = set(self._tracks.keys())

            # Construir todos los pares (IoU, det_idx, track_id)
            pairs: List[Tuple[float, int, int]] = []
            for di, det in enumerate(detections):
                for tid, trk in self._tracks.items():
                    iou_score = iou(det.bbox, trk.bbox)
                    pairs.append((iou_score, di, tid))

            # Ordenar por IoU descendente
            pairs.sort(reverse=True, key=lambda x: x[0])

            # Matching greedy
            for iou_score, di, tid in pairs:
                if iou_score < self.iou_threshold:
                    break  # No hay más matches posibles

                if di not in unmatched_det or tid not in unmatched_tracks:
                    continue

                # Match detección -> track
                det = detections[di]
                trk = self._tracks[tid]

                # Actualizar track
                trk.bbox = det.bbox
                trk.conf = det.conf
                trk.hits += 1
                trk.time_since_update = 0
                trk.state = "ACTIVE"

                # Guardar asignación
                self._last_assignments[di] = tid

                unmatched_det.remove(di)
                unmatched_tracks.remove(tid)

            # 3) Crear nuevos tracks para detecciones sin match
            # 🔒 SOLO crear tracks nuevos con alta confianza (low-conf puede matchear tracks existentes)
            for di in unmatched_det:
                det = detections[di]
                
                # Gate: no crear tracks nuevos con confianza baja
                if det.conf < self.min_conf_create:
                    continue
                
                tid = self._next_id
                self._next_id += 1

                self._tracks[tid] = Track(
                    track_id=tid,
                    bbox=det.bbox,
                    conf=det.conf,
                    age=1,
                    hits=1,
                    time_since_update=0,
                    state="ACTIVE",
                )
                
                # Guardar asignación (nuevo track)
                self._last_assignments[di] = tid

        # 4) Terminar tracks que excedieron max_age
        to_delete = []
        for tid, trk in self._tracks.items():
            if trk.time_since_update > self.max_age:
                trk.state = "ENDED"
                self._ended.add(tid)
                to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

        # 5) Retornar tracks activos
        active = [t for t in self._tracks.values() if t.state == "ACTIVE"]
        return active

    def get_last_assignments(self) -> Dict[int, int]:
        """
        Obtiene las asignaciones del último frame (det_idx -> track_id).

        Returns:
            Dict con {det_idx: track_id} del último update()
        """
        return dict(self._last_assignments)

    def is_valid_track(self, track: Track) -> bool:
        """
        Determina si un track es válido (tiene suficientes hits).

        Args:
            track: Track a validar

        Returns:
            True si hits >= min_hits
        """
        return track.hits >= self.min_hits

    def get_ended_track_ids(self) -> Set[int]:
        """
        Obtiene los IDs de tracks que terminaron desde la última actualización.
        
        P1.1: Edge-trigger explícito - consume el buffer de ended.
        Cada llamada retorna los ended del último update() y limpia el buffer.
        
        Returns:
            Set de track_ids que terminaron en el último update()
        """
        ended = set(self._ended)
        self._ended.clear()  # Consumir evento explícitamente (edge-trigger)
        return ended

    def flush_active_track_ids(self) -> Set[int]:
        """
        Finaliza todos los tracks activos y devuelve sus IDs.
        
        Útil para finalizar tracks al final del video cuando no alcanzaron max_age.
        Los tracks se marcan como ENDED y se limpian del estado interno.
        
        Returns:
            Set de track_ids que estaban activos y fueron finalizados
        """
        ids = set(self._tracks.keys())
        self._ended.update(ids)
        
        # Marcar todos como ENDED antes de limpiar
        for tid in ids:
            if tid in self._tracks:
                self._tracks[tid].state = "ENDED"
        
        # Limpiar estado interno (pero los IDs quedan en _ended para consulta)
        self._tracks.clear()
        
        return ids
