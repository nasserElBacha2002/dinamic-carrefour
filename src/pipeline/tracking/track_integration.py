#!/usr/bin/env python3
"""
Track Integration — Utilidades para integrar tracking en el pipeline.

Funciones helper para convertir entre formatos y exportar debug.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any

from src.tracking.track_types import Detection, Track


def bbox_rel_area(bbox: tuple, frame_w: int, frame_h: int) -> float:
    """
    Calcula el área relativa de un bbox respecto al frame completo.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        frame_w: Ancho del frame
        frame_h: Alto del frame
    
    Returns:
        Área relativa (0.0 a 1.0)
    """
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    return area / float(frame_w * frame_h)


def filter_detections_by_area(
    yolo_detections: List[Dict[str, Any]],
    frame_w: int,
    frame_h: int,
    min_rel_area: float = 0.002,
    max_rel_area: float = 0.25,
    border_px: int = 8,
) -> List[Dict[str, Any]]:
    """
    Filtra detecciones por área relativa del bbox y otros criterios.
    
    Elimina:
    - Bboxes muy pequeños (ruido)
    - Bboxes muy grandes (no son productos reales en góndola)
    - Bboxes que tocan bordes y son grandes (probablemente "grupo/estante")
    - Bboxes muy panorámicos (aspect ratio alto) y grandes
    
    Args:
        yolo_detections: Lista de detecciones YOLO
        frame_w: Ancho del frame
        frame_h: Alto del frame
        min_rel_area: Área relativa mínima (default: 0.002 = 0.2%)
        max_rel_area: Área relativa máxima (default: 0.25 = 25%)
        border_px: Píxeles de margen para considerar "toca borde" (default: 8)
    
    Returns:
        Lista filtrada de detecciones
    """
    filtradas = []
    for det in yolo_detections:
        if "bbox" not in det:
            continue
        
        bbox = det["bbox"]
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = bbox
        rel_area = bbox_rel_area(tuple(bbox), frame_w, frame_h)
        
        # Filtro 1: Área relativa mínima
        if rel_area < min_rel_area:
            continue  # Muy chico = ruido
        
        # Filtro 2: Área relativa máxima
        if rel_area > max_rel_area:
            continue  # Muy grande = no es producto real
        
        # Filtro 3: Toca bordes y es grande (probablemente "grupo/estante")
        touches_border = (
            x1 <= border_px or y1 <= border_px or
            x2 >= (frame_w - border_px) or y2 >= (frame_h - border_px)
        )
        if touches_border and rel_area > 0.12:
            continue  # Toca borde y es grande = probablemente grupo
        
        # Filtro 4: Aspecto muy panorámico y grande (probablemente "fila completa")
        bw = x2 - x1
        bh = y2 - y1
        if bh > 0:
            aspect = bw / float(bh)
            if aspect > 3.2 and rel_area > 0.10:
                continue  # Muy panorámico y grande = probablemente fila completa
        
        filtradas.append(det)
    
    return filtradas


def convert_yolo_detections_to_detections(yolo_detections: List[Dict[str, Any]]) -> List[Detection]:
    """
    Convierte detecciones de YOLO (formato dict) a Detection dataclass.

    Args:
        yolo_detections: Lista de dicts con formato YOLO:
            {
                "bbox": [x1, y1, x2, y2],
                "confianza": float,
                "class_id": int,
                "raw_label": str
            }

    Returns:
        Lista de Detection objects
    """
    detections = []
    for det in yolo_detections:
        if "bbox" not in det:
            continue

        bbox = tuple(det["bbox"])  # Convertir a tuple
        if len(bbox) != 4:
            continue

        detection = Detection(
            bbox=bbox,
            conf=float(det.get("confianza", 0.0)),
            class_id=int(det.get("class_id", -1)),
            raw_label=str(det.get("raw_label", "product")),
        )
        detections.append(detection)

    return detections


def export_tracks_debug(
    output_dir: Path,
    frame_idx: int,
    detections: List[Dict[str, Any]],
    tracks: List[Track],
    assignments: Dict[int, int],  # det_idx -> track_id del tracker
    min_hits: int = 3,  # Para marcar tracks válidos
) -> None:
    """
    Exporta información de debug de tracks por frame.

    Crea/actualiza tracks_debug.csv con:
    - frame_idx
    - track_id (o -1 si no tiene track)
    - bbox
    - confianza
    - estado del track
    - is_valid (hits >= min_hits)

    Args:
        output_dir: Directorio de output
        frame_idx: Índice del frame
        detections: Detecciones originales de YOLO
        tracks: Tracks activos
        assignments: Asignaciones del tracker (det_idx -> track_id)
        min_hits: Mínimo hits para considerar track válido
    """
    debug_file = output_dir / "tracks_debug.csv"
    file_exists = debug_file.exists()

    # Crear dict de tracks por ID para lookup rápido
    tracks_by_id = {t.track_id: t for t in tracks}

    with open(debug_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header solo si es nuevo archivo
        if not file_exists:
            writer.writerow([
                "frame_idx",
                "det_idx",
                "track_id",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "confianza",
                "track_state",
                "track_hits",
                "track_age",
                "track_valid",
            ])

        # Escribir cada detección
        for di, det_yolo in enumerate(detections):
            track_id = assignments.get(di, -1)
            track = tracks_by_id.get(track_id) if track_id > 0 else None

            bbox = det_yolo.get("bbox", [0, 0, 0, 0])
            track_valid = "YES" if (track and track.hits >= min_hits) else "NO"
            
            writer.writerow([
                frame_idx,
                di,
                track_id if track_id > 0 else -1,
                bbox[0] if len(bbox) >= 1 else 0,
                bbox[1] if len(bbox) >= 2 else 0,
                bbox[2] if len(bbox) >= 3 else 0,
                bbox[3] if len(bbox) >= 4 else 0,
                float(det_yolo.get("confianza", 0.0)),
                track.state if track else "NO_TRACK",
                track.hits if track else 0,
                track.age if track else 0,
                track_valid,
            ])
        
        # Si no hay detecciones pero hay tracks activos, registrar que el frame no tuvo detecciones
        # (útil para debug de lifecycle)
        if not detections and tracks:
            # Escribir una fila especial indicando que no hubo detecciones
            writer.writerow([
                frame_idx,
                -1,  # det_idx = -1 indica "sin detecciones"
                -1,  # track_id
                0, 0, 0, 0,  # bbox vacío
                0.0,  # confianza
                "NO_DETECTIONS",  # estado
                0, 0,  # hits, age
                "N/A",  # valid
            ])
