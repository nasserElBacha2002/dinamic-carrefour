#!/usr/bin/env python3
"""
BBox Quality — Métricas genéricas de calidad de bbox.

Calcula un score que indica la probabilidad de que un bbox esté "mezclado"
(contiene múltiples productos, carteles, reflejos, etc.).

Señales genéricas (no atadas a un producto específico):
- Aspect ratio extremo
- Área relativa al frame
- Confianza YOLO
- Distancia a bordes de ROI
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any


class BBoxQualityScorer:
    """
    Calcula un score de calidad genérico para un bbox.
    
    Score alto = probablemente mezclado/contaminado
    Score bajo = probablemente un solo producto limpio
    """
    
    def __init__(
        self,
        aspect_weight: float = 0.4,
        area_weight: float = 0.2,
        yolo_conf_weight: float = 0.3,
        edge_weight: float = 0.1,
    ):
        """
        Args:
            aspect_weight: Peso del aspect ratio en el score.
            area_weight: Peso del área relativa en el score.
            yolo_conf_weight: Peso de la confianza YOLO en el score.
            edge_weight: Peso de la distancia a bordes en el score.
        """
        self.aspect_weight = aspect_weight
        self.area_weight = area_weight
        self.yolo_conf_weight = yolo_conf_weight
        self.edge_weight = edge_weight
    
    def calcular_score(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        yolo_conf: float,
        roi: Optional[Tuple[float, float, float, float]] = None,
    ) -> float:
        """
        Calcula el score de calidad del bbox (0.0 = limpio, 1.0 = muy mezclado).
        
        Args:
            bbox: (x1, y1, x2, y2) del bbox.
            frame_shape: (height, width) del frame.
            yolo_conf: Confianza de YOLO para esta detección.
            roi: ROI normalizado (x1, y1, x2, y2) si existe.
        
        Returns:
            Score entre 0.0 y 1.0.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_shape
        
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        
        # 1. Aspect ratio score (bboxes muy anchos son sospechosos)
        aspect = box_w / float(box_h)
        # Normalizar: aspect > 1.0 es ancho, > 1.5 es muy ancho
        aspect_score = min(1.0, max(0.0, (aspect - 1.0) / 0.5))
        
        # 2. Área relativa score (bboxes muy grandes pueden incluir vecinos)
        area_bbox = box_w * box_h
        area_frame = h * w
        area_ratio = area_bbox / float(area_frame) if area_frame > 0 else 0.0
        # Normalizar: > 0.15 del frame es sospechoso
        area_score = min(1.0, area_ratio / 0.15)
        
        # 3. Confianza YOLO score (baja confianza = detección dudosa)
        # Invertir: baja confianza = alto score (más mezclado)
        yolo_score = 1.0 - min(1.0, max(0.0, yolo_conf))
        
        # 4. Distancia a bordes score (detecciones cerca de bordes son más sucias)
        edge_score = 0.0
        if roi:
            # Normalizar bbox a coordenadas relativas
            x1_rel = x1 / float(w)
            y1_rel = y1 / float(h)
            x2_rel = x2 / float(w)
            y2_rel = y2 / float(h)
            
            # Distancia a bordes del ROI
            roi_x1, roi_y1, roi_x2, roi_y2 = roi
            dist_left = abs(x1_rel - roi_x1)
            dist_right = abs(x2_rel - roi_x2)
            dist_top = abs(y1_rel - roi_y1)
            dist_bottom = abs(y2_rel - roi_y2)
            
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            # Cerca de bordes (< 0.05) = alto score
            edge_score = min(1.0, max(0.0, 1.0 - (min_dist / 0.05)))
        
        # Score combinado (ponderado)
        score = (
            self.aspect_weight * aspect_score +
            self.area_weight * area_score +
            self.yolo_conf_weight * yolo_score +
            self.edge_weight * edge_score
        )
        
        return min(1.0, max(0.0, score))
    
    def calcular_score_desde_deteccion(
        self,
        deteccion: Dict[str, Any],
        frame_shape: Tuple[int, int],
        roi: Optional[Tuple[float, float, float, float]] = None,
    ) -> float:
        """
        Calcula el score desde un dict de detección.
        
        Args:
            deteccion: Dict con 'bbox', 'confianza', etc.
            frame_shape: (height, width) del frame.
            roi: ROI normalizado si existe.
        
        Returns:
            Score entre 0.0 y 1.0.
        """
        bbox = deteccion.get("bbox", [0, 0, 0, 0])
        yolo_conf = float(deteccion.get("confianza", 0.5))
        
        return self.calcular_score(
            bbox=tuple(bbox),
            frame_shape=frame_shape,
            yolo_conf=yolo_conf,
            roi=roi,
        )
