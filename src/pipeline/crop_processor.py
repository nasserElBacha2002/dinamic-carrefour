#!/usr/bin/env python3
"""
Crop Processor — Heurísticas y procesamiento de crops para identificación.

Proporciona funciones para:
- Calcular padding dinámico y limitado
- Generar inner crops (recortes centrales)
- Detectar bboxes "doble producto" (muy anchos)
- Dividir crops verticalmente
"""

import numpy as np
from typing import Tuple


class CropProcessor:
    """
    Procesador de crops con heurísticas para mejorar la identificación.
    """

    @staticmethod
    def cap_padding(pad: int, box_w: int, box_h: int) -> int:
        """
        Limita el padding para que no incluya productos vecinos.

        Args:
            pad: Padding calculado originalmente.
            box_w: Ancho del bbox.
            box_h: Alto del bbox.

        Returns:
            Padding limitado (máximo 8% del tamaño mínimo del bbox).
        """
        pad_max = int(0.08 * min(box_w, box_h))  # 0.06–0.12 típico
        return max(0, min(int(pad), pad_max))

    @staticmethod
    def inner_crop_rect(
        x1: int, y1: int, x2: int, y2: int, ratio: float = 0.75
    ) -> Tuple[int, int, int, int]:
        """
        Calcula un rectángulo central (inner crop) dentro del bbox.

        Args:
            x1, y1, x2, y2: Coordenadas del bbox original.
            ratio: Ratio del tamaño a usar (0.75 = 75% central).

        Returns:
            (ix1, iy1, ix2, iy2) - Coordenadas del inner crop.
        """
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        iw = max(1, int(bw * ratio))
        ih = max(1, int(bh * ratio))

        cx = x1 + bw // 2
        cy = y1 + bh // 2

        ix1 = cx - iw // 2
        iy1 = cy - ih // 2
        ix2 = ix1 + iw
        iy2 = iy1 + ih

        return ix1, iy1, ix2, iy2

    @staticmethod
    def is_wide_box(box_w: int, box_h: int, threshold: float = 0.85) -> bool:
        """
        Detecta si un bbox parece contener "doble producto" (muy ancho).
        
        DEPRECATED: Usar BBoxQualityScorer en su lugar para métricas más genéricas.

        Args:
            box_w: Ancho del bbox.
            box_h: Alto del bbox.
            threshold: Umbral de aspect ratio (ancho/alto).

        Returns:
            True si el bbox es muy ancho (probablemente dos productos).
        """
        if box_h <= 0:
            return False
        aspect = box_w / float(box_h)
        return aspect > threshold

    @staticmethod
    def split_vertical(crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide un crop en 2 mitades verticales (izquierda/derecha).

        Args:
            crop: Imagen del crop (numpy array).

        Returns:
            (left_crop, right_crop) - Dos crops divididos verticalmente.
        """
        h, w = crop.shape[:2]
        mid = max(1, w // 2)
        left = crop[:, :mid].copy()
        right = crop[:, mid:].copy()
        return left, right

    @staticmethod
    def calcular_bbox_padded(
        x1: int, y1: int, x2: int, y2: int,
        pad_raw: int, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        """
        Calcula el bbox con padding aplicado y limitado.

        Args:
            x1, y1, x2, y2: Coordenadas del bbox original.
            pad_raw: Padding calculado originalmente.
            frame_w: Ancho del frame.
            frame_h: Alto del frame.

        Returns:
            (x1p, y1p, x2p, y2p) - Coordenadas del bbox con padding.
        """
        box_w = int(x2 - x1)
        box_h = int(y2 - y1)
        pad = CropProcessor.cap_padding(pad_raw, box_w, box_h)

        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(frame_w, x2 + pad)
        y2p = min(frame_h, y2 + pad)

        return x1p, y1p, x2p, y2p
