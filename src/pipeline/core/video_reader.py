#!/usr/bin/env python3
"""
Video Reader — Lectura eficiente de frames de video en RAM.

Proporciona un iterador que lee frames de un video, aplica muestreo temporal
y rotación opcional, todo sin escribir a disco.
"""

import cv2
import numpy as np
from typing import Iterator, Tuple, Optional


class VideoFrameReader:
    """
    Lee frames de un video de forma eficiente, aplicando muestreo temporal
    y rotación opcional.
    """

    def __init__(
        self,
        video_path: str,
        fps_extraction: float = 1.0,
        rotate: Optional[int] = None,
    ):
        """
        Args:
            video_path: Ruta al archivo de video.
            fps_extraction: FPS objetivo para extracción (ej: 1.0 = 1 frame por segundo).
            rotate: Rotación opcional en grados (90, 180, 270).
        """
        self.video_path = video_path
        self.fps_extraction = float(fps_extraction)
        self.rotate = rotate

    def iter_frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """
        Itera sobre los frames del video.

        Yields:
            (frame_index, timestamp_sec, frame_bgr)
            - frame_index: Índice del frame en la secuencia extraída (0, 1, 2, ...)
            - timestamp_sec: Tiempo en segundos desde el inicio del video
            - frame_bgr: Frame en formato BGR (numpy array)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0  # fallback razonable

        # Cada cuántos frames tomar 1
        step = max(1, int(round(fps / max(0.0001, self.fps_extraction))))

        idx = 0
        out_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if idx % step == 0:
                # Aplicar rotación opcional
                frame = self._apply_rotation(frame)

                t_sec = float(idx / fps)
                yield out_idx, t_sec, frame
                out_idx += 1

            idx += 1

        cap.release()

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Aplica rotación al frame si está configurada."""
        if self.rotate == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
