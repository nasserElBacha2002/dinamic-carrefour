#!/usr/bin/env python3
"""
Capa A — Detector genérico de productos con YOLOv8 (Retail-ready)

- NO depende de clases COCO.
- Maximiza recall (detecta instancias).
- Filtra basura con heurísticas baratas.
- Listo para: 1) MVP con yolov8n.pt 2) modelo propio 1-clase `product`
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

from ultralytics import YOLO

MODELO_DEFAULT = "yolov8n.pt"


class YOLODetector:
    """
    Salida por detección:
      - bbox: [x1,y1,x2,y2] int (coords en imagen original)
      - bbox_padded: [x1,y1,x2,y2] int (si se recorta con padding)
      - confianza: float
      - clase: "product" (normalizado)
      - class_id: int
      - raw_label: str (label del modelo)
      - crop: np.ndarray (si se generó)
      - crop_path: str (si crops_dir)
      - padding: int
    """

    def __init__(
        self,
        modelo: str = MODELO_DEFAULT,
        confianza_minima: float = 0.15,
        iou_nms: float = 0.60,
        device: str = "auto",
        imgsz: int = 960,
        half: bool = False,
        max_det: int = 300,
        roi: Optional[Tuple[float, float, float, float]] = None,  # (x1,y1,x2,y2) en [0..1]
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.90,
        min_aspect: float = 0.10,
        max_aspect: float = 10.0,
        padding_ratio: float = 0.06,
    ):
        self.confianza_minima = float(confianza_minima)
        self.iou_nms = float(iou_nms)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)
        self.padding_ratio = float(padding_ratio)
        self.roi = roi

        print(f"🔧 Cargando modelo YOLO: {modelo}")
        self.model = YOLO(modelo)

        # Determinar dispositivo
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        # FP16 solo en CUDA (robusto)
        self.half = bool(half) and self._device == "cuda"

        print(f"   Dispositivo: {self._device}")
        print(f"   Confianza mínima: {self.confianza_minima}")
        print(f"   IoU NMS: {self.iou_nms}")
        print(f"   imgsz: {self.imgsz}")
        print(f"   max_det: {self.max_det}")
        if self.roi:
            print(f"   ROI: {self.roi} (coords normalizadas)")
        print(f"   half: {self.half}")

    # ──────────────────────────────────────────────────────────────
    # Utilidades internas
    # ──────────────────────────────────────────────────────────────

    def _aplicar_roi(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        if not self.roi:
            return image, (0, 0)

        h, w = image.shape[:2]
        x1n, y1n, x2n, y2n = self.roi

        x1 = max(0, int(x1n * w))
        y1 = max(0, int(y1n * h))
        x2 = min(w, int(x2n * w))
        y2 = min(h, int(y2n * h))

        if x2 <= x1 or y2 <= y1:
            return image, (0, 0)

        return image[y1:y2, x1:x2].copy(), (x1, y1)

    def _post_filtrar(self, dets: List[Dict], w: int, h: int) -> List[Dict]:
        if not dets:
            return dets

        img_area = float(w * h)
        out: List[Dict] = []

        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            area_ratio = (bw * bh) / img_area
            aspect = bw / float(bh)

            if area_ratio < self.min_area_ratio:
                continue
            if area_ratio > self.max_area_ratio:
                continue
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            out.append(d)

        return out

    def _padding_dinamico(self, x1: int, y1: int, x2: int, y2: int) -> int:
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        p = int(max(bw, bh) * self.padding_ratio)
        return max(4, min(p, 80))

    def _infer(self, image_bgr: np.ndarray, conf: float) -> List[Dict]:
        # ROI opcional
        image_roi, (ox, oy) = self._aplicar_roi(image_bgr)

        results = self.model(
            image_roi,
            conf=conf,
            iou=self.iou_nms,
            imgsz=self.imgsz,
            device=self._device,
            verbose=False,
            max_det=self.max_det,
            half=self.half,
        )

        detecciones: List[Dict] = []
        for result in results:
            if result.boxes is None:
                continue

            names = getattr(result, "names", None) or getattr(self.model, "names", None)

            for box in result.boxes:
                class_id = int(box.cls[0]) if box.cls is not None else -1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x1i = int(x1) + ox
                y1i = int(y1) + oy
                x2i = int(x2) + ox
                y2i = int(y2) + oy

                raw_label = "product"
                if names and class_id in names:
                    raw_label = str(names[class_id])

                detecciones.append({
                    "bbox": [x1i, y1i, x2i, y2i],
                    "confianza": float(box.conf[0]) if box.conf is not None else 0.0,
                    "clase": "product",
                    "class_id": class_id,
                    "raw_label": raw_label,
                })

        h, w = image_bgr.shape[:2]
        return self._post_filtrar(detecciones, w=w, h=h)

    # ──────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────

    def detectar(
        self,
        imagen: Union[str, Path, np.ndarray],
        confianza_minima: Optional[float] = None
    ) -> List[Dict]:
        conf = float(confianza_minima) if confianza_minima is not None else self.confianza_minima

        if isinstance(imagen, (str, Path)):
            image = cv2.imread(str(imagen))
            if image is None:
                print(f"  ⚠️ No se pudo leer imagen: {imagen}")
                return []
        else:
            image = imagen

        return self._infer(image, conf)

    def detectar_y_recortar_frame(
        self,
        frame_bgr: np.ndarray,
        frame_id: str = "frame",
        crops_dir: Optional[str] = None,
        confianza_minima: Optional[float] = None,
    ) -> List[Dict]:
        """
        Detección + crops desde un frame ya cargado en RAM (retail real).
        """
        conf = float(confianza_minima) if confianza_minima is not None else self.confianza_minima

        detecciones = self._infer(frame_bgr, conf)
        if not detecciones:
            return []

        h, w = frame_bgr.shape[:2]
        if crops_dir:
            Path(crops_dir).mkdir(parents=True, exist_ok=True)

        for i, det in enumerate(detecciones):
            x1, y1, x2, y2 = det["bbox"]
            pad = self._padding_dinamico(x1, y1, x2, y2)

            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            if x2p <= x1p or y2p <= y1p:
                det["crop"] = None
                continue

            crop = frame_bgr[y1p:y2p, x1p:x2p].copy()
            det["crop"] = crop
            det["bbox_padded"] = [x1p, y1p, x2p, y2p]
            det["padding"] = pad

            if crops_dir:
                crop_name = f"{frame_id}_crop_{i:03d}.jpg"
                crop_path = str(Path(crops_dir) / crop_name)
                cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                det["crop_path"] = crop_path

        return detecciones

    def detectar_y_recortar(
        self,
        imagen_path: str,
        crops_dir: Optional[str] = None,
        confianza_minima: Optional[float] = None,
    ) -> List[Dict]:
        """
        Detección + crops desde archivo (compatibilidad).
        Lee la imagen UNA sola vez.
        """
        image = cv2.imread(imagen_path)
        if image is None:
            print(f"  ⚠️ No se pudo leer imagen: {imagen_path}")
            return []

        frame_id = Path(imagen_path).stem
        return self.detectar_y_recortar_frame(
            image,
            frame_id=frame_id,
            crops_dir=crops_dir,
            confianza_minima=confianza_minima,
        )

    def generar_imagen_anotada(
        self,
        imagen_path: str,
        detecciones: List[Dict],
        output_path: str,
        etiquetas_sku: Optional[Dict[int, str]] = None
    ) -> str:
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            return ""

        for i, det in enumerate(detecciones):
            x1, y1, x2, y2 = det.get("bbox_padded") or det["bbox"]
            conf = det.get("confianza", 0.0)

            color = (0, 200, 0)
            label = f'product {conf:.2f}'

            if etiquetas_sku and i in etiquetas_sku:
                sku_label = etiquetas_sku[i]
                if sku_label.startswith("UNKNOWN"):
                    color = (0, 0, 220)
                label = f"{sku_label} ({conf:.2f})"

            cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_top = max(0, y1 - th - 8)
            cv2.rectangle(imagen, (x1, y_top), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                imagen, label, (x1 + 2, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, imagen, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return output_path
