#!/usr/bin/env python3
"""
Pipeline Engine — Orquesta todo el flujo de procesamiento (Retail-ready, sin I/O innecesario).

Video (stream) → Frames (en RAM) → Detección (YOLO) → Crops → Identificación (CLIP) → Reporte

Genera:
  - output/<video>_<timestamp>/
      crops/                 # opcional (debug/review)
      reporte_deteccion/
          inventario_sku.csv
          frame_XXXX.jpg     # (si generar_anotaciones)
"""

from __future__ import annotations

import csv
import cv2
import numpy as np

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any
from datetime import datetime
from collections import Counter

from tqdm import tqdm

from src.analizar_video import analizar_video
from src.detector.yolo_detector import YOLODetector
from src.sku_identifier.identifier import SKUIdentifier

# DB opcional
try:
    from src.database.repository import ProductoRepository
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

# Learning Manager (dataset evolutivo)
try:
    from src.learning.manager import LearningManager
    _LEARNING_AVAILABLE = True
except Exception:
    _LEARNING_AVAILABLE = False


class PipelineEngine:
    def __init__(
        self,
        detector: YOLODetector,
        identificador: SKUIdentifier,
        output_base: str = "output",
        fps_extraccion: float = 1.0,
        rotar: Optional[int] = None,
        generar_anotaciones: bool = True,
        guardar_crops: bool = False,
        usar_db: bool = True,
        detector_conf: Optional[float] = None,
        detector_iou: Optional[float] = None,
        detector_roi: Optional[Tuple[float, float, float, float]] = None,  # (x1,y1,x2,y2) [0..1]
    ):
        self.detector = detector
        self.identificador = identificador
        self.output_base = output_base
        self.fps_extraccion = float(fps_extraccion)
        self.rotar = rotar
        self.generar_anotaciones = generar_anotaciones
        self.guardar_crops = guardar_crops

        # Persistencia
        self.usar_db = bool(usar_db) and _DB_AVAILABLE
        self._repo: Optional["ProductoRepository"] = None

        if self.usar_db:
            try:
                self._repo = ProductoRepository()
                print("   🗄️  Persistencia SQL Server activada")
            except Exception as e:
                print(f"   ⚠️  No se pudo conectar a SQL Server: {e}")
                print("       Los resultados se guardarán solo en CSV.")
                self.usar_db = False
                self._repo = None

        # Learning Manager (dataset evolutivo)
        self._learning_manager: Optional["LearningManager"] = None

        # Overrides detector
        if detector_conf is not None:
            self.detector.confianza_minima = float(detector_conf)
        if detector_iou is not None:
            self.detector.iou_nms = float(detector_iou)
        if detector_roi is not None:
            self.detector.roi = detector_roi

    # ──────────────────────────────────────────────────────────────
    # Lectura eficiente de frames en RAM (sin exportar a disco)
    # ──────────────────────────────────────────────────────────────
    def _iter_frames_video(self, video_path: str) -> Iterator[Tuple[int, float, np.ndarray]]:
        """
        Yields: (frame_index, timestamp_sec, frame_bgr)
        Muestrea a fps_extraccion (aprox).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0  # fallback razonable

        # cada cuántos frames tomar 1
        step = max(1, int(round(fps / max(0.0001, self.fps_extraccion))))

        idx = 0
        out_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if idx % step == 0:
                # rotación opcional
                if self.rotar == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotar == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotar == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                t_sec = float(idx / fps)
                yield out_idx, t_sec, frame
                out_idx += 1

            idx += 1

        cap.release()

    def procesar_video(self, video_path: str) -> Dict[str, Any]:
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_id = f"{video_name}_{timestamp}"
        output_dir = Path(self.output_base) / execution_id

        crops_dir = output_dir / "crops" if self.guardar_crops else None
        reporte_dir = output_dir / "reporte_deteccion"
        reporte_dir.mkdir(parents=True, exist_ok=True)

        if crops_dir:
            crops_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar Learning Manager
        if _LEARNING_AVAILABLE:
            try:
                execution_params: Dict[str, Any] = {
                    "fps_extraccion": self.fps_extraccion,
                    "confianza_minima": self.detector.confianza_minima,
                    "iou_nms": self.detector.iou_nms,
                    "sku_threshold": self.identificador.threshold,
                    "unknown_threshold": self.identificador.threshold_unknown,
                    "margen_ambiguedad": self.identificador.margen_ambiguedad,
                }
                self._learning_manager = LearningManager(
                    output_dir=output_dir,
                    execution_id=execution_id,
                    video_path=video_path,
                    execution_params=execution_params,
                )
                print("   📚 Learning Manager activado (dataset evolutivo)")
            except Exception as e:
                print(f"   ⚠️  No se pudo inicializar Learning Manager: {e}")
                self._learning_manager = None

        print("=" * 70)
        print(f"🚀 PIPELINE — Procesando: {video_path}")
        print(f"   Output: {output_dir}")
        print("=" * 70)

        info_video = analizar_video(video_path)
        if info_video is None:
            return {"error": "No se pudo analizar el video"}

        print("\n🔍 Detectando productos e identificando SKUs...")

        conteo_por_frame: List[Counter] = []
        conteo_acumulado: Counter = Counter()
        total_detecciones = 0
        frames_con_producto = 0
        frames_total = 0

        duration = float(info_video["video_info"].get("duration", 0.0) or 0.0)
        est_total = int(duration * self.fps_extraccion) if duration > 0 else None

        iterator = self._iter_frames_video(video_path)
        pbar = tqdm(iterator, total=est_total, desc="Procesando frames", unit="frame")

        for out_idx, t_sec, frame in pbar:
            frames_total += 1

            # 1) detectar (sin leer de disco)
            dets = self.detector.detectar(frame)
            if not dets:
                continue

            frames_con_producto += 1
            total_detecciones += len(dets)

            conteo_frame: Counter = Counter()
            etiquetas_sku: Dict[int, str] = {}

            # 2) recortar + identificar
            h, w = frame.shape[:2]

            for i, det in enumerate(dets):
                x1, y1, x2, y2 = det["bbox"]

                pad = self.detector._padding_dinamico(x1, y1, x2, y2)
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(w, x2 + pad)
                y2p = min(h, y2 + pad)
                if x2p <= x1p or y2p <= y1p:
                    continue

                crop = frame[y1p:y2p, x1p:x2p].copy()

                det["bbox_padded"] = [x1p, y1p, x2p, y2p]

                # opcional: guardar crop (debug)
                if crops_dir:
                    crop_name = f"frame_{out_idx:05d}_crop_{i:03d}.jpg"
                    cv2.imwrite(str(crops_dir / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                crop_id = f"frame_{out_idx:05d}_crop_{i:03d}"

                resultado_sku: Dict[str, Any] = self.identificador.identificar_crop_numpy(
                    crop,
                    crop_id=crop_id,
                )

                ean = str(resultado_sku.get("ean", "UNKNOWN"))
                status = str(resultado_sku.get("status", "matched"))

                conteo_frame[ean] += 1
                conteo_acumulado[ean] += 1

                # Guardar en Learning Manager si es dudoso
                if self._learning_manager and status in ("unknown", "ambiguous"):
                    detection_info = {
                        "bbox": det["bbox"],
                        "bbox_padded": det["bbox_padded"],
                        "yolo_conf": float(det.get("confianza", 0.0)),
                        "class_id": int(det.get("class_id", -1)),
                        "raw_label": str(det.get("raw_label", "product")),
                        "t_sec": float(t_sec),
                    }

                    packaging_info = {
                        "predicted": str(resultado_sku.get("categoria", "")),
                    }

                    top_matches = resultado_sku.get("top_matches") or []
                    sku_info = {
                        "decision": status,
                        "top_matches": top_matches,
                        "threshold_used": float(self.identificador.threshold),
                        "unknown_threshold": float(self.identificador.threshold_unknown),
                        "margen_ambiguedad": float(self.identificador.margen_ambiguedad),
                    }

                    # Si guardás anotaciones, la ruta real es reporte_deteccion/frame_XXXXX.jpg
                    frame_path_rel = (
                        f"reporte_deteccion/frame_{out_idx:05d}.jpg"
                        if self.generar_anotaciones
                        else None
                    )

                    self._learning_manager.guardar_crop_dudoso(
                        crop=crop,
                        crop_id=crop_id,
                        frame_path=frame_path_rel,
                        frame_idx=int(out_idx),
                        detection_info=detection_info,
                        packaging_info=packaging_info,
                        sku_info=sku_info,
                        decision=status,
                    )

                if ean != "UNKNOWN":
                    desc = (resultado_sku.get("descripcion") or "")
                    etiquetas_sku[i] = f"{ean} {str(desc)[:30]}".strip()
                else:
                    conf = float(resultado_sku.get("confianza") or 0.0)
                    etiquetas_sku[i] = f"UNKNOWN ({conf:.2f})"

            conteo_por_frame.append(conteo_frame)

            # 3) anotación (sobre frame en RAM)
            if self.generar_anotaciones:
                out_img = str(reporte_dir / f"frame_{out_idx:05d}.jpg")
                annotated = frame.copy()

                for i, det in enumerate(dets):
                    x1a, y1a, x2a, y2a = det.get("bbox_padded") or det["bbox"]
                    label = etiquetas_sku.get(i, f'product {float(det.get("confianza", 0.0)):.2f}')
                    color = (0, 200, 0) if not label.startswith("UNKNOWN") else (0, 0, 220)

                    cv2.rectangle(annotated, (x1a, y1a), (x2a, y2a), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y_top = max(0, y1a - th - 8)
                    cv2.rectangle(annotated, (x1a, y_top), (x1a + tw + 4, y1a), color, -1)
                    cv2.putText(
                        annotated, label, (x1a + 2, max(0, y1a - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )

                cv2.imwrite(out_img, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── Dedup + Reporte ─────────────────────────────────────────
        print("\n📊 Generando reporte (deduplicación por frame)...")

        conteo_dedup: Counter = Counter()
        for cf in conteo_por_frame:
            for ean, cant in cf.items():
                if conteo_dedup[ean] < cant:
                    conteo_dedup[ean] = cant

        fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_path = reporte_dir / "inventario_sku.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["EAN", "Cantidad", "Fecha"])
            for ean, cantidad in sorted(conteo_dedup.items()):
                if ean != "UNKNOWN":
                    writer.writerow([ean, int(cantidad), fecha_str])

        # ── Resumen ────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("📋 RESUMEN DEL PROCESAMIENTO")
        print("=" * 70)
        print(f"   Frames muestreados:    {frames_total}")
        print(f"   Frames con producto:   {frames_con_producto}")
        print(f"   Total detecciones raw: {total_detecciones}")
        skus_unicos = [k for k in conteo_dedup if k != "UNKNOWN"]
        print(f"   SKUs identificados:    {len(skus_unicos)}")

        learning_summary: Optional[Dict[str, Any]] = None
        if self._learning_manager:
            learning_summary = self._learning_manager.resumen()
            print("\n   📚 Dataset evolutivo:")
            print(f"      Crops guardados: {learning_summary.get('total_crops_saved')}")
            print(f"      UNKNOWN: {learning_summary.get('unknown_count')}")
            print(f"      AMBIGUOUS: {learning_summary.get('ambiguous_count')}")
            print(f"      Metadata: {learning_summary.get('metadata_file')}")

        print("=" * 70)

        resultado: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "frames_total": int(frames_total),
            "frames_con_producto": int(frames_con_producto),
            "total_detecciones": int(total_detecciones),
            "conteo_sku": dict(conteo_dedup),
            "conteo_raw": dict(conteo_acumulado),
            "csv_path": str(csv_path),
            "parametros": {
                "fps_extraccion": float(self.fps_extraccion),
                "rotar": self.rotar,
                "detector_conf": float(getattr(self.detector, "confianza_minima", 0.0)),
                "detector_iou": float(getattr(self.detector, "iou_nms", 0.0)),
                "detector_roi": getattr(self.detector, "roi", None),
                "sku_threshold": float(getattr(self.identificador, "threshold", 0.0)),
                "unknown_threshold": float(getattr(self.identificador, "threshold_unknown", 0.0)),
                "margen_ambiguedad": float(getattr(self.identificador, "margen_ambiguedad", 0.0)),
            },
            "duracion_segundos": float(info_video["video_info"].get("duration", 0.0) or 0.0),
        }

        if learning_summary is not None:
            resultado["learning"] = learning_summary

        if self.usar_db and self._repo:
            try:
                ej_id = self._repo.registrar_resultado_completo(
                    video_path=video_path,
                    resultado=resultado,
                )
                print(f"   🗄️  Resultado guardado en SQL Server (ejecución #{ej_id})")
            except Exception as e:
                print(f"   ⚠️  Error guardando en SQL Server: {e}")

        return resultado
