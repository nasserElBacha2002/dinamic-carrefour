#!/usr/bin/env python3
"""
Pipeline Engine — Orquesta todo el flujo de procesamiento (Retail-ready, sin I/O innecesario).

Video (stream) → Frames (en RAM) → Detección (YOLO) → Crops → Identificación (CLIP) → Reporte
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import Counter

from tqdm import tqdm

from src.analizar_video import analizar_video
from src.detector.yolo_detector import YOLODetector
from src.sku_identifier.identifier import SKUIdentifier

from src.pipeline.video_reader import VideoFrameReader
from src.pipeline.crop_processor import CropProcessor
from src.pipeline.detection_processor import DetectionProcessor
from src.pipeline.report_generator import ReportGenerator
from src.pipeline.learning_integration import LearningIntegration

# DB opcional
try:
    from src.database.repository import ProductoRepository
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False


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
        detector_roi: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.detector = detector
        self.identificador = identificador

        self.output_base = str(output_base)
        self.fps_extraccion = float(fps_extraccion)
        self.rotar = rotar
        self.generar_anotaciones = bool(generar_anotaciones)
        self.guardar_crops = bool(guardar_crops)

        # Overrides detector
        if detector_conf is not None:
            self.detector.confianza_minima = float(detector_conf)
        if detector_iou is not None:
            self.detector.iou_nms = float(detector_iou)
        if detector_roi is not None:
            self.detector.roi = detector_roi

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

    def procesar_video(self, video_path: str) -> Dict[str, Any]:
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_id = f"{video_name}_{timestamp}"

        output_dir = Path(self.output_base) / execution_id
        reporte_dir = output_dir / "reporte_deteccion"
        crops_dir = (output_dir / "crops") if self.guardar_crops else None

        reporte_dir.mkdir(parents=True, exist_ok=True)
        if crops_dir:
            crops_dir.mkdir(parents=True, exist_ok=True)

        execution_params: Dict[str, Any] = {
            "fps_extraccion": self.fps_extraccion,
            "confianza_minima": float(getattr(self.detector, "confianza_minima", 0.0) or 0.0),
            "iou_nms": float(getattr(self.detector, "iou_nms", 0.0) or 0.0),
            "sku_threshold": float(getattr(self.identificador, "threshold", 0.0) or 0.0),
            "unknown_threshold": float(getattr(self.identificador, "threshold_unknown", 0.0) or 0.0),
            "margen_ambiguedad": float(getattr(self.identificador, "margen_ambiguedad", 0.0) or 0.0),
        }

        learning = LearningIntegration(
            output_dir=output_dir,
            execution_id=execution_id,
            video_path=video_path,
            execution_params=execution_params,
        )

        report = ReportGenerator(reporte_dir=reporte_dir, generar_anotaciones=self.generar_anotaciones)

        crop_processor = CropProcessor()
        
        # Política de decisión genérica (configurable por perfil)
        from src.pipeline.decision_policy import DecisionPolicy, DecisionPolicyConfig
        decision_policy = DecisionPolicy(DecisionPolicyConfig.shelf_video())
        
        detection_processor = DetectionProcessor(
            identificador=self.identificador,
            crop_processor=crop_processor,
            guardar_crops=self.guardar_crops,
            crops_dir=crops_dir,
            decision_policy=decision_policy,
        )

        print("=" * 70)
        print(f"🚀 PIPELINE — Procesando: {video_path}")
        print(f"   Output: {output_dir}")
        print("=" * 70)

        info_video = analizar_video(video_path)
        if info_video is None:
            return {"error": "No se pudo analizar el video"}

        duration = float(info_video.get("video_info", {}).get("duration", 0.0) or 0.0)
        est_total = int(duration * self.fps_extraccion) if duration > 0 else None

        print("\n🔍 Detectando productos e identificando SKUs...")

        conteo_por_frame: List[Counter] = []
        conteo_acumulado: Counter = Counter()
        total_detecciones = 0
        frames_con_producto = 0
        frames_total = 0

        reader = VideoFrameReader(video_path=video_path, fps_extraccion=self.fps_extraccion, rotar=self.rotar)
        pbar = tqdm(reader.iter_frames(), total=est_total, desc="Procesando frames", unit="frame")

        for out_idx, t_sec, frame in pbar:
            frames_total += 1

            dets = self.detector.detectar(frame)
            if not dets:
                continue

            frames_con_producto += 1
            total_detecciones += len(dets)

            conteo_frame, etiquetas_sku, crops_para_learning = detection_processor.procesar_detecciones_en_frame(
                frame=frame,
                detecciones=dets,
                out_idx=int(out_idx),
                t_sec=float(t_sec),
                detector=self.detector,
            )

            conteo_por_frame.append(conteo_frame)
            conteo_acumulado.update(conteo_frame)

            report.anotar_frame(
                frame=frame,
                detecciones=dets,
                etiquetas_sku=etiquetas_sku,
                frame_idx=int(out_idx),
            )

            learning.guardar_crops_dudosos(
                crops_para_learning=crops_para_learning,
                generar_anotaciones=self.generar_anotaciones,
                identificador=self.identificador,
            )

        print("\n📊 Generando reporte (deduplicación por frame)...")
        conteo_dedup = ReportGenerator.deduplicar_por_frame(conteo_por_frame)
        csv_path = report.generar_csv_inventario(conteo_dedup)

        resultado_procesamiento = {
            "frames_total": int(frames_total),
            "frames_con_producto": int(frames_con_producto),
            "total_detecciones": int(total_detecciones),
        }

        learning_summary = learning.obtener_resumen()
        report.mostrar_resumen(resultado_procesamiento, conteo_dedup, learning_summary)

        learning.guardar_metricas_historicas(resultado_procesamiento, conteo_dedup)

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
                "detector_conf": float(getattr(self.detector, "confianza_minima", 0.0) or 0.0),
                "detector_iou": float(getattr(self.detector, "iou_nms", 0.0) or 0.0),
                "detector_roi": getattr(self.detector, "roi", None),
                "sku_threshold": float(getattr(self.identificador, "threshold", 0.0) or 0.0),
                "unknown_threshold": float(getattr(self.identificador, "threshold_unknown", 0.0) or 0.0),
                "margen_ambiguedad": float(getattr(self.identificador, "margen_ambiguedad", 0.0) or 0.0),
            },
            "duracion_segundos": duration,
        }

        if learning_summary is not None:
            resultado["learning"] = learning_summary

        if self.usar_db and self._repo:
            try:
                ej_id = self._repo.registrar_resultado_completo(video_path=video_path, resultado=resultado)
                print(f"   🗄️  Resultado guardado en SQL Server (ejecución #{ej_id})")
            except Exception as e:
                print(f"   ⚠️  Error guardando en SQL Server: {e}")

        return resultado