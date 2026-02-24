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

from src.pipeline.core.video_reader import VideoFrameReader
from src.pipeline.processing.crop_processor import CropProcessor
from src.pipeline.processing.detection_processor import DetectionProcessor
from src.pipeline.output.report_generator import ReportGenerator
from src.pipeline.output.learning_integration import LearningIntegration
from src.pipeline.tracking.track_setup import TrackSetup
from src.pipeline.tracking.track_runtime import TrackRuntime
from src.pipeline.tracking.track_exporter import TrackExporter
from src.pipeline.output.result_builder import ResultBuilder
from src.tracking.track_vote_accumulator import DecisionProfile

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
        identifier: SKUIdentifier,
        output_base: str = "output",
        fps_extraction: float = 1.0,
        rotate: Optional[int] = None,
        generate_annotations: bool = True,
        save_crops: bool = False,
        use_db: bool = True,
        detector_conf: Optional[float] = None,
        detector_iou: Optional[float] = None,
        detector_roi: Optional[Tuple[float, float, float, float]] = None,
        # Sprint 3: Tracking
        use_tracks: bool = False,
        track_iou: float = 0.4,
        track_min_hits: int = 3,
        track_max_age: int = 15,
    ):
        self.detector = detector
        self.identifier = identifier

        self.output_base = str(output_base)
        self.fps_extraction = float(fps_extraction)
        self.rotate = rotate
        self.generate_annotations = bool(generate_annotations)
        self.save_crops = bool(save_crops)

        # Overrides detector
        if detector_conf is not None:
            self.detector.confianza_minima = float(detector_conf)
        if detector_iou is not None:
            self.detector.iou_nms = float(detector_iou)
        if detector_roi is not None:
            self.detector.roi = detector_roi

        # Persistencia
        self.use_db = bool(use_db) and _DB_AVAILABLE
        self._repo: Optional["ProductoRepository"] = None
        if self.use_db:
            try:
                self._repo = ProductoRepository()
                print("   🗄️  Persistencia SQL Server activada")
            except Exception as e:
                print(f"   ⚠️  No se pudo conectar a SQL Server: {e}")
                print("       Los resultados se guardarán solo en CSV.")
                self.use_db = False
                self._repo = None

        # Sprint 3: Tracking (opcional)
        self.use_tracks = bool(use_tracks)
        self.track_runtime: Optional[TrackRuntime] = None
        self.track_exporter: Optional[TrackExporter] = None
        
        if self.use_tracks:
            # Setup de tracking (modularizado)
            tracker, vote_accumulator, track_config = TrackSetup.create(
                track_iou=float(track_iou),
                track_min_hits=int(track_min_hits),
                track_max_age=int(track_max_age),
                decision_profile=DecisionProfile.WAREHOUSE_BALANCED,
            )
            
            # Guardar para inicialización en process_video
            self._tracker = tracker
            self._vote_accumulator = vote_accumulator
            self._track_config = track_config

    def process_video(self, video_path: str) -> Dict[str, Any]:
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_id = f"{video_name}_{timestamp}"

        output_dir = Path(self.output_base) / execution_id
        report_dir = output_dir / "reporte_deteccion"
        crops_dir = (output_dir / "crops") if self.save_crops else None

        report_dir.mkdir(parents=True, exist_ok=True)
        if crops_dir:
            crops_dir.mkdir(parents=True, exist_ok=True)

        execution_params: Dict[str, Any] = {
            "fps_extraction": self.fps_extraction,
            "confianza_minima": float(getattr(self.detector, "confianza_minima", 0.0) or 0.0),
            "iou_nms": float(getattr(self.detector, "iou_nms", 0.0) or 0.0),
            "sku_threshold": float(getattr(self.identifier, "threshold", 0.0) or 0.0),
            "unknown_threshold": float(getattr(self.identifier, "threshold_unknown", 0.0) or 0.0),
            "margen_ambiguedad": float(getattr(self.identifier, "margen_ambiguedad", 0.0) or 0.0),
        }

        learning = LearningIntegration(
            output_dir=output_dir,
            execution_id=execution_id,
            video_path=video_path,
            execution_params=execution_params,
        )

        report = ReportGenerator(report_dir=report_dir, generate_annotations=self.generate_annotations)

        crop_processor = CropProcessor()
        
        # Política de decisión genérica (configurable por perfil)
        from src.pipeline.processing.decision_policy import DecisionPolicy, DecisionPolicyConfig
        decision_policy = DecisionPolicy(DecisionPolicyConfig.shelf_video())
        
        detection_processor = DetectionProcessor(
            identifier=self.identifier,
            crop_processor=crop_processor,
            save_crops=self.save_crops,
            crops_dir=crops_dir,
            decision_policy=decision_policy,
        )
        
        # Inicializar track runtime si está activado
        if self.use_tracks:
            self.track_runtime = TrackRuntime(
                tracker=self._tracker,
                vote_accumulator=self._vote_accumulator,
                detection_processor=detection_processor,
                detector=self.detector,
                min_rel_area=self._track_config["min_rel_area"],
                max_rel_area=self._track_config["max_rel_area"],
                output_dir=output_dir,
            )
            self.track_exporter = TrackExporter(
                output_dir=output_dir,
                report=report,
            )

        print("=" * 70)
        print(f"🚀 PIPELINE — Procesando: {video_path}")
        print(f"   Output: {output_dir}")
        print("=" * 70)

        video_info = analizar_video(video_path)
        if video_info is None:
            return {"error": "No se pudo analizar el video"}

        duration = float(video_info.get("video_info", {}).get("duration", 0.0) or 0.0)
        est_total = int(duration * self.fps_extraction) if duration > 0 else None

        print("\n🔍 Detectando productos e identificando SKUs...")

        count_per_frame: List[Counter] = []
        accumulated_count: Counter = Counter()
        total_detections = 0
        frames_with_products = 0
        frames_total = 0

        reader = VideoFrameReader(video_path=video_path, fps_extraction=self.fps_extraction, rotate=self.rotate)
        pbar = tqdm(reader.iter_frames(), total=est_total, desc="Procesando frames", unit="frame")

        for out_idx, t_sec, frame in pbar:
            frames_total += 1

            dets = self.detector.detectar(frame)
            
            # Sprint 3: Tracking (si está activado) - Modularizado
            sku_cache_by_det: Dict[int, Dict[str, Any]] = {}
            
            if self.use_tracks and self.track_runtime:
                dets_to_use, sku_cache_by_det, frame_info = self.track_runtime.process_frame(
                    frame=frame,
                    dets=dets,
                    out_idx=int(out_idx),
                    t_sec=float(t_sec),
                )
                # Actualizar tqdm con info de tracks
                pbar.set_postfix(frame_info)
            else:
                # Sin tracking: usar detecciones originales
                dets_to_use = dets if dets else []
            
            # Continuar con procesamiento normal solo si hay detecciones
            if not dets_to_use:
                continue

            frames_with_products += 1
            total_detections += len(dets_to_use)

            # Procesar detecciones con cache si está disponible (evita doble identificación)
            frame_count, sku_labels, crops_for_learning = detection_processor.process_detections_in_frame(
                frame=frame,
                detections=dets_to_use,
                out_idx=int(out_idx),
                t_sec=float(t_sec),
                detector=self.detector,
                sku_cache=sku_cache_by_det if sku_cache_by_det else None,  # Pasar cache
            )

            count_per_frame.append(frame_count)
            accumulated_count.update(frame_count)

            # P0.1: Anotar con las mismas detecciones que procesamos (consistencia)
            dets_for_annot = dets_to_use
            report.annotate_frame(
                frame=frame,
                detections=dets_for_annot,
                sku_labels=sku_labels,
                frame_idx=int(out_idx),
            )

            learning.save_dubious_crops(
                crops_for_learning=crops_for_learning,
                generate_annotations=self.generate_annotations,
                identifier=self.identifier,
            )

        # Sprint 3.2: Finalizar tracks y generar inventarios (modularizado)
        track_decisions: Optional[Dict[int, Any]] = None
        dedup_count: Counter
        
        if self.use_tracks and self.track_runtime:
            # Finalizar tracks activos
            self.track_runtime.finalize_remaining_tracks()
            track_decisions = self.track_runtime.get_track_decisions()
            
            # Exportar track_summary.json y generar inventarios
            if self.track_exporter and track_decisions:
                self.track_exporter.export_track_summary(track_decisions)
                csv_path_tracks, csv_path_frame = self.track_exporter.generate_inventories(
                    track_decisions=track_decisions,
                    count_per_frame=count_per_frame,
                )
                csv_path = csv_path_tracks  # Usar tracks como fuente principal
                # Calcular dedup_count para reporte (aunque no se use como fuente principal)
                dedup_count = ReportGenerator.deduplicate_by_frame(count_per_frame)
            else:
                # Fallback si no hay tracks
                print("\n📊 Generando reporte (deduplicación por frame)...")
                dedup_count = ReportGenerator.deduplicate_by_frame(count_per_frame)
                csv_path = report.generate_inventory_csv(dedup_count)
        else:
            # Sin tracking: usar deduplicación por frame (comportamiento original)
            print("\n📊 Generando reporte (deduplicación por frame)...")
            dedup_count = ReportGenerator.deduplicate_by_frame(count_per_frame)
            csv_path = report.generate_inventory_csv(dedup_count)

        processing_result = {
            "frames_total": int(frames_total),
            "frames_with_products": int(frames_with_products),
            "total_detections": int(total_detections),
        }

        learning_summary = learning.get_summary()
        report.show_summary(processing_result, dedup_count, learning_summary)

        learning.save_historical_metrics(processing_result, dedup_count)

        # Construir resultado final (modularizado)
        result = ResultBuilder.build(
            output_dir=output_dir,
            frames_total=frames_total,
            frames_with_products=frames_with_products,
            total_detections=total_detections,
            dedup_count=dedup_count,
            accumulated_count=accumulated_count,
            csv_path=csv_path,
            duration=duration,
            detector=self.detector,
            identifier=self.identifier,
            fps_extraction=self.fps_extraction,
            rotate=self.rotate,
            learning_summary=learning_summary,
            track_decisions=track_decisions,
        )

        if self.use_db and self._repo:
            try:
                exec_id = self._repo.registrar_resultado_completo(video_path=video_path, resultado=result)
                print(f"   🗄️  Resultado guardado en SQL Server (ejecución #{exec_id})")
            except Exception as e:
                print(f"   ⚠️  Error guardando en SQL Server: {e}")

        return result