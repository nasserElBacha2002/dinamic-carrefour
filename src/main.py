#!/usr/bin/env python3
"""
Punto de entrada del sistema de inventario de góndola — Sprint 2.

Arquitectura:
  Capa A: YOLOv8 (detección genérica de productos)
  Capa B: CLIP (identificación SKU por embeddings)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

from src.detector.yolo_detector import YOLODetector
from src.sku_identifier.embedder import CLIPEmbedder
from src.sku_identifier.vector_store import VectorStore
from src.sku_identifier.identifier import SKUIdentifier
from src.sku_identifier.categorizer import PackagingCategorizer
from src.pipeline.engine import PipelineEngine


def _parse_roi(roi_str: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """
    ROI en formato: "x1,y1,x2,y2" normalizado [0..1]
    Ej: "0.05,0.10,0.95,0.98"
    """
    if not roi_str:
        return None
    parts = [p.strip() for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI inválido. Formato esperado: x1,y1,x2,y2 (4 valores)")
    x1, y1, x2, y2 = map(float, parts)
    if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
        raise ValueError("ROI inválido. Los valores deben estar normalizados entre 0 y 1.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI inválido. x2> x1 e y2> y1.")
    return (x1, y1, x2, y2)


def construir_pipeline(args: argparse.Namespace) -> PipelineEngine:
    """
    Construye el pipeline completo a partir de los argumentos CLI.
    """
    # ── Capa A: Detector (retail-ready, SIN COCO mapping) ───────────────
    detector = YOLODetector(
        modelo=args.modelo_yolo,
        confianza_minima=args.confianza,
        iou_nms=args.det_iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        roi=_parse_roi(args.roi),
    )

    # ── Capa B: Identificador SKU ──────────────────────────────────────
    embedder = CLIPEmbedder(modelo=args.clip_model)

    store = VectorStore(
        embeddings_dir=args.embeddings_dir,
        dimension=embedder.dimension,
    )

    # Categorizador de packaging (usa el mismo modelo CLIP)
    categorizer = None
    if not args.sin_categorias:
        categorizer = PackagingCategorizer(
            model=embedder.model,
            device=embedder._device,
        )

    identificador = SKUIdentifier(
        embedder=embedder,
        vector_store=store,
        categorizer=categorizer,
        eans_file=args.eans_file,
        threshold=args.sku_threshold,
        threshold_unknown=args.unknown_threshold,
        margen_ambiguedad=args.margen_ambiguedad,
        review_dir=args.review_dir,
        guardar_review=args.guardar_review,
        verbose=args.verbose,
    )

    # ── Pipeline ───────────────────────────────────────────────────────
    engine = PipelineEngine(
        detector=detector,
        identificador=identificador,
        output_base=args.output,
        fps_extraccion=args.fps,
        rotar=args.rotar,
        generar_anotaciones=not args.sin_anotaciones,
        guardar_crops=args.guardar_crops,
        usar_db=args.db,
    )

    return engine


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sistema de Inventario de Góndola — Sprint 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run.py data/IMG_2196.MOV
  python run.py data/IMG_2196.MOV --fps 0.5 --confianza 0.20 --det-iou 0.55
  python run.py data/IMG_2196.MOV --roi 0.05,0.10,0.95,0.98 --imgsz 960 --max-det 300
        """,
    )

    # Video
    parser.add_argument("video", help="Ruta al archivo de video")

    # Extracción
    parser.add_argument("--fps", type=float, default=1.0, help="Frames por segundo a extraer (default: 1.0)")
    parser.add_argument("--rotar", type=int, choices=[90, 180, 270], default=None, help="Rotar frames")

    # Detección (Capa A)
    parser.add_argument("--modelo-yolo", default="yolov8n.pt", help="Modelo YOLO (default: yolov8n.pt)")
    parser.add_argument("--confianza", type=float, default=0.25, help="Confianza mínima YOLO (default: 0.25)")

    # NUEVO (retail real)
    parser.add_argument("--det-iou", type=float, default=0.55, help="IoU para NMS en YOLO (default: 0.55)")
    parser.add_argument("--max-det", type=int, default=300, help="Máx detecciones por frame (default: 300)")
    parser.add_argument("--imgsz", type=int, default=960, help="Tamaño de inferencia (default: 960)")
    parser.add_argument("--roi", default=None, help="ROI normalizado x1,y1,x2,y2 (ej: 0.05,0.10,0.95,0.98)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device (default: auto)")
    parser.add_argument("--half", action="store_true", help="Usar FP16 (si aplica)")

    # Identificación (Capa B)
    parser.add_argument("--clip-model", default="ViT-B/32", help="Modelo CLIP (default: ViT-B/32)")
    parser.add_argument("--sku-threshold", type=float, default=0.70, help="Umbral match SKU (default: 0.70)")
    parser.add_argument("--unknown-threshold", type=float, default=0.40, help="Por debajo = UNKNOWN (default: 0.40)")
    parser.add_argument("--margen-ambiguedad", type=float, default=0.005, help="Margen top1-top2 (default: 0.005)")
    parser.add_argument("--eans-file", default="eans.txt", help="Ruta a eans.txt (default: eans.txt)")
    parser.add_argument("--embeddings-dir", default="catalog/embeddings", help="Directorio embeddings (default: catalog/embeddings)")

    # Output
    parser.add_argument("--output", default="output", help="Directorio base output (default: output)")
    parser.add_argument("--guardar-crops", action="store_true", help="Guardar crops individuales en disco")
    parser.add_argument("--sin-anotaciones", action="store_true", help="No generar imágenes anotadas")

    # Review
    parser.add_argument("--review-dir", default="review", help="Directorio para crops dudosos (default: review)")
    parser.add_argument("--no-review", dest="guardar_review", action="store_false", help="No guardar crops dudosos")
    parser.set_defaults(guardar_review=True)

    # Categorización
    parser.add_argument("--sin_categorias", dest="sin_categorias", action="store_true",
                        help="Desactivar categorización de packaging (CLIP)")

    # Base de datos
    parser.add_argument("--db", action="store_true", help="Persistir resultados en SQL Server")

    # Debug
    parser.add_argument("--verbose", action="store_true", help="Diagnóstico detallado por crop")

    return parser


def main():
    parser = crear_parser()
    args = parser.parse_args()

    # Validar video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: No se encuentra el video: {args.video}")
        sys.exit(1)

    # Validar embeddings
    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.exists() or not list(emb_dir.glob("*.npy")):
        print(f"\n⚠️  No se encontraron embeddings en {args.embeddings_dir}")
        print("   Ejecutá primero: python scripts/agregar_sku.py --todos")
        sys.exit(1)

    # Construir y ejecutar pipeline
    engine = construir_pipeline(args)
    resultado = engine.procesar_video(str(video_path))

    if isinstance(resultado, dict) and "error" in resultado:
        print(f"❌ Error en pipeline: {resultado['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
