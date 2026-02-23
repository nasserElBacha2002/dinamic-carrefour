#!/usr/bin/env python3
"""
Report Generator — CSV + anotaciones + resumen.
"""

from __future__ import annotations

import csv
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import Counter


class ReportGenerator:
    def __init__(self, reporte_dir: Path, generar_anotaciones: bool = True):
        self.reporte_dir = reporte_dir
        self.generar_anotaciones = bool(generar_anotaciones)
        self.reporte_dir.mkdir(parents=True, exist_ok=True)

    def generar_csv_inventario(self, conteo_dedup: Counter, fecha_str: Optional[str] = None) -> Path:
        if fecha_str is None:
            fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        csv_path = self.reporte_dir / "inventario_sku.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["EAN", "Cantidad", "Fecha"])
            for ean, cantidad in sorted(conteo_dedup.items()):
                if ean != "UNKNOWN":
                    writer.writerow([ean, int(cantidad), fecha_str])
        return csv_path

    def anotar_frame(
        self,
        frame: np.ndarray,
        detecciones: List[Dict[str, Any]],
        etiquetas_sku: Dict[int, str],
        frame_idx: int,
    ) -> None:
        if not self.generar_anotaciones:
            return

        out_img = str(self.reporte_dir / f"frame_{frame_idx:05d}.jpg")
        annotated = frame.copy()

        for i, det in enumerate(detecciones):
            x1a, y1a, x2a, y2a = det.get("bbox_padded") or det["bbox"]
            label = etiquetas_sku.get(i, f'product {float(det.get("confianza", 0.0)):.2f}')
            color = (0, 200, 0) if not label.startswith("UNKNOWN") else (0, 0, 220)

            cv2.rectangle(annotated, (x1a, y1a), (x2a, y2a), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_top = max(0, y1a - th - 8)
            cv2.rectangle(annotated, (x1a, y_top), (x1a + tw + 4, y1a), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1a + 2, max(0, y1a - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # ✅ se escribe UNA sola vez (no dentro del loop)
        cv2.imwrite(out_img, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def mostrar_resumen(
        self,
        resultado_procesamiento: Dict[str, Any],
        conteo_dedup: Counter,
        learning_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        print("\n" + "=" * 70)
        print("📋 RESUMEN DEL PROCESAMIENTO")
        print("=" * 70)
        print(f"   Frames muestreados:    {resultado_procesamiento['frames_total']}")
        print(f"   Frames con producto:   {resultado_procesamiento['frames_con_producto']}")
        print(f"   Total detecciones raw: {resultado_procesamiento['total_detecciones']}")
        skus_unicos = [k for k in conteo_dedup if k != "UNKNOWN"]
        print(f"   SKUs identificados:    {len(skus_unicos)}")

        if learning_summary:
            print("\n   📚 Dataset evolutivo:")
            print(f"      Crops guardados: {learning_summary.get('total_crops_saved')}")
            print(f"      UNKNOWN: {learning_summary.get('unknown_count')}")
            print(f"      AMBIGUOUS: {learning_summary.get('ambiguous_count')}")
            print(f"      Metadata: {learning_summary.get('metadata_file')}")

        print("=" * 70)

    @staticmethod
    def deduplicar_por_frame(conteo_por_frame: List[Counter]) -> Counter:
        conteo_dedup: Counter = Counter()
        for cf in conteo_por_frame:
            for ean, cant in cf.items():
                if conteo_dedup[ean] < cant:
                    conteo_dedup[ean] = cant
        return conteo_dedup