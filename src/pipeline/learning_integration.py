#!/usr/bin/env python3
"""
Learning Integration — Integración con el Learning Manager.

- Inicializa LearningManager si está disponible
- Guarda crops dudosos (UNKNOWN/AMBIGUOUS)
- Guarda métricas históricas globales
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import Counter

try:
    from src.learning.manager import LearningManager
    _LEARNING_AVAILABLE = True
except Exception:
    _LEARNING_AVAILABLE = False


class LearningIntegration:
    def __init__(
        self,
        output_dir: Path,
        execution_id: str,
        video_path: str,
        execution_params: Dict[str, Any],
    ):
        self.learning_manager: Optional["LearningManager"] = None

        if _LEARNING_AVAILABLE:
            try:
                self.learning_manager = LearningManager(
                    output_dir=output_dir,
                    execution_id=execution_id,
                    video_path=video_path,
                    execution_params=execution_params,
                )
                print("   📚 Learning Manager activado (dataset evolutivo)")
            except Exception as e:
                print(f"   ⚠️  No se pudo inicializar Learning Manager: {e}")
                self.learning_manager = None

    def guardar_crops_dudosos(
        self,
        crops_para_learning: List[Dict[str, Any]],
        generar_anotaciones: bool,
        identificador: Any,  # SKUIdentifier
    ) -> None:
        if not self.learning_manager:
            return

        for crop_info in crops_para_learning:
            crop_padded = crop_info["crop_padded"]
            crop_id = crop_info["crop_id"]
            det = crop_info["det"]
            resultado_sku = crop_info["resultado_sku"]
            resultados_identificacion = crop_info["resultados_identificacion"]
            out_idx = int(crop_info["out_idx"])
            j = int(crop_info["j"])
            t_sec = float(crop_info.get("t_sec", 0.0))

            status = str(resultado_sku.get("status", "matched"))

            detection_info = {
                "bbox": det["bbox"],
                "bbox_padded": det.get("bbox_padded") or det["bbox"],
                "yolo_conf": float(det.get("confianza", 0.0)),
                "class_id": int(det.get("class_id", -1)),
                "raw_label": str(det.get("raw_label", "product")),
                "t_sec": t_sec,
                "split": (
                    "L" if (len(resultados_identificacion) == 2 and j == 0)
                    else "R" if (len(resultados_identificacion) == 2 and j == 1)
                    else None
                ),
            }

            packaging_info = {"predicted": str(resultado_sku.get("categoria", ""))}

            top_matches = resultado_sku.get("top_matches") or []
            sku_info = {
                "decision": status,
                "top_matches": top_matches,
                "threshold_used": float(getattr(identificador, "threshold", 0.0) or 0.0),
                "unknown_threshold": float(getattr(identificador, "threshold_unknown", 0.0) or 0.0),
                "margen_ambiguedad": float(getattr(identificador, "margen_ambiguedad", 0.0) or 0.0),
            }

            frame_path_rel = (
                f"reporte_deteccion/frame_{out_idx:05d}.jpg"
                if generar_anotaciones
                else None
            )

            save_crop_id = (
                f"{crop_id}_L" if (len(resultados_identificacion) == 2 and j == 0)
                else f"{crop_id}_R" if (len(resultados_identificacion) == 2 and j == 1)
                else crop_id
            )

            self.learning_manager.guardar_crop_dudoso(
                crop=crop_padded,
                crop_id=save_crop_id,
                frame_path=frame_path_rel,
                frame_idx=out_idx,
                detection_info=detection_info,
                packaging_info=packaging_info,
                sku_info=sku_info,
                decision=status,
            )

    def guardar_metricas_historicas(
        self,
        resultado_procesamiento: Dict[str, Any],
        conteo_dedup: Counter,
    ) -> None:
        if not self.learning_manager:
            return

        try:
            learning_summary = self.learning_manager.resumen()
            total_skus_dedup = sum(v for k, v in conteo_dedup.items() if k != "UNKNOWN")
            unknown_dedup = int(conteo_dedup.get("UNKNOWN", 0))
            skus_unicos = [k for k in conteo_dedup if k != "UNKNOWN"]

            metrics: Dict[str, Any] = {
                "frames_total": int(resultado_procesamiento["frames_total"]),
                "frames_con_producto": int(resultado_procesamiento["frames_con_producto"]),
                "detecciones_raw": int(resultado_procesamiento["total_detecciones"]),
                "skus_identificados_unicos": int(len(skus_unicos)),
                "conteo_dedup_total": int(total_skus_dedup),
                "conteo_dedup_unknown": int(unknown_dedup),
            }

            if learning_summary:
                total_ls = int(learning_summary.get("total_crops_saved") or 0)
                unk_ls = int(learning_summary.get("unknown_count") or 0)
                amb_ls = int(learning_summary.get("ambiguous_count") or 0)
                metrics.update({
                    "learning_total_crops_saved": total_ls,
                    "learning_unknown": unk_ls,
                    "learning_ambiguous": amb_ls,
                    "learning_unknown_rate": (unk_ls / total_ls) if total_ls else 0.0,
                    "learning_ambiguous_rate": (amb_ls / total_ls) if total_ls else 0.0,
                })

            hist_path = self.learning_manager.guardar_metricas_historicas(metrics)
            if hist_path:
                print(f"   📈 Métricas guardadas (histórico): {hist_path}")
        except Exception as e:
            print(f"   ⚠️  No se pudieron guardar métricas históricas: {e}")

    def obtener_resumen(self) -> Optional[Dict[str, Any]]:
        if self.learning_manager:
            return self.learning_manager.resumen()
        return None