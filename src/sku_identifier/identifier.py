#!/usr/bin/env python3
"""
Identificador de SKU — Orquesta embedder + vector store + decisión.

Recibe un crop de producto y devuelve:
  - EAN identificado (o UNKNOWN)
  - Confianza (similitud coseno)
  - Top-K candidatos
  - Status: matched / unknown / ambiguous
"""

import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.sku_identifier.embedder import CLIPEmbedder
from src.sku_identifier.vector_store import VectorStore
from src.sku_identifier.categorizer import PackagingCategorizer


class SKUIdentifier:
    """
    Identifica SKUs a partir de crops de productos.

    Flujo:
      1. Recibe crop → genera embedding con CLIP.
      2. Clasifica tipo de packaging (botella, lata, bolsa, etc.).
      3. Busca en vector store SOLO dentro de esa categoría → top-K candidatos.
      4. Aplica lógica de decisión → matched / unknown / ambiguous.
      5. Si dudoso, guarda el crop en review/ para revisión humana.
    """

    def __init__(
        self,
        embedder: CLIPEmbedder,
        vector_store: VectorStore,
        categorizer: Optional[PackagingCategorizer] = None,
        eans_file: str = "eans.txt",
        threshold: float = 0.75,
        threshold_unknown: float = 0.40,
        margen_ambiguedad: float = 0.05,
        review_dir: Optional[str] = "review",
        guardar_review: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            embedder: Instancia de CLIPEmbedder.
            vector_store: Instancia de VectorStore.
            categorizer: Clasificador de packaging (None = busca en todo).
            eans_file: Ruta a eans.txt para descripciones.
            threshold: Similitud mínima para match (0-1).
            threshold_unknown: Por debajo = UNKNOWN directo (0-1).
            margen_ambiguedad: Si top1 - top2 < margen → ambiguous.
            review_dir: Carpeta donde guardar crops dudosos.
            guardar_review: Si True, guarda crops UNKNOWN/ambiguous.
            verbose: Si True, imprime diagnóstico detallado por crop.
        """
        self.embedder = embedder
        self.store = vector_store
        self.categorizer = categorizer
        self.threshold = threshold
        self.threshold_unknown = threshold_unknown
        self.margen_ambiguedad = margen_ambiguedad
        self.guardar_review = guardar_review
        self.verbose = verbose

        # Directorio de review por sesión
        if review_dir:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.review_dir = Path(review_dir) / ts
        else:
            self.review_dir = None

        # Cargar descripciones y categorías
        self._descripciones, self._categorias_ean = self._cargar_catalogo(eans_file)

        # Contadores para la sesión
        self.stats = {
            "matched": 0, "unknown": 0, "ambiguous": 0, "total": 0,
            "categorias_detectadas": {}
        }

    def _cargar_catalogo(self, eans_file: str) -> tuple:
        """Lee eans.txt y devuelve ({ean: descripcion}, {ean: categoria})."""
        descripciones = {}
        categorias = {}
        path = Path(eans_file)
        if not path.exists():
            return descripciones, categorias

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                ean = parts[0].strip()
                descripciones[ean] = parts[1].strip()
                if len(parts) >= 3:
                    categorias[ean] = parts[2].strip()

        return descripciones, categorias

    def identificar(
        self,
        crop_path: str,
        top_k: int = 3,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Identifica el SKU de un crop.

        Args:
            crop_path: Ruta al crop del producto.
            top_k: Candidatos a retornar.
            threshold: Override del umbral.

        Returns:
            Dict con:
                - ean: str (EAN o "UNKNOWN")
                - descripcion: str
                - confianza: float (similitud del mejor match)
                - top_matches: List[Dict] (top-k candidatos con ean, similitud, desc)
                - status: "matched" | "unknown" | "ambiguous"
        """
        th = threshold if threshold is not None else self.threshold

        self.stats["total"] += 1

        # Generar embedding del crop
        embedding = self.embedder.embed(crop_path)
        if embedding is None:
            self.stats["unknown"] += 1
            return self._resultado_unknown(crop_path, "Error generando embedding")

        # Clasificar packaging (si disponible)
        categoria_detectada = None
        if self.categorizer:
            categoria_detectada = self.categorizer.clasificar(embedding)

        # Buscar en store (filtrado por categoría)
        candidatos = self.store.buscar(embedding, top_k=top_k, categoria=categoria_detectada)
        if not candidatos:
            self.stats["unknown"] += 1
            return self._resultado_unknown(crop_path, "Store vacío")

        top1_ean, top1_sim = candidatos[0]
        top_matches = [
            {
                "ean": ean,
                "similitud": round(sim, 4),
                "descripcion": self._descripciones.get(ean, ""),
            }
            for ean, sim in candidatos
        ]

        # Lógica de decisión
        if top1_sim < self.threshold_unknown:
            # Muy baja similitud → UNKNOWN
            status = "unknown"
            self.stats["unknown"] += 1
            self._guardar_en_review(crop_path, status, top_matches)
            return {
                "ean": "UNKNOWN",
                "descripcion": "",
                "confianza": round(top1_sim, 4),
                "top_matches": top_matches,
                "status": status,
            }

        if top1_sim < th:
            # Por debajo del threshold pero no es ultra-bajo → ambiguous
            status = "ambiguous"
            self.stats["ambiguous"] += 1
            self._guardar_en_review(crop_path, status, top_matches)
            return {
                "ean": "UNKNOWN",
                "descripcion": "",
                "confianza": round(top1_sim, 4),
                "top_matches": top_matches,
                "status": status,
            }

        # Chequear ambigüedad: top1 ≈ top2
        if len(candidatos) >= 2:
            top2_sim = candidatos[1][1]
            if (top1_sim - top2_sim) < self.margen_ambiguedad:
                status = "ambiguous"
                self.stats["ambiguous"] += 1
                self._guardar_en_review(crop_path, status, top_matches)
                return {
                    "ean": "UNKNOWN",
                    "descripcion": "",
                    "confianza": round(top1_sim, 4),
                    "top_matches": top_matches,
                    "status": status,
                }

        # Match confiable
        status = "matched"
        self.stats["matched"] += 1
        return {
            "ean": top1_ean,
            "descripcion": self._descripciones.get(top1_ean, ""),
            "confianza": round(top1_sim, 4),
            "top_matches": top_matches,
            "status": status,
        }

    def identificar_crop_numpy(
        self,
        crop: np.ndarray,
        crop_id: str = "crop",
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Identifica SKU desde un crop ya cargado en memoria (numpy BGR).

        Args:
            crop: Imagen numpy BGR.
            crop_id: Identificador para review/logging.
            top_k: Candidatos.
            threshold: Override.

        Returns:
            Mismo formato que identificar().
        """
        th = threshold if threshold is not None else self.threshold

        self.stats["total"] += 1

        embedding = self.embedder.embed_crop(crop)
        if embedding is None:
            self.stats["unknown"] += 1
            return {
                "ean": "UNKNOWN", "descripcion": "", "confianza": 0.0,
                "top_matches": [], "status": "unknown", "categoria": ""
            }

        # Clasificar packaging (si el categorizador está disponible)
        categoria_detectada = None
        if self.categorizer:
            categoria_detectada = self.categorizer.clasificar(embedding)
            # Contar categorías detectadas para estadísticas
            self.stats["categorias_detectadas"][categoria_detectada] = \
                self.stats["categorias_detectadas"].get(categoria_detectada, 0) + 1

        # Buscar en store (filtrado por categoría si está disponible)
        candidatos = self.store.buscar(
            embedding, top_k=top_k, categoria=categoria_detectada
        )
        if not candidatos:
            self.stats["unknown"] += 1
            return {
                "ean": "UNKNOWN", "descripcion": "", "confianza": 0.0,
                "top_matches": [], "status": "unknown",
                "categoria": categoria_detectada or ""
            }

        top1_ean, top1_sim = candidatos[0]
        top_matches = [
            {
                "ean": ean,
                "similitud": round(sim, 4),
                "descripcion": self._descripciones.get(ean, ""),
            }
            for ean, sim in candidatos
        ]

        # Decisión
        if top1_sim < self.threshold_unknown:
            status = "unknown"
        elif top1_sim < th:
            status = "ambiguous"
        elif len(candidatos) >= 2 and (top1_sim - candidatos[1][1]) < self.margen_ambiguedad:
            status = "ambiguous"
        else:
            status = "matched"

        self.stats[status] += 1

        ean_final = top1_ean if status == "matched" else "UNKNOWN"
        desc_final = self._descripciones.get(ean_final, "") if status == "matched" else ""

        # Diagnóstico verbose
        if self.verbose:
            icon = {"matched": "✅", "ambiguous": "⚠️", "unknown": "❌"}.get(status, "?")
            cat_str = f" [{categoria_detectada}]" if categoria_detectada else ""
            margin_str = ""
            if len(candidatos) >= 2:
                margin_str = f" Δ={top1_sim - candidatos[1][1]:.4f}"
            print(f"   {icon} {crop_id}{cat_str}: {status} → {ean_final} "
                  f"(sim={top1_sim:.4f}{margin_str})")
            for m in top_matches[:3]:
                tag = "👈" if m["ean"] == ean_final else "  "
                desc_short = m["descripcion"][:35] if m["descripcion"] else ""
                print(f"      {tag} {m['ean']} {m['similitud']:.4f}  {desc_short}")

        return {
            "ean": ean_final,
            "descripcion": desc_final,
            "confianza": round(top1_sim, 4),
            "top_matches": top_matches,
            "status": status,
            "categoria": categoria_detectada or "",
        }

    def _resultado_unknown(self, crop_path: str, razon: str) -> Dict:
        """Genera resultado UNKNOWN estándar."""
        return {
            "ean": "UNKNOWN",
            "descripcion": "",
            "confianza": 0.0,
            "top_matches": [],
            "status": "unknown",
        }

    def _guardar_en_review(
        self,
        crop_path: str,
        status: str,
        top_matches: List[Dict]
    ) -> None:
        """Guarda crop + metadata en review/ para revisión humana."""
        if not self.guardar_review or not self.review_dir:
            return

        self.review_dir.mkdir(parents=True, exist_ok=True)

        # Copiar crop
        src = Path(crop_path)
        if not src.exists():
            return

        count = len(list(self.review_dir.glob(f"{status}_*.jpg")))
        dst_name = f"{status}_{count:04d}{src.suffix}"
        dst = self.review_dir / dst_name
        shutil.copy2(str(src), str(dst))

        # Guardar metadata
        meta = {
            "status": status,
            "crop_original": str(src),
            "top_matches": top_matches,
            "timestamp": datetime.now().isoformat(),
        }
        meta_path = dst.with_suffix(".json")
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def resumen_sesion(self) -> str:
        """Resumen de la sesión de identificación."""
        s = self.stats
        total = s["total"] or 1
        lines = [
            f"🏷️  Identificación SKU — Resumen de sesión",
            f"   Total crops procesados: {s['total']}",
            f"   ✅ Matched:   {s['matched']} ({s['matched']*100/total:.0f}%)",
            f"   ❓ Ambiguous: {s['ambiguous']} ({s['ambiguous']*100/total:.0f}%)",
            f"   ❌ Unknown:   {s['unknown']} ({s['unknown']*100/total:.0f}%)",
        ]

        # Estadísticas por categoría de packaging
        cat_stats = s.get("categorias_detectadas", {})
        if cat_stats:
            lines.append(f"\n   📦 Detecciones por tipo de packaging:")
            for cat, count in sorted(cat_stats.items(), key=lambda x: -x[1]):
                lines.append(f"      {cat}: {count} ({count*100/total:.0f}%)")

        if self.review_dir and self.review_dir.exists():
            n_review = len(list(self.review_dir.glob("*.jpg")))
            if n_review:
                lines.append(f"   📁 Crops en review: {n_review} → {self.review_dir}")
        return "\n".join(lines)
