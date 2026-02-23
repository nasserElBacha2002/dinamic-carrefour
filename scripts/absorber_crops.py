#!/usr/bin/env python3
"""
Script para absorber crops revisados al catálogo.

Lee crops que fueron revisados y asignados a un EAN, los copia a la carpeta
de imágenes del SKU correspondiente, y recalcula embeddings.

Uso:
  python scripts/absorber_crops.py output/IMG_2199_20260219_171424/learning
  python scripts/absorber_crops.py output/IMG_2199_20260219_171424/learning --dry-run
  python scripts/absorber_crops.py output/IMG_2199_20260219_171424/learning --solo-ean 7793890258288
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Asegurar imports desde raíz del proyecto
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sku_identifier.embedder import CLIPEmbedder
from src.sku_identifier.vector_store import VectorStore
import os
import numpy as np

# -----------------------------
# Helpers (validación + dedup)
# -----------------------------

def cargar_eans_validos(eans_file: Path) -> Optional[set]:
    """Carga EANs válidos desde eans.txt (EAN<TAB>...). Retorna None si no existe."""
    try:
        if not eans_file.exists():
            return None
        validos = set()
        with open(eans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ean = line.split("\t")[0].strip()
                if ean:
                    validos.add(ean)
        return validos if validos else None
    except Exception:
        return None


def _cosine_max_sim(emb_new: np.ndarray, emb_matrix: np.ndarray) -> float:
    """Max cosine similarity between emb_new (D,) or (1,D) and emb_matrix (N,D)."""
    if emb_new.ndim == 2:
        emb_new = emb_new[0]
    if emb_matrix.size == 0:
        return -1.0
    # Normalize
    a = emb_matrix.astype(np.float32)
    b = emb_new.astype(np.float32)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b /= (np.linalg.norm(b) + 1e-12)
    sims = a @ b
    return float(np.max(sims)) if sims.size else -1.0


def cargar_embeddings_existentes(embeddings_dir: Path, ean: str) -> Optional[np.ndarray]:
    """Carga embeddings existentes de un SKU si están en disco (catalog/embeddings/<EAN>.npy)."""
    p = embeddings_dir / f"{ean}.npy"
    if not p.exists():
        return None
    try:
        arr = np.load(p)
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr
    except Exception:
        return None
    return None


def cargar_crops_revisados(learning_dir: Path) -> List[Dict]:
    """Carga crops que fueron revisados y tienen EAN asignado."""
    metadata_file = learning_dir / "metadata" / "crops_index.jsonl"
    if not metadata_file.exists():
        print(f"❌ No se encontró {metadata_file}")
        return []
    
    crops_revisados = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                crop_data = json.loads(line)
                review = crop_data.get("review", {})
                # Aceptar tanto "reviewed" como "labeled" (el UI usa "labeled")
                status = review.get("status")
                if status in ("reviewed", "labeled"):
                    ean = review.get("assigned_ean")
                    if ean and ean != "DESCARTED":
                        crop_data["ean_asignado"] = ean
                        crops_revisados.append(crop_data)
            except json.JSONDecodeError:
                continue
    
    return crops_revisados



def resolver_ruta_crop_abs(crop_data: Dict, learning_dir: Path) -> Optional[Path]:
    """Resuelve la ruta absoluta del crop en disco según metadata."""
    paths = crop_data.get("paths", {})
    crop_path_rel = paths.get("crop")
    if not crop_path_rel:
        return None
    crop_path_abs = learning_dir.parent / crop_path_rel
    return crop_path_abs if crop_path_abs.exists() else None


def copiar_crop_a_catalogo(
    crop_data: Dict,
    learning_dir: Path,
    imagenes_dir: Path,
    dry_run: bool = False
) -> Optional[str]:
    """
    Copia un crop revisado a la carpeta de imágenes del SKU.
    
    Returns:
        Ruta al archivo copiado o None si falla
    """
    ean = crop_data.get("ean_asignado")
    if not ean:
        return None
    
    # Ruta del crop original
    paths = crop_data.get("paths", {})
    crop_path_rel = paths.get("crop")
    if not crop_path_rel:
        return None
    
    # Resolver ruta absoluta
    crop_path_abs = learning_dir.parent / crop_path_rel
    if not crop_path_abs.exists():
        print(f"  ⚠️  Crop no encontrado: {crop_path_abs}")
        return None
    
    # Carpeta destino del SKU
    sku_dir = imagenes_dir / ean
    if not dry_run:
        sku_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombre del archivo destino (usar crop_id para evitar duplicados)
    crop_id = crop_data.get("crop_id", "crop")
    # Extraer número secuencial si existe
    nombre_base = crop_id.replace("_", "-")
    extension = crop_path_abs.suffix or ".jpg"
    nombre_destino = f"{nombre_base}{extension}"
    
    destino = sku_dir / nombre_destino
    
    # Si ya existe, agregar sufijo
    if destino.exists() and not dry_run:
        counter = 1
        while destino.exists():
            nombre_sin_ext = destino.stem
            destino = sku_dir / f"{nombre_sin_ext}_abs{counter}{extension}"
            counter += 1
    
    if dry_run:
        print(f"  [DRY-RUN] Copiaría: {crop_path_abs.name} → {sku_dir.name}/{destino.name}")
        return str(destino)
    
    try:
        shutil.copy2(crop_path_abs, destino)
        return str(destino)
    except Exception as e:
        print(f"  ❌ Error copiando {crop_path_abs.name}: {e}")
        return None


def absorber_crops(
    learning_dir: Path,
    imagenes_dir: Path = Path("imagenes"),
    embeddings_dir: Path = Path("catalog/embeddings"),
    eans_file: Path = Path("eans.txt"),
    clip_model: str = "ViT-B/32",
    solo_ean: Optional[str] = None,
    dry_run: bool = False,
    dedup_threshold: float = 0.95,
    no_dedup: bool = False,
) -> Dict:
    """
    Absorbe crops revisados al catálogo.
    
    Returns:
        Dict con estadísticas del proceso
    """
    print("=" * 70)
    print("🔄 ABSORBIENDO CROPS REVISADOS AL CATÁLOGO")
    print("=" * 70)
    
    if dry_run:
        print("   ⚠️  MODO DRY-RUN (no se realizarán cambios)")
    
    # Cargar crops revisados
    crops_revisados = cargar_crops_revisados(learning_dir)
    
    if not crops_revisados:
        print("✅ No hay crops revisados para absorber")
        return {"copiados": 0, "recalculados": 0, "errores": 0, "dedup_skipped": 0, "invalid_ean": 0}
    
    # Filtrar por EAN si se especifica
    if solo_ean:
        crops_revisados = [c for c in crops_revisados if c.get("ean_asignado") == solo_ean]
        print(f"   Filtrando por EAN: {solo_ean}")
    
    print(f"\n📋 {len(crops_revisados)} crops revisados encontrados")
    
    # Agrupar por EAN
    por_ean: Dict[str, List[Dict]] = {}
    for crop in crops_revisados:
        ean = crop.get("ean_asignado")
        if ean:
            por_ean.setdefault(ean, []).append(crop)
    
    print(f"   SKUs afectados: {len(por_ean)}")
    

    # Preparar validación de EANs (evita basura en catálogo)
    eans_validos = None if dry_run else cargar_eans_validos(eans_file)
    if eans_validos is not None:
        print(f"   ✅ Validación EAN habilitada ({len(eans_validos)} EANs en {eans_file})")
    else:
        if not dry_run:
            print(f"   ⚠️  No se pudo validar EAN (no existe o vacío): {eans_file} — se aceptarán EANs manuales")

    # Preparar deduplicación por embedding (crítico para evitar crecimiento basura)
    embedder_dedup = None
    embeddings_cache: Dict[str, np.ndarray] = {}
    dedup_skipped = 0
    invalid_ean = 0

    if (not dry_run) and (not no_dedup):
        embedder_dedup = CLIPEmbedder(modelo=clip_model)
        print(f"   ✅ Deduplicación por embedding activada (threshold={dedup_threshold:.2f})")
    elif not dry_run:
        print("   ⚠️  Deduplicación desactivada (--no-dedup)")

    # Copiar crops
    copiados = 0
    errores = 0
    
    for ean, crops in sorted(por_ean.items()):
        print(f"\n   📦 {ean} ({len(crops)} crops):")
        for crop in crops:
            # Validar EAN contra catálogo (si está disponible)
            if eans_validos is not None and ean not in eans_validos:
                invalid_ean += 1
                errores += 1
                print(f"      ❌ EAN inválido (no está en {eans_file.name}): {ean} — skip")
                continue

            # Resolver crop path (necesario para dedup)
            crop_path_abs = resolver_ruta_crop_abs(crop, learning_dir)
            if crop_path_abs is None:
                errores += 1
                print("      ⚠️  Crop no encontrado en disco — skip")
                continue

            # Deduplicación por embedding (antes de copiar)
            if embedder_dedup is not None:
                try:
                    emb_new = embedder_dedup.embed_batch([str(crop_path_abs)])
                    if emb_new is None or getattr(emb_new, "shape", (0,))[0] == 0:
                        raise RuntimeError("embedding vacío")
                    if ean not in embeddings_cache:
                        existing = cargar_embeddings_existentes(embeddings_dir, ean)
                        embeddings_cache[ean] = existing if existing is not None else np.empty((0, emb_new.shape[1]), dtype=np.float32)
                    max_sim = _cosine_max_sim(emb_new, embeddings_cache[ean])
                    if max_sim >= dedup_threshold:
                        dedup_skipped += 1
                        print(f"      ⏭️  DEDUP skip (max_sim={max_sim:.4f} ≥ {dedup_threshold:.2f}) — {crop_path_abs.name}")
                        continue
                    # Aceptado: agregar al cache para dedup intra-ejecución
                    embeddings_cache[ean] = np.vstack([embeddings_cache[ean], emb_new.astype(np.float32)])
                except Exception as e:
                    # Si falla dedup, no frenamos el proceso (pero avisamos)
                    print(f"      ⚠️  Dedup falló, se copiará igual: {e}")

            resultado = copiar_crop_a_catalogo(crop, learning_dir, imagenes_dir, dry_run)
            if resultado:
                copiados += 1
                print(f"      ✅ {Path(resultado).name}")
            else:
                errores += 1
    
    # Recalcular embeddings para SKUs afectados
    if not dry_run and copiados > 0:
        print(f"\n🔄 Recalculando embeddings para {len(por_ean)} SKUs...")
        
        embedder = CLIPEmbedder(modelo=clip_model)
        store = VectorStore(
            embeddings_dir=str(embeddings_dir),
            dimension=embedder.dimension,
        )
        
        recalculos = 0
        for ean in sorted(por_ean.keys()):
            sku_dir = imagenes_dir / ean
            if not sku_dir.exists():
                print(f"  ⚠️  Carpeta no existe: {sku_dir}")
                continue
            
            # Buscar todas las imágenes (incluyendo las nuevas)
            extensiones = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
            imagenes = [
                str(f) for f in sorted(sku_dir.iterdir())
                if f.suffix.lower() in extensiones and f.is_file()
            ]
            
            if not imagenes:
                print(f"  ⚠️  Sin imágenes para {ean}")
                continue
            
            print(f"  📸 {ean} — {len(imagenes)} imágenes → calculando embeddings...")
            
            # Calcular embeddings
            embeddings = embedder.embed_batch(imagenes)
            
            if embeddings.shape[0] == 0:
                print(f"  ❌ {ean} — No se pudo generar embeddings")
                continue
            
            # Guardar en store
            store.agregar_sku(
                ean=ean,
                embeddings=embeddings,
                metadata={
                    "n_imagenes": len(imagenes),
                    "actualizado_desde": "absorber_crops",
                },
            )
            
            recalculos += 1
            print(f"  ✅ {ean} — {embeddings.shape[0]} embeddings guardados")
        
        print(f"\n✅ Proceso completado:")
        print(f"   Crops copiados: {copiados}")
        print(f"   SKUs actualizados: {recalculos}")
        print(f"   Errores: {errores}")
        
        return {
            "copiados": copiados,
            "recalculados": recalculos,
            "errores": errores,
            "dedup_skipped": dedup_skipped,
            "invalid_ean": invalid_ean,
        }
    else:
        print(f"\n✅ Proceso completado (dry-run):")
        print(f"   Crops que se copiarían: {copiados}")
        print(f"   SKUs que se actualizarían: {len(por_ean)}")
        
        return {
            "copiados": copiados,
            "recalculados": 0,
            "errores": errores,
            "dedup_skipped": dedup_skipped,
            "invalid_ean": invalid_ean,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Absorber crops revisados al catálogo"
    )
    parser.add_argument(
        "learning_dir",
        type=str,
        help="Directorio learning/ de una ejecución"
    )
    parser.add_argument(
        "--imagenes-dir",
        default="imagenes",
        help="Directorio de imágenes de referencia (default: imagenes)"
    )
    parser.add_argument(
        "--embeddings-dir",
        default="catalog/embeddings",
        help="Directorio de embeddings (default: catalog/embeddings)"
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B/32",
        help="Modelo CLIP (default: ViT-B/32)"
    )

    parser.add_argument(
        "--eans-file",
        default="eans.txt",
        help="Archivo de EANs válidos para validación (default: eans.txt)"
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Threshold de deduplicación por embedding (default: 0.95)"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Desactivar deduplicación por embedding (no recomendado)"
    )

    parser.add_argument(
        "--solo-ean",
        type=str,
        help="Solo absorber crops de un EAN específico"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular sin hacer cambios"
    )
    
    args = parser.parse_args()
    
    learning_dir = Path(args.learning_dir)
    if not learning_dir.exists():
        print(f"❌ No se encontró directorio: {learning_dir}")
        sys.exit(1)
    
    imagenes_dir = Path(args.imagenes_dir)
    embeddings_dir = Path(args.embeddings_dir)
    
    absorber_crops(
        learning_dir=learning_dir,
        imagenes_dir=imagenes_dir,
        embeddings_dir=embeddings_dir,
        eans_file=Path(args.eans_file),
        clip_model=args.clip_model,
        solo_ean=args.solo_ean,
        dry_run=args.dry_run,
        dedup_threshold=args.dedup_threshold,
        no_dedup=args.no_dedup,
    )


if __name__ == "__main__":
    main()
