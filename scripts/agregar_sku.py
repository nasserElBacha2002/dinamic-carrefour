#!/usr/bin/env python3
"""
Script para agregar SKUs al vector store.

Lee eans.txt + imágenes de referencia → calcula embeddings CLIP → guarda en catalog/embeddings/.

Uso:
  # Calcular embeddings para TODOS los SKUs de eans.txt
  python scripts/agregar_sku.py --todos

  # Calcular embeddings para UN SKU específico
  python scripts/agregar_sku.py --ean 7750496

  # Recalcular todos (forzar, aunque ya existan)
  python scripts/agregar_sku.py --todos --forzar

  # Ver estado actual del catálogo
  python scripts/agregar_sku.py --status
"""

import argparse
import re
import sys
from pathlib import Path

# Asegurar imports desde raíz del proyecto
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sku_identifier.embedder import CLIPEmbedder
from src.sku_identifier.vector_store import VectorStore


def cargar_eans(eans_file: str = "eans.txt") -> dict:
    """Lee eans.txt → {ean: {descripcion, categoria}}."""
    eans = {}
    path = Path(eans_file)
    if not path.exists():
        print(f"❌ No se encontró {eans_file}")
        return eans

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Intentar primero con tabs (formato preferido)
        if "\t" in line:
            parts = line.split("\t")
        else:
            # Fallback: espacios múltiples (más robusto)
            parts = re.split(r"\s{2,}", line)
        
        if len(parts) >= 2:
            ean = parts[0].strip()
            desc = parts[1].strip()
            cat = parts[2].strip() if len(parts) >= 3 else ""
            eans[ean] = {"descripcion": desc, "categoria": cat}

    return eans


def buscar_imagenes_ean(ean: str, imagenes_dir: str = "imagenes") -> list:
    """Busca imágenes de referencia para un EAN."""
    dir_ean = Path(imagenes_dir) / ean
    if not dir_ean.exists():
        return []

    extensiones = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
    imagenes = [
        str(f) for f in sorted(dir_ean.iterdir())
        if f.suffix.lower() in extensiones and f.is_file()
    ]
    return imagenes


def procesar_ean(
    ean: str,
    info: dict,
    embedder: CLIPEmbedder,
    store: VectorStore,
    imagenes_dir: str = "imagenes",
    forzar: bool = False
) -> bool:
    """
    Calcula y guarda embeddings para un EAN.

    Args:
        ean: Código EAN.
        info: Dict con 'descripcion' y 'categoria'.
        embedder: Instancia CLIPEmbedder.
        store: Instancia VectorStore.
        imagenes_dir: Directorio base de imágenes.
        forzar: Recalcular aunque ya exista.

    Returns:
        True si se procesó, False si se saltó.
    """
    descripcion = info.get("descripcion", "")
    categoria = info.get("categoria", "")

    # Verificar si ya existe y no se fuerza
    if store.tiene_sku(ean) and not forzar:
        print(f"   ⏭️  {ean} ya tiene embeddings (usar --forzar para recalcular)")
        return False

    # Buscar imágenes
    imagenes = buscar_imagenes_ean(ean, imagenes_dir)
    if not imagenes:
        print(f"   ⚠️  {ean} — Sin imágenes en {imagenes_dir}/{ean}/")
        return False

    cat_label = f" [{categoria}]" if categoria else ""
    print(f"   📸 {ean}{cat_label} — {len(imagenes)} imágenes → calculando embeddings...")

    # Calcular embeddings en batch
    embeddings = embedder.embed_batch(imagenes)

    if embeddings.shape[0] == 0:
        print(f"   ❌ {ean} — No se pudo generar ningún embedding")
        return False

    # Guardar en store con categoría
    store.agregar_sku(
        ean=ean,
        embeddings=embeddings,
        metadata={
            "descripcion": descripcion,
            "categoria": categoria,
            "n_imagenes": len(imagenes),
        },
    )

    print(f"   ✅ {ean} — {embeddings.shape[0]} embeddings guardados")
    return True


def mostrar_status(store: VectorStore, eans: dict, imagenes_dir: str = "imagenes"):
    """Muestra el estado actual del catálogo."""
    print("\n" + "=" * 70)
    print("📦 ESTADO DEL CATÁLOGO")
    print("=" * 70)

    total_ok = 0
    total_sin_emb = 0
    total_sin_img = 0

    # Agrupar por categoría
    por_categoria = {}
    for ean, info in sorted(eans.items()):
        cat = info.get("categoria", "sin_categoria") or "sin_categoria"
        por_categoria.setdefault(cat, []).append((ean, info))

    for cat, items in sorted(por_categoria.items()):
        print(f"\n   📂 {cat} ({len(items)} SKUs):")
        for ean, info in items:
            desc = info.get("descripcion", "")
            imgs = buscar_imagenes_ean(ean, imagenes_dir)
            tiene_emb = store.tiene_sku(ean)

            if tiene_emb:
                status = "✅"
                total_ok += 1
            elif imgs:
                status = "📸 (sin embeddings)"
                total_sin_emb += 1
            else:
                status = "❌ (sin imágenes)"
                total_sin_img += 1

            print(f"      {status} {ean} ({len(imgs)} imgs) — {desc}")

    print(f"\n   Resumen: {total_ok} listos, {total_sin_emb} sin embeddings, {total_sin_img} sin imágenes")
    print(f"   {store.resumen()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Agregar SKUs al vector store de embeddings"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--todos", action="store_true",
                       help="Procesar todos los EANs de eans.txt")
    group.add_argument("--ean", type=str,
                       help="Procesar un EAN específico")
    group.add_argument("--status", action="store_true",
                       help="Mostrar estado actual del catálogo")

    parser.add_argument("--forzar", action="store_true",
                        help="Recalcular embeddings aunque ya existan")
    parser.add_argument("--eans-file", default="eans.txt",
                        help="Ruta a eans.txt")
    parser.add_argument("--imagenes-dir", default="imagenes",
                        help="Directorio de imágenes de referencia")
    parser.add_argument("--embeddings-dir", default="catalog/embeddings",
                        help="Directorio de embeddings")
    parser.add_argument("--clip-model", default="ViT-B/32",
                        help="Modelo CLIP (default: ViT-B/32)")

    args = parser.parse_args()

    # Cargar catálogo
    eans = cargar_eans(args.eans_file)
    if not eans:
        print("❌ No se encontraron EANs")
        sys.exit(1)

    print(f"📋 Catálogo: {len(eans)} EANs en {args.eans_file}")

    # Inicializar componentes
    embedder = CLIPEmbedder(modelo=args.clip_model)
    store = VectorStore(
        embeddings_dir=args.embeddings_dir,
        dimension=embedder.dimension,
    )

    if args.status:
        mostrar_status(store, eans, args.imagenes_dir)
        return

    if args.ean:
        # Procesar un solo EAN
        if args.ean not in eans:
            print(f"⚠️  EAN {args.ean} no está en {args.eans_file}")
            desc = input("   Descripción (Enter para vacía): ").strip()
            info = {"descripcion": desc, "categoria": ""}
        else:
            info = eans[args.ean]

        procesar_ean(args.ean, info, embedder, store, args.imagenes_dir, args.forzar)

    elif args.todos:
        # Procesar todos
        print(f"\n🔄 Procesando {len(eans)} SKUs...")
        procesados = 0
        errores = 0

        for ean, info in sorted(eans.items()):
            ok = procesar_ean(ean, info, embedder, store, args.imagenes_dir, args.forzar)
            if ok:
                procesados += 1
            elif not store.tiene_sku(ean):
                errores += 1

        print(f"\n✅ Procesamiento completado: {procesados} nuevos, {errores} errores")

    # Mostrar resumen final
    print(f"\n{store.resumen()}")


if __name__ == "__main__":
    main()
