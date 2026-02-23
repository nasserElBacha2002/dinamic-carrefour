#!/usr/bin/env python3
"""
Script de revisión rápida de crops dudosos (CLI simple).

Permite revisar crops UNKNOWN/AMBIGUOUS y asignarles el EAN correcto.
Luego, el sistema puede absorber estos crops para mejorar.

Uso:
  python scripts/revisar_crops.py output/IMG_2199_20260219_171424/learning
  python scripts/revisar_crops.py output/IMG_2199_20260219_171424/learning --solo-unknown
  python scripts/revisar_crops.py output/IMG_2199_20260219_171424/learning --solo-ambiguous
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    _VISUAL_AVAILABLE = True
except ImportError:
    _VISUAL_AVAILABLE = False
import os
import tempfile
from datetime import datetime


def _atomic_write_jsonl(path: Path, rows: List[Dict]) -> None:
    """Escritura atómica: escribe a temp y luego reemplaza el archivo original."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)  # atomic en mismo filesystem
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def cargar_metadata(learning_dir: Path) -> List[Dict]:
    """Carga todos los crops desde metadata JSONL."""
    metadata_file = learning_dir / "metadata" / "crops_index.jsonl"
    if not metadata_file.exists():
        print(f"❌ No se encontró {metadata_file}")
        return []
    
    crops = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                crop_data = json.loads(line)
                if crop_data.get("review", {}).get("status") != "reviewed":
                    crops.append(crop_data)
            except json.JSONDecodeError:
                continue
    
    return crops


def mostrar_crop_info(crop_data: Dict, idx: int, total: int):
    """Muestra información de un crop."""
    print("\n" + "=" * 70)
    print(f"Crop {idx + 1}/{total}: {crop_data['crop_id']}")
    print("=" * 70)
    
    # Detección
    det = crop_data.get("detection", {})
    print(f"🔍 Detección YOLO: conf={det.get('yolo_conf', 0.0):.3f}")
    
    # Packaging
    pack = crop_data.get("packaging", {})
    if pack.get("predicted"):
        print(f"📦 Packaging: {pack['predicted']}")
    
    # SKU Identification
    sku = crop_data.get("sku_identification", {})
    decision = sku.get("decision", "unknown")
    print(f"🎯 Decisión: {decision.upper()}")
    
    top_matches = sku.get("all_matches", [])
    if top_matches:
        print(f"\n   Top candidatos:")
        for i, match in enumerate(top_matches[:5], 1):
            ean = match.get("ean", "?")
            sim = match.get("similitud", 0.0)
            desc = match.get("descripcion", "")[:40]
            marker = "👈" if i == 1 else "  "
            print(f"   {marker} {i}. {ean} (sim={sim:.4f}) — {desc}")
    
    # Paths
    paths = crop_data.get("paths", {})
    crop_path = paths.get("crop")
    if crop_path:
        full_path = Path(crop_data.get("execution_id", "")).parent / crop_path
        print(f"\n   📁 Crop: {full_path}")


def revisar_interactivo(crops: List[Dict], learning_dir: Path, solo_tipo: Optional[str] = None):
    """Revisión interactiva de crops."""
    # Filtrar por tipo si se especifica
    if solo_tipo:
        crops = [c for c in crops if c.get("decision") == solo_tipo]
    
    if not crops:
        print(f"✅ No hay crops pendientes de revisión (tipo: {solo_tipo or 'todos'})")
        return
    
    print(f"\n📋 {len(crops)} crops pendientes de revisión")
    print("   Comandos:")
    print("   - EAN: Asignar EAN (ej: 7793890258288)")
    print("   - 's': Saltar este crop")
    print("   - 'q': Salir")
    print("   - 'd': Descartar (no es producto)")
    
    revisados = 0
    for idx, crop_data in enumerate(crops):
        mostrar_crop_info(crop_data, idx, len(crops))
        
        respuesta = input("\n   ➜ EAN correcto (o s/q/d): ").strip()
        
        if respuesta.lower() == "q":
            print("\n👋 Revisión interrumpida")
            break
        elif respuesta.lower() == "s":
            continue
        elif respuesta.lower() == "d":
            # Marcar como descartado
            crop_data["review"]["status"] = "reviewed"
            crop_data["review"]["assigned_ean"] = "DESCARTED"
            revisados += 1
        elif respuesta:
            # Asignar EAN
            ean = respuesta
            crop_data["review"]["status"] = "reviewed"
            crop_data["review"]["assigned_ean"] = ean
            crop_data["review"]["reviewed_at"] = datetime.now().isoformat()
            revisados += 1
            print(f"   ✅ Asignado: {ean}")
        else:
            continue
    
    # Guardar cambios
    if revisados > 0:
        metadata_file = learning_dir / "metadata" / "crops_index.jsonl"
        # Reescribir archivo completo
        todos_crops = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    crop = json.loads(line)
                    # Actualizar si fue modificado
                    for c in crops:
                        if c["crop_id"] == crop["crop_id"]:
                            crop = c
                            break
                    todos_crops.append(crop)
                except json.JSONDecodeError:
                    continue
        
        # Guardar (atomic write)
        _atomic_write_jsonl(metadata_file, todos_crops)
        
        print(f"\n✅ {revisados} crops revisados y guardados")
        print(f"   Próximo paso: ejecutar script para absorber crops al catálogo")


def main():
    parser = argparse.ArgumentParser(
        description="Revisión rápida de crops dudosos (UNKNOWN/AMBIGUOUS)"
    )
    parser.add_argument(
        "learning_dir",
        type=str,
        help="Directorio learning/ de una ejecución"
    )
    parser.add_argument(
        "--solo-unknown",
        action="store_true",
        help="Solo revisar crops UNKNOWN"
    )
    parser.add_argument(
        "--solo-ambiguous",
        action="store_true",
        help="Solo revisar crops AMBIGUOUS"
    )
    
    args = parser.parse_args()
    
    learning_dir = Path(args.learning_dir)
    if not learning_dir.exists():
        print(f"❌ No se encontró directorio: {learning_dir}")
        sys.exit(1)
    
    # Determinar tipo
    solo_tipo = None
    if args.solo_unknown:
        solo_tipo = "unknown"
    elif args.solo_ambiguous:
        solo_tipo = "ambiguous"
    
    # Cargar crops
    crops = cargar_metadata(learning_dir)
    
    if not crops:
        print("✅ No hay crops pendientes de revisión")
        return
    
    # Revisar
    revisar_interactivo(crops, learning_dir, solo_tipo)


if __name__ == "__main__":
    main()
