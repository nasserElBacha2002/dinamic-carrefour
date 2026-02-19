#!/usr/bin/env python3
"""
Genera/actualiza roboflow_label_map.json automáticamente desde:
  - eans.txt (EAN -> descripción)
  - ean_class_map.json (EAN -> class_name)

Objetivo:
  Evitar edición manual del label map cada vez que agregás EANs nuevos.

Uso:
  # Ver preview sin escribir
  python scripts/sync_roboflow_label_map.py

  # Escribir cambios en roboflow_label_map.json
  python scripts/sync_roboflow_label_map.py --write
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def parse_eans_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        ean = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        if ean:
            data[ean] = desc
    return data


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def build_label_map(
    eans_dict: Dict[str, str],
    ean_to_class: Dict[str, str],
    existing_map: Dict[str, Any],
    preserve_extra_labels: bool = True,
) -> Dict[str, Any]:
    """
    Construye el JSON final de label map.
    - actualiza/agrega todas las clases derivadas de ean_class_map
    - preserva metadatos existentes (_keys) y labels extra si se solicita
    """
    out: Dict[str, Any] = {}

    # Metadatos
    for k, v in existing_map.items():
        if isinstance(k, str) and k.startswith("_"):
            out[k] = v
    out["_sync_info"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_eans": "eans.txt",
        "source_class_map": "ean_class_map.json",
        "note": "Auto-generado por scripts/sync_roboflow_label_map.py",
    }

    # Preservar labels extra existentes (por ejemplo bottle / pepsi legacy)
    if preserve_extra_labels:
        mapped_classes = set(ean_to_class.values())
        for label, payload in existing_map.items():
            if not isinstance(label, str) or label.startswith("_"):
                continue
            if label in mapped_classes:
                continue  # Se regenera abajo
            if isinstance(payload, dict) and ("ean" in payload or "descripcion" in payload):
                out[label] = payload

    # Generar labels desde ean->class
    for ean, class_name in sorted(ean_to_class.items()):
        desc = eans_dict.get(ean, f"Producto EAN {ean}")
        out[class_name] = {
            "ean": ean,
            "descripcion": desc,
        }

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sincroniza roboflow_label_map.json desde eans.txt + ean_class_map.json")
    parser.add_argument("--eans-file", default="eans.txt", help="Ruta a eans.txt")
    parser.add_argument("--class-map-file", default="ean_class_map.json", help="Ruta a ean_class_map.json")
    parser.add_argument("--label-map-file", default="roboflow_label_map.json", help="Ruta a roboflow_label_map.json")
    parser.add_argument("--write", action="store_true", help="Escribir cambios en disco")
    parser.add_argument("--no-preserve-extra", action="store_true", help="No preservar labels legacy/extras existentes")
    parser.add_argument("--backup", action="store_true", help="Crear backup .bak antes de escribir")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    eans_path = (root / args.eans_file).resolve()
    class_map_path = (root / args.class_map_file).resolve()
    label_map_path = (root / args.label_map_file).resolve()

    eans_dict = parse_eans_file(eans_path)
    ean_to_class = load_json(class_map_path)
    existing_map = load_json(label_map_path)

    if not eans_dict:
        print(f"❌ eans vacío o inexistente: {eans_path}")
        return 1
    if not ean_to_class:
        print(f"❌ class-map vacío o inexistente: {class_map_path}")
        print("   Ejecutá primero: python scripts/upload_to_roboflow.py --info")
        return 1

    # Filtrar class-map a EANs presentes
    filtered_map = {ean: cls for ean, cls in ean_to_class.items() if ean in eans_dict and isinstance(cls, str)}
    missing_eans = sorted([ean for ean in eans_dict if ean not in filtered_map])

    generated = build_label_map(
        eans_dict=eans_dict,
        ean_to_class=filtered_map,
        existing_map=existing_map,
        preserve_extra_labels=not args.no_preserve_extra,
    )

    print("\n" + "=" * 68)
    print("SYNC ROBOFLOW LABEL MAP")
    print("=" * 68)
    print(f"EANs en archivo:            {len(eans_dict)}")
    print(f"EANs con clase asignada:    {len(filtered_map)}")
    print(f"Labels generados por EAN:   {len(filtered_map)}")
    if missing_eans:
        print(f"⚠️  EANs sin clase asignada ({len(missing_eans)}):")
        for e in missing_eans:
            print(f"   - {e}")
    print(f"Preservar labels extra:     {not args.no_preserve_extra}")
    print(f"Archivo destino:            {label_map_path}")

    if not args.write:
        print("\nPreview (primeros 12 labels):")
        shown = 0
        for k, v in generated.items():
            if k.startswith("_"):
                continue
            print(f"  - {k}: {v.get('ean')}")
            shown += 1
            if shown >= 12:
                break
        print("\nℹ️  Modo preview. Usá --write para guardar.")
        return 0

    if args.backup and label_map_path.exists():
        backup_path = label_map_path.with_suffix(label_map_path.suffix + ".bak")
        backup_path.write_text(label_map_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"💾 Backup: {backup_path}")

    label_map_path.write_text(
        json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"✅ Label map actualizado: {label_map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

