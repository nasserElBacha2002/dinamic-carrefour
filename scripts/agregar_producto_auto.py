#!/usr/bin/env python3
"""
Alta automática de producto nuevo al flujo Roboflow.

Hace en un solo comando:
1) Agrega EAN+descripción a eans.txt (si no existe)
2) Ejecuta sync_eans_to_roboflow.py (descarga imágenes + upload dataset)
3) Sincroniza roboflow_label_map.json
4) (Opcional) sube pre-anotaciones desde un video
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def run_cmd(cmd: list[str], cwd: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def ensure_ean_line(eans_path: Path, ean: str, descripcion: str) -> bool:
    """
    Agrega línea al eans.txt si no existe.
    Returns True si agregó, False si ya existía.
    """
    if not eans_path.exists():
        eans_path.write_text("", encoding="utf-8")

    lines = eans_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if line.strip().startswith(f"{ean}\t") or line.strip() == ean:
            return False

    with eans_path.open("a", encoding="utf-8") as f:
        if lines and lines[-1].strip():
            f.write("\n")
        f.write(f"{ean}\t{descripcion}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alta automática de producto nuevo (eans + sync dataset + label map)."
    )
    parser.add_argument("--ean", required=True, help="EAN del producto")
    parser.add_argument("--descripcion", required=True, help="Descripción comercial del producto")
    parser.add_argument("--eans-file", default="eans.txt")
    parser.add_argument("--per-ean", type=int, default=8, help="Imágenes a descargar para el EAN nuevo")
    parser.add_argument("--proyecto", default=None, help="Proyecto Roboflow (workspace/slug). Si no, usa ROBOFLOW_PROJECT")
    parser.add_argument("--api-key", default=None, help="API key Roboflow. Si no, usa ROBOFLOW_API_KEY")
    parser.add_argument("--video", default=None, help="Video opcional para subir pre-anotaciones")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS para pre-anotaciones de video")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    proyecto = args.proyecto or os.getenv("ROBOFLOW_PROJECT")
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")

    if not proyecto:
        print("❌ Falta --proyecto o ROBOFLOW_PROJECT en .env")
        return 1
    if not api_key:
        print("❌ Falta --api-key o ROBOFLOW_API_KEY en .env")
        return 1

    eans_path = (root / args.eans_file).resolve()

    if args.dry_run:
        print("⚠️  DRY RUN activo")
        print(f"Agregaría/validaría EAN: {args.ean}\t{args.descripcion}")
        print("Luego ejecutaría sync_eans_to_roboflow.py y sync_roboflow_label_map.py")
        if args.video:
            print(f"Y subiría pre-anotaciones desde video: {args.video}")
        return 0

    added = ensure_ean_line(eans_path, args.ean, args.descripcion)
    if added:
        print(f"✅ EAN agregado a {eans_path.name}: {args.ean}")
    else:
        print(f"ℹ️  EAN ya existía en {eans_path.name}: {args.ean}")

    rc = run_cmd(
        [
            sys.executable,
            str((root / "scripts" / "sync_eans_to_roboflow.py").resolve()),
            "--eans-file",
            str(eans_path),
            "--proyecto",
            proyecto,
            "--api-key",
            api_key,
            "--per-ean",
            str(args.per_ean),
        ],
        cwd=root,
    )
    if rc != 0:
        return rc

    rc = run_cmd(
        [
            sys.executable,
            str((root / "scripts" / "sync_roboflow_label_map.py").resolve()),
            "--eans-file",
            str(eans_path),
            "--write",
        ],
        cwd=root,
    )
    if rc != 0:
        return rc

    if args.video:
        rc = run_cmd(
            [
                sys.executable,
                str((root / "scripts" / "upload_to_roboflow.py").resolve()),
                "--modo",
                "frames",
                "--video",
                args.video,
                "--proyecto",
                proyecto,
                "--api-key",
                api_key,
                "--fps",
                str(args.fps),
            ],
            cwd=root,
        )
        if rc != 0:
            return rc

    print("\n✅ Alta automática completada")
    print("Siguiente paso: revisar anotaciones en Roboflow, entrenar versión nueva y probar inferencia.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

