#!/usr/bin/env python3
"""
Sincronización incremental: eans.txt -> imágenes -> Roboflow.

Objetivo:
  1) Detectar EANs nuevos agregados manualmente a eans.txt
  2) Descargar imágenes para esos EANs nuevos (buscarimagenes.py)
  3) Subir esas imágenes al dataset Roboflow con anotación automática
     (upload_to_roboflow.py --modo catalogo --solo-eans ...)

Estado:
  Guarda eans ya sincronizados en scripts/.eans_sync_state.json

Uso:
  python scripts/sync_eans_to_roboflow.py \
    --proyecto gondolacarrefour/gondola-dataset \
    --api-key TU_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dotenv(dotenv_path: Path) -> None:
    """Carga variables de .env sin dependencias externas."""
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


def read_eans(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
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


def load_state(path: Path) -> Dict:
    if not path.exists():
        return {"processed_eans": [], "history": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("processed_eans", [])
            data.setdefault("history", [])
            return data
    except Exception:
        pass
    return {"processed_eans": [], "history": []}


def save_state(path: Path, state: Dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_cmd(cmd: List[str], cwd: Path) -> int:
    printable = " ".join(cmd)
    print(f"\n$ {printable}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(
        description="Sincroniza EANs nuevos de eans.txt hacia Roboflow (descarga + upload automático)."
    )
    parser.add_argument("--eans-file", default="eans.txt", help="Archivo EAN<TAB>DESCRIPCION")
    parser.add_argument("--imagenes-dir", default="imagenes", help="Directorio de catálogo")
    parser.add_argument("--proyecto", default=os.getenv("ROBOFLOW_PROJECT"), help="Proyecto Roboflow (workspace/slug o slug)")
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"), help="API key Roboflow (si no se pasa, usa ROBOFLOW_API_KEY de .env)")
    parser.add_argument("--per-ean", type=int, default=8, help="Imágenes a descargar por EAN nuevo")
    parser.add_argument("--state-file", default="scripts/.eans_sync_state.json", help="Archivo de estado incremental")
    parser.add_argument("--class-map-file", default="ean_class_map.json", help="JSON EAN->clase para upload_to_roboflow.py")
    parser.add_argument("--label-map-file", default="roboflow_label_map.json", help="Archivo label map para inferencia")
    parser.add_argument("--skip-label-map-sync", action="store_true", help="No ejecutar sync_roboflow_label_map.py al final")
    parser.add_argument("--dry-run", action="store_true", help="No sube ni descarga; solo muestra qué haría")
    args = parser.parse_args()

    eans_path = (repo_root / args.eans_file).resolve()
    state_path = (repo_root / args.state_file).resolve()

    if not args.proyecto:
        print("❌ Falta --proyecto (o ROBOFLOW_PROJECT en .env)")
        return 1
    if not args.api_key:
        print("❌ Falta --api-key (o ROBOFLOW_API_KEY en .env)")
        return 1

    if not eans_path.exists():
        print(f"❌ No existe eans file: {eans_path}")
        return 1

    eans_all = read_eans(eans_path)
    if not eans_all:
        print(f"❌ eans.txt vacío o inválido: {eans_path}")
        return 1

    state = load_state(state_path)
    processed = set(state.get("processed_eans", []))
    current = set(eans_all.keys())
    nuevos = sorted(list(current - processed))

    print("\n" + "=" * 70)
    print("SYNC EANS -> ROBOFLOW")
    print("=" * 70)
    print(f"Total EANs en archivo: {len(current)}")
    print(f"EANs ya sincronizados: {len(processed & current)}")
    print(f"EANs nuevos detectados: {len(nuevos)}")

    if not nuevos:
        print("✅ No hay EANs nuevos para sincronizar.")
        return 0

    print("\nEANs nuevos:")
    for e in nuevos:
        print(f" - {e}\t{eans_all.get(e, '')}")

    if args.dry_run:
        print("\n⚠️  DRY RUN activo. No se ejecutan descargas ni uploads.")
        return 0

    # Crear archivo temporal de eans nuevos para buscarimagenes.py
    tmp_new = repo_root / "scripts" / ".tmp_new_eans.txt"
    with tmp_new.open("w", encoding="utf-8") as f:
        for e in nuevos:
            f.write(f"{e}\t{eans_all.get(e, '')}\n")

    # 1) Descargar imágenes para EANs nuevos
    rc = run_cmd([
        sys.executable,
        str((repo_root / "scripts" / "buscarimagenes.py").resolve()),
        "--input", str(tmp_new),
        "--out", str((repo_root / args.imagenes_dir).resolve()),
        "--per-ean", str(args.per_ean),
        "--shuffle-candidates",
        "--dedupe-global",
    ], cwd=repo_root)
    if rc != 0:
        print("❌ Falló buscarimagenes.py")
        return rc

    # 2) Subir solo EANs nuevos al dataset Roboflow
    solo_eans_csv = ",".join(nuevos)
    rc = run_cmd([
        sys.executable,
        str((repo_root / "scripts" / "upload_to_roboflow.py").resolve()),
        "--modo", "catalogo",
        "--proyecto", args.proyecto,
        "--api-key", args.api_key,
        "--catalogo", str((repo_root / args.imagenes_dir).resolve()),
        "--eans-file", str(eans_path),
        "--class-map-file", str((repo_root / args.class_map_file).resolve()),
        "--solo-eans", solo_eans_csv,
    ], cwd=repo_root)
    if rc != 0:
        print("❌ Falló upload_to_roboflow.py")
        return rc

    # 3) Sincronizar roboflow_label_map.json automáticamente
    if not args.skip_label_map_sync:
        rc = run_cmd([
            sys.executable,
            str((repo_root / "scripts" / "sync_roboflow_label_map.py").resolve()),
            "--eans-file", str(eans_path),
            "--class-map-file", str((repo_root / args.class_map_file).resolve()),
            "--label-map-file", str((repo_root / args.label_map_file).resolve()),
            "--write",
        ], cwd=repo_root)
        if rc != 0:
            print("❌ Falló sync_roboflow_label_map.py")
            return rc

    # Actualizar estado solo si todo salió bien
    processed_new = sorted(list(processed | set(nuevos)))
    state["processed_eans"] = processed_new
    state.setdefault("history", []).append({
        "ts": utc_now_iso(),
        "new_eans": nuevos,
        "count": len(nuevos),
        "project": args.proyecto,
        "per_ean": args.per_ean,
    })
    save_state(state_path, state)

    try:
        tmp_new.unlink(missing_ok=True)
    except Exception:
        pass

    print("\n✅ Sincronización completada")
    print(f"   EANs procesados en total: {len(processed_new)}")
    print(f"   Estado guardado en: {state_path}")
    print("\nPróximo paso:")
    print("  - Revisar anotaciones en Roboflow Annotate")
    print("  - Generar nueva versión y re-entrenar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

