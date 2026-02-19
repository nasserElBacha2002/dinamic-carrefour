#!/usr/bin/env python3
"""
Script de upload automático a Roboflow.

Modos de operación:
  --modo catalogo  : Sube las imágenes del catálogo (imagenes/EAN/*.jpg) con
                     anotaciones full-frame y la clase correcta para cada EAN.
                     Ideal para enseñarle al modelo Coca-Cola, nuevas variantes, etc.

  --modo frames    : Extrae frames de un video, corre el modelo Roboflow actual
                     para obtener predicciones, y las sube como pre-anotaciones.
                     Acelerá la revisión: en vez de anotar de cero, solo confirmás.

  --modo imagenes  : Toma imágenes existentes (por ejemplo output/*/reporte_deteccion)
                     y las sube con pre-anotaciones inferidas.

Uso básico:
  # Subir catálogo (fotos de referencia de productos)
  python scripts/upload_to_roboflow.py \
    --modo catalogo \
    --proyecto gondolacarrefour/gondola-dataset

  # Subir frames con pre-anotaciones del modelo actual
  python scripts/upload_to_roboflow.py \
    --modo frames \
    --video data/IMG_2196.MOV \
    --proyecto gondolacarrefour/gondola-dataset

Nota sobre clases:
  El script usa nombres de clase específicos (ver EAN_TO_CLASS abajo).
  Asegurate de que tu proyecto Roboflow tenga las mismas clases,
  o Roboflow las creará automáticamente.
"""

import os
import sys
import cv2
import json
import time
import base64
import re
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# ─── Configuración del proyecto ───────────────────────────────────────────────

# Mapeo EAN → nombre de clase visual en Roboflow
# Regla: clases específicas, sin espacios, en minúsculas
DEFAULT_EAN_TO_CLASS = {
    "7750496":      "pepsi_225",         # Pepsi Regular 2.25 L
    "7791813421719": "pepsi_15",          # Pepsi Regular 1.5 L
    "7791813423775": "pepsi_black_15",    # Pepsi Black 1.5 L
    "7790895000997": "cocacola_225",      # Coca-Cola Regular 2.25 L
    "7790895000430": "cocacola_15",       # Coca-Cola Regular 1.5 L
    "7790895001130": "cocacola_zero_15",  # Coca-Cola Zero 1.5 L
}

# Estado mutable (puede sobreescribirse por archivo eans + class-map)
EAN_TO_CLASS = dict(DEFAULT_EAN_TO_CLASS)

# Mapeo de clases antiguas del modelo (las que ya devuelve) → nuevas
# Se usa al convertir predicciones del modelo en pre-anotaciones
OLD_CLASS_TO_NEW = {
    "pepsi":      "pepsi_225",     # El modelo sabe que es pepsi, asumimos 2.25
    "pepsi black": "pepsi_black_15",
    "bottle":     "bottle",        # Genérico — sin cambios
}

# Clases completas ordenadas (el índice = class_id en formato YOLO)
DEFAULT_ALL_CLASSES = [
    "pepsi_225",       # 0
    "pepsi_15",        # 1
    "pepsi_black_15",  # 2
    "cocacola_225",    # 3
    "cocacola_15",     # 4
    "cocacola_zero_15",# 5
    "bottle",          # 6  (genérico, sin EAN)
]

ALL_CLASSES = list(DEFAULT_ALL_CLASSES)
CLASS_TO_ID = {name: idx for idx, name in enumerate(ALL_CLASSES)}

# ─── Constantes ───────────────────────────────────────────────────────────────

ROBOFLOW_UPLOAD_URL = "https://api.roboflow.com/dataset/{project_slug}/upload"
ROBOFLOW_ANNOTATE_URL = "https://api.roboflow.com/dataset/{project_slug}/annotate/{image_id}"
ROBOFLOW_WORKFLOW_URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "{workspace}/{workflow_id}"
)

# Workflow para inferencia (el que ya tenemos funcionando)
INFERENCE_WORKSPACE = "gondolacarrefour"
INFERENCE_WORKFLOW = "find-bottles-pepsis-pepsi-1s-pepsi-blacks-and-5-lts"

# ─── Utilidades ───────────────────────────────────────────────────────────────

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

def parse_eans_file(eans_file: str) -> Dict[str, str]:
    """
    Lee eans.txt (EAN<TAB>DESCRIPCION) y devuelve dict ean -> descripcion.
    """
    result: Dict[str, str] = {}
    path = Path(eans_file)
    if not path.exists():
        return result

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        ean = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        if ean:
            result[ean] = desc
    return result


def generate_class_name_from_ean(ean: str, descripcion: str = "") -> str:
    """
    Genera nombre de clase estable para nuevos EANs.
    Formato por defecto: ean_<codigo>.
    """
    ean_digits = re.sub(r"[^0-9]", "", ean)
    if ean_digits:
        return f"ean_{ean_digits}"
    safe = re.sub(r"[^a-z0-9]+", "_", (descripcion or ean).lower()).strip("_")
    return f"ean_{safe[:40]}" if safe else f"ean_{ean}"


def rebuild_class_tables(ean_to_class: Dict[str, str]) -> None:
    """
    Recalcula ALL_CLASSES y CLASS_TO_ID globales.
    """
    global EAN_TO_CLASS, ALL_CLASSES, CLASS_TO_ID
    EAN_TO_CLASS = dict(ean_to_class)

    classes: List[str] = []
    for cls in DEFAULT_ALL_CLASSES:
        if cls not in classes:
            classes.append(cls)
    for cls in EAN_TO_CLASS.values():
        if cls not in classes:
            classes.append(cls)
    for cls in OLD_CLASS_TO_NEW.values():
        if cls not in classes:
            classes.append(cls)

    ALL_CLASSES = classes
    CLASS_TO_ID = {name: idx for idx, name in enumerate(ALL_CLASSES)}


def load_or_build_ean_class_map(
    eans_file: str,
    class_map_file: str
) -> Dict[str, str]:
    """
    Construye el mapeo EAN->clase usando:
    1) defaults conocidos
    2) class-map json persistente (si existe)
    3) auto-generación para EANs nuevos (ean_<ean>)
    """
    eans_dict = parse_eans_file(eans_file)
    mapping = dict(DEFAULT_EAN_TO_CLASS)

    map_path = Path(class_map_file)
    if map_path.exists():
        try:
            loaded = json.loads(map_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                        mapping[k.strip()] = v.strip()
        except Exception as e:
            print(f"⚠️  No se pudo leer class-map '{class_map_file}': {e}")

    changed = False
    for ean, desc in eans_dict.items():
        if ean not in mapping:
            mapping[ean] = generate_class_name_from_ean(ean, desc)
            changed = True

    if changed or not map_path.exists():
        map_path.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"💾 Class map actualizado: {class_map_file} ({len(mapping)} EANs)")

    return mapping

def bbox_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
    class_name: str
) -> Optional[str]:
    """
    Convierte bbox en píxeles a formato YOLO (normalizado).
    Devuelve línea de texto: "class_id cx cy w h"
    """
    class_id = CLASS_TO_ID.get(class_name)
    if class_id is None:
        print(f"  ⚠️  Clase desconocida: '{class_name}' — omitida")
        return None

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    # Clamp a [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.001, min(1.0, w))
    h  = max(0.001, min(1.0, h))

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def full_frame_annotation(img_w: int, img_h: int, class_name: str) -> Optional[str]:
    """Genera anotación YOLO que cubre el 85% central de la imagen."""
    margin_x = int(img_w * 0.075)
    margin_y = int(img_h * 0.075)
    return bbox_to_yolo(
        margin_x, margin_y,
        img_w - margin_x, img_h - margin_y,
        img_w, img_h, class_name
    )


def normalize_project_slug(project: str) -> str:
    """
    Normaliza el parámetro --proyecto para endpoints de Dataset API.

    Importante:
      - Workflows API usa "workspace/workflow_id"
      - Dataset API usa SOLO "project_slug"

    Si el usuario pasa "workspace/project_slug", extraemos el último segmento.
    """
    project = project.strip().strip("/")
    if "/" in project:
        parts = [p for p in project.split("/") if p]
        if len(parts) >= 2:
            workspace = "/".join(parts[:-1])
            slug = parts[-1]
            print(f"ℹ️  Dataset API usa solo project slug.")
            print(f"   Recibido: '{project}' → usando slug: '{slug}' (workspace='{workspace}')")
            return slug
    return project


def upload_image(
    api_key: str,
    project: str,
    image_path: str,
    split: str = "train",
    upload_name: Optional[str] = None,
) -> Tuple[Optional[str], bool]:
    """
    Sube una imagen a Roboflow y devuelve (image_id, es_duplicada).
    """
    project_slug = normalize_project_slug(project)
    url = ROBOFLOW_UPLOAD_URL.format(project_slug=project_slug)
    filename = Path(image_path).name
    effective_name = upload_name or filename

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    try:
        resp = requests.post(
            url,
            params={"api_key": api_key},
            files={"file": (effective_name, img_bytes, "image/jpeg")},
            data={"name": effective_name, "split": split},
            timeout=60
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"    ❌ HTTP Error al subir imagen: {e}")
        print(f"    Respuesta: {resp.text[:300]}")
        return None, False
    except requests.exceptions.ConnectionError:
        print(f"    ❌ Sin conexión a internet")
        return None, False
    except Exception as e:
        print(f"    ❌ Error inesperado: {e}")
        return None, False

    data = resp.json()

    # Roboflow devuelve {"success": true, "id": "...", ...} o {"image": {"id": ...}}
    image_id = (
        data.get("id") or
        (data.get("image") or {}).get("id")
    )

    if not image_id:
        # La imagen ya existe (duplicado)
        if data.get("duplicate"):
            existing = data.get("image", {}).get("id")
            print(f"    ℹ️  Imagen duplicada, usando ID existente")
            return existing, True
        print(f"    ⚠️  No se pudo obtener image_id. Respuesta: {json.dumps(data)[:300]}")
        return None, False

    return image_id, False


def upload_annotation(
    api_key: str,
    project: str,
    image_id: str,
    image_name: str,
    yolo_annotation: str
) -> bool:
    """
    Sube anotaciones YOLO para una imagen ya subida.
    """
    project_slug = normalize_project_slug(project)
    url = ROBOFLOW_ANNOTATE_URL.format(project_slug=project_slug, image_id=image_id)

    # Roboflow detecta mejor el formato cuando el nombre de anotación termina en .txt
    ann_name = Path(image_name).with_suffix(".txt").name

    try:
        resp = requests.post(
            url,
            params={"api_key": api_key, "name": ann_name, "annotation_type": "yolo"},
            data=yolo_annotation,
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"    ❌ HTTP Error al subir anotación: {e}")
        print(f"    Respuesta: {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"    ❌ Error en anotación: {e}")
        return False

    return True


def _extract_predictions_from_workflow_response(data: Dict) -> List[Dict]:
    """Extrae predicciones desde distintos formatos de salida de Workflows."""
    if not isinstance(data, dict):
        return []
    outputs = data.get("outputs", [])
    if not outputs or not isinstance(outputs, list):
        return []
    first = outputs[0] if outputs else {}
    if not isinstance(first, dict):
        return []

    preds_data = first.get("predictions")
    if isinstance(preds_data, dict) and isinstance(preds_data.get("predictions"), list):
        return preds_data.get("predictions", [])
    if isinstance(preds_data, list):
        return preds_data

    model_preds = first.get("model_predictions")
    if isinstance(model_preds, dict) and isinstance(model_preds.get("predictions"), list):
        return model_preds.get("predictions", [])
    if isinstance(model_preds, list):
        return model_preds

    return []


def inferir_frame(api_key: str, image_path: str) -> List[Dict]:
    """
    Corre inferencia con el modelo Roboflow existente y devuelve predicciones.
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    workspace = os.getenv("ROBOFLOW_WORKSPACE", INFERENCE_WORKSPACE)
    workflow_id = os.getenv("ROBOFLOW_WORKFLOW", INFERENCE_WORKFLOW)
    url = ROBOFLOW_WORKFLOW_URL.format(workspace=workspace, workflow_id=workflow_id)

    try:
        resp = requests.post(
            url,
            json={
                "api_key": api_key,
                "inputs": {"image": {"type": "base64", "value": img_b64}}
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"    ⚠️  Error en inferencia: {e}")
        return []

    data = resp.json()
    return _extract_predictions_from_workflow_response(data)


def normalizar_label_prediccion(pred: Dict) -> str:
    """
    Convierte label devuelto por Roboflow a clase del proyecto.
    Soporta:
      - class_id numérico (0..N)
      - class textual legacy
      - class textual ya normalizado
    """
    raw = pred.get("class_id", None)
    if raw is None:
        raw = pred.get("class", "")

    label_raw = str(raw).strip()
    if not label_raw:
        return ""

    # Caso 1: ya viene con nombre de clase final
    if label_raw in CLASS_TO_ID:
        return label_raw

    # Caso 2: numérico -> usar tabla ALL_CLASSES por índice
    if label_raw.isdigit():
        idx = int(label_raw)
        if 0 <= idx < len(ALL_CLASSES):
            return ALL_CLASSES[idx]

    # Caso 3: clases antiguas / alias
    return OLD_CLASS_TO_NEW.get(label_raw, label_raw)


def extraer_frames(video_path: str, fps: float = 1.0) -> List[str]:
    """
    Extrae frames de un video y los guarda temporalmente.
    Devuelve lista de rutas a los frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {video_path}")
        return []

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    intervalo = max(1, int(fps_video / fps))

    video_name = Path(video_path).stem
    tmp_dir = Path(f"/tmp/roboflow_frames_{video_name}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    frames_guardados = []
    frame_idx = 0
    frame_count = 0

    print(f"   📹 {Path(video_path).name}: {total_frames} frames @ {fps_video:.1f}fps")
    print(f"   Extrayendo cada {intervalo} frames (~{fps} fps)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % intervalo == 0:
            t = frame_idx / fps_video
            fname = f"frame_{frame_count:04d}_t{t:.2f}s.jpg"
            fpath = tmp_dir / fname
            cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frames_guardados.append(str(fpath))
            frame_count += 1

        frame_idx += 1

    cap.release()
    print(f"   ✅ {len(frames_guardados)} frames extraídos → {tmp_dir}")
    return frames_guardados


# ─── Modo Catálogo ────────────────────────────────────────────────────────────

def modo_catalogo(
    api_key: str,
    project: str,
    catalogo_dir: str,
    split: str,
    dry_run: bool,
    solo_eans: Optional[Set[str]] = None,
    name_prefix: str = "",
):
    """
    Sube todas las imágenes del catálogo (imagenes/EAN/*.jpg) con
    anotaciones full-frame de la clase correspondiente.
    """
    print("\n" + "=" * 60)
    print("MODO CATÁLOGO — Subiendo imágenes de referencia")
    print("=" * 60)
    print(f"Proyecto: {project}")
    print(f"Directorio: {catalogo_dir}")
    print(f"Split: {split}")
    if dry_run:
        print("⚠️  DRY RUN — no se harán uploads reales")
    print()

    cat_path = Path(catalogo_dir)
    if not cat_path.exists():
        print(f"❌ No existe el directorio: {catalogo_dir}")
        return

    total_ok = 0
    total_dup = 0
    total_skip = 0
    total_err = 0

    # Recorrer cada carpeta EAN
    for ean_dir in sorted(cat_path.iterdir()):
        if not ean_dir.is_dir():
            continue

        ean = ean_dir.name
        if solo_eans and ean not in solo_eans:
            continue
        class_name = EAN_TO_CLASS.get(ean)

        if class_name is None:
            print(f"⚠️  EAN {ean}: sin clase definida en EAN_TO_CLASS — saltando")
            total_skip += len(list(ean_dir.glob("*.jpg")))
            continue

        imgs = sorted(ean_dir.glob("*.jpg")) + sorted(ean_dir.glob("*.png"))

        if not imgs:
            print(f"⚠️  {ean}: sin imágenes — saltando")
            continue

        print(f"\n📦 EAN {ean} → clase '{class_name}' ({len(imgs)} imgs)")

        for img_path in imgs:
            img_name = img_path.name
            print(f"   {img_name}...", end=" ", flush=True)

            # Leer dimensiones
            img = cv2.imread(str(img_path))
            if img is None:
                print("❌ no se pudo leer")
                total_err += 1
                continue

            h, w = img.shape[:2]

            # Generar anotación full-frame
            yolo_line = full_frame_annotation(w, h, class_name)
            if yolo_line is None:
                print("❌ clase no en ALL_CLASSES")
                total_err += 1
                continue

            if dry_run:
                print(f"✅ (dry-run) [{w}x{h}] → '{class_name}' | {yolo_line}")
                total_ok += 1
                continue

            # Subir imagen
            upload_name = f"{name_prefix}{img_name}" if name_prefix else img_name
            image_id, is_dup = upload_image(
                api_key, project, str(img_path), split, upload_name=upload_name
            )
            if not image_id:
                total_err += 1
                continue
            if is_dup:
                total_dup += 1

            # Subir anotación
            ok = upload_annotation(api_key, project, image_id, img_name, yolo_line)
            if ok:
                status = "DUP" if is_dup else "NEW"
                print(f"✅ [{status}] id={image_id[:8]}... clase='{class_name}'")
                total_ok += 1
            else:
                total_err += 1

            time.sleep(0.3)  # Respetar rate limit

    print("\n" + "─" * 60)
    print(f"✅ Subidas OK:     {total_ok}")
    print(f"♻️  Duplicadas:     {total_dup}")
    print(f"⏭️  Saltadas:      {total_skip}")
    print(f"❌ Con error:      {total_err}")
    print("─" * 60)

    if not dry_run and total_ok > 0:
        print(f"\n🎯 Próximos pasos en Roboflow:")
        print(f"   1. Ir a https://app.roboflow.com/{project.split('/')[0]}")
        print(f"   2. Revisar imágenes en Annotate → revisar las anotaciones")
        print(f"   3. Generar nueva versión del dataset")
        print(f"   4. Re-entrenar el modelo")


# ─── Modo Frames ──────────────────────────────────────────────────────────────

def modo_frames(
    api_key: str,
    project: str,
    video_path: str,
    fps: float,
    split: str,
    confianza_minima: float,
    dry_run: bool,
    name_prefix: str = "",
):
    """
    Extrae frames de un video, infiere con el modelo actual,
    y sube frames + predicciones como pre-anotaciones a Roboflow.
    """
    print("\n" + "=" * 60)
    print("MODO FRAMES — Subiendo frames con pre-anotaciones")
    print("=" * 60)
    print(f"Video:    {video_path}")
    print(f"Proyecto: {project}")
    print(f"FPS:      {fps}")
    print(f"Confianza mínima: {confianza_minima}")
    if dry_run:
        print("⚠️  DRY RUN — no se harán uploads reales")
    print()

    if not Path(video_path).exists():
        print(f"❌ Video no encontrado: {video_path}")
        return

    # Extraer frames
    print("📹 Extrayendo frames...")
    frames = extraer_frames(video_path, fps=fps)

    if not frames:
        print("❌ No se pudieron extraer frames")
        return

    total_ok = 0
    total_dup = 0
    total_sin_detecciones = 0
    total_err = 0

    print(f"\n🌐 Procesando {len(frames)} frames...")

    for i, frame_path in enumerate(frames):
        fname = Path(frame_path).name
        print(f"\n[{i+1}/{len(frames)}] {fname}")

        # Leer dimensiones
        img = cv2.imread(frame_path)
        if img is None:
            print(f"   ❌ no se pudo leer")
            total_err += 1
            continue

        h_img, w_img = img.shape[:2]

        # Inferir con modelo actual
        print(f"   🔍 Infiriendo...", end=" ", flush=True)
        predicciones = inferir_frame(api_key, frame_path)

        # Filtrar por confianza y convertir a YOLO
        yolo_lines = []
        for pred in predicciones:
            conf = pred.get("confidence", 0.0)
            if conf < confianza_minima:
                continue

            label_new = normalizar_label_prediccion(pred)

            if label_new not in CLASS_TO_ID:
                continue  # Clase no registrada

            # Roboflow devuelve centro + ancho/alto (en píxeles)
            cx_px = pred["x"]
            cy_px = pred["y"]
            w_px  = pred["width"]
            h_px  = pred["height"]

            x1 = cx_px - w_px / 2
            y1 = cy_px - h_px / 2
            x2 = cx_px + w_px / 2
            y2 = cy_px + h_px / 2

            line = bbox_to_yolo(x1, y1, x2, y2, w_img, h_img, label_new)
            if line:
                yolo_lines.append(line)

        print(f"{len(yolo_lines)} detecciones válidas")

        if not yolo_lines:
            total_sin_detecciones += 1
            # Subir frame de todas formas (sin anotación) para tener datos
            if not dry_run:
                base = Path(frame_path).name
                upload_name = f"{name_prefix}{base}" if name_prefix else base
                image_id, is_dup = upload_image(
                    api_key, project, frame_path, split, upload_name=upload_name
                )
                if image_id and is_dup:
                    total_dup += 1
                print(f"   ⬆️  Frame subido sin anotaciones")
            continue

        # Mostrar resumen de detecciones
        from collections import Counter
        clases_detectadas = []
        for line in yolo_lines:
            cid = int(line.split()[0])
            clases_detectadas.append(ALL_CLASSES[cid])
        conteo = Counter(clases_detectadas)
        print(f"   Classes: {dict(conteo)}")

        annotation_text = "\n".join(yolo_lines)

        if dry_run:
            print(f"   ✅ (dry-run) {len(yolo_lines)} anotaciones")
            print(f"   Annotation preview:")
            for line in yolo_lines[:3]:
                print(f"      {line}")
            total_ok += 1
            continue

        # Subir imagen
        base = Path(frame_path).name
        upload_name = f"{name_prefix}{base}" if name_prefix else base
        image_id, is_dup = upload_image(
            api_key, project, frame_path, split, upload_name=upload_name
        )
        if not image_id:
            total_err += 1
            continue
        if is_dup:
            total_dup += 1

        # Subir anotaciones
        ok = upload_annotation(api_key, project, image_id, fname, annotation_text)
        if ok:
            status = "DUP" if is_dup else "NEW"
            print(f"   ✅ [{status}] Subido con {len(yolo_lines)} anotaciones (id={image_id[:8]}...)")
            total_ok += 1
        else:
            total_err += 1

        time.sleep(0.5)  # Rate limit

    print("\n" + "─" * 60)
    print(f"✅ Subidos con anotaciones: {total_ok}")
    print(f"♻️  Duplicadas:            {total_dup}")
    print(f"⏭️  Sin detecciones:        {total_sin_detecciones}")
    print(f"❌ Con error:               {total_err}")
    print("─" * 60)

    if not dry_run and total_ok > 0:
        print(f"\n🎯 Próximos pasos:")
        print(f"   1. Ir a https://app.roboflow.com/{project.split('/')[0]}")
        print(f"   2. Annotate → revisar frames (las pre-anotaciones ya están)")
        print(f"   3. Corregir errores (especialmente clases 'bottle' que sean Coca-Cola)")
        print(f"   4. Aprobar → generar dataset → re-entrenar")


def modo_imagenes(
    api_key: str,
    project: str,
    imagenes_dir: str,
    split: str,
    confianza_minima: float,
    dry_run: bool,
    recursivo: bool = True,
    solo_clases: Optional[Set[str]] = None,
    name_prefix: str = "",
):
    """
    Sube imágenes existentes con pre-anotaciones inferidas.
    Ideal para reutilizar output/*/reporte_deteccion.
    """
    print("\n" + "=" * 60)
    print("MODO IMAGENES — Subiendo imágenes existentes con pre-anotaciones")
    print("=" * 60)
    print(f"Directorio: {imagenes_dir}")
    print(f"Proyecto:   {project}")
    print(f"Confianza mínima: {confianza_minima}")
    if solo_clases:
        print(f"Filtro de clases: {sorted(solo_clases)}")
    if dry_run:
        print("⚠️  DRY RUN — no se harán uploads reales")
    print()

    root = Path(imagenes_dir)
    if not root.exists():
        print(f"❌ Directorio no encontrado: {imagenes_dir}")
        return

    patterns = ("*.jpg", "*.jpeg", "*.png")
    imagenes: List[Path] = []
    if recursivo:
        for pat in patterns:
            imagenes.extend(root.rglob(pat))
    else:
        for pat in patterns:
            imagenes.extend(root.glob(pat))
    imagenes = sorted(set(imagenes))

    if not imagenes:
        print("❌ No se encontraron imágenes para subir.")
        return

    total_ok = 0
    total_dup = 0
    total_sin_detecciones = 0
    total_err = 0

    print(f"🖼️  Total imágenes encontradas: {len(imagenes)}")

    for i, img_path in enumerate(imagenes):
        fname = img_path.name
        print(f"\n[{i+1}/{len(imagenes)}] {fname}")

        img = cv2.imread(str(img_path))
        if img is None:
            print("   ❌ no se pudo leer")
            total_err += 1
            continue
        h_img, w_img = img.shape[:2]

        predicciones = inferir_frame(api_key, str(img_path))
        yolo_lines = []

        for pred in predicciones:
            conf = pred.get("confidence", 0.0)
            if conf < confianza_minima:
                continue

            label_new = normalizar_label_prediccion(pred)
            if not label_new or label_new not in CLASS_TO_ID:
                continue
            if solo_clases and label_new not in solo_clases:
                continue

            if not all(k in pred for k in ("x", "y", "width", "height")):
                continue

            cx_px = pred["x"]
            cy_px = pred["y"]
            w_px = pred["width"]
            h_px = pred["height"]
            x1 = cx_px - w_px / 2
            y1 = cy_px - h_px / 2
            x2 = cx_px + w_px / 2
            y2 = cy_px + h_px / 2

            line = bbox_to_yolo(x1, y1, x2, y2, w_img, h_img, label_new)
            if line:
                yolo_lines.append(line)

        if not yolo_lines:
            print("   ⏭️  sin detecciones válidas para clases objetivo")
            total_sin_detecciones += 1
            continue

        annotation_text = "\n".join(yolo_lines)

        if dry_run:
            print(f"   ✅ (dry-run) {len(yolo_lines)} anotaciones")
            total_ok += 1
            continue

        base = img_path.name
        upload_name = f"{name_prefix}{base}" if name_prefix else base
        image_id, is_dup = upload_image(
            api_key, project, str(img_path), split, upload_name=upload_name
        )
        if not image_id:
            total_err += 1
            continue
        if is_dup:
            total_dup += 1

        ok = upload_annotation(api_key, project, image_id, fname, annotation_text)
        if ok:
            status = "DUP" if is_dup else "NEW"
            print(f"   ✅ [{status}] Subido con {len(yolo_lines)} anotaciones (id={image_id[:8]}...)")
            total_ok += 1
        else:
            total_err += 1

        time.sleep(0.3)

    print("\n" + "─" * 60)
    print(f"✅ Subidos con anotaciones: {total_ok}")
    print(f"♻️  Duplicadas:            {total_dup}")
    print(f"⏭️  Sin detecciones válidas: {total_sin_detecciones}")
    print(f"❌ Con error:               {total_err}")
    print("─" * 60)


# ─── Validación de configuración ──────────────────────────────────────────────

def mostrar_configuracion():
    """Muestra el mapeo de clases configurado."""
    print("\n📋 CONFIGURACIÓN DE CLASES")
    print("─" * 50)
    print(f"{'ID':<4} {'Clase':<20} {'EANs mapeados'}")
    print("─" * 50)
    for idx, clase in enumerate(ALL_CLASSES):
        eans = [ean for ean, c in EAN_TO_CLASS.items() if c == clase]
        eans_str = ", ".join(eans) if eans else "—"
        print(f"{idx:<4} {clase:<20} {eans_str}")
    print("─" * 50)
    print(f"Total: {len(ALL_CLASSES)} clases")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(
        description="Upload automático de imágenes y anotaciones a Roboflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Ver clases configuradas
  python scripts/upload_to_roboflow.py --info

  # Dry-run: ver qué se subiría del catálogo sin hacer nada
  python scripts/upload_to_roboflow.py \\
    --modo catalogo \\
    --proyecto gondolacarrefour/gondola-dataset \\
    --dry-run

  # Subir catálogo real
  python scripts/upload_to_roboflow.py \\
    --modo catalogo \\
    --proyecto gondolacarrefour/gondola-dataset

  # Subir frames de un video con pre-anotaciones
  python scripts/upload_to_roboflow.py \\
    --modo frames \\
    --video data/IMG_2196.MOV \\
    --proyecto gondolacarrefour/gondola-dataset \\
    --fps 1.0

  # Dry-run del modo frames
  python scripts/upload_to_roboflow.py \\
    --modo frames \\
    --video data/IMG_2196.MOV \\
    --proyecto gondolacarrefour/gondola-dataset \\
    --dry-run
        """
    )

    parser.add_argument("--modo", choices=["catalogo", "frames", "imagenes"],
                        help="Modo de operación")
    parser.add_argument("--proyecto", default=os.getenv("ROBOFLOW_PROJECT"),
                        help="Proyecto Roboflow: workspace/project-slug")
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"),
                        help="API Key de Roboflow (si no se pasa, usa ROBOFLOW_API_KEY de .env)")
    parser.add_argument("--catalogo", default="imagenes",
                        help="Directorio del catálogo (default: imagenes/)")
    parser.add_argument("--eans-file", default="eans.txt",
                        help="Archivo EAN<TAB>DESCRIPCION (default: eans.txt)")
    parser.add_argument("--class-map-file", default="ean_class_map.json",
                        help="JSON persistente EAN->clase (default: ean_class_map.json)")
    parser.add_argument("--solo-eans", default=None,
                        help="Procesar solo estos EANs (coma separada), útil para sync incremental")
    parser.add_argument("--video", default=None,
                        help="Ruta al video (requerido en modo frames)")
    parser.add_argument("--imagenes-dir", default=None,
                        help="Directorio de imágenes existentes (requerido en modo imagenes)")
    parser.add_argument("--no-recursivo", action="store_true",
                        help="En modo imagenes, no buscar recursivamente")
    parser.add_argument("--solo-clases", default=None,
                        help="Filtrar clases objetivo (coma separada), ej: cocacola_225,cocacola_15,cocacola_zero_15,ean_7790315058201")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames por segundo a extraer (default: 1.0)")
    parser.add_argument("--split", default="train",
                        choices=["train", "valid", "test"],
                        help="Split de Roboflow (default: train)")
    parser.add_argument("--confianza", type=float, default=0.40,
                        help="Confianza mínima para incluir como pre-anotación (default: 0.40)")
    parser.add_argument("--name-prefix", default="",
                        help="Prefijo para nombre de imagen al subir (evita sobreescritura por nombre)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simular sin hacer uploads reales")
    parser.add_argument("--info", action="store_true",
                        help="Mostrar configuración de clases y salir")

    args = parser.parse_args()

    # Construir mapeo dinámico de clases desde eans + class-map
    try:
        dynamic_map = load_or_build_ean_class_map(args.eans_file, args.class_map_file)
        rebuild_class_tables(dynamic_map)
    except Exception as e:
        print(f"⚠️  No se pudo construir class-map dinámico ({e}). Se usan defaults.")
        rebuild_class_tables(DEFAULT_EAN_TO_CLASS)

    # Modo informativo
    if args.info:
        mostrar_configuracion()
        return

    # Validar args mínimos
    if not args.modo:
        parser.print_help()
        print("\n❌ Especificá --modo catalogo o --modo frames")
        sys.exit(1)

    if not args.proyecto:
        print("❌ Especificá --proyecto workspace/project-slug")
        print("   Ejemplo: --proyecto gondolacarrefour/gondola-dataset")
        print("\n   Para crear el proyecto:")
        print("   1. Ir a https://app.roboflow.com")
        print("   2. New Project → Object Detection")
        print("   3. Usar el slug en --proyecto")
        sys.exit(1)

    if not args.api_key:
        print("❌ Especificá --api-key o definí ROBOFLOW_API_KEY en .env")
        sys.exit(1)

    if args.modo == "frames" and not args.video:
        print("❌ En modo 'frames' debés especificar --video ruta/al/video.MOV")
        sys.exit(1)
    if args.modo == "imagenes" and not args.imagenes_dir:
        print("❌ En modo 'imagenes' debés especificar --imagenes-dir")
        sys.exit(1)

    # Ejecutar modo correspondiente
    solo_eans_set = None
    if args.solo_eans:
        solo_eans_set = {x.strip() for x in args.solo_eans.split(",") if x.strip()}
    solo_clases_set = None
    if args.solo_clases:
        solo_clases_set = {x.strip() for x in args.solo_clases.split(",") if x.strip()}

    if args.modo == "catalogo":
        modo_catalogo(
            api_key=args.api_key,
            project=args.proyecto,
            catalogo_dir=args.catalogo,
            split=args.split,
            dry_run=args.dry_run,
            solo_eans=solo_eans_set,
            name_prefix=args.name_prefix,
        )
    elif args.modo == "frames":
        modo_frames(
            api_key=args.api_key,
            project=args.proyecto,
            video_path=args.video,
            fps=args.fps,
            split=args.split,
            confianza_minima=args.confianza,
            dry_run=args.dry_run,
            name_prefix=args.name_prefix,
        )
    elif args.modo == "imagenes":
        modo_imagenes(
            api_key=args.api_key,
            project=args.proyecto,
            imagenes_dir=args.imagenes_dir,
            split=args.split,
            confianza_minima=args.confianza,
            dry_run=args.dry_run,
            recursivo=not args.no_recursivo,
            solo_clases=solo_clases_set,
            name_prefix=args.name_prefix,
        )


if __name__ == "__main__":
    main()
