#!/usr/bin/env python3
"""
buscarimagenes.py

Descarga imágenes de referencia por EAN/descripcion usando Bing Images Search
(con scraping simple) para armar un dataset "semilla" interno.

Mejoras respecto a la versión original:
- Descarga N imágenes por EAN (no solo 1)
- Parsing más robusto de Bing (extrae URLs reales "murl" desde elementos iusc)
- Validación de descarga: content-type, tamaño mínimo, resolución mínima (si Pillow está disponible)
- Dedupe por hash (evita repetir la misma imagen)
- Guarda metadata (JSONL) por imagen: query, source_url, tamaños, hash, status
- Timeouts + retries + rate limit
- CLI configurable

Uso:
  python buscarimagenes.py --input eans.txt --out imagenes --per-ean 10

Formato esperado de eans.txt (TAB):
  EAN<TAB>DESCRIPCION
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

try:
    from PIL import Image
    from io import BytesIO

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class EANItem:
    ean: str
    descripcion: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_eans_file(path: Path) -> List[EANItem]:
    items: List[EANItem] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        ean = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        if ean:
            items.append(EANItem(ean=ean, descripcion=desc))
    return items


def safe_filename(name: str) -> str:
    # Mantener simple: letras, números, guion y underscore
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", name)
    return name[:200] if len(name) > 200 else name


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def guess_extension(content_type: str) -> str:
    ct = (content_type or "").lower().split(";")[0].strip()
    return {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
    }.get(ct, "jpg")


def get_image_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    if not PIL_AVAILABLE:
        return None
    try:
        im = Image.open(BytesIO(data))
        return im.size  # (w, h)
    except Exception:
        return None


def build_bing_url(query: str) -> str:
    # q: query, form: HDRSC2 suele funcionar, pero no es crítico
    return f"https://www.bing.com/images/search?q={quote_plus(query)}&form=HDRSC2"


def extract_candidate_image_urls(html: str, limit: int = 80) -> List[str]:
    """
    Bing Images suele guardar la URL de la imagen real en el atributo JSON "murl"
    dentro de elementos <a class="iusc" m='{"murl":"..."}'>
    """
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []

    # 1) URLs reales desde iusc
    for a in soup.find_all("a", class_="iusc"):
        m_attr = a.get("m")
        if not m_attr:
            continue
        try:
            payload = json.loads(m_attr)
            murl = payload.get("murl")
            if isinstance(murl, str) and murl.startswith("http"):
                urls.append(murl)
        except Exception:
            continue
        if len(urls) >= limit:
            break

    if len(urls) >= limit:
        return urls[:limit]

    # 2) Fallback: img tags (menos confiable)
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if isinstance(src, str) and src.startswith("http"):
            urls.append(src)
        if len(urls) >= limit:
            break

    # Quitar duplicados manteniendo orden
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:limit]


def request_with_retries(
    session: requests.Session,
    url: str,
    headers: dict,
    timeout: int,
    retries: int,
) -> Optional[requests.Response]:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            return resp
        except Exception as e:
            last_exc = e
            # Backoff simple
            time.sleep(0.4 * (attempt + 1))
    return None


def fetch_bing_candidates(
    session: requests.Session,
    query: str,
    timeout: int,
    retries: int,
    user_agent: str,
) -> List[str]:
    url = build_bing_url(query)
    headers = {"User-Agent": user_agent, "Accept-Language": "es-AR,es;q=0.9,en;q=0.7"}
    resp = request_with_retries(session, url, headers=headers, timeout=timeout, retries=retries)
    if not resp or resp.status_code != 200:
        return []
    return extract_candidate_image_urls(resp.text)


def download_image(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int,
    user_agent: str,
) -> Tuple[Optional[bytes], Optional[str], Optional[int]]:
    headers = {
        "User-Agent": user_agent,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.bing.com/",
    }
    resp = request_with_retries(session, url, headers=headers, timeout=timeout, retries=retries)
    if not resp:
        return None, None, None

    ct = resp.headers.get("Content-Type", "")
    status = resp.status_code
    if status != 200:
        return None, ct, status

    data = resp.content
    return data, ct, status


def is_valid_image_blob(
    data: bytes,
    content_type: str,
    min_bytes: int,
    min_width: int,
    min_height: int,
) -> Tuple[bool, Optional[int], Optional[int], str]:
    # Content-Type
    ct = (content_type or "").lower()
    if "image/" not in ct:
        return False, None, None, "content_type_not_image"

    # Size threshold
    if len(data) < min_bytes:
        return False, None, None, "too_small_bytes"

    # Resolution threshold (si Pillow está disponible)
    dims = get_image_dimensions(data)
    if dims is None:
        # No podemos validar resolución; aceptamos si pasó los checks básicos
        return True, None, None, "ok_no_dims"

    w, h = dims
    if w < min_width or h < min_height:
        return False, w, h, "too_small_resolution"

    return True, w, h, "ok"


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metadata global
    meta_path = out_dir / "metadata.jsonl"
    errors_path = out_dir / "errors.jsonl"

    items = read_eans_file(input_path)
    if not items:
        print(f"No se encontraron EANs en: {input_path}")
        return 1

    # Dedupe global por hash para no duplicar imágenes entre EANs (opcional)
    global_hashes: set[str] = set()

    session = requests.Session()

    for idx, item in enumerate(items, start=1):
        # Query: EAN + descripcion (puede incluir marca/variante)
        query = f"{item.ean} {item.descripcion}".strip()

        print(f"[{idx}/{len(items)}] Buscando imágenes para EAN={item.ean} | {item.descripcion}")

        candidates = fetch_bing_candidates(
            session=session,
            query=query,
            timeout=args.timeout,
            retries=args.retries,
            user_agent=args.user_agent,
        )

        if not candidates:
            append_jsonl(
                errors_path,
                {
                    "ts": utc_now_iso(),
                    "ean": item.ean,
                    "descripcion": item.descripcion,
                    "query": query,
                    "stage": "search",
                    "error": "no_candidates",
                },
            )
            print(f"  Sin resultados candidatos (Bing) para: {item.ean}")
            continue

        # Para diversificar, mezclamos ligeramente
        if args.shuffle_candidates:
            random.shuffle(candidates)

        saved = 0
        attempted = 0

        # Carpeta por EAN
        ean_dir = out_dir / safe_filename(item.ean)
        ean_dir.mkdir(parents=True, exist_ok=True)

        for cand_url in candidates:
            if saved >= args.per_ean:
                break

            attempted += 1

            # Rate limit
            if args.sleep > 0:
                time.sleep(args.sleep)

            data, ct, status = download_image(
                session=session,
                url=cand_url,
                timeout=args.timeout,
                retries=args.retries,
                user_agent=args.user_agent,
            )

            if not data:
                append_jsonl(
                    errors_path,
                    {
                        "ts": utc_now_iso(),
                        "ean": item.ean,
                        "descripcion": item.descripcion,
                        "query": query,
                        "stage": "download",
                        "url": cand_url,
                        "status": status,
                        "content_type": ct,
                        "error": "download_failed",
                    },
                )
                continue

            ok, w, h, reason = is_valid_image_blob(
                data=data,
                content_type=ct or "",
                min_bytes=args.min_bytes,
                min_width=args.min_width,
                min_height=args.min_height,
            )

            if not ok:
                append_jsonl(
                    errors_path,
                    {
                        "ts": utc_now_iso(),
                        "ean": item.ean,
                        "descripcion": item.descripcion,
                        "query": query,
                        "stage": "validate",
                        "url": cand_url,
                        "status": status,
                        "content_type": ct,
                        "bytes": len(data),
                        "width": w,
                        "height": h,
                        "error": reason,
                    },
                )
                continue

            hsh = sha256_bytes(data)

            if args.dedupe_global and hsh in global_hashes:
                # Ya la bajamos para otro EAN
                continue

            # Guardar
            ext = guess_extension(ct or "")
            filename = f"{item.ean}__{saved:03d}.{ext}"
            file_path = ean_dir / filename

            try:
                file_path.write_bytes(data)
            except Exception as e:
                append_jsonl(
                    errors_path,
                    {
                        "ts": utc_now_iso(),
                        "ean": item.ean,
                        "descripcion": item.descripcion,
                        "query": query,
                        "stage": "save",
                        "url": cand_url,
                        "path": str(file_path),
                        "error": str(e),
                    },
                )
                continue

            global_hashes.add(hsh)
            saved += 1

            append_jsonl(
                meta_path,
                {
                    "ts": utc_now_iso(),
                    "ean": item.ean,
                    "descripcion": item.descripcion,
                    "query": query,
                    "source_url": cand_url,
                    "saved_path": str(file_path.relative_to(out_dir)),
                    "content_type": ct,
                    "bytes": len(data),
                    "width": w,
                    "height": h,
                    "sha256": hsh,
                    "validation": reason,
                },
            )

        print(f"  Guardadas {saved} imágenes (intentadas {attempted}, candidatas {len(candidates)})")

    print("Listo.")
    print(f"Imágenes: {out_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Errores:  {errors_path}")
    if not PIL_AVAILABLE:
        print("Nota: Pillow no está disponible; no se validó resolución mínima.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Descarga imágenes por EAN/descripcion desde Bing (semilla de dataset).")
    p.add_argument("--input", default="eans.txt", help="Archivo con EAN<TAB>DESCRIPCIÓN (default: eans.txt)")
    p.add_argument("--out", default="imagenes", help="Directorio de salida (default: imagenes)")
    p.add_argument("--per-ean", type=int, default=10, help="Cantidad de imágenes a guardar por EAN (default: 10)")
    p.add_argument("--timeout", type=int, default=12, help="Timeout de requests en segundos (default: 12)")
    p.add_argument("--retries", type=int, default=2, help="Reintentos por request (default: 2)")
    p.add_argument("--sleep", type=float, default=0.4, help="Pausa entre descargas (default: 0.4)")
    p.add_argument("--min-bytes", type=int, default=50_000, help="Tamaño mínimo de imagen en bytes (default: 50000)")
    p.add_argument("--min-width", type=int, default=300, help="Ancho mínimo (si Pillow está disponible) (default: 300)")
    p.add_argument("--min-height", type=int, default=300, help="Alto mínimo (si Pillow está disponible) (default: 300)")
    p.add_argument("--user-agent", default=DEFAULT_UA, help="User-Agent para requests")
    p.add_argument(
        "--shuffle-candidates",
        action="store_true",
        help="Mezcla candidatos para mayor diversidad (default: false)",
    )
    p.add_argument(
        "--dedupe-global",
        action="store_true",
        help="Evita repetir la misma imagen entre EANs usando hash (default: false)",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
