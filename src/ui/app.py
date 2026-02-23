from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
import json
import os
import sys
import subprocess

# Asegurar que la raíz del proyecto esté en el path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui.runner import ProcessRunner
from src.ui.state import StateStore, RunState
from src.ui.services.report import read_inventory_csv, list_frames
from src.ui.services.review_store import ReviewStore
from src.ui.services.db import buscar_productos

# Rutas relativas desde la raíz del proyecto
DATA_DIR = Path(project_root) / "data"
OUTPUT_DIR = Path(project_root) / "output"

app = FastAPI()
runner = ProcessRunner()
store = StateStore()

# Rutas relativas desde la raíz del proyecto
UI_STATIC_DIR = Path(__file__).parent / "static"
UI_TEMPLATES_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(UI_STATIC_DIR)), name="static")

def _list_videos():
    exts = {".mp4", ".mov", ".mkv", ".avi"}
    if not DATA_DIR.exists():
        return []
    return sorted([p.name for p in DATA_DIR.iterdir() if p.suffix.lower() in exts])

def _find_output_dir_for_video(video_name: str) -> str | None:
    # opcional: encuentra el último output del video
    if not OUTPUT_DIR.exists():
        return None
    prefix = Path(video_name).stem + "_"
    candidates = sorted([p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    return str(candidates[-1]) if candidates else None

@app.get("/", response_class=HTMLResponse)
def index():
    html = (UI_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_page(run_id: str):
    html = (UI_TEMPLATES_DIR / "run.html").read_text(encoding="utf-8").replace("{{RUN_ID}}", run_id)
    return HTMLResponse(html)

@app.get("/review/{run_id}", response_class=HTMLResponse)
def review_page(run_id: str):
    html = (UI_TEMPLATES_DIR / "review.html").read_text(encoding="utf-8").replace("{{RUN_ID}}", run_id)
    return HTMLResponse(html)

def _list_runs():
    """Lista todos los runs en output/."""
    if not OUTPUT_DIR.exists():
        return []
    runs = sorted([p.name for p in OUTPUT_DIR.iterdir() if p.is_dir()], reverse=True)
    return runs

def _run_dir(run_id: str) -> Path:
    """Retorna el Path absoluto del directorio de un run."""
    return (OUTPUT_DIR / run_id).resolve()

@app.get("/api/videos")
def api_videos():
    return {"videos": _list_videos(), "runs": _list_runs()}

@app.post("/api/run")
async def api_run(req: Request):
    body = await req.json()
    video = body["video"]
    guardar_crops = bool(body.get("guardar_crops", True))

    run_id = f"{Path(video).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = RunState(run_id=run_id, video=video, status="running")
    store.upsert(run)

    # Usar run.py desde la raíz del proyecto
    run_py_path = Path(project_root) / "run.py"
    cmd = ["python", str(run_py_path), str(DATA_DIR / video)]
    if guardar_crops:
        cmd.append("--guardar-crops")

    # respeta CLIP_MODEL del entorno si está seteado
    env = os.environ.copy()

    runner.start(run_id=run_id, cmd=cmd, cwd=str(project_root), env=env)
    return {"run_id": run_id}

@app.get("/api/run/{run_id}/logs")
def api_logs(run_id: str):
    def gen():
        for line in runner.iter_logs(run_id):
            yield (line + "\n")

        rc = runner.return_code(run_id)
        run = store.get(run_id)
        if run:
            if rc == 0:
                run.status = "done"
                run.output_dir = _find_output_dir_for_video(run.video)
            else:
                run.status = "error"
            store.upsert(run)

    return StreamingResponse(gen(), media_type="text/plain")

@app.get("/api/run/{run_id}/report")
def api_report(run_id: str):
    """Endpoint para obtener el reporte completo de un run."""
    rd = _run_dir(run_id)
    if not rd.exists():
        return JSONResponse({"error": "Run no encontrado"}, status_code=404)
    
    reporte_dir = rd / "reporte_deteccion"
    csv_path = reporte_dir / "inventario_sku.csv"
    
    rows = read_inventory_csv(csv_path)
    frames = list_frames(reporte_dir)
    has_learning = (rd / "learning").exists()
    
    return {
        "run_id": run_id,
        "csv_path": str(csv_path) if csv_path.exists() else None,
        "rows": rows,
        "frames": frames,
        "has_learning": has_learning
    }

@app.get("/api/run/{run_id}/download_csv")
def api_download_csv(run_id: str):
    """Descarga el CSV de inventario."""
    rd = _run_dir(run_id)
    csv_path = rd / "reporte_deteccion" / "inventario_sku.csv"
    if not csv_path.exists():
        return JSONResponse({"error": "CSV no encontrado"}, status_code=404)
    return FileResponse(str(csv_path), filename=f"{run_id}_inventario_sku.csv")

@app.get("/media/run/{run_id}/frame/{name}")
def media_frame(run_id: str, name: str):
    """Sirve un frame anotado."""
    p = _run_dir(run_id) / "reporte_deteccion" / name
    if not p.exists() or not p.is_file():
        return JSONResponse({"error": "Frame no encontrado"}, status_code=404)
    return FileResponse(str(p))

@app.get("/media/run/{run_id}/crop/{idx}")
def media_crop(run_id: str, idx: int):
    """Sirve un crop de learning."""
    try:
        rd = _run_dir(run_id)
        store = ReviewStore(rd)
        items = store.list_items()
        if idx < 0 or idx >= len(items):
            return JSONResponse({"error": "Índice fuera de rango"}, status_code=404)
        
        item = items[idx]
        crop_path = Path(item.crop_path)
        if not crop_path.exists():
            return JSONResponse({"error": "Crop no encontrado"}, status_code=404)
        
        return FileResponse(str(crop_path))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Review API ───────────────────────────────────

@app.get("/api/review/{run_id}/items")
def api_review_items(run_id: str):
    """Lista todos los items para revisión."""
    try:
        rd = _run_dir(run_id)
        store = ReviewStore(rd)
        items = store.list_items()
        prog = store.progress()
        
        return {
            "progress": prog,
            "items": [{
                "idx": it.idx,
                "crop_id": it.crop_id,
                "status": it.status,
                "predicted_ean": it.predicted_ean,
            } for it in items]
        }
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/review/{run_id}/item/{idx}")
def api_review_item(run_id: str, idx: int):
    """Obtiene un item específico con toda su información."""
    try:
        rd = _run_dir(run_id)
        store = ReviewStore(rd)
        items = store.list_items()
        
        if idx < 0 or idx >= len(items):
            return JSONResponse({"error": "Índice fuera de rango"}, status_code=404)
        
        it = items[idx]
        return {
            "idx": it.idx,
            "crop_id": it.crop_id,
            "status": it.status,
            "predicted_ean": it.predicted_ean,
            "top_matches": it.top_matches,
            "crop_url": f"/media/run/{run_id}/crop/{idx}",
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/review/{run_id}/set_ean")
async def api_review_set_ean(run_id: str, req: Request):
    """Asigna un EAN a un item."""
    try:
        body = await req.json()
        idx = int(body["idx"])
        ean = str(body["ean"])
        
        rd = _run_dir(run_id)
        store = ReviewStore(rd)
        store.set_ean(idx, ean)
        
        return {"ok": True, "progress": store.progress()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/review/{run_id}/skip")
async def api_review_skip(run_id: str, req: Request):
    """Saltea un item."""
    try:
        body = await req.json()
        idx = int(body["idx"])
        
        rd = _run_dir(run_id)
        store = ReviewStore(rd)
        store.skip(idx)
        
        return {"ok": True, "progress": store.progress()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/db/search_products")
def api_search_products(q: str = ""):
    """Busca productos en la base de datos para autocomplete."""
    return {"results": buscar_productos(q)}

# ── Absorb ───────────────────────────────────────

@app.post("/api/review/{run_id}/absorb")
async def api_absorb(run_id: str):
    """Absorbe los crops revisados al catálogo."""
    try:
        rd = _run_dir(run_id)
        learning_dir = rd / "learning"
        
        if not learning_dir.exists():
            return JSONResponse({"error": "No hay learning/ en este run"}, status_code=400)
        
        absorber_script = Path(project_root) / "scripts" / "absorber_crops.py"
        cmd = ["python", str(absorber_script), str(learning_dir)]
        
        # Ejecutar de forma bloqueante para obtener output
        env = os.environ.copy()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root), env=env)
        
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout[-8000:] if p.stdout else "",
            "stderr": p.stderr[-8000:] if p.stderr else "",
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)