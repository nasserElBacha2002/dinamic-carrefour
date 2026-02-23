#!/usr/bin/env python3
"""
Review Store — Gestiona la lectura y edición del JSONL de learning con atomic writes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import tempfile
from datetime import datetime


@dataclass
class ReviewItem:
    idx: int
    crop_path: str
    crop_id: str
    status: str
    predicted_ean: str
    top_matches: List[Dict[str, Any]]
    data: Dict[str, Any]  # línea completa


def _find_learning_jsonl(learning_dir: Path) -> Optional[Path]:
    """Busca el archivo JSONL de metadata en el directorio learning."""
    if not learning_dir.exists():
        return None
    
    # Prioriza metadata/crops_index.jsonl (estructura actual)
    meta_dir = learning_dir / "metadata"
    if meta_dir.exists():
        jsonl_path = meta_dir / "crops_index.jsonl"
        if jsonl_path.exists():
            return jsonl_path
    
    # Fallback: buscar cualquier .jsonl en learning/
    candidates = sorted(learning_dir.glob("**/*.jsonl"))
    return candidates[0] if candidates else None


def _atomic_write_text(path: Path, text: str) -> None:
    """Escribe texto a un archivo de forma atómica (sin corrupción)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic en POSIX
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


class ReviewStore:
    """
    Carga items desde JSONL del LearningManager y permite:
    - setear ean
    - skip
    Guardando con atomic write.
    """
    
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.learning_dir = self.run_dir / "learning"
        self.jsonl_path = _find_learning_jsonl(self.learning_dir)
        
        if not self.jsonl_path:
            raise FileNotFoundError(f"No se encontró JSONL en: {self.learning_dir}")
        
        self.items: List[Dict[str, Any]] = self._load_lines()
    
    def _load_lines(self) -> List[Dict[str, Any]]:
        """Carga todas las líneas del JSONL."""
        lines: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    lines.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    # Si hay corrupción, loguear pero continuar
                    print(f"⚠️ Error parseando línea JSONL: {e}")
                    continue
        return lines
    
    def _save(self) -> None:
        """Guarda el JSONL de forma atómica."""
        text = "\n".join(json.dumps(x, ensure_ascii=False) for x in self.items) + "\n"
        _atomic_write_text(self.jsonl_path, text)
    
    def list_items(self) -> List[ReviewItem]:
        """Lista todos los items para revisión."""
        out: List[ReviewItem] = []
        
        for i, d in enumerate(self.items):
            # Extraer crop_id
            crop_id = d.get("crop_id") or d.get("id") or f"item_{i}"
            
            # Extraer status (decision o review.status)
            decision = d.get("decision") or "unknown"
            review_status = d.get("review", {}).get("status", "pending")
            if review_status == "labeled":
                status = "labeled"
            elif review_status == "skipped":
                status = "skipped"
            else:
                status = decision
            
            # Extraer EAN predicho
            sku_info = d.get("sku_identification", {})
            top1 = sku_info.get("top1") or {}
            predicted_ean = top1.get("ean") or d.get("predicted_ean") or "UNKNOWN"
            
            # Extraer top_matches
            top_matches = []
            if "top_matches" in sku_info:
                top_matches = sku_info["top_matches"]
            else:
                # Construir desde top1, top2, top3 si existen
                for key in ["top1", "top2", "top3"]:
                    if key in sku_info:
                        top_matches.append(sku_info[key])
            
            # Resolver ruta del crop
            crop_path = None
            paths = d.get("paths", {})
            if "crop" in paths:
                crop_rel = paths["crop"]
                crop_path = str((self.run_dir / crop_rel).resolve())
            elif "crop_path" in d:
                crop_path = str((self.run_dir / d["crop_path"]).resolve())
            
            # Si no existe, intentar construir desde decision y crop_id
            if not crop_path or not Path(crop_path).exists():
                decision_dir = "unknown" if decision == "unknown" else "ambiguous"
                crop_filename = f"{crop_id}.jpg"
                crop_path = str((self.learning_dir / decision_dir / crop_filename).resolve())
            
            out.append(ReviewItem(
                idx=i,
                crop_path=crop_path,
                crop_id=str(crop_id),
                status=str(status),
                predicted_ean=str(predicted_ean),
                top_matches=list(top_matches),
                data=d
            ))
        
        return out
    
    def set_ean(self, idx: int, ean: str) -> None:
        """Asigna un EAN a un item."""
        if idx < 0 or idx >= len(self.items):
            raise IndexError(f"Índice {idx} fuera de rango")
        
        d = self.items[idx]
        
        # Actualizar review
        if "review" not in d:
            d["review"] = {}
        
        d["review"]["action"] = "set_ean"
        d["review"]["assigned_ean"] = str(ean)
        d["review"]["status"] = "labeled"
        d["review"]["reviewed_at"] = datetime.now().isoformat()
        
        # Marcar decision
        d["decision"] = "labeled"
        
        self._save()
    
    def skip(self, idx: int) -> None:
        """Saltea un item."""
        if idx < 0 or idx >= len(self.items):
            raise IndexError(f"Índice {idx} fuera de rango")
        
        d = self.items[idx]
        
        # Actualizar review
        if "review" not in d:
            d["review"] = {}
        
        d["review"]["action"] = "skip"
        d["review"]["status"] = "skipped"
        d["review"]["reviewed_at"] = datetime.now().isoformat()
        
        # Marcar decision
        d["decision"] = "skipped"
        
        self._save()
    
    def progress(self) -> Dict[str, int]:
        """Retorna estadísticas de progreso."""
        total = len(self.items)
        labeled = 0
        skipped = 0
        pending = 0
        
        for d in self.items:
            review = d.get("review", {})
            status = review.get("status", "pending")
            
            if status == "labeled":
                labeled += 1
            elif status == "skipped":
                skipped += 1
            else:
                pending += 1
        
        return {
            "total": total,
            "labeled": labeled,
            "skipped": skipped,
            "pending": pending
        }
