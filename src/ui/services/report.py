#!/usr/bin/env python3
"""
Report Service — Lee CSV de inventario y lista frames anotados.
"""

from pathlib import Path
import csv
from typing import List, Dict, Any


def read_inventory_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Lee el CSV de inventario y retorna lista de diccionarios."""
    if not csv_path.exists():
        return []
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"⚠️ Error leyendo CSV: {e}")
        return []


def list_frames(reporte_dir: Path) -> List[str]:
    """Lista los nombres de los frames anotados."""
    if not reporte_dir.exists():
        return []
    
    frames = sorted([
        p.name for p in reporte_dir.glob("frame_*.jpg")
        if p.is_file()
    ])
    return frames
