#!/usr/bin/env python3
"""
Report Service — Lee CSV de inventario y lista frames anotados.
"""

from pathlib import Path
import csv
from typing import List, Dict, Any
import sys

# Asegurar que la raíz del proyecto esté en el path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


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


def enrich_inventory_with_product_names(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enriquece las filas del inventario con el nombre del producto desde la base de datos.
    
    Args:
        rows: Lista de diccionarios del CSV (con EAN, Cantidad, Fecha, etc.)
    
    Returns:
        Lista de diccionarios enriquecidos con columna "Nombre" o "Descripcion"
    """
    try:
        from src.database.repository import ProductoRepository
        
        repo = ProductoRepository()
        
        # Crear un cache de productos para evitar múltiples queries
        productos_cache = {}
        
        enriched = []
        for row in rows:
            # Buscar EAN en diferentes columnas posibles
            ean = row.get("EAN") or row.get("ean") or row.get("sku") or ""
            
            # Si no hay EAN, copiar la fila sin modificar
            if not ean:
                enriched.append(row)
                continue
            
            # Buscar en cache primero
            if ean not in productos_cache:
                producto = repo.obtener_producto(ean)
                if producto:
                    # Obtener descripción (puede estar en diferentes campos)
                    nombre = (
                        producto.get("descripcion") or
                        producto.get("descripcion_larga") or
                        producto.get("nombre") or
                        ""
                    )
                    productos_cache[ean] = nombre
                else:
                    productos_cache[ean] = ""  # No encontrado
            
            # Agregar nombre a la fila
            row_enriched = dict(row)
            row_enriched["Nombre"] = productos_cache.get(ean, "")
            enriched.append(row_enriched)
        
        return enriched
    
    except Exception as e:
        # Si falla la DB, retornar filas sin enriquecer
        print(f"⚠️ Error enriqueciendo inventario con nombres: {e}")
        return rows


def list_frames(reporte_dir: Path) -> List[str]:
    """Lista los nombres de los frames anotados."""
    if not reporte_dir.exists():
        return []
    
    frames = sorted([
        p.name for p in reporte_dir.glob("frame_*.jpg")
        if p.is_file()
    ])
    return frames
