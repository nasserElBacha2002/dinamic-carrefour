#!/usr/bin/env python3
"""
DB Service — Búsqueda de productos desde la base de datos para autocomplete.
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def buscar_productos(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Busca productos en la base de datos por EAN o descripción.
    
    Args:
        query: Texto de búsqueda
        limit: Máximo de resultados
    
    Returns:
        Lista de diccionarios con {ean, descripcion}
    """
    try:
        from src.database.repository import ProductoRepository
        
        repo = ProductoRepository()
        productos = repo.listar_productos()
        
        q = (query or "").lower().strip()
        if not q:
            # Si no hay query, retornar primeros N
            return [
                {
                    "ean": str(p.get("ean") or p.get("EAN") or ""),
                    "descripcion": str(p.get("descripcion") or p.get("descripcion_larga") or p.get("nombre") or "")
                }
                for p in productos[:limit]
            ]
        
        # Filtrar por query
        out = []
        for p in productos:
            ean = str(p.get("ean") or p.get("EAN") or "")
            desc = str(p.get("descripcion") or p.get("descripcion_larga") or p.get("nombre") or "")
            
            if q in ean.lower() or q in desc.lower():
                out.append({
                    "ean": ean,
                    "descripcion": desc
                })
                if len(out) >= limit:
                    break
        
        return out
    
    except Exception as e:
        print(f"⚠️ Error buscando productos: {e}")
        return []
