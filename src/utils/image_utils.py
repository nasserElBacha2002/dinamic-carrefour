"""
Utilidades para procesamiento de imágenes
Módulo de helpers para detección de productos
"""

import cv2
from pathlib import Path
from typing import List, Tuple, Optional

__all__ = [
    'cargar_imagen_segura',
    'buscar_imagenes',
    'clamp_bbox',
    'validar_bbox',
    'buscar_frame'
]


def cargar_imagen_segura(ruta: str):
    """
    Carga una imagen con validación de errores
    
    Args:
        ruta: Ruta a la imagen
    
    Returns:
        Imagen OpenCV o None si falla
    """
    imagen = cv2.imread(ruta)
    if imagen is None:
        print(f"❌ Error: No se pudo leer la imagen {ruta}")
    return imagen


def buscar_imagenes(directorio: str, extensiones: List[str] = None) -> List[Path]:
    """
    Busca todas las imágenes en un directorio
    
    Args:
        directorio: Directorio donde buscar
        extensiones: Lista de extensiones (default: ['.jpg', '.jpeg', '.png'])
    
    Returns:
        Lista ordenada de rutas a imágenes
    """
    if extensiones is None:
        extensiones = ['.jpg', '.jpeg', '.png']
    
    imagenes = []
    for ext in extensiones:
        imagenes.extend(Path(directorio).glob(f"*{ext}"))
    
    return sorted(imagenes)


def clamp_bbox(bbox: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    """
    Clamp bounding box a los límites de la imagen
    
    Args:
        bbox: [x1, y1, x2, y2] coordenadas originales
        w: Ancho de la imagen
        h: Alto de la imagen
    
    Returns:
        Tupla (x1, y1, x2, y2) con coordenadas clampeadas
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    return x1, y1, x2, y2


def validar_bbox(x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Valida que un bounding box sea válido
    
    Args:
        x1, y1, x2, y2: Coordenadas del bbox
    
    Returns:
        True si es válido (x2 > x1 y y2 > y1)
    """
    return x2 > x1 and y2 > y1


def buscar_frame(frame_name: str, frames_dir: Optional[str] = None) -> Path:
    """
    Busca un frame en varios directorios posibles
    
    Args:
        frame_name: Nombre del frame a buscar
        frames_dir: Directorio preferido donde buscar
    
    Returns:
        Path al frame (puede no existir)
    """
    # Si se especifica directorio, buscar ahí primero
    if frames_dir:
        return Path(frames_dir) / frame_name
    
    # Intentar como ruta directa
    frame_path = Path(frame_name)
    if frame_path.exists():
        return frame_path
    
    # Buscar en directorios comunes
    posibles_dirs = [
        Path("frames_extraidos"),
        Path("output") / "frames_extraidos",
        Path.cwd()
    ]
    
    for dir_base in posibles_dirs:
        posible_path = dir_base / frame_name
        if posible_path.exists():
            return posible_path
    
    # Retornar path original aunque no exista
    return frame_path
