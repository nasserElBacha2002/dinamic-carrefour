#!/usr/bin/env python3
"""
Configuración del sistema - Rutas y parámetros
"""

import os
from pathlib import Path

# Directorio base del proyecto (raíz, no src/)
BASE_DIR = Path(__file__).parent.parent

# Rutas de modelos
MODELOS_DIR = BASE_DIR / "modelos"
MODELO_DEFAULT = MODELOS_DIR / "yolov8_gondola_mvp.pt"

# Directorios de salida
OUTPUT_DIR = BASE_DIR / "output"
FRAMES_DIR = BASE_DIR / "frames_extraidos"
DATA_DIR = BASE_DIR / "data"

# Crear directorios si no existen
MODELOS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuración de detección
CONFIANZA_MINIMA_DEFAULT = 0.25

# Configuración de extracción de frames
FPS_EXTRACCION_DEFAULT = 1.0

# Configuración de reconocimiento de marcas
MARCAS_CONOCIDAS_FILE = BASE_DIR / "marcas_conocidas.txt"  # Archivo opcional con marcas conocidas

def cargar_marcas_conocidas() -> list:
    """
    Carga marcas conocidas desde archivo de configuración (opcional)
    
    Returns:
        Lista de marcas conocidas, o lista vacía si no existe el archivo
    """
    if MARCAS_CONOCIDAS_FILE.exists():
        try:
            with open(MARCAS_CONOCIDAS_FILE, 'r', encoding='utf-8') as f:
                marcas = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            return marcas
        except Exception as e:
            print(f"⚠️  Error al cargar marcas conocidas: {e}")
            return []
    return []

def obtener_ruta_modelo(nombre_modelo=None):
    """
    Obtiene la ruta completa al modelo
    
    Args:
        nombre_modelo: Nombre del archivo del modelo (ej: 'yolov8_gondola.pt')
                      Si es None, usa el modelo por defecto
    
    Returns:
        Path al archivo del modelo
    """
    if nombre_modelo is None:
        return MODELO_DEFAULT
    
    # Si es ruta absoluta, usarla directamente
    if os.path.isabs(nombre_modelo):
        return Path(nombre_modelo)
    
    # Si es relativa, buscar en directorio de modelos
    modelo_path = MODELOS_DIR / nombre_modelo
    
    # Si no existe en modelos/, asumir que es ruta relativa al proyecto
    if not modelo_path.exists():
        modelo_path = BASE_DIR / nombre_modelo
    
    return modelo_path

