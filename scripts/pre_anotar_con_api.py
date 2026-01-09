#!/usr/bin/env python3
"""
Script para pre-anotar imágenes usando APIs de detección

Este script usa APIs comerciales (Google Vision, AWS Rekognition) para
pre-anotar imágenes, reduciendo el tiempo de anotación manual.

Las anotaciones se exportan en formato YOLO para usar con entrenar_modelo.py
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def pre_anotar_google_vision(imagen_path: str, api_key: str) -> List[Dict]:
    """
    Pre-anota una imagen usando Google Cloud Vision API
    
    Args:
        imagen_path: Ruta a la imagen
        api_key: API key de Google Cloud Vision
    
    Returns:
        Lista de detecciones en formato YOLO
    """
    try:
        from google.cloud import vision
        from google.oauth2 import service_account
    except ImportError:
        print("❌ Error: google-cloud-vision no está instalado")
        print("   Instala con: pip install google-cloud-vision")
        return []
    
    # Cargar imagen
    with open(imagen_path, 'rb') as image_file:
        content = image_file.read()
    
    # Inicializar cliente
    client = vision.ImageAnnotatorClient(credentials=service_account.Credentials.from_service_account_file(api_key))
    
    # Detectar objetos
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    
    # Convertir a formato YOLO
    detecciones = []
    if response.localized_object_annotations:
        # Obtener dimensiones de imagen
        img = cv2.imread(imagen_path)
        h, w = img.shape[:2]
        
        for obj in response.localized_object_annotations:
            # Convertir bounding box a formato YOLO
            bbox = obj.bounding_poly.normalized_vertices
            if len(bbox) >= 2:
                x_min = min(v.x for v in bbox)
                x_max = max(v.x for v in bbox)
                y_min = min(v.y for v in bbox)
                y_max = max(v.y for v in bbox)
                
                # Calcular centro y dimensiones normalizadas
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                detecciones.append({
                    'clase': obj.name,
                    'confianza': obj.score,
                    'bbox': [x_center, y_center, width, height]
                })
    
    return detecciones


def exportar_anotaciones_yolo(
    detecciones: List[Dict],
    imagen_path: Path,
    labels_dir: Path,
    clases_map: Dict[str, int]
):
    """
    Exporta anotaciones en formato YOLO (.txt)
    
    Args:
        detecciones: Lista de detecciones
        imagen_path: Ruta a la imagen original
        labels_dir: Directorio donde guardar las anotaciones
        clases_map: Mapeo de nombres de clases a IDs
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombre del archivo de anotación (mismo que imagen pero .txt)
    label_file = labels_dir / f"{imagen_path.stem}.txt"
    
    with open(label_file, 'w') as f:
        for det in detecciones:
            clase_nombre = det['clase']
            if clase_nombre in clases_map:
                clase_id = clases_map[clase_nombre]
                bbox = det['bbox']
                # Formato YOLO: clase_id x_center y_center width height
                f.write(f"{clase_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Pre-anotar imágenes usando APIs de detección',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTA: Este script requiere configuración de APIs externas.

Para Google Cloud Vision:
  1. Crear proyecto en Google Cloud Console
  2. Habilitar Vision API
  3. Crear service account y descargar JSON key
  4. Usar: python scripts/pre_anotar_con_api.py --google-key ruta/key.json --imagenes data/

Para AWS Rekognition:
  1. Configurar AWS CLI
  2. Usar: python scripts/pre_anotar_con_api.py --aws-region us-east-1 --imagenes data/
        """
    )
    
    parser.add_argument('--imagenes', required=True,
                       help='Directorio con imágenes a pre-anotar')
    parser.add_argument('--output', default=None,
                       help='Directorio de salida (default: mismo que imágenes)')
    parser.add_argument('--google-key', default=None,
                       help='Ruta al archivo JSON de service account de Google Cloud')
    parser.add_argument('--aws-region', default=None,
                       help='Región de AWS para Rekognition')
    parser.add_argument('--clases', nargs='+',
                       help='Lista de clases a detectar (filtro)')
    
    args = parser.parse_args()
    
    if not args.google_key and not args.aws_region:
        print("❌ Error: Debes proporcionar --google-key o --aws-region")
        print("\n💡 Ver ayuda con: python scripts/pre_anotar_con_api.py --help")
        sys.exit(1)
    
    # Procesar imágenes
    imagenes_dir = Path(args.imagenes)
    if not imagenes_dir.exists():
        print(f"❌ Error: Directorio no existe: {imagenes_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else imagenes_dir.parent / "pre_anotado"
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapeo de clases (temporal, se puede mejorar)
    clases_map = {}
    if args.clases:
        clases_map = {clase: i for i, clase in enumerate(args.clases)}
    
    print("=" * 70)
    print("PRE-ANOTACIÓN CON API")
    print("=" * 70)
    print(f"\n📁 Imágenes: {imagenes_dir}")
    print(f"📁 Salida: {output_dir}")
    print()
    
    # Procesar cada imagen
    imagenes = list(imagenes_dir.glob("*.jpg")) + list(imagenes_dir.glob("*.png"))
    total = len(imagenes)
    
    print(f"📊 Encontradas {total} imágenes")
    
    for i, img_path in enumerate(imagenes, 1):
        print(f"\n[{i}/{total}] Procesando: {img_path.name}")
        
        if args.google_key:
            detecciones = pre_anotar_google_vision(str(img_path), args.google_key)
        else:
            print("⚠️  AWS Rekognition aún no implementado")
            continue
        
        if detecciones:
            print(f"   ✓ Detectados {len(detecciones)} objetos")
            exportar_anotaciones_yolo(detecciones, img_path, labels_dir, clases_map)
        else:
            print("   ⚠️  No se detectaron objetos")
    
    print("\n" + "=" * 70)
    print("✅ PRE-ANOTACIÓN COMPLETADA")
    print("=" * 70)
    print(f"\n📁 Anotaciones guardadas en: {labels_dir}")
    print("\n💡 Próximos pasos:")
    print("   1. Revisar y corregir anotaciones manualmente")
    print("   2. Organizar en train/ y val/")
    print("   3. Entrenar con: python scripts/entrenar_modelo.py --dataset ...")


if __name__ == "__main__":
    main()

