#!/usr/bin/env python3
"""
Script para descargar modelo YOLOv8 pre-entrenado
Útil para empezar a probar el sistema antes de entrenar un modelo personalizado
"""

import os
import sys
from pathlib import Path

# Agregar src al path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODELOS_DIR, MODELO_DEFAULT

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ Error: ultralytics no está instalado")
    print("   Instala con: pip install ultralytics")
    exit(1)


def descargar_modelo_preentrenado(tamaño='n', destino=None):
    """
    Descarga un modelo YOLOv8 pre-entrenado
    
    Args:
        tamaño: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        destino: Ruta donde guardar el modelo. Si es None, usa MODELO_DEFAULT
    """
    if destino is None:
        destino = MODELO_DEFAULT
    
    destino = Path(destino)
    
    # Verificar si ya existe
    if destino.exists():
        print(f"✅ El modelo ya existe en: {destino}")
        respuesta = input("¿Deseas sobrescribirlo? (s/n): ")
        if respuesta.lower() != 's':
            print("❌ Operación cancelada")
            return False
    
    print(f"\n📥 Descargando modelo YOLOv8{tamaño}...")
    print(f"   Destino: {destino}")
    
    try:
        # Crear modelo (esto descarga los pesos automáticamente)
        modelo = YOLO(f'yolov8{tamaño}.pt')
        
        # Guardar en la ubicación deseada
        modelo.save(str(destino))
        
        print(f"\n✅ Modelo descargado exitosamente!")
        print(f"   Ubicación: {destino}")
        print(f"   Tamaño: {destino.stat().st_size / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error al descargar modelo: {e}")
        return False


def listar_modelos_disponibles():
    """Lista los modelos YOLOv8 disponibles"""
    modelos = {
        'n': 'YOLOv8n (nano) - Más rápido, menos preciso',
        's': 'YOLOv8s (small) - Balanceado',
        'm': 'YOLOv8m (medium) - Más preciso',
        'l': 'YOLOv8l (large) - Muy preciso',
        'x': 'YOLOv8x (xlarge) - Máxima precisión'
    }
    
    print("\n📋 Modelos YOLOv8 disponibles:")
    for key, desc in modelos.items():
        print(f"   {key}: {desc}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Descargar modelo YOLOv8 pre-entrenado para pruebas'
    )
    parser.add_argument(
        '--tamaño', '-t',
        choices=['n', 's', 'm', 'l', 'x'],
        default='n',
        help='Tamaño del modelo (default: n - nano)'
    )
    parser.add_argument(
        '--destino', '-d',
        default=None,
        help='Ruta donde guardar el modelo (default: modelos/yolov8_gondola_mvp.pt)'
    )
    parser.add_argument(
        '--listar',
        action='store_true',
        help='Listar modelos disponibles'
    )
    
    args = parser.parse_args()
    
    if args.listar:
        listar_modelos_disponibles()
        return
    
    print("=" * 60)
    print("DESCARGAR MODELO YOLOv8 PRE-ENTRENADO")
    print("=" * 60)
    print("\n⚠️  NOTA: Este es un modelo genérico (COCO dataset)")
    print("   Detecta objetos comunes (personas, autos, etc.)")
    print("   Para detectar productos específicos, necesitas entrenar tu propio modelo")
    print()
    
    # Mostrar modelos disponibles
    listar_modelos_disponibles()
    
    if descargar_modelo_preentrenado(args.tamaño, args.destino):
        print("\n" + "=" * 60)
        print("✅ MODELO LISTO PARA USAR")
        print("=" * 60)
        print("\nAhora puedes probar el sistema:")
        print("  python3 probar_deteccion.py")
        print("\nO usar en tu código:")
        print("  from detectar_productos import DetectorProductos")
        print("  detector = DetectorProductos()")


if __name__ == "__main__":
    main()

