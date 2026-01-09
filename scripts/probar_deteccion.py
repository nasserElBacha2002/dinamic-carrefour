#!/usr/bin/env python3
"""
Script de prueba para verificar que el modelo local funciona correctamente
"""

import os
import sys
from pathlib import Path

# Agregar src al path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODELO_DEFAULT, MODELOS_DIR

print("=" * 60)
print("PRUEBA DE MODELO LOCAL")
print("=" * 60)

# Verificar que existe el modelo
print(f"\n📁 Verificando modelo en: {MODELO_DEFAULT}")

if not MODELO_DEFAULT.exists():
    print(f"❌ Modelo no encontrado: {MODELO_DEFAULT}")
    print("\n💡 Opciones:")
    print("   1. Descargar modelo pre-entrenado:")
    print("      python3 descargar_modelo.py")
    print("\n   2. Colocar tu modelo entrenado en:")
    print(f"      {MODELO_DEFAULT}")
    exit(1)

print(f"✅ Modelo encontrado")
print(f"   Tamaño: {MODELO_DEFAULT.stat().st_size / (1024*1024):.2f} MB")

# Verificar ultralytics
try:
    from ultralytics import YOLO
    print("\n✅ ultralytics instalado")
except ImportError:
    print("\n❌ ultralytics no está instalado")
    print("   Instala con: pip install ultralytics")
    exit(1)

# Intentar cargar el modelo
print("\n🔄 Cargando modelo...")
try:
    modelo = YOLO(str(MODELO_DEFAULT))
    print("✅ Modelo cargado exitosamente")
    
    # Mostrar información del modelo
    print(f"\n📊 Información del modelo:")
    print(f"   Clases: {len(modelo.names)}")
    print(f"   Tipo: {type(modelo).__name__}")
    
    if hasattr(modelo, 'names'):
        print(f"\n   Primeras 5 clases:")
        for i, (id_clase, nombre) in enumerate(list(modelo.names.items())[:5]):
            print(f"      {id_clase}: {nombre}")
    
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    exit(1)

# Probar con DetectorProductos
print("\n" + "=" * 60)
print("PROBANDO CON DetectorProductos")
print("=" * 60)

try:
    from src.detectar_productos import DetectorProductos
    
    print("\n🔄 Inicializando DetectorProductos...")
    detector = DetectorProductos()
    
    if detector.modelo is not None:
        print("✅ DetectorProductos inicializado correctamente")
        print(f"   Confianza mínima: {detector.confianza_minima}")
        print(f"   Modelo cargado: Sí")
    else:
        print("⚠️  DetectorProductos inicializado pero sin modelo")
        print("   Esto es normal si el modelo no existe o hay un error")
        
except Exception as e:
    print(f"❌ Error al inicializar DetectorProductos: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Verificar si hay frames para probar
frames_dir = Path("output/IMG_1838_20260109_104525/frames_extraidos")
if frames_dir.exists():
    frames = list(frames_dir.glob("*.jpg"))
    if frames:
        print(f"\n📸 Frames disponibles para probar: {len(frames)}")
        print(f"   Ejemplo: {frames[0].name}")
        print("\n💡 Para probar detección en un frame:")
        print(f"   python3 -c \"from detectar_productos import DetectorProductos; d = DetectorProductos(); print(d.detectar_en_imagen('{frames[0]}'))\"")
    else:
        print("\n⚠️  No hay frames disponibles para probar")
else:
    print("\n⚠️  Directorio de frames no encontrado")
    print("   Genera frames primero con: python3 main.py IMG_1838.MOV")

print("\n" + "=" * 60)
print("✅ VERIFICACIÓN COMPLETADA")
print("=" * 60)
print("\nEl modelo está listo para usar!")

