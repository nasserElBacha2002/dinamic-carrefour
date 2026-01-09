#!/usr/bin/env python3
"""
Script para entrenar modelo YOLOv8 personalizado para detección de productos en góndolas

REQUISITOS:
1. Dataset en formato YOLO (imágenes + archivos .txt con anotaciones)
2. Archivo de configuración dataset.yaml
3. Modelo base (puede ser pre-entrenado como yolov8n.pt)

ESTRUCTURA DEL DATASET:
datos/
  train/
    images/
      imagen1.jpg
      imagen2.jpg
    labels/
      imagen1.txt
      imagen2.txt
  val/
    images/
      imagen3.jpg
    labels/
      imagen3.txt

FORMATO DE ANOTACIONES (.txt):
  clase_id x_center y_center width height
  (valores normalizados 0-1)
  
  Ejemplo:
  0 0.5 0.5 0.2 0.3
  (clase 0, centro en 50% x, 50% y, ancho 20%, alto 30%)
"""

import os
import sys
import argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODELOS_DIR

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ Error: ultralytics no está instalado")
    print("   Instala con: pip install ultralytics")
    sys.exit(1)


def crear_config_dataset(dataset_path: Path, clases: list, output_path: Path = None):
    """
    Crea archivo de configuración dataset.yaml para YOLOv8
    
    Args:
        dataset_path: Ruta al directorio del dataset
        clases: Lista de nombres de clases
        output_path: Ruta donde guardar el archivo (default: dataset_path/dataset.yaml)
    """
    if output_path is None:
        output_path = dataset_path / "dataset.yaml"
    
    # Convertir a rutas absolutas
    dataset_path = dataset_path.resolve()
    
    # Crear contenido del archivo YAML
    contenido = f"""# Dataset configuration for YOLOv8 training
# Paths are relative to this file

path: {dataset_path}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path')
test:  # test images (optional)

# Classes
names:
"""
    for i, clase in enumerate(clases):
        contenido += f"  {i}: {clase}\n"
    
    # Guardar archivo
    with open(output_path, 'w') as f:
        f.write(contenido)
    
    print(f"✅ Archivo de configuración creado: {output_path}")
    return output_path


def entrenar_modelo(
    dataset_yaml: str,
    modelo_base: str = 'yolov8n.pt',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    output_dir: Path = None
):
    """
    Entrena un modelo YOLOv8 personalizado
    
    Args:
        dataset_yaml: Ruta al archivo dataset.yaml
        modelo_base: Modelo base para transfer learning (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Número de épocas de entrenamiento
        imgsz: Tamaño de imagen (640, 1280, etc.)
        batch: Tamaño del batch
        output_dir: Directorio donde guardar el modelo entrenado
    """
    if output_dir is None:
        output_dir = MODELOS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ENTRENAMIENTO DE MODELO YOLOv8 PERSONALIZADO")
    print("=" * 70)
    print(f"\n📋 Configuración:")
    print(f"   Dataset: {dataset_yaml}")
    print(f"   Modelo base: {modelo_base}")
    print(f"   Épocas: {epochs}")
    print(f"   Tamaño imagen: {imgsz}x{imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Salida: {output_dir}")
    print()
    
    # Verificar que existe el archivo de configuración
    dataset_yaml_path = Path(dataset_yaml)
    if not dataset_yaml_path.exists():
        print(f"❌ Error: No se encuentra el archivo de configuración: {dataset_yaml}")
        return False
    
    try:
        # Cargar modelo base
        print(f"📥 Cargando modelo base: {modelo_base}")
        modelo = YOLO(modelo_base)
        
        # Entrenar modelo
        print(f"\n🚀 Iniciando entrenamiento...")
        print("   (Esto puede tomar varias horas dependiendo del dataset y hardware)")
        print()
        
        resultados = modelo.train(
            data=str(dataset_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(output_dir),
            name='gondola_training',
            save=True,
            save_period=10,  # Guardar checkpoint cada 10 épocas
            val=True,  # Validar durante entrenamiento
            plots=True,  # Generar gráficos
            verbose=True
        )
        
        # El modelo entrenado se guarda automáticamente en:
        # output_dir/gondola_training/weights/best.pt
        modelo_entrenado = output_dir / 'gondola_training' / 'weights' / 'best.pt'
        
        if modelo_entrenado.exists():
            print("\n" + "=" * 70)
            print("✅ ENTRENAMIENTO COMPLETADO")
            print("=" * 70)
            print(f"\n📦 Modelo entrenado guardado en:")
            print(f"   {modelo_entrenado}")
            print(f"\n📊 Métricas de entrenamiento:")
            print(f"   mAP50: {resultados.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   mAP50-95: {resultados.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"\n💡 Para usar el modelo entrenado:")
            print(f"   python run.py video.MOV --modelo {modelo_entrenado}")
            
            return True
        else:
            print("⚠️  Entrenamiento completado pero no se encontró el modelo final")
            return False
            
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Función principal con interfaz de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo YOLOv8 personalizado para detección de productos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS DE USO:

  # Entrenar con dataset.yaml existente
  python scripts/entrenar_modelo.py --dataset datos/dataset.yaml

  # Crear dataset.yaml y entrenar
  python scripts/entrenar_modelo.py --crear-config datos/ --clases botella bidon

  # Entrenar con parámetros personalizados
  python scripts/entrenar_modelo.py --dataset datos/dataset.yaml --epochs 200 --batch 32

REQUISITOS DEL DATASET:
  - Formato YOLO: imágenes en train/images/ y val/images/
  - Anotaciones en train/labels/ y val/labels/ (formato .txt)
  - Archivo dataset.yaml con configuración

FORMATO DE ANOTACIONES (.txt):
  clase_id x_center y_center width height
  (valores normalizados 0-1)
        """
    )
    
    # Argumentos principales
    parser.add_argument('--dataset', '-d', required=False,
                       help='Ruta al archivo dataset.yaml')
    parser.add_argument('--modelo-base', '-m', default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='Modelo base para transfer learning (default: yolov8n.pt)')
    
    # Opciones de entrenamiento
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Número de épocas (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Tamaño de imagen (default: 640)')
    parser.add_argument('--batch', '-b', type=int, default=16,
                       help='Tamaño del batch (default: 16)')
    parser.add_argument('--output', '-o', default=None,
                       help='Directorio de salida (default: modelos/)')
    
    # Opciones para crear configuración
    parser.add_argument('--crear-config', action='store_true',
                       help='Crear archivo dataset.yaml')
    parser.add_argument('--clases', nargs='+',
                       help='Lista de nombres de clases (ej: botella bidon taza)')
    
    args = parser.parse_args()
    
    # Si no se proporciona ningún argumento relevante, mostrar ayuda
    if not args.dataset and not args.crear_config:
        parser.print_help()
        print("\n" + "=" * 70)
        print("💡 INICIO RÁPIDO")
        print("=" * 70)
        print("\n1. Si ya tienes un dataset anotado en formato YOLO:")
        print("   python scripts/entrenar_modelo.py --crear-config datos/ --clases botella bidon")
        print("   python scripts/entrenar_modelo.py --dataset datos/dataset.yaml")
        print("\n2. Si necesitas preparar el dataset primero:")
        print("   - Usa herramientas como Roboflow (https://roboflow.com)")
        print("   - O LabelImg (https://github.com/heartexlabs/labelImg)")
        print("   - Exporta en formato YOLO")
        print("\n3. Ver más información:")
        print("   python scripts/entrenar_modelo.py --help")
        sys.exit(0)
    
    # Si se solicita crear configuración
    if args.crear_config:
        if not args.dataset:
            print("❌ Error: --crear-config requiere --dataset con ruta al directorio del dataset")
            sys.exit(1)
        if not args.clases:
            print("❌ Error: --crear-config requiere --clases con lista de nombres")
            sys.exit(1)
        
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"❌ Error: Directorio no existe: {dataset_path}")
            sys.exit(1)
        
        config_path = crear_config_dataset(dataset_path, args.clases)
        print(f"\n✅ Configuración creada. Ahora puedes entrenar con:")
        print(f"   python scripts/entrenar_modelo.py --dataset {config_path}")
        return
    
    # Validar que se proporcionó dataset
    if not args.dataset:
        print("❌ Error: Debes proporcionar --dataset con ruta al archivo dataset.yaml")
        print("\n💡 Si no tienes dataset.yaml, créalo con:")
        print("   python scripts/entrenar_modelo.py --crear-config datos/ --clases botella bidon")
        sys.exit(1)
    
    # Entrenar modelo
    success = entrenar_modelo(
        dataset_yaml=args.dataset,
        modelo_base=args.modelo_base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        output_dir=Path(args.output) if args.output else None
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

