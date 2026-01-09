#!/usr/bin/env python3
"""
Script para organizar un dataset descargado manualmente de Roboflow

USO:
1. Descarga el ZIP de Roboflow manualmente
2. Extrae el ZIP en algún lugar
3. Ejecuta este script apuntando a la carpeta extraída

python scripts/organizar_dataset_descargado.py --ruta /ruta/al/dataset/extraido
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR


def organizar_dataset(ruta_dataset: Path, nombre_proyecto: str = None):
    """
    Organiza un dataset descargado de Roboflow en la estructura del proyecto
    
    Args:
        ruta_dataset: Ruta al directorio del dataset extraído
        nombre_proyecto: Nombre del proyecto (si no se proporciona, usa el nombre de la carpeta)
    """
    ruta_dataset = Path(ruta_dataset)
    
    if not ruta_dataset.exists():
        print(f"❌ Error: No se encuentra el directorio: {ruta_dataset}")
        return False
    
    # Buscar data.yaml o dataset.yaml
    dataset_yaml = None
    for yaml_file in ruta_dataset.rglob("*.yaml"):
        if yaml_file.name in ['data.yaml', 'dataset.yaml']:
            dataset_yaml = yaml_file
            break
    
    if not dataset_yaml:
        print(f"⚠️  No se encontró data.yaml en: {ruta_dataset}")
        print("   Buscando en subdirectorios...")
        # Buscar en subdirectorios comunes
        for subdir in ['train', 'valid', 'val']:
            subdir_path = ruta_dataset / subdir
            if subdir_path.exists():
                yaml_file = subdir_path / "data.yaml"
                if yaml_file.exists():
                    dataset_yaml = yaml_file
                    break
    
    if not dataset_yaml:
        print("❌ Error: No se encontró archivo data.yaml")
        print("\n💡 Asegúrate de haber descargado el dataset completo de Roboflow")
        return False
    
    # Determinar nombre del proyecto
    if nombre_proyecto is None:
        nombre_proyecto = ruta_dataset.name
    
    # Destino final
    destino = DATA_DIR / "datasets" / nombre_proyecto
    destino.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ORGANIZANDO DATASET DE ROBOFLOW")
    print("=" * 70)
    print(f"\n📁 Origen: {ruta_dataset}")
    print(f"📁 Destino: {destino}")
    print(f"📄 Dataset YAML: {dataset_yaml}")
    print()
    
    # Copiar dataset completo
    print("📦 Copiando archivos...")
    
    # Copiar estructura completa
    if destino.exists():
        respuesta = input(f"⚠️  El directorio {destino} ya existe. ¿Sobrescribir? (s/n): ")
        if respuesta.lower() != 's':
            print("❌ Operación cancelada")
            return False
        shutil.rmtree(destino)
    
    # Copiar todo
    shutil.copytree(ruta_dataset, destino)
    
    # Verificar que data.yaml está en la raíz del destino
    dataset_yaml_destino = destino / dataset_yaml.name
    if not dataset_yaml_destino.exists():
        # Si está en un subdirectorio, moverlo a la raíz
        yaml_en_subdir = list(destino.rglob("data.yaml"))
        if yaml_en_subdir:
            shutil.copy2(yaml_en_subdir[0], dataset_yaml_destino)
    
    print(f"\n✅ Dataset organizado exitosamente!")
    print(f"   Ubicación: {destino}")
    print(f"   Archivo de configuración: {dataset_yaml_destino}")
    
    # Verificar estructura
    print("\n📊 Estructura del dataset:")
    train_dir = destino / "train"
    valid_dir = destino / "valid" or destino / "val"
    
    if train_dir.exists():
        train_images = len(list((train_dir / "images").glob("*"))) if (train_dir / "images").exists() else 0
        train_labels = len(list((train_dir / "labels").glob("*.txt"))) if (train_dir / "labels").exists() else 0
        print(f"   Train: {train_images} imágenes, {train_labels} anotaciones")
    
    if valid_dir.exists():
        valid_images = len(list((valid_dir / "images").glob("*"))) if (valid_dir / "images").exists() else 0
        valid_labels = len(list((valid_dir / "labels").glob("*.txt"))) if (valid_dir / "labels").exists() else 0
        print(f"   Valid: {valid_images} imágenes, {valid_labels} anotaciones")
    
    print(f"\n💡 Ahora puedes entrenar con:")
    print(f"   python scripts/entrenar_modelo.py --dataset {dataset_yaml_destino}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Organizar dataset descargado manualmente de Roboflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS:

  # Organizar dataset extraído
  python scripts/organizar_dataset_descargado.py --ruta ~/Downloads/find-packages-and-bottles-2

  # Especificar nombre del proyecto
  python scripts/organizar_dataset_descargado.py \\
    --ruta ~/Downloads/dataset \\
    --nombre gondola-carrefour
        """
    )
    
    parser.add_argument('--ruta', required=True,
                       help='Ruta al directorio del dataset extraído')
    parser.add_argument('--nombre', default=None,
                       help='Nombre del proyecto (default: nombre de la carpeta)')
    
    args = parser.parse_args()
    
    success = organizar_dataset(Path(args.ruta), args.nombre)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

