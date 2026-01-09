#!/usr/bin/env python3
"""
Script para integrar Roboflow con el sistema de entrenamiento

Roboflow facilita:
- Anotación visual de imágenes
- Preprocesamiento y augmentación
- Exportación en formato YOLOv8
- Gestión de datasets

USO:
1. Crear cuenta en https://roboflow.com (gratis)
2. Crear proyecto y subir imágenes
3. Anotar productos en la interfaz web
4. Exportar dataset en formato YOLOv8
5. Usar este script para descargar y preparar el dataset
"""

import os
import sys
import argparse
import requests
from pathlib import Path
import zipfile
import shutil

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, MODELOS_DIR

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("⚠️  Roboflow no está instalado")
    print("   Instala con: pip install roboflow")


def descargar_dataset_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    formato: str = "yolov8",
    destino: Path = None
):
    """
    Descarga un dataset de Roboflow
    
    Args:
        api_key: API key de Roboflow
        workspace: Nombre del workspace en Roboflow
        project: Nombre del proyecto
        version: Versión del dataset
        formato: Formato de exportación (yolov8, coco, etc.)
        destino: Directorio donde guardar el dataset
    """
    if not ROBOFLOW_AVAILABLE:
        print("❌ Error: Roboflow no está instalado")
        print("   Instala con: pip install roboflow")
        return False
    
    if destino is None:
        destino = DATA_DIR / "datasets" / project
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DESCARGANDO DATASET DE ROBOFLOW")
    print("=" * 70)
    print(f"\n📋 Configuración:")
    print(f"   Workspace: {workspace}")
    print(f"   Proyecto: {project}")
    print(f"   Versión: {version}")
    print(f"   Formato: {formato}")
    print(f"   Destino: {destino}")
    print()
    
    try:
        # Inicializar Roboflow
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(formato)
        
        print(f"\n✅ Dataset descargado exitosamente!")
        print(f"   Ubicación: {dataset.location}")
        
        # Mover a destino si es necesario
        if Path(dataset.location) != destino:
            print(f"\n📦 Moviendo dataset a: {destino}")
            if destino.exists():
                shutil.rmtree(destino)
            shutil.move(dataset.location, destino)
        
        # Buscar dataset.yaml
        dataset_yaml = destino / "data.yaml"
        if not dataset_yaml.exists():
            # Buscar en subdirectorios
            for yaml_file in destino.rglob("*.yaml"):
                dataset_yaml = yaml_file
                break
        
        if dataset_yaml.exists():
            print(f"\n✅ Archivo de configuración encontrado: {dataset_yaml}")
            print(f"\n💡 Ahora puedes entrenar con:")
            print(f"   python scripts/entrenar_modelo.py --dataset {dataset_yaml}")
            return True
        else:
            print(f"\n⚠️  No se encontró dataset.yaml")
            print(f"   Busca manualmente en: {destino}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error al descargar dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def listar_proyectos_roboflow(api_key: str, workspace: str):
    """
    Lista proyectos disponibles en Roboflow
    """
    if not ROBOFLOW_AVAILABLE:
        print("❌ Error: Roboflow no está instalado")
        return
    
    try:
        rf = Roboflow(api_key=api_key)
        workspace_obj = rf.workspace(workspace)
        proyectos = workspace_obj.projects()
        
        print(f"\n📋 Proyectos en workspace '{workspace}':")
        for proyecto in proyectos:
            print(f"   - {proyecto.name}")
            print(f"     Versiones: {proyecto.versions()}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Integrar Roboflow con sistema de entrenamiento',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS DE USO:

  # Descargar dataset de Roboflow
  python scripts/integrar_roboflow.py \\
    --api-key TU_API_KEY \\
    --workspace mi-workspace \\
    --project gondola-products \\
    --version 1

  # Listar proyectos disponibles
  python scripts/integrar_roboflow.py \\
    --api-key TU_API_KEY \\
    --workspace mi-workspace \\
    --listar

OBTENER API KEY:
  1. Crear cuenta en https://roboflow.com (gratis)
  2. Ir a Account Settings → API
  3. Copiar tu API key

FLUJO RECOMENDADO:
  1. Subir imágenes a Roboflow (interfaz web)
  2. Anotar productos (interfaz web)
  3. Generar dataset → Exportar formato YOLOv8
  4. Usar este script para descargar
  5. Entrenar con: python scripts/entrenar_modelo.py --dataset ...
        """
    )
    
    parser.add_argument('--api-key', required=False,
                       help='API key de Roboflow (o usar variable ROBOFLOW_API_KEY)')
    parser.add_argument('--workspace', required=False,
                       help='Nombre del workspace en Roboflow')
    parser.add_argument('--project', required=False,
                       help='Nombre del proyecto')
    parser.add_argument('--version', type=int, default=1,
                       help='Versión del dataset (default: 1)')
    parser.add_argument('--formato', default='yolov8',
                       choices=['yolov8', 'yolov5', 'coco'],
                       help='Formato de exportación (default: yolov8)')
    parser.add_argument('--destino', default=None,
                       help='Directorio donde guardar el dataset')
    parser.add_argument('--listar', action='store_true',
                       help='Listar proyectos disponibles')
    
    args = parser.parse_args()
    
    # Obtener API key
    api_key = args.api_key or os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        print("❌ Error: Debes proporcionar --api-key o configurar ROBOFLOW_API_KEY")
        print("\n💡 Obtén tu API key en: https://roboflow.com → Account Settings → API")
        sys.exit(1)
    
    # Si solo listar proyectos
    if args.listar:
        if not args.workspace:
            print("❌ Error: --listar requiere --workspace")
            sys.exit(1)
        listar_proyectos_roboflow(api_key, args.workspace)
        return
    
    # Validar argumentos para descarga
    if not args.workspace or not args.project:
        print("❌ Error: Debes proporcionar --workspace y --project")
        print("\n💡 O usa --listar para ver proyectos disponibles")
        sys.exit(1)
    
    # Descargar dataset
    destino = Path(args.destino) if args.destino else None
    success = descargar_dataset_roboflow(
        api_key=api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        formato=args.formato,
        destino=destino
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

