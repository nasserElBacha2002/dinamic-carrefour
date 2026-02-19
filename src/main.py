#!/usr/bin/env python3
"""
Sistema Principal - Inventario de Góndolas MVP
Pipeline completo: Análisis → Frames → Detección → Reporte

MEJORADO: Implementa Dependency Inversion Principle (DIP)
- Usa ComponentFactory para crear dependencias
- Acepta detector inyectado
- Desacoplado de implementaciones concretas
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Importar módulos del proyecto
from src.analizar_video import analizar_video, exportar_frames

# Importar factory para crear componentes (Dependency Inversion)
from src.factory import ComponentFactory

# Importar protocolos (abstracciones)
try:
    from src.protocols import DetectorProtocol, IdentificadorSKUProtocol
except ImportError:
    # Backward compatibility
    DetectorProtocol = None
    IdentificadorSKUProtocol = None


def _load_dotenv(dotenv_path: Path) -> None:
    """
    Carga variables de .env de forma simple (sin dependencias externas).
    """
    if not dotenv_path.exists():
        return

    for raw in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class SistemaInventarioGondola:
    """
    Sistema principal para procesamiento completo de videos de góndolas
    Pipeline: Análisis → Extracción → Detección → Reporte
    
    SOLID Improvements:
    - Dependency Inversion: Acepta detector e identificador inyectados
    - Open/Closed: Extensible sin modificar código existente
    """
    
    def __init__(
        self,
        detector: object = None,  # DetectorProtocol
        identificador_sku: object = None,  # IdentificadorSKUProtocol
        sku_threshold: float = 0.80
    ):
        """
        Inicializa el sistema con inyección de dependencias
        
        Args:
            detector: Instancia de detector (inyección)
            identificador_sku: Instancia de identificador SKU (inyección)
            sku_threshold: Umbral de similitud para SKU
        """
        self.detector = detector
        self.identificador_sku = identificador_sku
        self.sku_threshold = sku_threshold
    
    def procesar_video(self, video_path, output_base_dir="output", 
                       fps_extraccion=1.0, rotar_frames=False,
                       detectar=True, generar_anotaciones=True,
                       guardar_crops=False):
        """
        Procesa un video completo: análisis → extracción → detección → identificación SKU → reporte
        
        Args:
            video_path: Ruta al archivo de video
            output_base_dir: Directorio base para todos los outputs
            fps_extraccion: Frames por segundo a extraer
            rotar_frames: Si True, rota los frames
            detectar: Si True, ejecuta detección de productos
            generar_anotaciones: Si True, genera imágenes anotadas
            guardar_crops: Si True, guarda crops de cada detección
        
        Returns:
            Diccionario con resultados del procesamiento
        """
        print("\n" + "=" * 70)
        print("🚀 SISTEMA DE INVENTARIO DE GÓNDOLAS - MVP")
        print("=" * 70)
        print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📹 Video: {video_path}")
        if self.identificador_sku:
            print("🔍 Identificación SKU: ACTIVADA")
        print("=" * 70)
        
        # Verificar que el video existe
        if not os.path.exists(video_path):
            print(f"❌ Error: No se encuentra el video {video_path}")
            return None
        
        # Determinar si hay que guardar crops (antes de cualquier condicional)
        debe_guardar_crops = guardar_crops or (self.identificador_sku is not None)
        
        # Crear estructura de directorios
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        session_dir = Path(output_base_dir) / f"{video_name}_{timestamp}"
        
        frames_dir = session_dir / "frames_extraidos"
        reporte_dir = session_dir / "reporte_deteccion"
        crops_dir = session_dir / "crops"
        
        frames_dir.mkdir(parents=True, exist_ok=True)
        reporte_dir.mkdir(parents=True, exist_ok=True)
        if debe_guardar_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)
        
        resultados = {
            'video_path': video_path,
            'session_dir': str(session_dir),
            'fecha_procesamiento': datetime.now().isoformat()
        }
        
        # PASO 1: Análisis del video
        print("\n" + "─" * 70)
        print("PASO 1: ANÁLISIS DEL VIDEO")
        print("─" * 70)
        
        analisis = analizar_video(video_path)
        
        if not analisis:
            print("❌ Error en el análisis del video")
            return None
        
        resultados['analisis'] = analisis
        
        # Guardar análisis en JSON
        analisis_path = session_dir / "analisis_video.json"
        with open(analisis_path, 'w', encoding='utf-8') as f:
            json.dump(analisis, f, indent=2, ensure_ascii=False)
        print(f"💾 Análisis guardado en: {analisis_path}")
        
        # PASO 2: Extracción de frames
        print("\n" + "─" * 70)
        print("PASO 2: EXTRACCIÓN DE FRAMES")
        print("─" * 70)
        
        frames_exportados = exportar_frames(
            video_path,
            output_dir=str(frames_dir),
            fps_extraccion=fps_extraccion,
            rotar=rotar_frames
        )
        
        if not frames_exportados:
            print("⚠️  No se exportaron frames")
            resultados['frames_exportados'] = 0
            if detectar:
                print("⚠️  No se puede ejecutar detección sin frames")
                detectar = False
        else:
            resultados['frames_exportados'] = len(frames_exportados)
            resultados['frames_dir'] = str(frames_dir)
            print(f"✅ {len(frames_exportados)} frames exportados")
        
        # PASO 3: Detección de productos (opcional)
        if detectar:
            print("\n" + "─" * 70)
            print("PASO 3: DETECCIÓN DE PRODUCTOS")
            print("─" * 70)
            
            # Verificar que hay detector inyectado
            if self.detector is None:
                print("⚠️  No hay detector disponible. Omitiendo detección.")
                detectar = False
            else:
                # Procesar frames (con crops si está habilitado)
                resultados_deteccion = self.detector.procesar_frames(
                    str(frames_dir),
                    guardar_crops=debe_guardar_crops,
                    crops_dir=str(crops_dir) if debe_guardar_crops else None
                )
            
            if not resultados_deteccion:
                print("⚠️  No se detectaron productos o no hay frames para procesar")
                resultados['deteccion'] = None
            else:
                resultados['deteccion'] = {
                    'total_frames_procesados': len(resultados_deteccion),
                    'total_detecciones': sum(len(dets) for dets in resultados_deteccion.values())
                }
                
                # PASO 3.5: Identificación SKU (si está habilitado)
                if self.identificador_sku and debe_guardar_crops:
                    print("\n" + "─" * 70)
                    print("PASO 3.5: IDENTIFICACIÓN DE SKU")
                    print("─" * 70)
                    
                    # Recolectar todos los crops
                    todos_crops = []
                    for frame_dets in resultados_deteccion.values():
                        for det in frame_dets:
                            if 'crop_path' in det:
                                todos_crops.append(det)
                    
                    print(f"🔍 Identificando {len(todos_crops)} crops...")
                    
                    # Identificar SKUs
                    sku_identificados = 0
                    for det in todos_crops:
                        resultado_sku = self.identificador_sku.identificar(
                            det['crop_path'],
                            threshold=self.sku_threshold
                        )
                        det['ean'] = resultado_sku['ean']
                        det['confianza_sku'] = resultado_sku['confianza']
                        det['top_matches_sku'] = resultado_sku['top_matches'][:3]
                        
                        if resultado_sku['ean'] != 'UNKNOWN':
                            sku_identificados += 1
                    
                    print(f"✅ SKUs identificados: {sku_identificados}/{len(todos_crops)}")
                    resultados['sku_identificados'] = sku_identificados
                    resultados['crops_procesados'] = len(todos_crops)
                
                # Contar productos
                conteo = self.detector.contar_productos(resultados_deteccion)
                resultados['conteo_productos'] = conteo
                
                # PASO 4: Generar reporte completo
                print("\n" + "─" * 70)
                print("PASO 4: GENERACIÓN DE REPORTE")
                print("─" * 70)
                
                self.detector.generar_reporte_completo(
                    resultados_deteccion,
                    output_dir=str(reporte_dir),
                    frames_dir=str(frames_dir),
                    generar_anotaciones=generar_anotaciones
                )
                
                # Guardar reporte SKU si se identificaron
                if self.identificador_sku and todos_crops:
                    reporte_sku_path = reporte_dir / "inventario_sku.csv"
                    self._generar_reporte_sku(todos_crops, reporte_sku_path)
                
                resultados['reporte_dir'] = str(reporte_dir)
        
        # Resumen final
        print("\n" + "=" * 70)
        print("✅ PROCESAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"📁 Directorio de sesión: {session_dir}")
        print(f"📊 Frames exportados: {resultados['frames_exportados']}")
        
        if detectar and resultados.get('deteccion'):
            print(f"🔍 Total de detecciones: {resultados['deteccion']['total_detecciones']}")
            if resultados.get('conteo_productos'):
                conteo = resultados['conteo_productos']
                print(f"📦 SKUs detectados: {len(conteo)}")
                print(f"📈 Total de productos: {sum(conteo.values())}")
            if self.identificador_sku and resultados.get('sku_identificados'):
                print(f"🏷️  EANs identificados: {resultados['sku_identificados']}/{resultados.get('crops_procesados', 0)}")
        
        print("\n📋 Archivos generados:")
        print(f"   - Análisis: {analisis_path.name}")
        print(f"   - Frames: {frames_dir.name}/")
        if debe_guardar_crops:
            print(f"   - Crops: {crops_dir.name}/")
        if detectar and resultados.get('reporte_dir'):
            print(f"   - Inventario CSV: reporte_deteccion/inventario.csv")
            if self.identificador_sku:
                print(f"   - Inventario SKU: reporte_deteccion/inventario_sku.csv")
            if generar_anotaciones:
                print(f"   - Imágenes anotadas: reporte_deteccion/*_detectado.jpg")
        print("=" * 70)
        
        return resultados
    
    def _generar_reporte_sku(self, detecciones_con_sku, output_path):
        """Genera CSV con inventario por EAN"""
        import csv
        
        # Agrupar por EAN
        conteo_ean = {}
        for det in detecciones_con_sku:
            ean = det.get('ean', 'UNKNOWN')
            if ean not in conteo_ean:
                conteo_ean[ean] = 0
            conteo_ean[ean] += 1
        
        # Escribir CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['EAN', 'Cantidad', 'Fecha'])
            
            fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for ean, cantidad in sorted(conteo_ean.items()):
                writer.writerow([ean, cantidad, fecha])
        
        print(f"💾 Reporte SKU guardado: {output_path.name}")
    


def main():
    """Función principal con interfaz de línea de comandos (Roboflow-only)."""
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description='Sistema de Inventario de Góndolas - Roboflow Only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Pipeline completo con Roboflow (API key desde .env)
  python3 main.py video.MOV

  # Con mapeo de labels a EANs
  python3 main.py video.MOV --label-map roboflow_label_map.json

  # Solo análisis y frames (sin detección)
  python3 main.py video.MOV --sin-deteccion

  # Guardar crops de cada detección
  python3 main.py video.MOV --guardar-crops
        """
    )
    
    # Argumentos principales
    parser.add_argument('video', help='Ruta al archivo de video')
    
    # Roboflow (obligatorio)
    parser.add_argument('--roboflow-api-key', default=os.getenv('ROBOFLOW_API_KEY'),
                       help='API Key de Roboflow (si no se pasa, usa ROBOFLOW_API_KEY de .env)')
    parser.add_argument('--roboflow-workspace', default=os.getenv('ROBOFLOW_WORKSPACE', 'gondolacarrefour'),
                       help='Workspace de Roboflow (default: gondolacarrefour)')
    parser.add_argument('--roboflow-workflow',
                       default=os.getenv('ROBOFLOW_WORKFLOW', 'find-bottles-pepsis-pepsi-1s-pepsi-blacks-and-5-lts'),
                       help='Workflow ID de Roboflow')
    parser.add_argument('--label-map', default='roboflow_label_map.json',
                       help='Archivo JSON con mapeo label → EAN (default: roboflow_label_map.json)')

    # Detección / extracción
    parser.add_argument('--confianza', type=float, default=None,
                       help='Confianza mínima para detecciones (0-1, default: 0.25)')
    parser.add_argument('--sin-deteccion', action='store_true',
                       help='Omitir detección de productos (solo análisis y frames)')
    parser.add_argument('--sin-anotaciones', action='store_true',
                       help='No generar imágenes anotadas (más rápido)')
    parser.add_argument('--guardar-crops', action='store_true',
                       help='Guardar crops de cada detección')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Frames por segundo a extraer (default: 1.0)')
    parser.add_argument('--rotar', action='store_true',
                       help='Rotar frames 90° (para videos verticales)')
    
    # Salida
    parser.add_argument('--output', default='output',
                       help='Directorio base para outputs (default: output)')
    
    args = parser.parse_args()
    
    # Validación
    if not os.path.exists(args.video):
        print(f"❌ Error: No se encuentra el video {args.video}")
        sys.exit(1)
    
    if not args.roboflow_api_key:
        print("❌ Falta API key. Definila en .env como ROBOFLOW_API_KEY o pasá --roboflow-api-key")
        sys.exit(1)

    # Crear componentes usando Factory (solo Roboflow)
    print("\n🔧 Configurando componentes del sistema (Roboflow)...")

    config = {
        'detector_tipo': 'roboflow',
        'confianza_minima': args.confianza,
        'roboflow_api_key': args.roboflow_api_key,
        'roboflow_workspace': args.roboflow_workspace,
        'roboflow_workflow': args.roboflow_workflow,
        'label_map_path': args.label_map,
    }

    componentes = ComponentFactory.desde_config(config)

    sistema = SistemaInventarioGondola(
        detector=componentes['detector'],
        identificador_sku=None
    )
    
    # Ejecutar procesamiento
    try:
        resultados = sistema.procesar_video(
            args.video,
            output_base_dir=args.output,
            fps_extraccion=args.fps,
            rotar_frames=args.rotar,
            detectar=not args.sin_deteccion,
            generar_anotaciones=not args.sin_anotaciones,
            guardar_crops=args.guardar_crops
        )
        
        if resultados:
            print("\n✅ Proceso completado exitosamente")
            sys.exit(0)
        else:
            print("\n⚠️  Proceso completado con advertencias")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

