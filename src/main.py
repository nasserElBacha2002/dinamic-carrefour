#!/usr/bin/env python3
"""
Sistema Principal - Inventario de Góndolas MVP
Pipeline completo: Análisis → Frames → Detección → Reporte
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Importar módulos del proyecto
from src.analizar_video import analizar_video, exportar_frames
from src.detectar_productos import DetectorProductos


class SistemaInventarioGondola:
    """
    Sistema principal para procesamiento completo de videos de góndolas
    Pipeline: Análisis → Extracción → Detección → Reporte
    """
    
    def __init__(self, modelo_path=None, confianza_minima=None, reconocer_marcas=True):
        """
        Inicializa el sistema
        
        Args:
            modelo_path: Ruta al modelo YOLOv8 (None = usa modelo por defecto)
            confianza_minima: Confianza mínima para detecciones (None = usa default)
            reconocer_marcas: Si True, intenta reconocer marcas usando OCR
        """
        self.detector = None
        self.modelo_path = modelo_path
        self.confianza_minima = confianza_minima
        self.reconocer_marcas = reconocer_marcas
    
    def procesar_video(self, video_path, output_base_dir="output", 
                       fps_extraccion=1.0, rotar_frames=False,
                       detectar=True, generar_anotaciones=True):
        """
        Procesa un video completo: análisis → extracción → detección → reporte
        
        Args:
            video_path: Ruta al archivo de video
            output_base_dir: Directorio base para todos los outputs
            fps_extraccion: Frames por segundo a extraer
            rotar_frames: Si True, rota los frames
            detectar: Si True, ejecuta detección de productos
            generar_anotaciones: Si True, genera imágenes anotadas
        
        Returns:
            Diccionario con resultados del procesamiento
        """
        print("\n" + "=" * 70)
        print("🚀 SISTEMA DE INVENTARIO DE GÓNDOLAS - MVP")
        print("=" * 70)
        print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📹 Video: {video_path}")
        print("=" * 70)
        
        # Verificar que el video existe
        if not os.path.exists(video_path):
            print(f"❌ Error: No se encuentra el video {video_path}")
            return None
        
        # Crear estructura de directorios
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        session_dir = Path(output_base_dir) / f"{video_name}_{timestamp}"
        
        frames_dir = session_dir / "frames_extraidos"
        reporte_dir = session_dir / "reporte_deteccion"
        
        frames_dir.mkdir(parents=True, exist_ok=True)
        reporte_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            # Inicializar detector si no está inicializado
            if self.detector is None:
                self.detector = DetectorProductos(
                    modelo_path=self.modelo_path,
                    confianza_minima=self.confianza_minima,
                    reconocer_marcas=self.reconocer_marcas
                )
            
            # Procesar frames
            resultados_deteccion = self.detector.procesar_frames(str(frames_dir))
            
            if not resultados_deteccion:
                print("⚠️  No se detectaron productos o no hay frames para procesar")
                resultados['deteccion'] = None
            else:
                resultados['deteccion'] = {
                    'total_frames_procesados': len(resultados_deteccion),
                    'total_detecciones': sum(len(dets) for dets in resultados_deteccion.values())
                }
                
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
                    frames_dir=str(frames_dir)
                )
                
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
        
        print("\n📋 Archivos generados:")
        print(f"   - Análisis: {analisis_path.name}")
        print(f"   - Frames: {frames_dir.name}/")
        if detectar and resultados.get('reporte_dir'):
            print(f"   - Inventario CSV: reporte_deteccion/inventario.csv")
            if generar_anotaciones:
                print(f"   - Imágenes anotadas: reporte_deteccion/*_detectado.jpg")
        print("=" * 70)
        
        return resultados
    


def main():
    """Función principal con interfaz de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Sistema de Inventario de Góndolas - MVP (Pipeline Completo)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Procesar video completo (análisis + frames + detección + reporte)
  python3 main.py video.MOV

  # Solo análisis y frames (sin detección)
  python3 main.py video.MOV --sin-deteccion

  # Especificar modelo personalizado
  python3 main.py video.MOV --modelo modelos/mi_modelo.pt

  # Ajustar confianza de detección
  python3 main.py video.MOV --confianza 0.3

  # Extraer más frames por segundo
  python3 main.py video.MOV --fps 2.0

  # Rotar frames (para videos verticales)
  python3 main.py video.MOV --rotar

  # Sin generar imágenes anotadas (más rápido)
  python3 main.py video.MOV --sin-anotaciones
        """
    )
    
    # Argumentos principales
    parser.add_argument('video', help='Ruta al archivo de video')
    
    # Opciones de modelo y detección
    parser.add_argument('--modelo', default=None,
                       help='Ruta al modelo YOLOv8 (.pt). Si no se especifica, usa modelo por defecto')
    parser.add_argument('--confianza', type=float, default=None,
                       help='Confianza mínima para detecciones (0-1, default: 0.25)')
    parser.add_argument('--sin-deteccion', action='store_true',
                       help='Omitir detección de productos (solo análisis y frames)')
    parser.add_argument('--sin-anotaciones', action='store_true',
                       help='No generar imágenes anotadas (más rápido)')
    parser.add_argument('--sin-marcas', action='store_true',
                       help='Desactivar reconocimiento de marcas (más rápido)')
    
    # Opciones de extracción de frames
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Frames por segundo a extraer (default: 1.0)')
    parser.add_argument('--rotar', action='store_true',
                       help='Rotar frames 90° (para videos verticales)')
    
    # Opciones de salida
    parser.add_argument('--output', default='output',
                       help='Directorio base para outputs (default: output)')
    
    args = parser.parse_args()
    
    # Validación
    if not os.path.exists(args.video):
        print(f"❌ Error: No se encuentra el video {args.video}")
        sys.exit(1)
    
    # Crear sistema
    sistema = SistemaInventarioGondola(
        modelo_path=args.modelo,
        confianza_minima=args.confianza,
        reconocer_marcas=not args.sin_marcas
    )
    
    # Ejecutar procesamiento
    try:
        resultados = sistema.procesar_video(
            args.video,
            output_base_dir=args.output,
            fps_extraccion=args.fps,
            rotar_frames=args.rotar,
            detectar=not args.sin_deteccion,
            generar_anotaciones=not args.sin_anotaciones
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

