#!/usr/bin/env python3
"""
Script de análisis del video de góndola para el MVP
Extrae información técnica y prepara frames para análisis
"""

import cv2
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from tqdm import tqdm

def analizar_video(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Analiza el video y extrae información relevante
    
    Args:
        video_path: Ruta al archivo de video
        
    Returns:
        Diccionario con información del video y análisis, o None si falla
    """
    print("=" * 60)
    print("ANÁLISIS DEL VIDEO DE GÓNDOLA")
    print("=" * 60)
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video {video_path}")
        return None
    
    # Información básica
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"\n📹 INFORMACIÓN DEL VIDEO:")
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total de frames: {frame_count}")
    print(f"   Duración: {duration:.2f} segundos")
    
    # Analizar algunos frames clave
    frames_analisis = []
    frame_indices = [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
    
    print(f"\n🖼️  ANALIZANDO FRAMES CLAVE:")
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Información del frame
            timestamp = idx / fps if fps > 0 else 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Métricas de calidad
            brightness = gray.mean()
            contrast = gray.std()
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()  # Detectar blur
            
            frames_analisis.append({
                'frame': idx,
                'timestamp': timestamp,
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'width': width,
                'height': height
            })
            
            print(f"   Frame {idx:4d} ({timestamp:.2f}s): Brillo={brightness:.1f}, Contraste={contrast:.1f}, Nitidez={sharpness:.1f}")
    
    cap.release()
    
    # Resumen
    print(f"\n📊 RESUMEN:")
    avg_brightness = sum(f['brightness'] for f in frames_analisis) / len(frames_analisis)
    avg_contrast = sum(f['contrast'] for f in frames_analisis) / len(frames_analisis)
    avg_sharpness = sum(f['sharpness'] for f in frames_analisis) / len(frames_analisis)
    
    print(f"   Brillo promedio: {avg_brightness:.1f}")
    print(f"   Contraste promedio: {avg_contrast:.1f}")
    print(f"   Nitidez promedio: {avg_sharpness:.1f}")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES PARA EL MVP:")
    
    if avg_brightness < 100:
        print("   ⚠️  Brillo bajo - considerar ajuste de exposición")
    elif avg_brightness > 200:
        print("   ⚠️  Brillo alto - posible sobreexposición")
    else:
        print("   ✅ Brillo adecuado")
    
    if avg_contrast < 30:
        print("   ⚠️  Contraste bajo - puede afectar detección")
    else:
        print("   ✅ Contraste adecuado")
    
    if avg_sharpness < 100:
        print("   ⚠️  Video borroso - puede afectar detección y OCR")
    else:
        print("   ✅ Nitidez adecuada")
    
    # Orientación
    if height > width:
        print(f"   📱 Video en modo vertical (portrait)")
        print(f"   💡 Considerar rotación para análisis horizontal")
    else:
        print(f"   📱 Video en modo horizontal (landscape)")
    
    # Frecuencia de muestreo recomendada
    frames_por_segundo = 1  # 1 frame por segundo como sugiere el MVP
    frames_totales_necesarios = int(duration * frames_por_segundo)
    print(f"\n🎯 PARA EL MVP:")
    print(f"   Frames a extraer (1 fps): ~{frames_totales_necesarios}")
    print(f"   Intervalo recomendado: cada {int(fps)} frames")
    
    return {
        'video_info': {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        },
        'frames_analysis': frames_analisis,
        'summary': {
            'avg_brightness': avg_brightness,
            'avg_contrast': avg_contrast,
            'avg_sharpness': avg_sharpness
        }
    }


def exportar_frames(
    video_path: str, 
    output_dir: str = "frames_extraidos", 
    fps_extraccion: float = 1.0, 
    rotar: Optional[int] = None,
    formato: str = 'jpg',
    calidad: int = 95,
    filtrar_borrosos: bool = False,
    umbral_nitidez: float = 100.0
) -> List[str]:
    """
    Exporta frames del video como imágenes
    
    Args:
        video_path: Ruta al archivo de video
        output_dir: Directorio donde guardar los frames
        fps_extraccion: Frames por segundo a extraer (default: 1 para MVP)
        rotar: Grados de rotación (None, 90, 180, 270)
        formato: Formato de salida ('jpg' o 'png')
        calidad: Calidad de compresión JPEG (1-100, default: 95)
        filtrar_borrosos: Si True, descarta frames con baja nitidez
        umbral_nitidez: Umbral mínimo de nitidez (default: 100.0)
    
    Returns:
        Lista de rutas de los frames exportados
    """
    print("\n" + "=" * 60)
    print("EXPORTANDO FRAMES DEL VIDEO")
    print("=" * 60)
    
    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video {video_path}")
        return []
    
    # Información del video
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calcular intervalo de frames (protección contra división por cero y fps_extraccion > fps_video)
    if fps_video <= 0 or fps_extraccion <= 0:
        frame_interval = 1
    else:
        frame_interval = max(1, int(fps_video / fps_extraccion))
    
    total_frames_extraer = int(frame_count / frame_interval) + 1
    
    print(f"\n📹 Configuración de extracción:")
    print(f"   FPS del video: {fps_video:.2f}")
    print(f"   FPS de extracción: {fps_extraccion}")
    print(f"   Intervalo: cada {frame_interval} frames")
    print(f"   Frames a extraer: ~{total_frames_extraer}")
    print(f"   Formato: {formato.upper()} (calidad: {calidad})")
    if filtrar_borrosos:
        print(f"   Filtro de nitidez: ACTIVADO (umbral: {umbral_nitidez})")
    if rotar:
        print(f"   Rotación: {rotar}°")
    print(f"   Directorio de salida: {output_dir}")
    
    frames_exportados = []
    frame_num = 0
    frames_guardados = 0
    frames_descartados = 0
    
    print(f"\n💾 Guardando frames...")
    
    # Barra de progreso
    with tqdm(total=frame_count, desc="Extrayendo", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Extraer frame si corresponde al intervalo
            if frame_num % frame_interval == 0:
                # Rotar si es necesario
                if rotar == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotar == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotar == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Filtrar frames borrosos si está activado
                if filtrar_borrosos:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    nitidez = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if nitidez < umbral_nitidez:
                        frames_descartados += 1
                        frame_num += 1
                        pbar.update(1)
                        continue

                # Calcular timestamp
                timestamp = frame_num / fps_video if fps_video > 0 else 0

                # Nombre del archivo
                nombre_frame = f"frame_{frames_guardados:04d}_t{timestamp:.2f}s.{formato}"
                ruta_completa = os.path.join(output_dir, nombre_frame)

                # Guardar frame con formato y calidad especificados
                if formato.lower() in ("jpg", "jpeg"):
                    cv2.imwrite(ruta_completa, frame, [cv2.IMWRITE_JPEG_QUALITY, calidad])
                elif formato.lower() == "png":
                    # PNG: calidad invertida (0=mejor, 9=peor compresión)
                    compresion = int((100 - calidad) / 10)
                    cv2.imwrite(ruta_completa, frame, [cv2.IMWRITE_PNG_COMPRESSION, compresion])
                else:
                    cv2.imwrite(ruta_completa, frame)

                frames_exportados.append(ruta_completa)
                frames_guardados += 1

            frame_num += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"\n✅ Exportación completada:")
    print(f"   Total de frames guardados: {frames_guardados}")
    if filtrar_borrosos and frames_descartados > 0:
        print(f"   Frames descartados (borrosos): {frames_descartados}")
    print(f"   Ubicación: {os.path.abspath(output_dir)}")
    
    return frames_exportados

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar video de góndola y extraer frames')
    parser.add_argument('video', nargs='?', default='IMG_1838.MOV', help='Ruta al archivo de video')
    parser.add_argument('--exportar-frames', action='store_true', help='Exportar frames como imágenes')
    parser.add_argument('--output-dir', default='frames_extraidos', help='Directorio para frames exportados')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames por segundo a extraer (default: 1.0)')
    parser.add_argument('--rotar', type=int, choices=[90, 180, 270], help='Rotar frames (90, 180 o 270 grados)')
    parser.add_argument('--formato', choices=['jpg', 'png'], default='jpg', help='Formato de imagen (default: jpg)')
    parser.add_argument('--calidad', type=int, default=95, help='Calidad de compresión 1-100 (default: 95)')
    parser.add_argument('--filtrar-borrosos', action='store_true', help='Descartar frames borrosos automáticamente')
    parser.add_argument('--umbral-nitidez', type=float, default=100.0, help='Umbral de nitidez mínima (default: 100.0)')
    
    args = parser.parse_args()
    video_path = args.video
    
    if not os.path.exists(video_path):
        print(f"❌ Error: No se encuentra el video {video_path}")
        exit(1)
    
    # Análisis del video
    resultado = analizar_video(video_path)
    
    # Guardar resultado en JSON
    if resultado:
        with open("analisis_video.json", "w") as f:
            json.dump(resultado, f, indent=2)
        print(f"\n💾 Resultado guardado en: analisis_video.json")
    
    # Exportar frames si se solicita
    if args.exportar_frames:
        frames = exportar_frames(
            video_path, 
            output_dir=args.output_dir,
            fps_extraccion=args.fps,
            rotar=args.rotar,
            formato=args.formato,
            calidad=args.calidad,
            filtrar_borrosos=args.filtrar_borrosos,
            umbral_nitidez=args.umbral_nitidez
        )
        
        # Guardar lista de frames exportados en JSON
        if frames:
            frames_info = {
                'total_frames': len(frames),
                'frames': [os.path.basename(f) for f in frames],
                'directorio': args.output_dir
            }
            with open(os.path.join(args.output_dir, "frames_info.json"), "w") as f:
                json.dump(frames_info, f, indent=2)
            print(f"   📄 Lista de frames guardada en: {os.path.join(args.output_dir, 'frames_info.json')}")

