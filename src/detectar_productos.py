#!/usr/bin/env python3
"""
Sistema de detección de productos en góndolas usando YOLOv8
MVP - Sistema de Inventario de Góndolas

MEJORADO: Implementa Dependency Inversion Principle (DIP)
- Acepta reconocedor de marcas inyectado
- Acepta exportador de reportes inyectado
- Desacoplado de implementaciones concretas
"""

import cv2
import os
import json
import csv
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Importar utilidades de procesamiento de imágenes
from src.utils.image_utils import (
    cargar_imagen_segura,
    buscar_imagenes,
    clamp_bbox,
    validar_bbox,
    buscar_frame
)

# Importar exportadores (Open/Closed Principle)
from src.exporters import ReporteExporterBase, CSVExporter

# Importar protocolo de reconocedor de marcas (Dependency Inversion)
try:
    from src.protocols import ReconocedorMarcasProtocol
except ImportError:
    # Fallback si protocols no existe (backward compatibility)
    ReconocedorMarcasProtocol = None

# Importar configuración local
try:
    from src.config import obtener_ruta_modelo, MODELO_DEFAULT, CONFIANZA_MINIMA_DEFAULT, cargar_marcas_conocidas
except ImportError:
    # Fallback si config.py no existe
    def obtener_ruta_modelo(nombre=None):
        if nombre:
            return Path(nombre)
        return Path("modelos/yolov8_gondola_mvp.pt")
    MODELO_DEFAULT = Path("modelos/yolov8_gondola_mvp.pt")
    CONFIANZA_MINIMA_DEFAULT = 0.25
    def cargar_marcas_conocidas():
        return []

# YOLOv8 se importará cuando esté disponible
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Advertencia: ultralytics no está instalado. Instalar con: pip install ultralytics")

# Importar reconocedor de marcas (opcional, con backward compatibility)
try:
    from src.reconocer_marcas import ReconocedorMarcas
    RECONOCIMIENTO_MARCAS_AVAILABLE = True
except ImportError:
    RECONOCIMIENTO_MARCAS_AVAILABLE = False
    print("⚠️  Reconocimiento de marcas no disponible")


class DetectorProductos:
    """
    Clase principal para detección de productos en imágenes de góndolas
    
    SOLID Improvements:
    - Dependency Inversion: Acepta reconocedor de marcas y exportador inyectados
    - Single Responsibility: Se enfoca en detección, delega OCR y exportación
    """
    
    def __init__(
        self, 
        modelo_path: Optional[str] = None, 
        confianza_minima: float = None,
        reconocedor_marcas: Optional[object] = None,  # ReconocedorMarcasProtocol
        exporter: Optional[ReporteExporterBase] = None
    ):
        """
        Inicializa el detector de productos con inyección de dependencias
        
        Args:
            modelo_path: Ruta al modelo YOLOv8 (.pt)
            confianza_minima: Umbral mínimo de confianza (0-1)
            reconocedor_marcas: Instancia de reconocedor de marcas (inyección)
            exporter: Instancia de exportador de reportes (inyección)
        """
        self.confianza_minima = confianza_minima if confianza_minima is not None else CONFIANZA_MINIMA_DEFAULT
        self.modelo = None
        self.clases_detectadas = {}
        
        # Dependency Injection: reconocedor de marcas
        self.reconocedor_marcas = reconocedor_marcas
        
        # Dependency Injection: exportador (default CSVExporter)
        self.exporter = exporter if exporter is not None else CSVExporter()
        
        # Si no se especifica modelo_path, usar el por defecto
        if modelo_path is None:
            modelo_path = str(MODELO_DEFAULT)
        
        # Resolver ruta del modelo
        modelo_path_resuelto = obtener_ruta_modelo(modelo_path)
        
        if YOLO_AVAILABLE:
            if modelo_path_resuelto.exists():
                self.cargar_modelo(str(modelo_path_resuelto))
            else:
                print(f"⚠️  Modelo no encontrado en: {modelo_path_resuelto}")
                print(f"ℹ️  Modo simulación: Coloca el modelo en {MODELO_DEFAULT} o especifica --modelo")
        else:
            print("⚠️  No se puede cargar el modelo: ultralytics no está instalado")
    
    def cargar_modelo(self, modelo_path: str):
        """
        Carga el modelo YOLOv8 entrenado
        
        Args:
            modelo_path: Ruta al archivo .pt del modelo
        """
        if not YOLO_AVAILABLE:
            print("❌ Error: ultralytics no está instalado")
            return False
        
        if not os.path.exists(modelo_path):
            print(f"❌ Error: No se encuentra el modelo en {modelo_path}")
            return False
        
        try:
            self.modelo = YOLO(modelo_path)
            print(f"✅ Modelo cargado desde: {modelo_path}")
            return True
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            return False
    
    def detectar_en_imagen(self, ruta_imagen: str, guardar_crops: bool = False,
                          crops_dir: Optional[str] = None) -> List[Dict]:
        """
        Detecta productos en una imagen
        
        Args:
            ruta_imagen: Ruta a la imagen a procesar
            guardar_crops: Si True, guarda cada detección como imagen individual (crop)
            crops_dir: Directorio donde guardar los crops (si None, usa 'crops/')
        
        Returns:
            Lista de detecciones, cada una con:
            - clase: nombre del SKU/producto
            - confianza: nivel de confianza (0-1)
            - bbox: [x1, y1, x2, y2] coordenadas del bounding box
            - crop_path: ruta al crop guardado (si guardar_crops=True)
        """
        if not os.path.exists(ruta_imagen):
            print(f"❌ Error: No se encuentra la imagen {ruta_imagen}")
            return []
        
        if self.modelo is None:
            print("⚠️  Modo simulación: No hay modelo cargado")
            return self._simular_deteccion(ruta_imagen)
        
        # Cargar imagen con validación
        imagen = cargar_imagen_segura(ruta_imagen)
        if imagen is None:
            return []
        
        # Ejecutar detección con YOLOv8
        resultados = self.modelo(ruta_imagen, conf=self.confianza_minima)
        
        detecciones = []
        
        for resultado in resultados:
            boxes = resultado.boxes
            
            for box in boxes:
                # Obtener información de la detección
                clase_id = int(box.cls[0])
                confianza = float(box.conf[0])
                nombre_clase = self.modelo.names[clase_id]
                
                # Coordenadas del bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                deteccion = {
                    'clase': nombre_clase,
                    'clase_id': clase_id,
                    'confianza': confianza,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'imagen': ruta_imagen  # Guardar ruta completa
                }
                
                detecciones.append(deteccion)
        
        # Guardar crops si está habilitado
        if guardar_crops and detecciones:
            crops_guardados = self._guardar_crops(imagen, detecciones, ruta_imagen, crops_dir)
            # Actualizar detecciones con rutas de crops
            for i, crop_path in enumerate(crops_guardados):
                if i < len(detecciones):
                    detecciones[i]['crop_path'] = crop_path
        
        # Reconocer marcas si está habilitado (usa dependencia inyectada)
        if self.reconocedor_marcas and detecciones:
            print(f"   🔍 Reconociendo marcas en {len(detecciones)} productos...")
            # Cargar marcas conocidas desde configuración (opcional)
            marcas_conocidas = cargar_marcas_conocidas()
            detecciones = self.reconocedor_marcas.procesar_detecciones(
                ruta_imagen, detecciones, marcas_conocidas if marcas_conocidas else None
            )
        
        # Actualizar contador de clases (después de reconocimiento de marcas)
        for deteccion in detecciones:
            clase_final = deteccion.get('clase', 'unknown')
            if clase_final not in self.clases_detectadas:
                self.clases_detectadas[clase_final] = 0
            self.clases_detectadas[clase_final] += 1
        
        return detecciones
    
    def _guardar_crops(self, imagen, detecciones: List[Dict], ruta_imagen_original: str,
                      crops_dir: Optional[str] = None) -> List[str]:
        """
        Guarda cada detección como una imagen individual (crop)
        
        Args:
            imagen: Imagen OpenCV (numpy array)
            detecciones: Lista de detecciones
            ruta_imagen_original: Ruta de la imagen original
            crops_dir: Directorio donde guardar crops
        
        Returns:
            Lista de rutas donde se guardaron los crops
        """
        if crops_dir is None:
            crops_dir = "crops"
        
        crops_path = Path(crops_dir)
        crops_path.mkdir(parents=True, exist_ok=True)
        
        frame_name = Path(ruta_imagen_original).stem
        crops_guardados = []
        
        # Obtener dimensiones de imagen
        h, w = imagen.shape[:2]
        
        for i, deteccion in enumerate(detecciones):
            # Clamp bbox a límites de imagen
            x1, y1, x2, y2 = clamp_bbox(deteccion['bbox'], w, h)
            
            # Validar que el bbox es válido
            if not validar_bbox(x1, y1, x2, y2):
                print(f"⚠️  Bbox inválido para detección {i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                crops_guardados.append("")
                continue
            
            # Extraer crop
            crop = imagen[y1:y2, x1:x2]
            
            # Validar que el crop no esté vacío
            if crop.size == 0:
                print(f"⚠️  Crop vacío para detección {i}")
                crops_guardados.append("")
                continue
            
            # Generar nombre de archivo
            clase = deteccion['clase'].replace('/', '_').replace(' ', '_')
            crop_filename = f"crop_{frame_name}_{i:03d}_{clase}.jpg"
            crop_path = crops_path / crop_filename
            
            # Guardar crop
            try:
                cv2.imwrite(str(crop_path), crop)
                crops_guardados.append(str(crop_path))
            except Exception as e:
                print(f"⚠️  Error guardando crop {crop_filename}: {e}")
                crops_guardados.append("")
        
        if crops_guardados:
            crops_validos = sum(1 for c in crops_guardados if c)
            print(f"   📦 Guardados {crops_validos}/{len(detecciones)} crops en: {crops_dir}")
        
        return crops_guardados
    
    def _simular_deteccion(self, ruta_imagen: str) -> List[Dict]:
        """
        Simula detecciones cuando no hay modelo disponible (para desarrollo)
        """
        print(f"🔍 [SIMULACIÓN] Analizando {ruta_imagen}")
        return []
    
    def procesar_frames(self, directorio_frames: str, guardar_crops: bool = False,
                       crops_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Procesa todos los frames en un directorio
        
        Args:
            directorio_frames: Directorio con las imágenes de frames
            guardar_crops: Si True, guarda crops de cada detección
            crops_dir: Directorio donde guardar crops (si None, usa 'crops/')
        
        Returns:
            Diccionario con frame -> lista de detecciones
        """
        print("\n" + "=" * 60)
        print("PROCESANDO FRAMES PARA DETECCIÓN")
        print("=" * 60)
        
        if not os.path.exists(directorio_frames):
            print(f"❌ Error: No se encuentra el directorio {directorio_frames}")
            return {}
        
        # Buscar todas las imágenes
        imagenes = buscar_imagenes(directorio_frames)
        
        if not imagenes:
            print(f"⚠️  No se encontraron imágenes en {directorio_frames}")
            return {}
        
        print(f"\n📸 Encontradas {len(imagenes)} imágenes para procesar")
        if guardar_crops:
            print(f"📦 Generación de crops: ACTIVADA → {crops_dir or 'crops/'}")
        
        resultados = {}
        total_detecciones = 0
        total_crops = 0
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            print(f"\n[{i}/{len(imagenes)}] Procesando: {ruta_imagen.name}")
            
            detecciones = self.detectar_en_imagen(
                str(ruta_imagen),
                guardar_crops=guardar_crops,
                crops_dir=crops_dir
            )
            # Usar nombre del archivo como clave pero guardar ruta completa en detecciones
            resultados[ruta_imagen.name] = detecciones
            total_detecciones += len(detecciones)
            
            # Contar crops guardados
            if guardar_crops:
                crops_en_frame = sum(1 for det in detecciones if det.get('crop_path'))
                total_crops += crops_en_frame
            
            print(f"   ✓ Detectados {len(detecciones)} productos")
        
        print(f"\n✅ Procesamiento completado:")
        print(f"   Frames procesados: {len(imagenes)}")
        print(f"   Total de detecciones: {total_detecciones}")
        if guardar_crops:
            print(f"   Total de crops guardados: {total_crops}")
        
        return resultados
    
    def contar_productos(self, resultados: Dict[str, List[Dict]]) -> Dict[str, int]:
        """
        Cuenta productos por SKU/clase
        
        Args:
            resultados: Diccionario con resultados de detección
        
        Returns:
            Diccionario con SKU -> cantidad
        """
        conteo = {}
        
        for frame, detecciones in resultados.items():
            for deteccion in detecciones:
                sku = deteccion['clase']
                if sku not in conteo:
                    conteo[sku] = 0
                conteo[sku] += 1
        
        return conteo
    
    def generar_imagen_anotada(self, ruta_imagen: str, detecciones: List[Dict], 
                               output_path: Optional[str] = None) -> str:
        """
        Genera imagen con bounding boxes y etiquetas
        
        Args:
            ruta_imagen: Ruta a la imagen original
            detecciones: Lista de detecciones
            output_path: Ruta donde guardar (si None, se genera automáticamente)
        
        Returns:
            Ruta de la imagen guardada
        """
        imagen = cargar_imagen_segura(ruta_imagen)
        if imagen is None:
            return ""
        
        # Dibujar bounding boxes
        for deteccion in detecciones:
            x1, y1, x2, y2 = [int(coord) for coord in deteccion['bbox']]
            clase = deteccion['clase']
            confianza = deteccion['confianza']
            
            # Color según clase (hash determinístico con MD5)
            color_hash = int(hashlib.md5(clase.encode('utf-8')).hexdigest()[:8], 16) % 256
            color = (
                (color_hash * 7) % 256,
                (color_hash * 11) % 256,
                (color_hash * 13) % 256
            )
            
            # Dibujar rectángulo
            cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta con clase y confianza
            etiqueta = f"{clase} {confianza:.2f}"
            
            # Tamaño del texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Fondo para el texto
            (text_width, text_height), _ = cv2.getTextSize(etiqueta, font, font_scale, thickness)
            cv2.rectangle(imagen, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Texto
            cv2.putText(imagen, etiqueta, (x1, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # Generar nombre de salida
        if output_path is None:
            base_name = Path(ruta_imagen).stem
            output_dir = Path(ruta_imagen).parent / "detecciones"
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"{base_name}_detectado.jpg")
        
        # Guardar imagen
        cv2.imwrite(output_path, imagen)
        return output_path
    
    def exportar_csv(self, conteo: Dict[str, int], output_path: str = "inventario.csv"):
        """
        Exporta el conteo de productos usando el exportador inyectado
        Usa Dependency Injection para soportar diferentes formatos
        
        Args:
            conteo: Diccionario con SKU/Marca -> cantidad
            output_path: Ruta del archivo a generar
        """
        print(f"\n💾 Exportando inventario a: {output_path}")
        
        # Usar exportador inyectado (Open/Closed Principle)
        self.exporter.exportar(conteo, output_path)
    
    def generar_reporte_completo(self, resultados: Dict[str, List[Dict]], 
                                output_dir: str = "reporte_deteccion",
                                frames_dir: Optional[str] = None,
                                generar_anotaciones: bool = True):
        """
        Genera reporte completo con imágenes anotadas y CSV
        
        Args:
            resultados: Resultados de detección por frame
            output_dir: Directorio donde guardar el reporte
            frames_dir: Directorio donde están las imágenes originales (opcional)
            generar_anotaciones: Si True, genera imágenes anotadas con bounding boxes
        """
        print("\n" + "=" * 60)
        print("GENERANDO REPORTE COMPLETO")
        print("=" * 60)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Contar productos
        conteo = self.contar_productos(resultados)
        
        # Exportar CSV
        csv_path = os.path.join(output_dir, "inventario.csv")
        self.exportar_csv(conteo, csv_path)
        
        # Generar imágenes anotadas (si está habilitado)
        imagenes_anotadas = []
        if generar_anotaciones:
            print(f"\n🖼️  Generando imágenes anotadas...")
        
        for frame_name, detecciones in resultados.items():
                # Buscar imagen original usando función helper
                frame_path = buscar_frame(frame_name, frames_dir)

                if frame_path and frame_path.exists():
                img_anotada = self.generar_imagen_anotada(
                    str(frame_path), 
                    detecciones,
                    os.path.join(output_dir, f"{Path(frame_name).stem}_detectado.jpg")
                )
                if img_anotada:
                    imagenes_anotadas.append(img_anotada)
            else:
                print(f"⚠️  No se encontró imagen: {frame_path}")
        else:
            print(f"\n⏩ Omitiendo generación de imágenes anotadas")
        
        # Guardar metadatos
        metadata = {
            'fecha': datetime.now().isoformat(),
            'total_frames': len(resultados),
            'total_skus': len(conteo),
            'total_productos': sum(conteo.values()),
            'conteo': conteo,
            'imagenes_anotadas': [os.path.basename(img) for img in imagenes_anotadas]
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Reporte completo generado en: {output_dir}")
        print(f"   - Inventario CSV: {os.path.basename(csv_path)}")
        print(f"   - Imágenes anotadas: {len(imagenes_anotadas)}")
        print(f"   - Metadatos: metadata.json")


def main():
    """Función principal para ejecución desde línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detectar productos en frames de góndola')
    parser.add_argument('--modelo', default=None, help='Ruta al modelo YOLOv8 (.pt)')
    parser.add_argument('--frames', required=True, help='Directorio con frames a procesar')
    parser.add_argument('--confianza', type=float, default=0.25, help='Confianza mínima (0-1)')
    parser.add_argument('--output', default='reporte_deteccion', help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Crear detector
    detector = DetectorProductos(
        modelo_path=args.modelo,
        confianza_minima=args.confianza
    )
    
    # Procesar frames
    resultados = detector.procesar_frames(args.frames)
    
    if resultados:
        # Generar reporte completo
        detector.generar_reporte_completo(resultados, args.output)
    else:
        print("⚠️  No se procesaron frames. Verificar directorio y formato de imágenes.")


if __name__ == "__main__":
    main()

