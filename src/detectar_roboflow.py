#!/usr/bin/env python3
"""
Detector de productos usando Roboflow Workflows API.
Implementa DetectorProtocol (Dependency Inversion Principle).

A diferencia de DetectorProductos (YOLOv8 local + ResNet50 retrieval),
este detector usa un modelo custom entrenado en Roboflow que detecta
y clasifica productos en un solo paso via API serverless.
"""

import os
import cv2
import json
import csv
import hashlib
import base64
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from src.utils.image_utils import (
    cargar_imagen_segura,
    buscar_imagenes,
    clamp_bbox,
    validar_bbox,
    buscar_frame
)


class DetectorRoboflow:
    """
    Detector de productos en góndolas usando Roboflow Workflows API.
    
    Implementa la misma interfaz que DetectorProductos (DetectorProtocol)
    para ser intercambiable via Dependency Injection.
    
    Ventajas vs detector local:
    - Modelo custom entrenado para góndolas → mayor precisión
    - Detecta Y clasifica en un solo paso (no necesita retrieval)
    - Serverless → sin GPU local
    - Fácil de re-entrenar desde interfaz web de Roboflow
    """
    
    def __init__(
        self,
        api_key: str,
        workspace_name: str = "gondolacarrefour",
        workflow_id: str = "find-bottles-pepsis-pepsi-1s-pepsi-blacks-and-5-lts",
        api_url: str = "https://serverless.roboflow.com",
        label_map_path: Optional[str] = None,
        label_to_ean: Optional[Dict] = None,
        confianza_minima: float = 0.25
    ):
        """
        Inicializa el detector Roboflow.
        
        Args:
            api_key: API Key de Roboflow
            workspace_name: Nombre del workspace en Roboflow
            workflow_id: ID del workflow a ejecutar
            api_url: URL base de la API
            label_map_path: Ruta a JSON con mapeo label → EAN
            label_to_ean: Dict directo de mapeo (alternativa a label_map_path)
            confianza_minima: Confianza mínima para filtrar detecciones (0-1)
        """
        self.api_key = api_key
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.api_url = api_url.rstrip('/')
        self.confianza_minima = confianza_minima
        self._warned_output_format = False
        self._warned_conf_filter = False
        self._warned_empty_predictions = False
        self._warned_null_image_shape = False
        
        # URL del endpoint
        self.workflow_url = (
            f"{self.api_url}/infer/workflows/"
            f"{self.workspace_name}/{self.workflow_id}"
        )
        
        # Cargar mapeo label → EAN
        self.label_map = self._cargar_label_map(label_map_path, label_to_ean)
        
        print(f"🌐 Detector Roboflow configurado")
        print(f"   Workspace: {self.workspace_name}")
        print(f"   Workflow:  {self.workflow_id}")
        print(f"   Labels mapeados: {len(self.label_map)}")
        print(f"   Confianza mínima: {self.confianza_minima}")

    @staticmethod
    def _extract_predictions_from_response(data: Dict) -> Tuple[List[Dict], bool]:
        """
        Extrae predicciones de distintos formatos de respuesta de Roboflow Workflows.
        Returns:
            (predictions, parsed_ok)
        """
        if not isinstance(data, dict):
            return [], False

        # Formatos top-level alternativos
        top_preds = data.get("predictions")
        if isinstance(top_preds, list):
            return top_preds, True
        if isinstance(top_preds, dict):
            nested = top_preds.get("predictions")
            if isinstance(nested, list):
                return nested, True

        outputs = data.get("outputs", [])
        if not outputs or not isinstance(outputs, list):
            return [], False

        first = outputs[0] if outputs else {}
        if not isinstance(first, dict):
            return [], False

        # Caso clásico: outputs[0].predictions.predictions
        preds_data = first.get("predictions")
        if isinstance(preds_data, dict):
            nested = preds_data.get("predictions")
            if isinstance(nested, list):
                return nested, True
        if isinstance(preds_data, list):
            return preds_data, True

        # Caso alternativo: outputs[0].model_predictions.predictions
        model_preds = first.get("model_predictions")
        if isinstance(model_preds, dict):
            nested = model_preds.get("predictions")
            if isinstance(nested, list):
                return nested, True
        if isinstance(model_preds, list):
            return model_preds, True

        # Escaneo defensivo: buscar primer campo que sea lista de dicts con bbox
        for value in first.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if {"x", "y", "width", "height"}.issubset(value[0].keys()):
                    return value, True
            if isinstance(value, dict):
                nested = value.get("predictions")
                if isinstance(nested, list) and nested and isinstance(nested[0], dict):
                    return nested, True

        return [], False
    
    def _cargar_label_map(
        self,
        label_map_path: Optional[str],
        label_to_ean: Optional[Dict]
    ) -> Dict[str, Dict]:
        """
        Carga el mapeo de labels de Roboflow a EANs.
        
        Returns:
            Dict {label: {ean: str|None, descripcion: str}}
        """
        if label_to_ean:
            # Normalizar keys a string para evitar mismatch int vs str
            return {str(k): v for k, v in label_to_ean.items()}
        
        if label_map_path and Path(label_map_path).exists():
            with open(label_map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = {str(k): v for k, v in data.items()}
            print(f"📋 Label map cargado desde: {label_map_path}")
            return data
        
        # Default vacío
        print("⚠️  Sin label map — los labels de Roboflow se usarán como clase directamente")
        return {}
    
    def _label_a_ean(self, label: str) -> Optional[str]:
        """Convierte un label de Roboflow a EAN usando el label map."""
        label = str(label)
        if label in self.label_map:
            return self.label_map[label].get('ean')
        return None
    
    def _label_a_clase(self, label: str) -> str:
        """
        Retorna el nombre de clase para reportes.
        Si tiene EAN mapeado, usa la descripción; si no, usa el label directo.
        """
        label = str(label)
        if label in self.label_map:
            desc = self.label_map[label].get('descripcion', label)
            return desc if desc else label
        return label
    
    def _enviar_imagen(self, imagen_path: str) -> List[Dict]:
        """
        Envía una imagen a la API de Roboflow y retorna las detecciones.
        
        Args:
            imagen_path: Ruta a la imagen
            
        Returns:
            Lista de predicciones raw de Roboflow
        """
        # Leer y codificar imagen
        with open(imagen_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            'api_key': self.api_key,
            'inputs': {
                'image': {
                    'type': 'base64',
                    'value': img_b64
                }
            }
        }
        
        try:
            resp = requests.post(
                self.workflow_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"❌ Sin conexión a internet — no se puede usar Roboflow")
            return []
        except requests.exceptions.Timeout:
            print(
                f"⚠️  Timeout al enviar {Path(imagen_path).name} a workflow "
                f"'{self.workspace_name}/{self.workflow_id}'"
            )
            return []
        except requests.exceptions.HTTPError as e:
            body = ""
            try:
                body = resp.text[:240].replace("\n", " ")
            except Exception:
                body = "<sin body>"
            print(
                f"⚠️  Error HTTP {resp.status_code} en '{self.workflow_url}': {e}. "
                f"Respuesta: {body}"
            )
            return []
        
        data = resp.json()

        predictions, parsed_ok = self._extract_predictions_from_response(data)
        if not parsed_ok and not self._warned_output_format:
            top_keys = list(data.keys()) if isinstance(data, dict) else []
            output_keys = []
            if isinstance(data, dict) and isinstance(data.get("outputs"), list) and data["outputs"]:
                first = data["outputs"][0]
                if isinstance(first, dict):
                    output_keys = list(first.keys())
            print("⚠️  Respuesta de Roboflow sin predicciones parseables.")
            print(f"   - Keys top-level: {top_keys}")
            print(f"   - Keys outputs[0]: {output_keys}")
            print("   - Revisá en Workflows que el output esté conectado a predicciones del modelo.")
            self._warned_output_format = True
        elif parsed_ok and not predictions and not self._warned_empty_predictions:
            print(
                "ℹ️  El workflow respondió correctamente, pero con 0 predicciones para la imagen "
                f"de prueba ({Path(imagen_path).name})."
            )
            print("   - Esto suele indicar modelo no entrenado/publicado para esa clase o umbral alto en Roboflow.")
            self._warned_empty_predictions = True

        # Diagnóstico adicional: en algunos workflows mal conectados llega image.width/height = null
        try:
            outputs = data.get("outputs", [])
            if outputs and isinstance(outputs[0], dict):
                p = outputs[0].get("predictions")
                if isinstance(p, dict):
                    image_info = p.get("image")
                    if (
                        isinstance(image_info, dict)
                        and image_info.get("width") is None
                        and image_info.get("height") is None
                        and not self._warned_null_image_shape
                    ):
                        print("⚠️  El workflow reporta image.width/height = null.")
                        print("   - Revisá el nodo Input del workflow y que la variable 'image' llegue al modelo.")
                        self._warned_null_image_shape = True
        except Exception:
            pass

        return predictions
    
    def _parsear_prediccion(self, pred: Dict, imagen_shape: Tuple[int, int]) -> Dict:
        """
        Convierte una predicción de Roboflow al formato interno.
        
        Roboflow devuelve (x_center, y_center, width, height).
        Convertimos a (x1, y1, x2, y2) para compatibilidad.
        
        Args:
            pred: Predicción raw de Roboflow
            imagen_shape: (height, width) de la imagen
            
        Returns:
            Dict con formato estándar de detección
        """
        h_img, w_img = imagen_shape
        
        # Roboflow usa centro + ancho/alto
        cx = pred['x']
        cy = pred['y']
        w = pred['width']
        h = pred['height']
        
        # Convertir a esquinas
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Clamp a límites de imagen
        x1, y1, x2, y2 = clamp_bbox([x1, y1, x2, y2], w_img, h_img)
        
        # Preferimos class_id para mantener clases numéricas estables (0,1,2,...).
        # Si no existe class_id, usamos class textual como fallback.
        raw_class_id = pred.get('class_id', None)
        raw_class = pred.get('class', 'unknown')
        if raw_class_id is not None:
            label = str(raw_class_id)
        else:
            label = str(raw_class)
        confidence = pred.get('confidence', 0.0)
        ean = self._label_a_ean(label)
        
        return {
            'clase': label,
            'confianza': round(confidence, 4),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'ean': ean,
            'class_raw': raw_class,
            'class_id_raw': raw_class_id,
            'detection_id': pred.get('detection_id', ''),
        }
    
    def detectar_en_imagen(
        self,
        ruta_imagen: str,
        guardar_crops: bool = False,
        crops_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Detecta productos en una imagen usando Roboflow API.
        
        Args:
            ruta_imagen: Ruta a la imagen
            guardar_crops: Si True, guarda crops individuales
            crops_dir: Directorio para guardar crops
            
        Returns:
            Lista de detecciones con bbox, clase, confianza, ean
        """
        # Verificar imagen
        img = cargar_imagen_segura(ruta_imagen)
        if img is None:
            return []
        
        h_img, w_img = img.shape[:2]
        
        # Enviar a Roboflow
        predicciones_raw = self._enviar_imagen(ruta_imagen)
        
        if not predicciones_raw:
            return []
        
        # Parsear y filtrar por confianza
        detecciones = []
        descartadas_por_confianza = 0
        for pred in predicciones_raw:
            det = self._parsear_prediccion(pred, (h_img, w_img))
            
            if det['confianza'] < self.confianza_minima:
                descartadas_por_confianza += 1
                continue
            
            detecciones.append(det)

        if (
            predicciones_raw
            and not detecciones
            and descartadas_por_confianza == len(predicciones_raw)
            and not self._warned_conf_filter
        ):
            print(
                "⚠️  El workflow devolvió predicciones, pero todas quedaron por debajo "
                f"de la confianza mínima ({self.confianza_minima})."
            )
            print("   - Probá bajar --confianza (ej. 0.05)")
            self._warned_conf_filter = True
        
        # Guardar crops si se solicita
        if guardar_crops and crops_dir and detecciones:
            self._guardar_crops(img, detecciones, ruta_imagen, crops_dir)
        
        return detecciones
    
    def _guardar_crops(
        self,
        img,
        detecciones: List[Dict],
        ruta_imagen: str,
        crops_dir: str
    ):
        """Guarda crops individuales de cada detección."""
        Path(crops_dir).mkdir(parents=True, exist_ok=True)
        frame_name = Path(ruta_imagen).stem
        
        for i, det in enumerate(detecciones):
            x1, y1, x2, y2 = det['bbox']
            
            if not validar_bbox(x1, y1, x2, y2):
                continue
            
            # Verificar tamaño mínimo
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            clase = det['clase'].replace(' ', '_')
            crop_name = f"{frame_name}_{clase}_{i:03d}.jpg"
            crop_path = os.path.join(crops_dir, crop_name)
            
            cv2.imwrite(crop_path, crop)
            det['crop_path'] = crop_path
    
    def procesar_frames(
        self,
        directorio_frames: str,
        guardar_crops: bool = False,
        crops_dir: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Procesa todos los frames de un directorio.
        
        Args:
            directorio_frames: Directorio con imágenes
            guardar_crops: Si True, guarda crops
            crops_dir: Directorio para crops
            
        Returns:
            Dict {nombre_frame: [detecciones]}
        """
        imagenes = buscar_imagenes(directorio_frames)
        
        if not imagenes:
            print(f"⚠️  No se encontraron imágenes en: {directorio_frames}")
            return {}
        
        print(f"\n🌐 Procesando {len(imagenes)} frames con Roboflow API...")
        
        resultados = {}
        total_detecciones = 0
        
        for i, img_path in enumerate(imagenes):
            nombre = Path(img_path).name
            print(f"   [{i+1}/{len(imagenes)}] {nombre}...", end=" ", flush=True)
            
            detecciones = self.detectar_en_imagen(
                str(img_path),
                guardar_crops=guardar_crops,
                crops_dir=crops_dir
            )
            
            resultados[nombre] = detecciones
            total_detecciones += len(detecciones)
            
            # Resumen por frame
            clases = {}
            for d in detecciones:
                c = d['clase']
                clases[c] = clases.get(c, 0) + 1
            
            clases_str = ", ".join(f"{c}:{n}" for c, n in sorted(clases.items()))
            print(f"{len(detecciones)} detecciones ({clases_str})")
        
        # Diagnóstico rápido: modelo colapsado a una sola clase
        conteo_global_clase = {}
        for detecciones in resultados.values():
            for det in detecciones:
                c = det.get('clase', 'UNKNOWN')
                conteo_global_clase[c] = conteo_global_clase.get(c, 0) + 1
        if len(conteo_global_clase) == 1 and total_detecciones > 0:
            unica = next(iter(conteo_global_clase.keys()))
            print(f"⚠️  Diagnóstico: el workflow devolvió solo una clase ({unica}) en todo el video.")
            print("   Revisar entrenamiento/balance de clases o umbral de confianza en Roboflow.")

        print(f"\n✅ Total: {total_detecciones} detecciones en {len(resultados)} frames")
        if total_detecciones == 0:
            print("⚠️  Diagnóstico: 0 detecciones en todo el video.")
            print(f"   - Workflow usado: {self.workspace_name}/{self.workflow_id}")
            print(f"   - Confianza mínima local: {self.confianza_minima}")
            print("   - Verificá que el workflow esté publicado y apunte al modelo/version correcta.")
            print("   - Si el workflow cambió de formato de salida, revisar alertas previas de parseo.")
        
        return resultados
    
    def contar_productos(self, resultados: Dict[str, List[Dict]]) -> Dict[str, int]:
        """
        Cuenta productos agrupando por EAN cuando está disponible.
        Si no hay EAN mapeado, usa clave SIN_EAN_<clase>.
        
        Args:
            resultados: Dict frame → detecciones
            
        Returns:
            Dict {ean_o_sin_ean: cantidad_total}
        """
        conteo = {}
        for detecciones in resultados.values():
            for det in detecciones:
                ean = det.get('ean')
                if ean:
                    key = str(ean)
                else:
                    key = f"SIN_EAN_{det.get('clase', 'UNKNOWN')}"
                conteo[key] = conteo.get(key, 0) + 1
        return conteo
    
    def generar_reporte_completo(
        self,
        resultados: Dict[str, List[Dict]],
        output_dir: str,
        frames_dir: str = "",
        generar_anotaciones: bool = True
    ):
        """
        Genera reporte completo: CSV + imágenes anotadas + metadata.
        
        Args:
            resultados: Dict frame → detecciones
            output_dir: Directorio de salida
            frames_dir: Directorio con frames originales
            generar_anotaciones: Si True, genera imágenes anotadas
        """
        print("\n" + "=" * 60)
        print("GENERANDO REPORTE COMPLETO (ROBOFLOW)")
        print("=" * 60)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Contar productos
        conteo = self.contar_productos(resultados)
        
        # Exportar CSV de inventario
        csv_path = os.path.join(output_dir, "inventario.csv")
        self._exportar_csv(conteo, csv_path)
        
        # Exportar CSV de inventario por EAN
        sku_csv_path = os.path.join(output_dir, "inventario_sku.csv")
        self._exportar_csv_sku(resultados, sku_csv_path)
        
        # Generar imágenes anotadas
        imagenes_anotadas = []
        if generar_anotaciones and frames_dir:
            print(f"\n🖼️  Generando imágenes anotadas...")
            
            for frame_name, detecciones in resultados.items():
                frame_path = buscar_frame(frame_name, frames_dir)
                
                if frame_path and frame_path.exists():
                    out_path = os.path.join(
                        output_dir,
                        f"{Path(frame_name).stem}_detectado.jpg"
                    )
                    img_anotada = self._generar_imagen_anotada(
                        str(frame_path), detecciones, out_path
                    )
                    if img_anotada:
                        imagenes_anotadas.append(img_anotada)
                else:
                    print(f"⚠️  No se encontró imagen: {frame_name}")
        else:
            print(f"\n⏩ Omitiendo generación de imágenes anotadas")
        
        # Guardar metadata
        metadata = {
            'fecha': datetime.now().isoformat(),
            'detector': 'roboflow',
            'workspace': self.workspace_name,
            'workflow': self.workflow_id,
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
        print(f"   - Inventario SKU: {os.path.basename(sku_csv_path)}")
        print(f"   - Imágenes anotadas: {len(imagenes_anotadas)}")
        print(f"   - Metadatos: metadata.json")
    
    def _exportar_csv(self, conteo: Dict[str, int], output_path: str):
        """Exporta conteo principal a CSV (por EAN)."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['EAN', 'Cantidad', 'Fecha'])
            fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for ean, cantidad in sorted(conteo.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([ean, cantidad, fecha])
        print(f"💾 Inventario guardado: {os.path.basename(output_path)}")
    
    def _exportar_csv_sku(self, resultados: Dict[str, List[Dict]], output_path: str):
        """Exporta inventario agrupado por EAN."""
        # Agrupar por EAN
        conteo_ean = {}
        for detecciones in resultados.values():
            for det in detecciones:
                ean = det.get('ean')
                if ean:
                    conteo_ean[ean] = conteo_ean.get(ean, 0) + 1
                else:
                    label = det.get('clase', 'UNKNOWN')
                    key = f"SIN_EAN_{label}"
                    conteo_ean[key] = conteo_ean.get(key, 0) + 1
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['EAN', 'Cantidad', 'Fecha'])
            fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for ean, cantidad in sorted(conteo_ean.items()):
                writer.writerow([ean, cantidad, fecha])
        
        print(f"💾 Inventario SKU guardado: {os.path.basename(output_path)}")
    
    def _generar_imagen_anotada(
        self,
        ruta_imagen: str,
        detecciones: List[Dict],
        output_path: str
    ) -> Optional[str]:
        """Genera imagen con bounding boxes y labels anotados."""
        img = cargar_imagen_segura(ruta_imagen)
        if img is None:
            return None
        
        for det in detecciones:
            x1, y1, x2, y2 = det['bbox']
            clase = det['clase']
            confianza = det['confianza']
            ean = det.get('ean', '')
            
            # Color determinístico por clase
            color = self._color_para_clase(clase)
            
            # Dibujar bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label con EAN si está disponible
            if ean:
                label_text = f"{clase} [{ean}] {confianza:.0%}"
            else:
                label_text = f"{clase} {confianza:.0%}"
            
            # Fondo para texto
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        cv2.imwrite(output_path, img)
        return output_path
    
    @staticmethod
    def _color_para_clase(clase: str) -> Tuple[int, int, int]:
        """Genera un color BGR determinístico para una clase."""
        h = hashlib.md5(clase.encode()).hexdigest()
        r = int(h[:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        # Asegurar que el color sea visible (no muy oscuro)
        r = max(r, 80)
        g = max(g, 80)
        b = max(b, 80)
        return (b, g, r)  # BGR para OpenCV
