#!/usr/bin/env python3
"""
Módulo para reconocimiento de marcas usando OCR
Identifica marcas automáticamente leyendo texto de las etiquetas de productos
"""

import cv2
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract no está instalado. Instalar con: pip install pytesseract")
    print("   También necesitas instalar Tesseract OCR: brew install tesseract (macOS)")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class ReconocedorMarcas:
    """
    Clase para reconocer marcas de productos usando OCR
    """
    
    def __init__(self, metodo='tesseract'):
        """
        Inicializa el reconocedor de marcas
        
        Args:
            metodo: 'tesseract' o 'easyocr'
        """
        self.metodo = metodo
        self.reader = None
        
        if metodo == 'easyocr' and EASYOCR_AVAILABLE:
            print("🔄 Inicializando EasyOCR...")
            self.reader = easyocr.Reader(['es', 'en'], gpu=False)
            print("✅ EasyOCR listo")
        elif metodo == 'tesseract' and not TESSERACT_AVAILABLE:
            print("⚠️  Tesseract no disponible, intentando EasyOCR...")
            if EASYOCR_AVAILABLE:
                self.metodo = 'easyocr'
                self.reader = easyocr.Reader(['es', 'en'], gpu=False)
            else:
                print("❌ Error: No hay OCR disponible. Instala pytesseract o easyocr")
    
    def extraer_texto_region(self, imagen, bbox):
        """
        Extrae texto de una región específica de la imagen
        
        Args:
            imagen: Imagen completa (numpy array)
            bbox: [x1, y1, x2, y2] coordenadas del bounding box
        
        Returns:
            Texto extraído
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Asegurar que las coordenadas están dentro de la imagen
        h, w = imagen.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Extraer región
        region = imagen[y1:y2, x1:x2]
        
        if region.size == 0:
            return ""
        
        # Preprocesar imagen para mejor OCR
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Redimensionar si es muy pequeño
        if gray.shape[0] < 50 or gray.shape[1] < 50:
            scale = max(50 / gray.shape[0], 50 / gray.shape[1])
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Extraer texto
        if self.metodo == 'tesseract' and TESSERACT_AVAILABLE:
            texto = pytesseract.image_to_string(gray, lang='spa+eng')
        elif self.metodo == 'easyocr' and self.reader:
            resultados = self.reader.readtext(gray)
            texto = ' '.join([result[1] for result in resultados])
        else:
            return ""
        
        return texto.strip()
    
    def extraer_texto_con_confianza(self, imagen, bbox) -> List[Tuple[str, float]]:
        """
        Extrae texto con niveles de confianza usando EasyOCR
        
        Args:
            imagen: Imagen completa
            bbox: Coordenadas del bounding box
        
        Returns:
            Lista de tuplas (texto, confianza)
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        h, w = imagen.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        region = imagen[y1:y2, x1:x2]
        if region.size == 0:
            return []
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        if gray.shape[0] < 50 or gray.shape[1] < 50:
            scale = max(50 / gray.shape[0], 50 / gray.shape[1])
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        textos = []
        if self.metodo == 'easyocr' and self.reader:
            resultados = self.reader.readtext(gray)
            textos = [(result[1], result[2]) for result in resultados]
        elif self.metodo == 'tesseract' and TESSERACT_AVAILABLE:
            texto = pytesseract.image_to_string(gray, lang='spa+eng')
            if texto.strip():
                textos = [(texto.strip(), 0.8)]  # Confianza estimada para Tesseract
        
        return textos
    
    def _similitud_palabras(self, palabra1: str, palabra2: str) -> float:
        """
        Calcula similitud entre dos palabras (para manejar errores de OCR)
        Usa distancia de Levenshtein simple mejorada
        """
        palabra1 = palabra1.upper()
        palabra2 = palabra2.upper()
        
        if palabra1 == palabra2:
            return 1.0
        
        # Si una está contenida en la otra (más estricto)
        if len(palabra1) >= 4 and len(palabra2) >= 4:
            if palabra1 in palabra2:
                return 0.85
            if palabra2 in palabra1:
                return 0.85
        
        # Calcular diferencia de caracteres
        len_diff = abs(len(palabra1) - len(palabra2))
        max_len = max(len(palabra1), len(palabra2))
        min_len = min(len(palabra1), len(palabra2))
        
        if max_len == 0:
            return 0.0
        
        # Contar caracteres comunes en las mismas posiciones
        comunes = sum(1 for a, b in zip(palabra1, palabra2) if a == b)
        
        # También contar caracteres comunes sin importar posición (para errores de orden)
        chars1 = set(palabra1)
        chars2 = set(palabra2)
        comunes_set = len(chars1 & chars2)
        
        # Similitud por posición
        similitud_posicion = comunes / max_len
        
        # Similitud por caracteres comunes
        similitud_chars = comunes_set / max(len(chars1), len(chars2)) if chars1 or chars2 else 0
        
        # Combinar ambas métricas
        similitud = (similitud_posicion * 0.6 + similitud_chars * 0.4)
        
        # Penalizar por diferencia de longitud (pero menos severo)
        if len_diff > 0:
            similitud *= (1 - len_diff / (max_len * 2))
        
        return similitud
    
    def identificar_marca(self, textos_con_confianza: List[Tuple[str, float]], 
                         marcas_conocidas: Optional[List[str]] = None) -> Optional[str]:
        """
        Identifica la marca automáticamente basándose en el texto extraído
        
        Estrategia genérica:
        1. Si hay marcas conocidas, buscar coincidencias (tolerante a errores de OCR)
        2. Si no, identificar palabras prominentes que probablemente sean marcas
        3. Filtrar palabras comunes (artículos, preposiciones, números)
        
        Args:
            textos_con_confianza: Lista de tuplas (texto, confianza) extraídos por OCR
            marcas_conocidas: Lista opcional de marcas conocidas para buscar
        
        Returns:
            Nombre de la marca identificada o None
        """
        if not textos_con_confianza:
            return None
        
        # Palabras comunes a ignorar (incluye palabras descriptivas que no son marcas)
        palabras_comunes = {
            'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'con', 'por', 'para',
            'ml', 'lt', 'l', 'kg', 'g', 'gr', 'cm', 'm', 'mm',
            'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for',
            'agua', 'water', 'bebida', 'drink', 'producto', 'product',
            'retornable', 'no', 'yes', 'si', 'no', 'envasad', 'envasada', 'envase',
            'botella', 'bottle', 'bidon', 'bidón'
        }
        
        # Números y códigos comunes
        patrones_ignorar = [
            r'^\d+$',  # Solo números
            r'^\d+[ml|lt|kg|g]$',  # Cantidades
            r'^[A-Z]{2,}\d+$',  # Códigos como "SKU123"
        ]
        
        # Primero, buscar marcas conocidas con tolerancia a errores de OCR
        if marcas_conocidas:
            texto_completo = ' '.join([texto for texto, _ in textos_con_confianza])
            texto_upper = texto_completo.upper()
            
            # Buscar coincidencias exactas primero
            for marca in marcas_conocidas:
                marca_upper = marca.upper()
                if marca_upper in texto_upper:
                    return marca
            
            # Buscar coincidencias aproximadas (para manejar errores de OCR)
            mejor_similitud = 0.0
            mejor_marca = None
            
            # Extraer todas las palabras del texto
            palabras_texto = re.findall(r'\b[A-Za-zÁÉÍÓÚáéíóúÑñ]{3,}\b', texto_completo)
            
            for marca in marcas_conocidas:
                marca_upper = marca.upper()
                for palabra in palabras_texto:
                    palabra_upper = palabra.upper()
                    # Solo comparar palabras de longitud similar (más de 3 caracteres)
                    if len(palabra_upper) < 3 or abs(len(palabra_upper) - len(marca_upper)) > 2:
                        continue
                    similitud = self._similitud_palabras(palabra_upper, marca_upper)
                    if similitud > 0.65 and similitud > mejor_similitud:  # Umbral más bajo para capturar más errores
                        mejor_similitud = similitud
                        mejor_marca = marca
            
            if mejor_marca:
                return mejor_marca
        
        # Si no hay marcas conocidas o no se encontró ninguna, buscar candidatos genéricos
        candidatos_marca = []
        
        # Extraer palabras candidatas (palabras con buena confianza y no comunes)
        for texto, confianza in textos_con_confianza:
            if confianza < 0.3:  # Umbral más bajo para capturar más texto
                continue
            
            # Dividir en palabras
            palabras = re.findall(r'\b[A-Za-zÁÉÍÓÚáéíóúÑñ]{3,}\b', texto)
            
            for palabra in palabras:
                palabra_clean = palabra.strip().upper()
                
                # Ignorar palabras comunes
                if palabra_clean.lower() in palabras_comunes:
                    continue
                
                # Ignorar patrones comunes
                if any(re.match(patron, palabra_clean, re.IGNORECASE) for patron in patrones_ignorar):
                    continue
                
                # Preferir palabras más largas (marcas suelen ser nombres propios)
                # Filtrar palabras que parecen descriptivas en lugar de marcas
                palabras_descriptivas = {'RETORNABLE', 'ENVASAD', 'ENVASADA', 'BOTELLA', 'BIDON', 'BIDÓN'}
                
                if len(palabra_clean) >= 4 and palabra_clean not in palabras_descriptivas:
                    # Priorizar palabras que empiezan con mayúscula (más probable que sean marcas)
                    es_mayuscula = palabra[0].isupper() if palabra else False
                    score_adicional = 0.2 if es_mayuscula else 0
                    candidatos_marca.append((palabra_clean, confianza + score_adicional, len(palabra_clean)))
        
        if not candidatos_marca:
            return None
        
        # Ordenar por confianza y longitud (más confianza y más larga = más probable que sea marca)
        candidatos_marca.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Filtrar candidatos que son muy similares entre sí (evitar duplicados)
        candidatos_filtrados = []
        for candidato in candidatos_marca:
            es_duplicado = False
            for existente in candidatos_filtrados:
                if self._similitud_palabras(candidato[0], existente[0]) > 0.8:
                    es_duplicado = True
                    break
            if not es_duplicado:
                candidatos_filtrados.append(candidato)
        
        if not candidatos_filtrados:
            return None
        
        # Tomar el mejor candidato
        mejor_candidato = candidatos_filtrados[0][0]
        
        # Normalizar (capitalizar primera letra)
        marca_normalizada = mejor_candidato.capitalize()
        
        return marca_normalizada
    
    def clasificar_deteccion(self, imagen_path: str, deteccion: Dict, 
                            marcas_conocidas: Optional[List[str]] = None) -> Dict:
        """
        Clasifica una detección agregando información de marca
        
        Args:
            imagen_path: Ruta a la imagen
            deteccion: Diccionario con información de detección (debe incluir 'bbox')
            marcas_conocidas: Lista opcional de marcas conocidas para buscar
        
        Returns:
            Detección actualizada con información de marca
        """
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            return deteccion
        
        # Extraer texto con confianza de la región del producto
        textos_con_confianza = self.extraer_texto_con_confianza(imagen, deteccion['bbox'])
        
        # Identificar marca de forma genérica
        marca = self.identificar_marca(textos_con_confianza, marcas_conocidas)
        
        # Combinar todos los textos extraídos para mostrar
        texto_completo = ' '.join([texto for texto, _ in textos_con_confianza])
        
        # Actualizar detección
        deteccion['texto_extraido'] = texto_completo
        deteccion['marca'] = marca if marca else None
        
        # Si se identificó marca, actualizar clase
        if marca:
            deteccion['clase'] = f"{deteccion['clase']}_{marca}"
        
        return deteccion
    
    def procesar_detecciones(self, imagen_path: str, detecciones: List[Dict],
                            marcas_conocidas: Optional[List[str]] = None) -> List[Dict]:
        """
        Procesa múltiples detecciones agregando información de marca
        
        Args:
            imagen_path: Ruta a la imagen
            detecciones: Lista de detecciones
            marcas_conocidas: Lista opcional de marcas conocidas para buscar
        
        Returns:
            Lista de detecciones actualizadas con marcas
        """
        detecciones_clasificadas = []
        
        for deteccion in detecciones:
            # Solo procesar si es un producto relevante (bottle, etc.)
            if deteccion['clase'] in ['bottle', 'cup', 'bowl', 'can', 'jar']:
                deteccion_clasificada = self.clasificar_deteccion(
                    imagen_path, deteccion, marcas_conocidas
                )
                detecciones_clasificadas.append(deteccion_clasificada)
            else:
                detecciones_clasificadas.append(deteccion)
        
        return detecciones_clasificadas

