#!/usr/bin/env python3
"""
Estrategias de OCR - Strategy Pattern
Implementa Open/Closed Principle para reconocimiento de texto
"""

import cv2
from typing import List, Tuple
from abc import ABC, abstractmethod

# Imports opcionales
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class OCRStrategyBase(ABC):
    """Clase base abstracta para estrategias de OCR"""
    
    @abstractmethod
    def extraer_texto(self, imagen, bbox) -> str:
        """Extrae texto de región de imagen"""
        pass
    
    @abstractmethod
    def extraer_texto_con_confianza(self, imagen, bbox) -> List[Tuple[str, float]]:
        """Extrae texto con niveles de confianza"""
        pass
    
    def _preprocesar_region(self, imagen, bbox):
        """
        Preprocesa región para mejor OCR
        Método común a todas las estrategias
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Clamp a límites de imagen
        h, w = imagen.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Extraer región
        region = imagen[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Redimensionar si es muy pequeño
        if gray.shape[0] < 50 or gray.shape[1] < 50:
            scale = max(50 / gray.shape[0], 50 / gray.shape[1])
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return gray


class TesseractOCRStrategy(OCRStrategyBase):
    """Estrategia de OCR usando Tesseract"""
    
    def __init__(self, lang='spa+eng'):
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract no disponible. Instalar: pip install pytesseract")
        self.lang = lang
    
    def extraer_texto(self, imagen, bbox) -> str:
        """Extrae texto usando Tesseract"""
        region = self._preprocesar_region(imagen, bbox)
        if region is None:
            return ""
        
        texto = pytesseract.image_to_string(region, lang=self.lang)
        return texto.strip()
    
    def extraer_texto_con_confianza(self, imagen, bbox) -> List[Tuple[str, float]]:
        """Extrae texto con confianza estimada"""
        texto = self.extraer_texto(imagen, bbox)
        if texto:
            # Tesseract no proporciona confianza fácilmente, usar valor estimado
            return [(texto, 0.8)]
        return []


class EasyOCRStrategy(OCRStrategyBase):
    """Estrategia de OCR usando EasyOCR"""
    
    def __init__(self, languages=['es', 'en'], gpu=False):
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR no disponible. Instalar: pip install easyocr")
        print(f"🔄 Inicializando EasyOCR ({', '.join(languages)})...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("✅ EasyOCR listo")
    
    def extraer_texto(self, imagen, bbox) -> str:
        """Extrae texto usando EasyOCR"""
        region = self._preprocesar_region(imagen, bbox)
        if region is None:
            return ""
        
        resultados = self.reader.readtext(region)
        texto = ' '.join([result[1] for result in resultados])
        return texto.strip()
    
    def extraer_texto_con_confianza(self, imagen, bbox) -> List[Tuple[str, float]]:
        """Extrae texto con confianzas"""
        region = self._preprocesar_region(imagen, bbox)
        if region is None:
            return []
        
        resultados = self.reader.readtext(region)
        return [(result[1], result[2]) for result in resultados]


class DummyOCRStrategy(OCRStrategyBase):
    """
    Estrategia dummy para testing o cuando no hay OCR disponible
    """
    
    def extraer_texto(self, imagen, bbox) -> str:
        """Retorna texto vacío"""
        return ""
    
    def extraer_texto_con_confianza(self, imagen, bbox) -> List[Tuple[str, float]]:
        """Retorna lista vacía"""
        return []


def crear_ocr_strategy(metodo: str = 'easyocr', **kwargs) -> OCRStrategyBase:
    """
    Factory function para crear estrategias de OCR
    
    Args:
        metodo: 'tesseract', 'easyocr', o 'dummy'
        **kwargs: Argumentos adicionales para la estrategia
        
    Returns:
        Instancia de OCRStrategyBase
    """
    if metodo == 'tesseract' and TESSERACT_AVAILABLE:
        return TesseractOCRStrategy(**kwargs)
    elif metodo == 'easyocr' and EASYOCR_AVAILABLE:
        return EasyOCRStrategy(**kwargs)
    elif metodo == 'dummy':
        return DummyOCRStrategy()
    else:
        # Fallback
        if EASYOCR_AVAILABLE:
            print(f"⚠️  '{metodo}' no disponible, usando EasyOCR")
            return EasyOCRStrategy(**kwargs)
        elif TESSERACT_AVAILABLE:
            print(f"⚠️  '{metodo}' no disponible, usando Tesseract")
            return TesseractOCRStrategy(**kwargs)
        else:
            print(f"⚠️  No hay OCR disponible, usando DummyOCR")
            return DummyOCRStrategy()
