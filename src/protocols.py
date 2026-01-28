#!/usr/bin/env python3
"""
Protocolos (abstracciones) para componentes del sistema
Implementa Dependency Inversion Principle (DIP)
"""

from typing import Protocol, List, Dict, Optional
from pathlib import Path


class DetectorProtocol(Protocol):
    """
    Protocolo para detectores de productos
    Define el contrato que debe cumplir cualquier implementación de detector
    """
    
    def detectar_en_imagen(
        self, 
        ruta_imagen: str, 
        guardar_crops: bool = False,
        crops_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Detecta productos en una imagen
        
        Args:
            ruta_imagen: Ruta a la imagen
            guardar_crops: Si guardar crops de detecciones
            crops_dir: Directorio para crops
            
        Returns:
            Lista de detecciones con bbox, clase, confianza
        """
        ...
    
    def procesar_frames(
        self,
        directorio_frames: str,
        guardar_crops: bool = False,
        crops_dir: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Procesa múltiples frames
        
        Args:
            directorio_frames: Directorio con imágenes
            guardar_crops: Si guardar crops
            crops_dir: Directorio para crops
            
        Returns:
            Diccionario frame -> detecciones
        """
        ...


class ReconocedorMarcasProtocol(Protocol):
    """
    Protocolo para reconocedores de marcas
    Define el contrato para OCR y reconocimiento
    """
    
    def procesar_detecciones(
        self,
        imagen_path: str,
        detecciones: List[Dict],
        marcas_conocidas: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Procesa detecciones agregando información de marca
        
        Args:
            imagen_path: Ruta a la imagen
            detecciones: Lista de detecciones
            marcas_conocidas: Marcas conocidas para búsqueda
            
        Returns:
            Detecciones con información de marca
        """
        ...


class IdentificadorSKUProtocol(Protocol):
    """
    Protocolo para identificadores de SKU
    Define el contrato para identificación visual
    """
    
    def identificar(
        self,
        crop_path: str,
        top_k: int = 3,
        threshold: float = 0.5
    ) -> Dict:
        """
        Identifica SKU de un crop
        
        Args:
            crop_path: Ruta al crop
            top_k: Candidatos a retornar
            threshold: Umbral de similitud
            
        Returns:
            Dict con ean, confianza, top_matches
        """
        ...


class OCRStrategy(Protocol):
    """
    Protocolo para estrategias de OCR
    Implementa Strategy Pattern para Open/Closed Principle
    """
    
    def extraer_texto(self, imagen, bbox) -> str:
        """
        Extrae texto de una región de imagen
        
        Args:
            imagen: Imagen (numpy array)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Texto extraído
        """
        ...
    
    def extraer_texto_con_confianza(self, imagen, bbox) -> List[tuple]:
        """
        Extrae texto con niveles de confianza
        
        Args:
            imagen: Imagen (numpy array)
            bbox: Bounding box
            
        Returns:
            Lista de (texto, confianza)
        """
        ...


class ReporteExporter(Protocol):
    """
    Protocolo para exportadores de reportes
    Implementa Strategy Pattern para diferentes formatos
    """
    
    def exportar(
        self,
        conteo: Dict[str, int],
        output_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Exporta conteo de productos a archivo
        
        Args:
            conteo: Diccionario SKU -> cantidad
            output_path: Ruta de salida
            metadata: Metadata adicional
            
        Returns:
            Ruta del archivo generado
        """
        ...


class VisualizadorProtocol(Protocol):
    """
    Protocolo para visualizadores de detecciones
    Separa visualización de detección (ISP)
    """
    
    def generar_imagen_anotada(
        self,
        ruta_imagen: str,
        detecciones: List[Dict],
        output_path: Optional[str] = None
    ) -> str:
        """
        Genera imagen con bounding boxes
        
        Args:
            ruta_imagen: Ruta a imagen original
            detecciones: Lista de detecciones
            output_path: Ruta de salida
            
        Returns:
            Ruta de imagen generada
        """
        ...
