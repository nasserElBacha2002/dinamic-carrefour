#!/usr/bin/env python3
"""
Factory para crear componentes del sistema
Implementa Factory Pattern y encapsula lógica de creación
Centraliza inyección de dependencias
"""

from typing import Optional, Dict, Any
from pathlib import Path

# Importar clases concretas
from src.detectar_productos import DetectorProductos
from src.reconocer_marcas import ReconocedorMarcas
from src.ocr_strategies import crear_ocr_strategy, OCRStrategyBase
from src.exporters import crear_exporter, ReporteExporterBase

# Importar identificador SKU si está disponible
try:
    from src.identificar_sku_retrieval import IdentificadorSKURetrieval
    IDENTIFICADOR_SKU_AVAILABLE = True
except ImportError:
    IDENTIFICADOR_SKU_AVAILABLE = False


class ComponentFactory:
    """
    Factory para crear componentes del sistema con dependencias inyectadas
    
    SOLID Benefits:
    - Dependency Inversion: Encapsula creación de dependencias
    - Single Responsibility: Solo se encarga de crear objetos
    - Open/Closed: Fácil agregar nuevos tipos de componentes
    """
    
    @staticmethod
    def crear_reconocedor_marcas(
        ocr_metodo: str = 'easyocr',
        **kwargs
    ) -> ReconocedorMarcas:
        """
        Crea reconocedor de marcas con estrategia OCR configurada
        
        Args:
            ocr_metodo: 'easyocr', 'tesseract', o 'dummy'
            **kwargs: Argumentos adicionales para la estrategia OCR
            
        Returns:
            Instancia de ReconocedorMarcas
        """
        # Crear estrategia OCR
        ocr_strategy = crear_ocr_strategy(ocr_metodo, **kwargs)
        
        # Inyectar estrategia en reconocedor
        return ReconocedorMarcas(ocr_strategy=ocr_strategy)
    
    @staticmethod
    def crear_detector(
        modelo_path: Optional[str] = None,
        confianza_minima: Optional[float] = None,
        with_reconocedor_marcas: bool = True,
        ocr_metodo: str = 'easyocr',
        export_formato: str = 'csv'
    ) -> DetectorProductos:
        """
        Crea detector de productos con todas sus dependencias
        
        Args:
            modelo_path: Ruta al modelo YOLO
            confianza_minima: Umbral de confianza
            with_reconocedor_marcas: Si incluir reconocedor de marcas
            ocr_metodo: Método OCR para reconocedor
            export_formato: Formato de exportación ('csv', 'json', 'multi')
            
        Returns:
            Instancia de DetectorProductos con dependencias inyectadas
        """
        # Crear reconocedor de marcas si se solicita
        reconocedor = None
        if with_reconocedor_marcas:
            try:
                reconocedor = ComponentFactory.crear_reconocedor_marcas(ocr_metodo)
                print("✅ Reconocimiento de marcas configurado")
            except Exception as e:
                print(f"⚠️  No se pudo crear reconocedor de marcas: {e}")
        
        # Crear exportador
        exporter = crear_exporter(export_formato)
        
        # Crear detector con dependencias inyectadas
        return DetectorProductos(
            modelo_path=modelo_path,
            confianza_minima=confianza_minima,
            reconocedor_marcas=reconocedor,
            exporter=exporter
        )
    
    @staticmethod
    def crear_identificador_sku(
        catalogo_dir: str,
        embeddings_path: Optional[str] = None,
        modelo: str = 'resnet50'
    ):
        """
        Crea identificador SKU si está disponible
        
        Args:
            catalogo_dir: Directorio con catálogo de imágenes
            embeddings_path: Ruta a embeddings pre-calculados
            modelo: Modelo de features ('resnet50', 'mobilenet_v2')
            
        Returns:
            Instancia de IdentificadorSKURetrieval o None
        """
        if not IDENTIFICADOR_SKU_AVAILABLE:
            print("⚠️  IdentificadorSKURetrieval no disponible")
            return None
        
        if not Path(catalogo_dir).exists():
            print(f"⚠️  Catálogo no encontrado: {catalogo_dir}")
            return None
        
        try:
            # Auto-generar path de embeddings si no se proporciona
            if embeddings_path is None:
                embeddings_path = str(Path(catalogo_dir).parent / "embeddings.pkl")
            
            return IdentificadorSKURetrieval(
                catalogo_dir=catalogo_dir,
                embeddings_path=embeddings_path,
                modelo=modelo
            )
        except Exception as e:
            print(f"⚠️  Error creando identificador SKU: {e}")
            return None
    
    @staticmethod
    def desde_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea componentes desde diccionario de configuración
        
        Args:
            config: Diccionario con configuración
                - modelo_path: str
                - confianza_minima: float
                - reconocer_marcas: bool
                - ocr_metodo: str
                - export_formato: str
                - identificar_sku: bool
                - catalogo_imagenes: str
                
        Returns:
            Dict con componentes creados:
                - detector: DetectorProductos
                - identificador_sku: IdentificadorSKURetrieval o None
        """
        # Crear detector
        detector = ComponentFactory.crear_detector(
            modelo_path=config.get('modelo_path'),
            confianza_minima=config.get('confianza_minima'),
            with_reconocedor_marcas=config.get('reconocer_marcas', True),
            ocr_metodo=config.get('ocr_metodo', 'easyocr'),
            export_formato=config.get('export_formato', 'csv')
        )
        
        # Crear identificador SKU si se solicita
        identificador_sku = None
        if config.get('identificar_sku', False):
            identificador_sku = ComponentFactory.crear_identificador_sku(
                catalogo_dir=config.get('catalogo_imagenes', ''),
                embeddings_path=config.get('embeddings_path')
            )
        
        return {
            'detector': detector,
            'identificador_sku': identificador_sku
        }


def ejemplo_uso():
    """Ejemplo de uso del factory"""
    
    # Método 1: Crear componentes individuales
    print("=== Método 1: Componentes individuales ===")
    reconocedor = ComponentFactory.crear_reconocedor_marcas(ocr_metodo='easyocr')
    detector = ComponentFactory.crear_detector(
        with_reconocedor_marcas=True,
        export_formato='json'
    )
    
    # Método 2: Desde configuración
    print("\n=== Método 2: Desde configuración ===")
    config = {
        'modelo_path': 'modelos/yolov8_gondola_mvp.pt',
        'confianza_minima': 0.25,
        'reconocer_marcas': True,
        'ocr_metodo': 'easyocr',
        'export_formato': 'multi',  # CSV + JSON
        'identificar_sku': True,
        'catalogo_imagenes': 'imagenes/'
    }
    
    componentes = ComponentFactory.desde_config(config)
    print(f"Detector: {componentes['detector']}")
    print(f"Identificador SKU: {componentes['identificador_sku']}")


if __name__ == "__main__":
    ejemplo_uso()
