#!/usr/bin/env python3
"""
Protocolos (abstracciones) para componentes del sistema — Sprint 2

Define los contratos que deben cumplir las implementaciones.
Implementa Dependency Inversion Principle (DIP).
"""

from typing import Protocol, List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np


class DetectorProtocol(Protocol):
    """
    Protocolo para detectores genéricos de productos.
    
    El detector localiza productos en una imagen sin clasificarlos por SKU.
    Devuelve bounding boxes + confianza. La clase es genérica ("producto").
    """

    def detectar(
        self,
        imagen_path: str,
        confianza_minima: float = 0.25
    ) -> List[Dict]:
        """
        Detecta productos en una imagen.

        Args:
            imagen_path: Ruta a la imagen.
            confianza_minima: Umbral mínimo de confianza (0-1).

        Returns:
            Lista de detecciones, cada una con:
                - bbox: [x1, y1, x2, y2] en píxeles
                - confianza: float (0-1)
                - clase: str (ej: "producto", "botella")
        """
        ...


class EmbedderProtocol(Protocol):
    """
    Protocolo para extractores de embeddings.
    
    Convierte una imagen (o crop) en un vector numérico de dimensión fija.
    """

    def embed(self, imagen_path: str) -> np.ndarray:
        """
        Genera embedding de una imagen.

        Args:
            imagen_path: Ruta a la imagen.

        Returns:
            Vector numpy de dimensión fija (ej: 512 para CLIP ViT-B/32).
        """
        ...

    def embed_batch(self, imagenes: List[str]) -> np.ndarray:
        """
        Genera embeddings de múltiples imágenes.

        Args:
            imagenes: Lista de rutas a imágenes.

        Returns:
            Matriz numpy (N, D) con un embedding por fila.
        """
        ...

    @property
    def dimension(self) -> int:
        """Dimensión del vector de embedding."""
        ...


class VectorStoreProtocol(Protocol):
    """
    Protocolo para almacenes de vectores (embeddings de SKUs).
    
    Guarda embeddings de referencia por SKU y busca los más similares.
    """

    def agregar_sku(
        self,
        ean: str,
        embeddings: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Agrega o actualiza embeddings de un SKU.

        Args:
            ean: Código EAN del producto.
            embeddings: Matriz (N, D) con N embeddings de referencia.
            metadata: Info adicional (descripción, etc).
        """
        ...

    def buscar(
        self,
        query: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Busca los SKUs más similares a un embedding query.

        Args:
            query: Vector embedding (D,) del crop a identificar.
            top_k: Número de candidatos a devolver.

        Returns:
            Lista de (ean, similaridad) ordenada de mayor a menor.
        """
        ...

    def tiene_sku(self, ean: str) -> bool:
        """Verifica si un SKU ya existe en el store."""
        ...

    @property
    def total_skus(self) -> int:
        """Número total de SKUs en el store."""
        ...


class IdentificadorSKUProtocol(Protocol):
    """
    Protocolo para identificadores de SKU.
    
    Recibe un crop y devuelve el EAN más probable con su confianza.
    """

    def identificar(
        self,
        crop_path: str,
        top_k: int = 3,
        threshold: float = 0.5
    ) -> Dict:
        """
        Identifica el SKU de un crop.

        Args:
            crop_path: Ruta al crop del producto.
            top_k: Candidatos a retornar.
            threshold: Umbral mínimo de similitud.

        Returns:
            Dict con:
                - ean: str (EAN identificado o "UNKNOWN")
                - confianza: float (similitud del mejor match)
                - top_matches: List[Tuple[str, float]] (top-k candidatos)
                - status: str ("matched", "unknown", "ambiguous")
        """
        ...


class ReporteExporter(Protocol):
    """
    Protocolo para exportadores de reportes.
    """

    def exportar(
        self,
        conteo: Dict[str, int],
        output_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Exporta conteo de productos a archivo.

        Args:
            conteo: Diccionario EAN -> cantidad.
            output_path: Ruta de salida.
            metadata: Metadata adicional.

        Returns:
            Ruta del archivo generado.
        """
        ...
