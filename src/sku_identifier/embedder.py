#!/usr/bin/env python3
"""
Extractor de embeddings con CLIP.

Convierte una imagen (o crop de producto) en un vector numérico de 512 dimensiones.
Usa OpenAI CLIP ViT-B/32 pre-entrenado — no requiere entrenamiento adicional.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional

import torch
import clip
from PIL import Image


# Modelos CLIP disponibles (de más liviano a más pesado)
MODELOS_CLIP = {
    "ViT-B/32": 512,   # Rápido, bueno para CPU
    "ViT-B/16": 512,   # Mejor precisión, más lento
    "ViT-L/14": 768,   # Máxima precisión, requiere GPU
}

MODELO_DEFAULT = "ViT-B/16"


class CLIPEmbedder:
    """
    Extractor de embeddings usando CLIP.

    Convierte imágenes a vectores de dimensión fija para comparación
    por similitud coseno.
    """

    def __init__(
        self,
        modelo: str = MODELO_DEFAULT,
        device: str = "auto"
    ):
        """
        Args:
            modelo: Nombre del modelo CLIP ("ViT-B/32", "ViT-B/16", "ViT-L/14").
            device: "auto", "cpu", "cuda", "mps".
        """
        # Determinar dispositivo
        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        self._modelo_nombre = modelo
        self._dimension = MODELOS_CLIP.get(modelo, 512)

        print(f"🔧 Cargando CLIP: {modelo}")
        print(f"   Dispositivo: {self._device}")
        print(f"   Dimensión embedding: {self._dimension}")

        self.model, self.preprocess = clip.load(modelo, device=self._device)
        self.model.eval()

        print(f"   ✅ CLIP cargado")

    @property
    def dimension(self) -> int:
        """Dimensión del vector de embedding."""
        return self._dimension

    def embed(self, imagen_path: str) -> Optional[np.ndarray]:
        """
        Genera embedding de una imagen.

        Args:
            imagen_path: Ruta a la imagen (jpg, png, etc).

        Returns:
            Vector numpy normalizado (D,) o None si falla.
        """
        try:
            image = Image.open(imagen_path).convert("RGB")
        except Exception as e:
            print(f"  ⚠️ Error leyendo {imagen_path}: {e}")
            return None

        image_input = self.preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self.model.encode_image(image_input)

        # Normalizar a norma unitaria (para que cosine similarity = dot product)
        features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten()

    def embed_batch(self, imagenes: List[str]) -> np.ndarray:
        """
        Genera embeddings de múltiples imágenes en batch.

        Args:
            imagenes: Lista de rutas a imágenes.

        Returns:
            Matriz numpy (N, D) con un embedding normalizado por fila.
            Si alguna imagen falla, se omite (la matriz puede tener menos filas).
        """
        images_pil = []
        indices_validos = []

        for i, img_path in enumerate(imagenes):
            try:
                img = Image.open(img_path).convert("RGB")
                images_pil.append(img)
                indices_validos.append(i)
            except Exception as e:
                print(f"  ⚠️ Saltando {img_path}: {e}")

        if not images_pil:
            return np.zeros((0, self._dimension), dtype=np.float32)

        # Preprocesar y apilar
        batch = torch.stack([self.preprocess(img) for img in images_pil]).to(self._device)

        with torch.no_grad():
            features = self.model.encode_image(batch)

        # Normalizar
        features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()

    def embed_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Genera embedding de un crop ya cargado en memoria (numpy BGR).

        Args:
            crop: Imagen numpy BGR (OpenCV format).

        Returns:
            Vector numpy normalizado (D,) o None si falla.
        """
        import cv2

        try:
            # Convertir BGR → RGB → PIL
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
        except Exception as e:
            print(f"  ⚠️ Error convirtiendo crop: {e}")
            return None

        image_input = self.preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self.model.encode_image(image_input)

        features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten()
