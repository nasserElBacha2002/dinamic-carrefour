#!/usr/bin/env python3
"""
Extractor de embeddings con CLIP.

Convierte una imagen (o crop de producto) en un vector numérico de 512 dimensiones.
Usa OpenAI CLIP ViT-B/32 pre-entrenado — no requiere entrenamiento adicional.
"""

import os
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


def _obtener_modelo_desde_env() -> str:
    """
    Obtiene el modelo CLIP desde variable de entorno.
    
    Variable de entorno: CLIP_MODEL
    Valores válidos: "ViT-B/32", "ViT-B/16", "ViT-L/14"
    
    Returns:
        Nombre del modelo a usar (validado).
    """
    modelo_env = os.getenv("CLIP_MODEL", "").strip()
    
    if not modelo_env:
        return MODELO_DEFAULT
    
    # Validar que el modelo sea válido
    if modelo_env in MODELOS_CLIP:
        return modelo_env
    
    # Si no es válido, mostrar warning y usar default
    print(f"   ⚠️  Modelo CLIP inválido en CLIP_MODEL: '{modelo_env}'")
    print(f"       Modelos válidos: {', '.join(MODELOS_CLIP.keys())}")
    print(f"       Usando default: {MODELO_DEFAULT}")
    return MODELO_DEFAULT


class CLIPEmbedder:
    """
    Extractor de embeddings usando CLIP.

    Convierte imágenes a vectores de dimensión fija para comparación
    por similitud coseno.
    """

    def __init__(
        self,
        modelo: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Args:
            modelo: Nombre del modelo CLIP ("ViT-B/32", "ViT-B/16", "ViT-L/14").
                   Si es None, se lee desde variable de entorno CLIP_MODEL.
                   Si CLIP_MODEL no está definida, usa el default.
            device: "auto", "cpu", "cuda", "mps".
        """
        # Determinar modelo: parámetro > variable de entorno > default
        if modelo is None:
            modelo = _obtener_modelo_desde_env()
            # Determinar fuente: env o default
            self._modelo_fuente = "variable de entorno CLIP_MODEL" if os.getenv("CLIP_MODEL") else "default"
        else:
            self._modelo_fuente = "parámetro explícito"
        
        # Validar modelo
        if modelo not in MODELOS_CLIP:
            raise ValueError(
                f"Modelo CLIP inválido: '{modelo}'. "
                f"Modelos válidos: {', '.join(MODELOS_CLIP.keys())}"
            )
        
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
        self._dimension = MODELOS_CLIP[modelo]

        print(f"🔧 Cargando CLIP: {modelo}")
        print(f"   Fuente: {self._modelo_fuente}")
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
