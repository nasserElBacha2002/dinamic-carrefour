#!/usr/bin/env python3
"""
Categorizador de packaging — Clasifica crops por tipo de envase usando CLIP zero-shot.

En vez de buscar un crop contra TODOS los SKUs del catálogo, primero se
clasifica el tipo de packaging (botella, lata, bolsa, etc.) y después se
busca solo dentro de esa categoría.

Esto:
  - Reduce falsos positivos (una lata nunca se confunde con una botella).
  - Acelera la búsqueda (menos comparaciones).
  - Escala mejor con más SKUs.

Usa CLIP zero-shot: compara el embedding visual del crop contra embeddings
de texto que describen cada tipo de packaging. No requiere entrenamiento.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import clip


# ─── Categorías de packaging ────────────────────────────────────────────────
# Cada categoría tiene múltiples descripciones textuales para que CLIP
# entienda mejor qué buscar (prompt engineering zero-shot).

CATEGORIAS_PACKAGING: Dict[str, List[str]] = {
    "botella": [
        "a plastic bottle on a shelf",
        "a PET bottle of soda",
        "a water bottle",
        "a beverage bottle",
        "a plastic drink bottle",
    ],
    "lata": [
        "a metal can on a shelf",
        "an aluminum soda can",
        "a beer can",
        "a canned drink",
        "a tin can of food",
    ],
    "bolsa": [
        "a bag of chips on a shelf",
        "a plastic bag of snacks",
        "a bag of bread",
        "a flexible plastic package",
        "a bag of food on a shelf",
    ],
    "caja": [
        "a cardboard box on a shelf",
        "a cereal box",
        "a box of cookies",
        "a rectangular cardboard package",
        "a tea box",
    ],
    "paquete": [
        "a sealed package of yerba mate",
        "a foil package of coffee",
        "a sealed bag of rice",
        "a metallic sealed package",
        "a vacuum sealed food package",
    ],
    "tubo": [
        "a tube of toothpaste",
        "a plastic tube on a shelf",
        "a squeezable tube",
        "a cosmetic tube",
        "a hygiene product tube",
    ],
    "frasco": [
        "a glass jar on a shelf",
        "a jar of jam",
        "a sauce bottle",
        "a glass container with lid",
        "a small glass jar of food",
    ],
}

# Categorías válidas (para validación)
CATEGORIAS_VALIDAS = set(CATEGORIAS_PACKAGING.keys())


class PackagingCategorizer:
    """
    Clasifica un crop de producto por tipo de packaging usando CLIP zero-shot.

    Flujo:
      1. Pre-calcula embeddings de texto para cada categoría (una sola vez).
      2. Para cada crop, genera embedding visual.
      3. Compara contra embeddings de texto → la categoría con mayor similitud gana.
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        categorias: Optional[Dict[str, List[str]]] = None
    ):
        """
        Args:
            model: Modelo CLIP ya cargado (compartido con el embedder).
            device: Dispositivo ("cpu", "cuda", "mps").
            categorias: Override de categorías (default: CATEGORIAS_PACKAGING).
        """
        self._model = model
        self._device = device
        self._categorias = categorias or CATEGORIAS_PACKAGING

        # Pre-calcular embeddings de texto para cada categoría
        self._text_embeddings: Dict[str, np.ndarray] = {}
        self._precalcular_text_embeddings()

    def _precalcular_text_embeddings(self) -> None:
        """
        Calcula y promedia embeddings de texto por categoría.
        Se ejecuta una sola vez al inicializar.
        """
        for cat, textos in self._categorias.items():
            tokens = clip.tokenize(textos).to(self._device)
            with torch.no_grad():
                text_features = self._model.encode_text(tokens)

            # Normalizar cada embedding
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Promediar y normalizar el centroide
            centroide = text_features.mean(dim=0)
            centroide = centroide / centroide.norm()

            self._text_embeddings[cat] = centroide.cpu().numpy()

        n_cats = len(self._text_embeddings)
        print(f"   📦 Categorizador: {n_cats} categorías de packaging cargadas")

    def clasificar_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Clasifica un embedding por tipo de packaging.

        Args:
            embedding: Vector embedding normalizado (D,) de un crop.
            top_k: Cantidad de categorías a devolver.

        Returns:
            Lista de (categoria, similitud) ordenada de mayor a menor.
        """
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return [("botella", 0.0)]  # fallback
        embedding = embedding / norm

        resultados = []
        for cat, text_emb in self._text_embeddings.items():
            sim = float(np.dot(embedding, text_emb))
            resultados.append((cat, sim))

        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_k]

    def clasificar(
        self,
        embedding: np.ndarray,
        min_confianza: float = 0.15
    ) -> str:
        """
        Devuelve la categoría más probable.

        Args:
            embedding: Vector embedding normalizado (D,).
            min_confianza: Si la mejor categoría tiene sim < min_confianza,
                          devuelve "otro".

        Returns:
            Nombre de la categoría ("botella", "lata", etc.) o "otro".
        """
        resultados = self.clasificar_embedding(embedding, top_k=1)
        if not resultados:
            return "otro"

        cat, sim = resultados[0]
        if sim < min_confianza:
            return "otro"

        return cat

    @property
    def categorias_disponibles(self) -> List[str]:
        """Lista de categorías configuradas."""
        return list(self._categorias.keys())
