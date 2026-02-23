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

Las categorías se cargan desde la base de datos (tabla packaging_types).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import clip


def _cargar_categorias_desde_db() -> Optional[Dict[str, List[str]]]:
    """
    Carga categorías de packaging desde la base de datos.
    
    Returns:
        Dict {id_categoria: [prompts]} o None si falla.
    """
    try:
        from src.database.repository import ProductoRepository
        
        repo = ProductoRepository()
        packaging_types = repo.listar_packaging_types()
        
        if not packaging_types:
            return None
        
        categorias: Dict[str, List[str]] = {}
        for pt in packaging_types:
            cat_id = str(pt.get("id", ""))
            prompts = pt.get("prompts_clip", [])
            
            # Validar que prompts sea una lista de strings
            if isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
                categorias[cat_id] = prompts
            elif prompts:
                # Si viene como string, intentar parsear
                import json
                try:
                    if isinstance(prompts, str):
                        prompts = json.loads(prompts)
                    if isinstance(prompts, list):
                        categorias[cat_id] = [str(p) for p in prompts]
                except (json.JSONDecodeError, TypeError):
                    pass
        
        if categorias:
            return categorias
        return None
        
    except Exception as e:
        print(f"   ⚠️  No se pudieron cargar categorías desde DB: {e}")
        return None


class PackagingCategorizer:
    """
    Clasifica un crop de producto por tipo de packaging usando CLIP zero-shot.

    Flujo:
      1. Carga categorías desde DB (tabla packaging_types).
      2. Pre-calcula embeddings de texto para cada categoría (una sola vez).
      3. Para cada crop, genera embedding visual.
      4. Compara contra embeddings de texto → la categoría con mayor similitud gana.
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        categorias: Optional[Dict[str, List[str]]] = None,
        usar_db: bool = True,
    ):
        """
        Args:
            model: Modelo CLIP ya cargado (compartido con el embedder).
            device: Dispositivo ("cpu", "cuda", "mps").
            categorias: Override de categorías (si se proporciona, se usa directamente).
            usar_db: Si True, intenta cargar desde DB. Si False, requiere categorias.
        
        Raises:
            ValueError: Si no se pueden cargar categorías desde DB y no se proporcionan manualmente.
        """
        self._model = model
        self._device = device
        
        # Cargar categorías: override manual > DB
        if categorias is not None:
            self._categorias = categorias
            self._fuente = "override"
        elif usar_db:
            categorias_db = _cargar_categorias_desde_db()
            print(f"   Categorías cargadas desde DB: {categorias_db}")
            if categorias_db:
                self._categorias = categorias_db
                self._fuente = "database"
            else:
                raise ValueError(
                    "No se pudieron cargar categorías desde la base de datos. "
                    "Asegúrate de que la tabla 'packaging_types' exista y tenga datos, "
                    "o proporciona categorías manualmente con el parámetro 'categorias'."
                )
        else:
            raise ValueError(
                "Se requiere cargar categorías desde DB (usar_db=True) o proporcionarlas "
                "manualmente (categorias=...)."
            )

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
        fuente_str = {
            "database": "desde DB",
            "override": "override manual",
        }.get(self._fuente, "desconocida")
        print(f"   📦 Categorizador: {n_cats} categorías de packaging cargadas ({fuente_str})")

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
            # Si el embedding es inválido, retornar la primera categoría disponible con similitud 0
            if self._text_embeddings:
                primera_cat = list(self._text_embeddings.keys())[0]
                return [(primera_cat, 0.0)]
            return []
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
