"""Capa B — Identificación de SKU por embeddings."""
from .embedder import CLIPEmbedder
from .vector_store import VectorStore
from .identifier import SKUIdentifier
from .categorizer import PackagingCategorizer

__all__ = ["CLIPEmbedder", "VectorStore", "SKUIdentifier", "PackagingCategorizer"]
