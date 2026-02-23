#!/usr/bin/env python3
"""
Vector Store — Base de embeddings de SKUs.

Almacena embeddings de referencia por EAN y realiza búsqueda
por similitud coseno para identificar productos.

Persistencia: cada SKU se guarda como <EAN>.npy en catalog/embeddings/.
Un índice general se guarda en catalog/index.json.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class VectorStore:
    """
    Almacén de embeddings de referencia para identificación SKU.

    Cada SKU tiene:
      - N embeddings de referencia (uno por imagen de catálogo).
      - Un centroide (promedio normalizado de los embeddings).
      - Metadata (descripción, nro imágenes, fecha de cálculo).
    """

    def __init__(
        self,
        embeddings_dir: str = "catalog/embeddings",
        dimension: int = 512
    ):
        """
        Args:
            embeddings_dir: Directorio donde se persisten los .npy.
            dimension: Dimensión esperada de cada embedding.
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension

        # Estado en memoria
        self._skus: Dict[str, Dict] = {}  # ean -> {embeddings, centroide, metadata}
        self._index_path = self.embeddings_dir.parent / "index.json"

        # Cargar índice existente
        self._cargar_indice()
        
        # Validar dimensiones al cargar
        self._validar_dimensiones()

    def _cargar_indice(self) -> None:
        """Carga embeddings existentes desde disco."""
        if self._index_path.exists():
            try:
                index_data = json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                index_data = {}
        else:
            index_data = {}

        cargados = 0
        for ean, meta in index_data.items():
            npy_path = self.embeddings_dir / f"{ean}.npy"
            if npy_path.exists():
                try:
                    embeddings = np.load(str(npy_path))
                    centroide = embeddings.mean(axis=0)
                    centroide = centroide / (np.linalg.norm(centroide) + 1e-10)

                    self._skus[ean] = {
                        "embeddings": embeddings,
                        "centroide": centroide,
                        "metadata": meta,
                    }
                    cargados += 1
                except Exception as e:
                    print(f"  ⚠️ Error cargando {ean}.npy: {e}")

        if cargados > 0:
            print(f"📦 Vector store: {cargados} SKUs cargados desde {self.embeddings_dir}")
    
    def _validar_dimensiones(self) -> None:
        """Valida que todos los embeddings tengan la dimensión correcta."""
        errores = []
        for ean, data in self._skus.items():
            emb = data.get("embeddings")
            if emb is not None and emb.shape[1] != self.dimension:
                errores.append(f"{ean}: esperado {self.dimension}, tiene {emb.shape[1]}")
        
        if errores:
            print(f"   ⚠️  ADVERTENCIA: {len(errores)} SKUs con dimensiones incorrectas:")
            for err in errores[:5]:  # Mostrar solo los primeros 5
                print(f"      {err}")
            if len(errores) > 5:
                print(f"      ... y {len(errores) - 5} más")
            print(f"   💡 Regenerá embeddings con: python scripts/agregar_sku.py --todos --forzar")

    def _guardar_indice(self) -> None:
        """Persiste el índice JSON a disco."""
        index_data = {}
        for ean, data in self._skus.items():
            meta = dict(data.get("metadata", {}))
            # Convertir tipos no serializables
            meta["n_embeddings"] = int(data["embeddings"].shape[0])
            index_data[ean] = meta

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(
            json.dumps(index_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def agregar_sku(
        self,
        ean: str,
        embeddings: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Agrega o reemplaza embeddings de un SKU.

        Args:
            ean: Código EAN del producto.
            embeddings: Matriz (N, D) con N embeddings de referencia.
            metadata: Info adicional (descripcion, etc).
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimensión incorrecta: esperado {self.dimension}, "
                f"recibido {embeddings.shape[1]}"
            )

        # Normalizar cada embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms

        # Centroide normalizado
        centroide = embeddings.mean(axis=0)
        centroide = centroide / (np.linalg.norm(centroide) + 1e-10)

        meta = metadata or {}
        meta["actualizado"] = datetime.now().isoformat()

        self._skus[ean] = {
            "embeddings": embeddings,
            "centroide": centroide,
            "metadata": meta,
        }

        # Persistir .npy
        npy_path = self.embeddings_dir / f"{ean}.npy"
        np.save(str(npy_path), embeddings)

        # Actualizar índice
        self._guardar_indice()

    def eliminar_sku(self, ean: str) -> bool:
        """Elimina un SKU del store."""
        if ean not in self._skus:
            return False

        del self._skus[ean]

        npy_path = self.embeddings_dir / f"{ean}.npy"
        if npy_path.exists():
            npy_path.unlink()

        self._guardar_indice()
        return True

    def buscar(
        self,
        query: np.ndarray,
        top_k: int = 3,
        categoria: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Busca los SKUs más similares a un embedding query.

        Compara contra TODOS los embeddings individuales de cada SKU
        y usa la MÁXIMA similitud encontrada como score del SKU.
        Esto es más preciso que comparar solo contra el centroide,
        ya que no diluye rasgos distintivos.

        Si se especifica una categoría, solo compara contra SKUs de esa categoría.
        Esto reduce falsos positivos y acelera la búsqueda.

        Args:
            query: Vector embedding (D,) del crop a identificar.
            top_k: Número de candidatos a devolver.
            categoria: Si se especifica, filtra solo SKUs de esa categoría.

        Returns:
            Lista de (ean, similitud) ordenada de mayor a menor similitud.
        """
        if not self._skus:
            return []

        # Normalizar query
        query = query.flatten()
        
        # Validar dimensión del query
        if query.shape[0] != self.dimension:
            print(f"   ⚠️  Dimensión de query incorrecta: esperado {self.dimension}, recibido {query.shape[0]}")
            print(f"   💡 Verificá que CLIP_MODEL coincida con el usado para generar embeddings")
            return []
        
        norm = np.linalg.norm(query)
        if norm < 1e-10:
            return []
        query = query / norm

        # Filtrar por categoría si se especifica
        skus_a_buscar = self._skus
        candidatos_en_categoria = 0
        if categoria:
            skus_a_buscar = {
                ean: data for ean, data in self._skus.items()
                if data.get("metadata", {}).get("categoria", "").lower() == categoria.lower()
            }
            candidatos_en_categoria = len(skus_a_buscar)
            # Si no hay SKUs en la categoría, buscar en todos (fallback)
            # Esto se maneja en el identifier para logging, pero aquí también como seguridad
            if not skus_a_buscar:
                skus_a_buscar = self._skus

        # Cosine similarity: max contra todos los embeddings individuales
        resultados = []
        for ean, data in skus_a_buscar.items():
            # Similitud con cada embedding individual del SKU
            sims = data["embeddings"] @ query  # (N,) dot products
            sim_max = float(sims.max())
            resultados.append((ean, sim_max))

        # Ordenar de mayor a menor similitud
        resultados.sort(key=lambda x: x[1], reverse=True)

        return resultados[:top_k]

    def buscar_detallado(
        self,
        query: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Búsqueda detallada: compara contra todos los embeddings individuales,
        no solo los centroides. Más preciso pero más lento.

        Returns:
            Lista de dicts con ean, similitud_centroide, similitud_max, similitud_mean.
        """
        if not self._skus:
            return []

        query = query.flatten()
        norm = np.linalg.norm(query)
        if norm < 1e-10:
            return []
        query = query / norm

        resultados = []
        for ean, data in self._skus.items():
            # Similitud con centroide
            sim_centroide = float(np.dot(query, data["centroide"]))

            # Similitud con cada embedding individual
            sims = data["embeddings"] @ query
            sim_max = float(sims.max())
            sim_mean = float(sims.mean())

            resultados.append({
                "ean": ean,
                "similitud_centroide": sim_centroide,
                "similitud_max": sim_max,
                "similitud_mean": sim_mean,
                "descripcion": data.get("metadata", {}).get("descripcion", ""),
            })

        resultados.sort(key=lambda x: x["similitud_centroide"], reverse=True)
        return resultados[:top_k]

    def tiene_sku(self, ean: str) -> bool:
        """Verifica si un SKU ya existe en el store."""
        return ean in self._skus

    @property
    def total_skus(self) -> int:
        """Número total de SKUs en el store."""
        return len(self._skus)

    @property
    def total_embeddings(self) -> int:
        """Número total de embeddings en el store."""
        return sum(d["embeddings"].shape[0] for d in self._skus.values())

    def listar_skus(self) -> List[Dict]:
        """Lista todos los SKUs con su metadata."""
        resultado = []
        for ean, data in sorted(self._skus.items()):
            resultado.append({
                "ean": ean,
                "n_embeddings": data["embeddings"].shape[0],
                "descripcion": data.get("metadata", {}).get("descripcion", ""),
                "categoria": data.get("metadata", {}).get("categoria", ""),
            })
        return resultado

    def skus_por_categoria(self) -> Dict[str, List[str]]:
        """Agrupa EANs por categoría."""
        categorias: Dict[str, List[str]] = {}
        for ean, data in self._skus.items():
            cat = data.get("metadata", {}).get("categoria", "sin_categoria")
            categorias.setdefault(cat, []).append(ean)
        return categorias

    def resumen(self) -> str:
        """Resumen legible del estado del store."""
        lines = [f"📦 Vector Store: {self.total_skus} SKUs, {self.total_embeddings} embeddings"]

        # Agrupar por categoría
        por_cat = self.skus_por_categoria()
        if len(por_cat) > 1 or (len(por_cat) == 1 and "sin_categoria" not in por_cat):
            for cat, eans in sorted(por_cat.items()):
                lines.append(f"\n   📂 {cat} ({len(eans)} SKUs):")
                for ean in sorted(eans):
                    data = self._skus[ean]
                    desc = data.get("metadata", {}).get("descripcion", "")
                    n = data["embeddings"].shape[0]
                    lines.append(f"      {ean} ({n} imgs) — {desc}")
        else:
            for info in self.listar_skus():
                lines.append(f"   {info['ean']} ({info['n_embeddings']} imgs) — {info['descripcion']}")

        return "\n".join(lines)
