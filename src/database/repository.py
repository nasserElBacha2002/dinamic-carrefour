#!/usr/bin/env python3
"""
Repositorio de datos — Operaciones CRUD sobre SQL Server.

Centraliza todas las operaciones de lectura/escritura de la base de datos.
El resto del sistema usa este módulo en vez de hacer queries directos.

Tablas (modelo nuevo):
  - packaging_types: Tipos de packaging (botella, lata, bolsa, etc.)
  - productos: Catálogo de SKUs con FK a packaging_types + metadata (brand/size/variant)
  - ejecuciones: Historial de procesamiento de videos
  - detecciones: Resultados por SKU de cada ejecución (incluye tracking_id)
  - review_queue: Cola de revisión para casos ambiguos / desconocidos
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from src.database.connection import DatabaseConnection


class ProductoRepository:
    """
    Repositorio para gestionar productos y packaging types en SQL Server.
    """

    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        env_path: str = ".env"
    ):
        self._conn_params = {
            "server": server,
            "database": database,
            "username": username,
            "password": password,
            "env_path": env_path,
        }

    def _get_conn(self) -> DatabaseConnection:
        return DatabaseConnection(**self._conn_params)

    # ──────────────────────────────────────────────────────────────
    # Helpers: evitar int(None) / float(None) y defaults consistentes
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_int(x, default: int = 0) -> int:
        try:
            return default if x is None else int(x)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(x, default: float = 0.0) -> float:
        try:
            return default if x is None else float(x)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_bool(x, default: bool = False) -> bool:
        if x is None:
            return default
        return bool(x)

    # ═══════════════════════════════════════════════════════════════
    #  PACKAGING TYPES (antes: categorias)
    # ═══════════════════════════════════════════════════════════════

    def listar_packaging_types(self) -> List[Dict]:
        """Lista todos los packaging types activos."""
        with self._get_conn() as db:
            db.execute("""
                SELECT id, nombre, descripcion, prompts_clip
                FROM packaging_types
                WHERE activo = 1
                ORDER BY nombre
            """)
            rows = db.fetchall()

        return [
            {
                "id": row[0],
                "nombre": row[1],
                "descripcion": row[2],
                "prompts_clip": json.loads(row[3]) if row[3] else [],
            }
            for row in rows
        ]

    def obtener_packaging_type(self, packaging_type_id: str) -> Optional[Dict]:
        """Obtiene un packaging type por ID."""
        with self._get_conn() as db:
            db.execute(
                """
                SELECT id, nombre, descripcion, prompts_clip
                FROM packaging_types
                WHERE id = ?
                """,
                (packaging_type_id,)
            )
            row = db.fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "nombre": row[1],
            "descripcion": row[2],
            "prompts_clip": json.loads(row[3]) if row[3] else [],
        }

    def agregar_packaging_type(
        self,
        id: str,
        nombre: str,
        descripcion: str = "",
        prompts_clip: Optional[List[str]] = None
    ) -> bool:
        """Agrega un packaging type."""
        prompts_json = json.dumps(prompts_clip or [], ensure_ascii=False)

        with self._get_conn() as db:
            db.execute("SELECT 1 FROM packaging_types WHERE id = ?", (id,))
            if db.fetchone():
                return False

            db.execute(
                """
                INSERT INTO packaging_types (id, nombre, descripcion, prompts_clip)
                VALUES (?, ?, ?, ?)
                """,
                (id, nombre, descripcion, prompts_json)
            )

        return True

    # Backward compatibility (si tu pipeline todavía llama "categoria")
    def listar_categorias(self) -> List[Dict]:
        return self.listar_packaging_types()

    def obtener_categoria(self, categoria_id: str) -> Optional[Dict]:
        return self.obtener_packaging_type(categoria_id)

    def agregar_categoria(
        self,
        id: str,
        nombre: str,
        descripcion: str = "",
        prompts_clip: Optional[List[str]] = None
    ) -> bool:
        return self.agregar_packaging_type(id, nombre, descripcion, prompts_clip)

    # ═══════════════════════════════════════════════════════════════
    #  PRODUCTOS
    # ═══════════════════════════════════════════════════════════════

    def listar_productos(self, solo_activos: bool = True) -> List[Dict]:
        """Lista todos los productos."""
        filtro = "WHERE p.activo = 1" if solo_activos else ""

        with self._get_conn() as db:
            db.execute(f"""
                SELECT
                    p.ean,
                    p.descripcion,
                    p.packaging_type_id,
                    pt.nombre AS packaging_type_nombre,
                    p.brand,
                    p.size,
                    p.variant,
                    p.n_imagenes,
                    p.embeddings_calculados,
                    p.embeddings_path,
                    p.fecha_alta,
                    p.fecha_embeddings
                FROM productos p
                JOIN packaging_types pt ON p.packaging_type_id = pt.id
                {filtro}
                ORDER BY p.packaging_type_id, p.descripcion
            """)
            rows = db.fetchall()

        return [
            {
                "ean": row[0],
                "descripcion": row[1],
                "packaging_type_id": row[2],
                "packaging_type_nombre": row[3],
                "brand": row[4],
                "size": row[5],
                "variant": row[6],
                "n_imagenes": self._safe_int(row[7], 0),
                "embeddings_calculados": self._safe_bool(row[8], False),
                "embeddings_path": row[9],
                "fecha_alta": row[10],
                "fecha_embeddings": row[11],
            }
            for row in rows
        ]

    def productos_por_packaging(self, packaging_type_id: str) -> List[Dict]:
        """Lista productos de un packaging type específico."""
        with self._get_conn() as db:
            db.execute("""
                SELECT
                    ean, descripcion, brand, size, variant,
                    n_imagenes, embeddings_calculados, embeddings_path,
                    fecha_alta, fecha_embeddings
                FROM productos
                WHERE packaging_type_id = ? AND activo = 1
                ORDER BY descripcion
            """, (packaging_type_id,))
            rows = db.fetchall()

        return [
            {
                "ean": row[0],
                "descripcion": row[1],
                "brand": row[2],
                "size": row[3],
                "variant": row[4],
                "n_imagenes": self._safe_int(row[5], 0),
                "embeddings_calculados": self._safe_bool(row[6], False),
                "embeddings_path": row[7],
                "fecha_alta": row[8],
                "fecha_embeddings": row[9],
            }
            for row in rows
        ]

    # Backward compatibility
    def productos_por_categoria(self, categoria_id: str) -> List[Dict]:
        return self.productos_por_packaging(categoria_id)

    def obtener_producto(self, ean: str) -> Optional[Dict]:
        """Obtiene un producto por EAN."""
        with self._get_conn() as db:
            db.execute("""
                SELECT
                    p.ean,
                    p.descripcion,
                    p.packaging_type_id,
                    pt.nombre,
                    p.brand,
                    p.size,
                    p.variant,
                    p.n_imagenes,
                    p.embeddings_calculados,
                    p.embeddings_path,
                    p.fecha_alta,
                    p.fecha_embeddings
                FROM productos p
                JOIN packaging_types pt ON p.packaging_type_id = pt.id
                WHERE p.ean = ?
            """, (ean,))
            row = db.fetchone()

        if not row:
            return None

        return {
            "ean": row[0],
            "descripcion": row[1],
            "packaging_type_id": row[2],
            "packaging_type_nombre": row[3],
            "brand": row[4],
            "size": row[5],
            "variant": row[6],
            "n_imagenes": self._safe_int(row[7], 0),
            "embeddings_calculados": self._safe_bool(row[8], False),
            "embeddings_path": row[9],
            "fecha_alta": row[10],
            "fecha_embeddings": row[11],
        }

    def agregar_producto(
        self,
        ean: str,
        descripcion: str,
        packaging_type_id: str,
        brand: Optional[str] = None,
        size: Optional[str] = None,
        variant: Optional[str] = None,
        n_imagenes: int = 0,
        embeddings_path: Optional[str] = None
    ) -> bool:
        """Agrega un producto al catálogo."""
        with self._get_conn() as db:
            db.execute("SELECT 1 FROM productos WHERE ean = ?", (ean,))
            if db.fetchone():
                return False

            db.execute(
                """
                INSERT INTO productos
                    (ean, descripcion, packaging_type_id, brand, size, variant, n_imagenes, embeddings_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ean,
                    descripcion,
                    packaging_type_id,
                    brand,
                    size,
                    variant,
                    self._safe_int(n_imagenes, 0),
                    embeddings_path,
                )
            )

        return True

    # Backward compatibility
    def agregar_producto_legacy(
        self,
        ean: str,
        descripcion: str,
        categoria_id: str,
        n_imagenes: int = 0,
        embeddings_path: Optional[str] = None
    ) -> bool:
        return self.agregar_producto(
            ean=ean,
            descripcion=descripcion,
            packaging_type_id=categoria_id,
            n_imagenes=n_imagenes,
            embeddings_path=embeddings_path
        )

    def actualizar_producto(self, ean: str, **campos) -> bool:
        """Actualiza campos de un producto."""
        campos_permitidos = {
            "descripcion",
            "packaging_type_id",
            "brand",
            "size",
            "variant",
            "n_imagenes",
            "embeddings_calculados",
            "embeddings_path",
            "fecha_embeddings",
            "activo",
        }

        # aceptar categoria_id como alias
        if "categoria_id" in campos and "packaging_type_id" not in campos:
            campos["packaging_type_id"] = campos.pop("categoria_id")

        updates = {k: v for k, v in campos.items() if k in campos_permitidos}
        if not updates:
            return False

        # saneo básico
        if "n_imagenes" in updates:
            updates["n_imagenes"] = self._safe_int(updates["n_imagenes"], 0)
        if "embeddings_calculados" in updates and updates["embeddings_calculados"] is not None:
            updates["embeddings_calculados"] = bool(updates["embeddings_calculados"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [ean]

        with self._get_conn() as db:
            db.execute(
                f"UPDATE productos SET {set_clause} WHERE ean = ?",
                tuple(values)
            )

        return True

    def marcar_embeddings_calculados(self, ean: str, embeddings_path: str, n_imagenes: int) -> None:
        """Marca un producto como con embeddings calculados."""
        self.actualizar_producto(
            ean,
            embeddings_calculados=True,
            embeddings_path=embeddings_path,
            n_imagenes=self._safe_int(n_imagenes, 0),
            fecha_embeddings=datetime.now(),
        )

    def eliminar_producto(self, ean: str, soft: bool = True) -> bool:
        """Elimina un producto (soft delete recomendado)."""
        with self._get_conn() as db:
            if soft:
                db.execute("UPDATE productos SET activo = 0 WHERE ean = ?", (ean,))
            else:
                db.execute("DELETE FROM detecciones WHERE ean = ?", (ean,))
                db.execute("DELETE FROM productos WHERE ean = ?", (ean,))

        return True

    def contar_productos(self, packaging_type_id: Optional[str] = None) -> int:
        """Cuenta productos activos, opcionalmente filtrado por packaging."""
        with self._get_conn() as db:
            if packaging_type_id:
                db.execute(
                    "SELECT COUNT(*) FROM productos WHERE activo = 1 AND packaging_type_id = ?",
                    (packaging_type_id,)
                )
            else:
                db.execute("SELECT COUNT(*) FROM productos WHERE activo = 1")
            return self._safe_int(db.fetchval(), 0)

    # ═══════════════════════════════════════════════════════════════
    #  EJECUCIONES
    # ═══════════════════════════════════════════════════════════════

    def registrar_ejecucion(
        self,
        video_path: str,
        frames_procesados: int = 0,
        frames_con_producto: int = 0,
        total_detecciones: int = 0,
        skus_identificados: int = 0,
        duracion_segundos: float = 0.0,
        parametros: Optional[Dict] = None,
        output_dir: Optional[str] = None,
        csv_path: Optional[str] = None
    ) -> int:
        """Registra una ejecución del pipeline (usa OUTPUT INSERTED.id)."""
        video_nombre = Path(video_path).name
        params_json = json.dumps(parametros or {}, ensure_ascii=False)

        frames_procesados = self._safe_int(frames_procesados, 0)
        frames_con_producto = self._safe_int(frames_con_producto, 0)
        total_detecciones = self._safe_int(total_detecciones, 0)
        skus_identificados = self._safe_int(skus_identificados, 0)
        duracion_segundos = self._safe_float(duracion_segundos, 0.0)

        with self._get_conn() as db:
            db.execute(
                """
                INSERT INTO ejecuciones
                    (video_path, video_nombre, frames_procesados, frames_con_producto,
                     total_detecciones, skus_identificados, duracion_segundos,
                     parametros, output_dir, csv_path)
                OUTPUT INSERTED.id
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_path,
                    video_nombre,
                    frames_procesados,
                    frames_con_producto,
                    total_detecciones,
                    skus_identificados,
                    duracion_segundos,
                    params_json,
                    output_dir,
                    csv_path,
                )
            )
            new_id = db.fetchval()

        if new_id is None:
            raise RuntimeError("No se pudo obtener el id de la ejecución (INSERT devolvió None).")

        return int(new_id)

    def registrar_deteccion(
        self,
        ejecucion_id: int,
        ean: str,
        cantidad_dedup: int,
        cantidad_raw: int = 0,
        confianza_promedio: float = 0.0,
        tracking_id: Optional[int] = None
    ) -> None:
        """Registra una detección de SKU en una ejecución (incluye tracking_id)."""
        ejecucion_id = self._safe_int(ejecucion_id, 0)
        cantidad_dedup = self._safe_int(cantidad_dedup, 0)
        cantidad_raw = self._safe_int(cantidad_raw, 0)
        confianza_promedio = self._safe_float(confianza_promedio, 0.0)
        tracking_id = None if tracking_id is None else self._safe_int(tracking_id, 0)

        with self._get_conn() as db:
            db.execute(
                """
                INSERT INTO detecciones
                    (ejecucion_id, ean, cantidad_dedup, cantidad_raw, confianza_promedio, tracking_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ejecucion_id, ean, cantidad_dedup, cantidad_raw, confianza_promedio, tracking_id)
            )

    def registrar_resultado_completo(self, video_path: str, resultado: Dict) -> int:
        """
        Registra un resultado completo del pipeline (ejecución + detecciones).
        """
        frames_total = self._safe_int(resultado.get("frames_total"), 0)
        frames_con_producto = self._safe_int(resultado.get("frames_con_producto"), 0)
        total_detecciones = self._safe_int(resultado.get("total_detecciones"), 0)
        duracion_segundos = self._safe_float(resultado.get("duracion_segundos"), 0.0)

        conteo_sku = resultado.get("conteo_sku") or {}
        conteo_raw = resultado.get("conteo_raw") or {}

        skus_identificados = len([k for k in conteo_sku.keys() if k != "UNKNOWN"])

        ejecucion_id = self.registrar_ejecucion(
            video_path=video_path,
            frames_procesados=frames_total,
            frames_con_producto=frames_con_producto,
            total_detecciones=total_detecciones,
            skus_identificados=skus_identificados,
            duracion_segundos=duracion_segundos,
            parametros=resultado.get("parametros") or None,
            output_dir=resultado.get("output_dir") or None,
            csv_path=resultado.get("csv_path") or None,
        )

        for ean, cantidad in conteo_sku.items():
            if ean == "UNKNOWN":
                continue
            self.registrar_deteccion(
                ejecucion_id=ejecucion_id,
                ean=ean,
                cantidad_dedup=self._safe_int(cantidad, 0),
                cantidad_raw=self._safe_int(conteo_raw.get(ean), 0),
                tracking_id=None,
            )

        return ejecucion_id

    def listar_ejecuciones(self, limite: int = 20) -> List[Dict]:
        """Lista las últimas ejecuciones."""
        with self._get_conn() as db:
            db.execute(f"""
                SELECT TOP {limite}
                    id, video_nombre, fecha, frames_procesados,
                    total_detecciones, skus_identificados, duracion_segundos
                FROM ejecuciones
                ORDER BY fecha DESC
            """)
            rows = db.fetchall()

        return [
            {
                "id": row[0],
                "video": row[1],
                "fecha": row[2],
                "frames": self._safe_int(row[3], 0),
                "detecciones": self._safe_int(row[4], 0),
                "skus": self._safe_int(row[5], 0),
                "duracion": self._safe_float(row[6], 0.0),
            }
            for row in rows
        ]

    def detalle_ejecucion(self, ejecucion_id: int) -> Optional[Dict]:
        """Obtiene detalle completo de una ejecución con sus detecciones."""
        with self._get_conn() as db:
            db.execute("SELECT * FROM ejecuciones WHERE id = ?", (ejecucion_id,))
            ej_row = db.fetchone()
            if not ej_row:
                return None

            db.execute("""
                SELECT
                    d.ean,
                    p.descripcion,
                    p.packaging_type_id,
                    d.cantidad_dedup,
                    d.cantidad_raw,
                    d.confianza_promedio,
                    d.tracking_id
                FROM detecciones d
                JOIN productos p ON d.ean = p.ean
                WHERE d.ejecucion_id = ?
                ORDER BY d.cantidad_dedup DESC
            """, (ejecucion_id,))
            det_rows = db.fetchall()

        return {
            "id": ej_row[0],
            "video_path": ej_row[1],
            "video_nombre": ej_row[2],
            "fecha": ej_row[3],
            "frames_procesados": self._safe_int(ej_row[4], 0),
            "frames_con_producto": self._safe_int(ej_row[5], 0),
            "total_detecciones": self._safe_int(ej_row[6], 0),
            "skus_identificados": self._safe_int(ej_row[7], 0),
            "duracion_segundos": self._safe_float(ej_row[8], 0.0),
            "detecciones": [
                {
                    "ean": row[0],
                    "descripcion": row[1],
                    "packaging_type": row[2],
                    "cantidad_dedup": self._safe_int(row[3], 0),
                    "cantidad_raw": self._safe_int(row[4], 0),
                    "confianza": self._safe_float(row[5], 0.0),
                    "tracking_id": row[6],
                }
                for row in det_rows
            ],
        }

    # ═══════════════════════════════════════════════════════════════
    #  REVIEW QUEUE
    # ═══════════════════════════════════════════════════════════════

    def agregar_a_review_queue(
        self,
        ejecucion_id: int,
        motivo: str,
        crop_path: Optional[str] = None,
        ean_predicho: Optional[str] = None,
        similitud: Optional[float] = None
    ) -> None:
        """Inserta un item en la cola de revisión."""
        ejecucion_id = self._safe_int(ejecucion_id, 0)
        similitud = None if similitud is None else self._safe_float(similitud, 0.0)

        with self._get_conn() as db:
            db.execute(
                """
                INSERT INTO review_queue (ejecucion_id, ean_predicho, similitud, crop_path, motivo)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ejecucion_id, ean_predicho, similitud, crop_path, motivo)
            )

    def listar_review_queue(self, limite: int = 50, solo_pendientes: bool = True) -> List[Dict]:
        """Lista items de review."""
        filtro = "WHERE revisado = 0" if solo_pendientes else ""
        with self._get_conn() as db:
            db.execute(f"""
                SELECT TOP {limite}
                    id, ejecucion_id, ean_predicho, similitud, crop_path, motivo, revisado, fecha_creacion
                FROM review_queue
                {filtro}
                ORDER BY fecha_creacion DESC
            """)
            rows = db.fetchall()

        return [
            {
                "id": row[0],
                "ejecucion_id": row[1],
                "ean_predicho": row[2],
                "similitud": row[3],
                "crop_path": row[4],
                "motivo": row[5],
                "revisado": self._safe_bool(row[6], False),
                "fecha_creacion": row[7],
            }
            for row in rows
        ]

    def marcar_review_como_revisado(self, review_id: int, revisado: bool = True) -> None:
        """Marca un item de review como revisado."""
        review_id = self._safe_int(review_id, 0)
        with self._get_conn() as db:
            db.execute(
                "UPDATE review_queue SET revisado = ? WHERE id = ?",
                (1 if revisado else 0, review_id)
            )

    # ═══════════════════════════════════════════════════════════════
    #  CONSULTAS ÚTILES
    # ═══════════════════════════════════════════════════════════════

    def resumen_catalogo(self) -> Dict:
        """Resumen del estado del catálogo (vista nueva)."""
        with self._get_conn() as db:
            db.execute("""
                SELECT packaging_type, packaging_type_nombre,
                       total_productos, total_imagenes, con_embeddings, sin_embeddings
                FROM v_catalogo_resumen
                ORDER BY packaging_type
            """)
            rows = db.fetchall()

        categorias = []
        total_productos = 0
        total_con_embeddings = 0

        for row in rows:
            cat = {
                "packaging_type": row[0],
                "nombre": row[1],
                "total_productos": self._safe_int(row[2], 0),
                "total_imagenes": self._safe_int(row[3], 0),
                "con_embeddings": self._safe_int(row[4], 0),
                "sin_embeddings": self._safe_int(row[5], 0),
            }
            categorias.append(cat)
            total_productos += cat["total_productos"]
            total_con_embeddings += cat["con_embeddings"]

        return {
            "total_productos": total_productos,
            "total_con_embeddings": total_con_embeddings,
            "categorias": categorias,
        }

    def productos_sin_embeddings(self) -> List[Dict]:
        """Lista productos que aún no tienen embeddings calculados."""
        with self._get_conn() as db:
            db.execute("""
                SELECT ean, descripcion, packaging_type_id, n_imagenes
                FROM productos
                WHERE activo = 1 AND embeddings_calculados = 0
                ORDER BY packaging_type_id, descripcion
            """)
            rows = db.fetchall()

        return [
            {
                "ean": row[0],
                "descripcion": row[1],
                "packaging_type_id": row[2],
                "n_imagenes": self._safe_int(row[3], 0),
            }
            for row in rows
        ]

    def estadisticas_detecciones(self, ean: str) -> Dict:
        """Estadísticas históricas de detección de un SKU."""
        with self._get_conn() as db:
            db.execute("""
                SELECT
                    COUNT(*)                         AS veces_detectado,
                    AVG(d.cantidad_dedup)            AS promedio_unidades,
                    MAX(d.cantidad_dedup)            AS max_unidades,
                    MIN(d.cantidad_dedup)            AS min_unidades,
                    AVG(d.confianza_promedio)        AS confianza_media
                FROM detecciones d
                WHERE d.ean = ?
            """, (ean,))
            row = db.fetchone()

        if not row or self._safe_int(row[0], 0) == 0:
            return {"ean": ean, "veces_detectado": 0}

        return {
            "ean": ean,
            "veces_detectado": self._safe_int(row[0], 0),
            "promedio_unidades": round(self._safe_float(row[1], 0.0), 1),
            "max_unidades": self._safe_int(row[2], 0),
            "min_unidades": self._safe_int(row[3], 0),
            "confianza_media": round(self._safe_float(row[4], 0.0), 4),
        }

    def sincronizar_desde_eans_txt(self, eans_file: str = "eans.txt") -> Dict:
        """
        Sincroniza la base de datos con el archivo eans.txt.

        Formato esperado (TSV):
          EAN \t DESCRIPCION \t PACKAGING_TYPE_ID [\t BRAND] [\t SIZE] [\t VARIANT]
        """
        path = Path(eans_file)
        if not path.exists():
            return {"error": f"No se encontró {eans_file}"}

        agregados = 0
        actualizados = 0
        errores = 0

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                print(f"  ⚠️ Línea inválida (se requieren 3 columnas): {line[:80]}")
                errores += 1
                continue

            ean = parts[0].strip()
            descripcion = parts[1].strip()
            packaging_type_id = parts[2].strip()

            brand = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else None
            size = parts[4].strip() if len(parts) >= 5 and parts[4].strip() else None
            variant = parts[5].strip() if len(parts) >= 6 and parts[5].strip() else None

            pt = self.obtener_packaging_type(packaging_type_id)
            if not pt:
                print(f"  ⚠️ Packaging '{packaging_type_id}' no existe para EAN {ean}")
                errores += 1
                continue

            existente = self.obtener_producto(ean)
            if not existente:
                self.agregar_producto(
                    ean=ean,
                    descripcion=descripcion,
                    packaging_type_id=packaging_type_id,
                    brand=brand,
                    size=size,
                    variant=variant,
                )
                agregados += 1
                print(f"  ✅ Agregado: {ean} [{packaging_type_id}] — {descripcion}")
            else:
                cambios = {}
                if existente["descripcion"] != descripcion:
                    cambios["descripcion"] = descripcion
                if existente["packaging_type_id"] != packaging_type_id:
                    cambios["packaging_type_id"] = packaging_type_id

                if (existente.get("brand") or None) != brand:
                    cambios["brand"] = brand
                if (existente.get("size") or None) != size:
                    cambios["size"] = size
                if (existente.get("variant") or None) != variant:
                    cambios["variant"] = variant

                if cambios:
                    self.actualizar_producto(ean, **cambios)
                    actualizados += 1
                    print(f"  🔄 Actualizado: {ean} — {cambios}")

        return {
            "agregados": agregados,
            "actualizados": actualizados,
            "errores": errores,
        }
