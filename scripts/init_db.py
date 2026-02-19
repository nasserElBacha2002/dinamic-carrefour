#!/usr/bin/env python3
"""
Inicialización de la base de datos SQL Server.

Crea la base de datos, tablas, vistas e inserta datos iniciales.
También sincroniza el catálogo desde eans.txt.

Uso:
  python scripts/init_db.py --crear
  python scripts/init_db.py --sync
  python scripts/init_db.py --crear --sync
  python scripts/init_db.py --status
  python scripts/init_db.py --test
"""

import argparse
import re
import sys
import time
from pathlib import Path

# Asegurar imports desde raíz del proyecto
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.connection import DatabaseConnection
from src.database.repository import ProductoRepository


def test_conexion() -> bool:
    """Prueba la conexión a SQL Server."""
    print("\n🔌 Probando conexión a SQL Server...")
    db = DatabaseConnection()
    print(f"   Server:   {db.server}")
    print(f"   Database: {db.database}")
    print(f"   Driver:   {db.driver}")

    if db.test_connection():
        print("   ✅ Conexión exitosa")
        return True

    print("   ❌ No se pudo conectar")
    print("\n   Verificá:")
    print("   1. SQL Server está corriendo")
    print("   2. El archivo .env tiene las credenciales correctas:")
    print("      SQL_SERVER=localhost")
    print("      SQL_DATABASE=dinamic_carrefour")
    print("      SQL_USERNAME=sa")
    print("      SQL_PASSWORD=tu_password")
    print("      SQL_DRIVER={ODBC Driver 17 for SQL Server}")
    print("   3. El driver ODBC está instalado:")
    print("      macOS:   brew install --cask microsoft-odbc-driver-17")
    print("      Ubuntu:  sudo apt install msodbcsql17")
    return False


def _split_batches(schema_sql: str) -> list[str]:
    """
    Divide un .sql estilo SQL Server por GO (línea sola).
    Tolera GO con espacios y distintos saltos.
    """
    # Normalizar saltos
    s = schema_sql.replace("\r\n", "\n").replace("\r", "\n")
    batches = re.split(r"\n\s*GO\s*(?:\n|$)", s, flags=re.IGNORECASE)
    return [b.strip() for b in batches if b and b.strip()]


def crear_base_datos() -> bool:
    """Ejecuta el script SQL de creación de la base de datos."""
    print("\n🗄️  Creando base de datos SQL Server...")

    schema_path = Path(__file__).parent.parent / "src" / "database" / "schema.sql"
    if not schema_path.exists():
        print(f"   ❌ No se encontró {schema_path}")
        return False

    schema_sql = schema_path.read_text(encoding="utf-8", errors="ignore")
    batches = _split_batches(schema_sql)

    # 1) Crear DB en master (si el schema lo incluye)
    print("   📦 Creando base de datos (si corresponde)...")
    db_master = DatabaseConnection()
    db_master.database = "master"

    try:
        with db_master as conn:
            for batch in batches:
                if "CREATE DATABASE" in batch.upper():
                    try:
                        conn.execute(batch)
                        conn.commit()
                        print("      ✅ Batch CREATE DATABASE ejecutado")
                    except Exception as e:
                        msg = str(e).lower()
                        if "already exists" in msg or "ya existe" in msg:
                            print("      ℹ️  Base de datos ya existe")
                        else:
                            raise
                    break

        # 2) Ejecutar el resto sobre dinamic_carrefour
        print("   📋 Creando tablas, índices, vistas y datos iniciales...")
        time.sleep(1)

        db = DatabaseConnection()  # por .env apunta a dinamic_carrefour

        with db as conn:
            conn.execute("SELECT DB_NAME()")
            current_db = conn.fetchval()
            if current_db != "dinamic_carrefour":
                print(f"      ⚠️  Estamos en DB: {current_db}, cambiando a dinamic_carrefour...")
                conn.execute("USE dinamic_carrefour")
                conn.commit()

            for i, batch in enumerate(batches, 1):
                up = batch.strip().upper()

                # Saltar CREATE DATABASE y USE (ya manejado arriba)
                if "CREATE DATABASE" in up or up.startswith("USE "):
                    continue

                # Saltar lotes que son solo comentarios
                if re.fullmatch(r"(--.*\n?)+", batch.strip(), flags=re.MULTILINE):
                    continue

                try:
                    conn.execute(batch)
                    conn.commit()
                except Exception as e:
                    msg = str(e).lower()

                    # Ignorar errores típicos de idempotencia
                    ign = [
                        "already exists",
                        "ya existe",
                        "there is already an object",
                        "cannot create",
                        "is already an object",
                        "duplicate",
                        "violat",  # violate unique (si re-ejecutás inserts/merge)
                    ]
                    if any(x in msg for x in ign):
                        continue

                    # PRINT puede fallar dependiendo del driver; no lo tomamos como fatal
                    if "print" in up:
                        continue

                    print(f"      ❌ Batch {i} ERROR: {e}")
                    print(f"         Preview: {batch[:140].replace(chr(10), ' ')}...")
                    raise

        # 3) Verificar tablas
        print("   🔍 Verificando tablas creadas...")
        with db as conn:
            conn.execute("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """)
            tablas = [row[0] for row in conn.fetchall()]
            if tablas:
                print(f"      ✅ Tablas encontradas: {', '.join(tablas)}")
            else:
                print("      ⚠️  No se encontraron tablas (revisar schema.sql)")

        print("   ✅ Base de datos creada correctamente")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def sincronizar_catalogo(eans_file: str = "eans.txt") -> bool:
    """
    Sincroniza el catálogo desde eans.txt a la base de datos.

    Formato esperado (TSV):
      EAN \t DESCRIPCION \t PACKAGING_TYPE_ID [\t BRAND] [\t SIZE] [\t VARIANT]
    """
    print(f"\n🔄 Sincronizando catálogo desde {eans_file}...")

    repo = ProductoRepository()
    resultado = repo.sincronizar_desde_eans_txt(eans_file)

    if "error" in resultado:
        print(f"   ❌ {resultado['error']}")
        return False

    print("\n   📊 Resultado:")
    print(f"      Agregados:    {resultado.get('agregados', 0)}")
    print(f"      Actualizados: {resultado.get('actualizados', 0)}")
    print(f"      Errores:      {resultado.get('errores', 0)}")
    return True


def mostrar_status() -> None:
    """Muestra el estado actual del catálogo en la DB."""
    print("\n📦 ESTADO DEL CATÁLOGO (SQL Server)")
    print("=" * 60)

    repo = ProductoRepository()

    try:
        resumen = repo.resumen_catalogo()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   ¿Ejecutaste init_db.py --crear primero?")
        return

    print(f"   Total productos: {resumen.get('total_productos', 0)}")
    print(f"   Con embeddings:  {resumen.get('total_con_embeddings', 0)}")
    print()

    for cat in resumen.get("categorias", []):
        total_prod = cat.get("total_productos", 0)
        icon = "📂" if total_prod > 0 else "📁"

        # Vista nueva: packaging_type / nombre
        cat_id = cat.get("packaging_type") or cat.get("categoria") or ""
        print(f"   {icon} {cat.get('nombre', '')} ({cat_id})")
        print(f"      Productos: {total_prod}")
        print(f"      Imágenes:  {cat.get('total_imagenes', 0) or 0}")
        print(f"      Con embeddings: {cat.get('con_embeddings', 0)}")
        if (cat.get("sin_embeddings") or 0) > 0:
            print(f"      ⚠️  Sin embeddings: {cat['sin_embeddings']}")

    # Productos
    productos = repo.listar_productos()
    if productos:
        print("\n   📋 Productos:")
        pack_actual = ""
        for p in productos:
            pack_id = p.get("packaging_type_id") or p.get("categoria_id") or ""
            if pack_id != pack_actual:
                pack_actual = pack_id
                print(f"\n   [{pack_actual}]")

            emb_icon = "✅" if p.get("embeddings_calculados") else "❌"

            # Mostrar metadatos si están
            meta = []
            if p.get("brand"):
                meta.append(p["brand"])
            if p.get("variant"):
                meta.append(p["variant"])
            if p.get("size"):
                meta.append(p["size"])
            meta_txt = f" — {' / '.join(meta)}" if meta else ""

            print(f"      {emb_icon} {p.get('ean')} — {p.get('descripcion')} ({p.get('n_imagenes', 0)} imgs){meta_txt}")

    # Últimas ejecuciones
    ejecuciones = repo.listar_ejecuciones(limite=5)
    if ejecuciones:
        print("\n   📹 Últimas ejecuciones:")
        for ej in ejecuciones:
            print(
                f"      {ej.get('fecha')} — {ej.get('video')} — "
                f"{ej.get('detecciones', 0)} detecciones, {ej.get('skus', 0)} SKUs"
            )

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inicialización de base de datos SQL Server")
    parser.add_argument("--crear", action="store_true", help="Crear base de datos y tablas")
    parser.add_argument("--sync", action="store_true", help="Sincronizar catálogo desde eans.txt")
    parser.add_argument("--status", action="store_true", help="Mostrar estado del catálogo")
    parser.add_argument("--test", action="store_true", help="Probar conexión a SQL Server")
    parser.add_argument("--eans-file", default="eans.txt", help="Ruta a eans.txt (default: eans.txt)")

    args = parser.parse_args()

    if not any([args.crear, args.sync, args.status, args.test]):
        parser.print_help()
        return

    if args.test:
        test_conexion()
        return

    if args.crear:
        ok = test_conexion()
        if not ok:
            return
        ok = crear_base_datos()
        if not ok:
            return

    if args.sync:
        ok = sincronizar_catalogo(args.eans_file)
        if not ok:
            return

    if args.status:
        mostrar_status()


if __name__ == "__main__":
    main()
