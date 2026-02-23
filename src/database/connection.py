#!/usr/bin/env python3
"""
Conexión a SQL Server — Gestión centralizada de la conexión a la base de datos.

Usa pyodbc para conectarse a SQL Server. La configuración se lee desde
variables de entorno o un archivo .env.

Ejemplo de .env:
    SQL_SERVER=localhost
    SQL_DATABASE=dinamic_carrefour
    SQL_USERNAME=sa
    SQL_PASSWORD=tu_password
    SQL_DRIVER={ODBC Driver 17 for SQL Server}
"""

import os
from typing import Optional
from pathlib import Path

try:
    import pyodbc
except ImportError:
    pyodbc = None


def _cargar_env(env_path: str = ".env") -> None:
    """Carga variables de entorno desde un archivo .env (sin dependencia de dotenv)."""
    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class DatabaseConnection:
    """
    Gestiona la conexión a SQL Server.

    Soporta context manager (with) para manejo automático de conexiones.

    Uso:
        with DatabaseConnection() as db:
            db.execute("SELECT * FROM productos")
            rows = db.fetchall()

    O sin context manager:
        db = DatabaseConnection()
        db.connect()
        # ... operaciones ...
        db.close()
    """

    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver: Optional[str] = None,
        env_path: str = ".env"
    ):
        """
        Args:
            server: Dirección del servidor SQL Server.
            database: Nombre de la base de datos.
            username: Usuario.
            password: Contraseña.
            driver: Driver ODBC.
            env_path: Ruta al archivo .env.

        Si no se pasan parámetros, se leen desde variables de entorno:
            SQL_SERVER, SQL_DATABASE, SQL_USERNAME, SQL_PASSWORD, SQL_DRIVER
        """
        if pyodbc is None:
            raise ImportError(
                "pyodbc no está instalado. Ejecutá: pip install pyodbc\n"
                "También necesitás el ODBC Driver para SQL Server:\n"
                "  macOS:   brew install microsoft/mssql-release/msodbcsql17\n"
                "  Ubuntu:  sudo apt install msodbcsql17\n"
                "  Windows: Viene incluido con SQL Server."
            )

        # Cargar .env
        _cargar_env(env_path)

        self.server = server or os.environ.get("SQL_SERVER", "localhost")
        self.database = database or os.environ.get("SQL_DATABASE", "dinamic_carrefour")
        self.username = username or os.environ.get("SQL_USERNAME", "sa")
        self.password = password or os.environ.get("SQL_PASSWORD", "")
        self.driver = driver or os.environ.get(
            "SQL_DRIVER", "{ODBC Driver 17 for SQL Server}"
        )

        self._connection = None
        self._cursor = None

    @property
    def connection_string(self) -> str:
        """Genera el connection string para pyodbc."""
        return (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate=yes;"
        )

    def connect(self) -> "DatabaseConnection":
        """Establece conexión con SQL Server."""
        try:
            self._connection = pyodbc.connect(self.connection_string)
            self._cursor = self._connection.cursor()
            return self
        except pyodbc.Error as e:
            error_msg = str(e)
            print(f"❌ Error conectando a SQL Server: {error_msg}")
            print(f"   Server:   {self.server}")
            print(f"   Database: {self.database}")
            print(f"   Driver:   {self.driver}")
            print(f"\n   Verificá:")
            print(f"   1. SQL Server está corriendo")
            print(f"   2. Las credenciales en .env son correctas")
            print(f"   3. El driver ODBC está instalado")
            raise

    def close(self) -> None:
        """Cierra la conexión."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "DatabaseConnection":
        """Context manager: abre conexión."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager: cierra conexión."""
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()

    def execute(self, query: str, params: tuple = ()) -> "DatabaseConnection":
        """
        Ejecuta una consulta SQL.

        Args:
            query: Consulta SQL con placeholders (?).
            params: Parámetros para la consulta.
        """
        if not self._cursor:
            raise RuntimeError("No hay conexión activa. Usá connect() primero.")
        self._cursor.execute(query, params)
        return self

    def executemany(self, query: str, params_list: list) -> "DatabaseConnection":
        """Ejecuta una consulta para múltiples sets de parámetros."""
        if not self._cursor:
            raise RuntimeError("No hay conexión activa.")
        self._cursor.executemany(query, params_list)
        return self

    def fetchone(self):
        """Devuelve una fila."""
        if not self._cursor:
            return None
        return self._cursor.fetchone()

    def fetchall(self) -> list:
        """Devuelve todas las filas."""
        if not self._cursor:
            return []
        return self._cursor.fetchall()

    def fetchval(self):
        """Devuelve el primer valor de la primera fila."""
        row = self.fetchone()
        return row[0] if row else None

    def commit(self) -> None:
        """Confirma la transacción."""
        if self._connection:
            self._connection.commit()

    def rollback(self) -> None:
        """Revierte la transacción."""
        if self._connection:
            self._connection.rollback()

    @property
    def connected(self) -> bool:
        """Verifica si hay conexión activa."""
        return self._connection is not None

    def test_connection(self) -> bool:
        """
        Prueba la conexión a SQL Server.

        Returns:
            True si la conexión es exitosa.
        """
        try:
            self.connect()
            self.execute("SELECT 1")
            result = self.fetchval()
            self.close()
            return result == 1
        except Exception:
            return False
