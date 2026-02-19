#!/usr/bin/env python3
"""
Script wrapper para ejecutar el sistema principal.
Ejecuta desde la raíz del proyecto para que los imports src.* funcionen.
"""

import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.main import main

if __name__ == "__main__":
    main()
