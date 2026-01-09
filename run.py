#!/usr/bin/env python3
"""
Script wrapper para ejecutar el sistema principal
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Ejecutar main
from main import main

if __name__ == "__main__":
    main()

