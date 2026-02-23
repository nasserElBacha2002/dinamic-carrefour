#!/usr/bin/env python3
"""
Script para iniciar la UI web del sistema.

Ejecuta uvicorn con la configuración correcta para importar el módulo src.ui.app.
"""

import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.ui.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root)],
    )
