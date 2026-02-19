"""
Módulo de aprendizaje continuo — Dataset evolutivo por ejecución.

Cada ejecución genera un dataset estructurado que captura:
- Casos dudosos (UNKNOWN, AMBIGUOUS)
- Metadata completa de cada decisión
- Información para revisión humana rápida
"""

from .manager import LearningManager

__all__ = ["LearningManager"]
