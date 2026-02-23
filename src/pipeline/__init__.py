"""
Pipeline module — Procesamiento de video para detección e identificación de SKUs.

Módulos:
- engine: Motor principal del pipeline (orquestador de alto nivel)
- video_reader: Lectura eficiente de frames
- crop_processor: Procesamiento y heurísticas de crops
- detection_processor: Procesamiento de detecciones individuales
- report_generator: Generación de reportes y anotaciones
- learning_integration: Integración con el Learning Manager
"""

from .engine import PipelineEngine
from .video_reader import VideoFrameReader
from .crop_processor import CropProcessor
from .detection_processor import DetectionProcessor
from .report_generator import ReportGenerator
from .learning_integration import LearningIntegration

__all__ = [
    "PipelineEngine",
    "VideoFrameReader",
    "CropProcessor",
    "DetectionProcessor",
    "ReportGenerator",
    "LearningIntegration",
]
