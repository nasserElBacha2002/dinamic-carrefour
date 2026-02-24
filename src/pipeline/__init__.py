"""
Pipeline module — Main entry point for pipeline components.

This module provides backward-compatible imports for the main components.
For better organization, use direct imports from submodules:
- src.pipeline.core (engine, video_reader)
- src.pipeline.processing (detection, crops, decisions)
- src.pipeline.tracking (track runtime, setup, export)
- src.pipeline.output (reports, results, learning)
"""

# Backward compatibility: export main components
from .core import PipelineEngine, VideoFrameReader
from .processing import (
    DetectionProcessor,
    CropProcessor,
    DecisionPolicy,
    DecisionPolicyConfig,
    BBoxQualityScorer,
)
from .tracking import (
    TrackRuntime,
    TrackSetup,
    TrackExporter,
    convert_yolo_detections_to_detections,
    export_tracks_debug,
    filter_detections_by_area,
)
from .output import (
    ReportGenerator,
    ResultBuilder,
    LearningIntegration,
)

__all__ = [
    # Core
    "PipelineEngine",
    "VideoFrameReader",
    # Processing
    "DetectionProcessor",
    "CropProcessor",
    "DecisionPolicy",
    "DecisionPolicyConfig",
    "BBoxQualityScorer",
    # Tracking
    "TrackRuntime",
    "TrackSetup",
    "TrackExporter",
    "convert_yolo_detections_to_detections",
    "export_tracks_debug",
    "filter_detections_by_area",
    # Output
    "ReportGenerator",
    "ResultBuilder",
    "LearningIntegration",
]
