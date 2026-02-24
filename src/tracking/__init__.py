"""Tracking module — Asignación de identidad persistente a detecciones."""

from .track_types import Detection, Track, BBox
from .tracker_base import TrackerBase
from .sort_like_tracker import SortLikeTracker
from .track_vote_accumulator import TrackVoteAccumulator, TrackDecision, DecisionProfile

__all__ = [
    "Detection",
    "Track",
    "BBox",
    "TrackerBase",
    "SortLikeTracker",
    "TrackVoteAccumulator",
    "TrackDecision",
    "DecisionProfile",
]
