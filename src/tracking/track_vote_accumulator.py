#!/usr/bin/env python3
"""
Track Vote Accumulator — Sprint 3.2 (liviano).

Consolida identificaciones por track mediante votación simple.
Objetivo: producir una decisión estable y auditable por track sin complejidad excesiva.

Diseñado para ser compatible con Sprint 4 (depósito logístico).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
from enum import Enum


class DecisionProfile(Enum):
    """Perfiles de decisión para diferentes escenarios."""
    SHELF_STRICT = "shelf_strict"
    WAREHOUSE_LENIENT = "warehouse_lenient"
    WAREHOUSE_BALANCED = "warehouse_balanced"
    
    @classmethod
    def get_config(cls, profile: "DecisionProfile") -> Dict[str, Any]:
        """
        Retorna configuración para un perfil específico.
        
        Returns:
            Dict con: min_frames, min_confidence, min_avg_sim, min_sim_vote, high_sim_threshold
        """
        configs = {
            cls.SHELF_STRICT: {
                "min_frames": 3,
                "min_confidence": 0.60,
                "min_avg_sim": 0.28,
                "min_sim_vote": 0.30,  # Solo votar si sim >= 0.30
                "high_sim_threshold": 0.90,  # Para regla dual
            },
            cls.WAREHOUSE_LENIENT: {
                "min_frames": 2,
                "min_confidence": 0.50,
                "min_avg_sim": 0.25,
                "min_sim_vote": 0.30,
                "high_sim_threshold": 0.90,
            },
            cls.WAREHOUSE_BALANCED: {
                "min_frames": 2,
                "min_confidence": 0.55,
                "min_avg_sim": 0.26,
                "min_sim_vote": 0.30,
                "high_sim_threshold": 0.88,
            },
        }
        return configs.get(profile, configs[cls.SHELF_STRICT])


@dataclass
class TrackDecision:
    """
    Decisión final consolidada para un track.
    
    Attributes:
        track_id: ID del track
        final_sku: EAN final (o "UNKNOWN")
        confidence: Confianza de la decisión (0.0-1.0)
        avg_sim: Similitud promedio del SKU ganador
        frames_validos: Cantidad de frames con predicción válida
        votes: Dict[sku -> cantidad de votos]
        top_matches: Lista de top-N candidatos con sus votos
        ended_reason: Razón de finalización ("max_age" | "video_end" | "lost")
    """
    track_id: int
    final_sku: str
    confidence: float
    avg_sim: float
    frames_validos: int
    frames_scored: int = 0  # Frames que aportaron votos reales (no UNKNOWN)
    votes: Dict[str, int] = field(default_factory=dict)
    top_matches: List[Dict[str, Any]] = field(default_factory=list)
    ended_reason: str = "unknown"


class TrackVoteAccumulator:
    """
    Acumula votos de identificación por track con política de decisión mejorada.
    
    Mejoras:
    - Separa frames_seen vs frames_scored (no penaliza frames sin señal)
    - Regla dual de aceptación (confianza OR similitud promedio alta)
    - Filtro min_sim_vote para evitar votos basura
    - Perfiles configurables (shelf vs warehouse)
    """
    
    def __init__(
        self,
        min_frames: int = 3,
        min_confidence: float = 0.60,
        min_avg_sim: float = 0.25,
        min_sim_vote: float = 0.30,
        high_sim_threshold: float = 0.90,
        profile: Optional[DecisionProfile] = None,
    ):
        """
        Args:
            min_frames: Mínimo de frames scored para aceptar decisión
            min_confidence: Mínimo de confianza (votes[top1] / frames_scored)
            min_avg_sim: Mínimo de similitud promedio del SKU ganador
            min_sim_vote: Mínimo de similitud para considerar un voto válido
            high_sim_threshold: Umbral de similitud alta para regla dual
            profile: Perfil predefinido (sobrescribe otros parámetros si se proporciona)
        """
        # Si se proporciona un perfil, usar su configuración
        if profile:
            config = DecisionProfile.get_config(profile)
            self.min_frames = int(config["min_frames"])
            self.min_confidence = float(config["min_confidence"])
            self.min_avg_sim = float(config["min_avg_sim"])
            self.min_sim_vote = float(config["min_sim_vote"])
            self.high_sim_threshold = float(config["high_sim_threshold"])
        else:
            self.min_frames = int(min_frames)
            self.min_confidence = float(min_confidence)
            self.min_avg_sim = float(min_avg_sim)
            self.min_sim_vote = float(min_sim_vote)
            self.high_sim_threshold = float(high_sim_threshold)
        
        # Estado por track_id
        # votes[track_id][sku] = cantidad de votos
        self.votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # score_sum[track_id][sku] = suma de similitudes (para calcular promedio)
        self.score_sum: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # frames_seen[track_id] = cantidad total de frames donde existió el track
        self.frames_seen: Dict[int, int] = defaultdict(int)
        
        # frames_scored[track_id] = cantidad de frames que aportaron votos reales
        self.frames_scored: Dict[int, int] = defaultdict(int)
        
        # last_seen_frame[track_id] = último frame donde se vio el track
        self.last_seen_frame: Dict[int, int] = {}
        
        # ended_tracks: tracks que ya fueron finalizados
        self.ended_tracks: set[int] = set()
    
    def add(
        self,
        track_id: int,
        sku_pred: str,
        sim: float,
        frame_idx: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Agrega un voto de identificación para un track.
        
        Mejora: separa frames_seen (todos) vs frames_scored (solo con votos válidos).
        
        Args:
            track_id: ID del track
            sku_pred: EAN predicho (o "UNKNOWN")
            sim: Similitud del match (0.0-1.0)
            frame_idx: Índice del frame
            meta: Metadata adicional (opcional)
        """
        if track_id in self.ended_tracks:
            # No agregar votos a tracks ya finalizados
            return
        
        # Siempre contar frame visto
        self.frames_seen[track_id] += 1
        self.last_seen_frame[track_id] = max(self.last_seen_frame.get(track_id, 0), frame_idx)
        
        # Solo contar como "scored" si:
        # 1. No es UNKNOWN
        # 2. Similitud supera min_sim_vote (filtro de calidad)
        if sku_pred and sku_pred != "UNKNOWN" and sim >= self.min_sim_vote:
            self.votes[track_id][sku_pred] += 1
            self.score_sum[track_id][sku_pred] += sim
            self.frames_scored[track_id] += 1
    
    def finalize(
        self,
        track_id: int,
        ended_reason: str = "unknown",
    ) -> Optional[TrackDecision]:
        """
        Finaliza un track y calcula la decisión final.
        
        Args:
            track_id: ID del track a finalizar
            ended_reason: Razón de finalización ("max_age" | "video_end" | "lost")
        
        Returns:
            TrackDecision si el track tiene evidencia suficiente, None si no
        """
        if track_id in self.ended_tracks:
            # Ya fue finalizado
            return None
        
        frames_seen = self.frames_seen.get(track_id, 0)
        frames_scored = self.frames_scored.get(track_id, 0)
        
        if frames_scored == 0:
            # Track sin votos válidos - limpiar memoria
            self._cleanup_track(track_id)
            self.ended_tracks.add(track_id)
            decision = TrackDecision(
                track_id=track_id,
                final_sku="UNKNOWN",
                confidence=0.0,
                avg_sim=0.0,
                frames_validos=frames_seen,
                frames_scored=0,
                votes={},
                top_matches=[],
                ended_reason=ended_reason,
            )
            return decision
        
        votes_track = self.votes.get(track_id, {})
        score_sum_track = self.score_sum.get(track_id, {})
        
        if not votes_track:
            # No hay votos válidos (todo UNKNOWN)
            self._cleanup_track(track_id)
            self.ended_tracks.add(track_id)
            decision = TrackDecision(
                track_id=track_id,
                final_sku="UNKNOWN",
                confidence=0.0,
                avg_sim=0.0,
                frames_validos=frames_seen,
                frames_scored=frames_scored,
                votes={},
                top_matches=[],
                ended_reason=ended_reason,
            )
            return decision
        
        # Encontrar top1 (SKU con más votos)
        top1_sku = max(votes_track.items(), key=lambda x: x[1])[0]
        top1_votes = votes_track[top1_sku]
        
        # Calcular confianza usando frames_scored (no frames_seen)
        confidence = top1_votes / float(frames_scored) if frames_scored > 0 else 0.0
        
        # Calcular similitud promedio del top1
        avg_sim = score_sum_track.get(top1_sku, 0.0) / float(top1_votes) if top1_votes > 0 else 0.0
        
        # REGLA DUAL DE ACEPTACIÓN:
        # (A) Confianza alta + frames suficientes
        # (B) Similitud muy alta + al menos 1 voto (para tracks cortos)
        rule_a = (
            frames_scored >= self.min_frames
            and confidence >= self.min_confidence
            and avg_sim >= self.min_avg_sim
        )
        
        rule_b = (
            avg_sim >= self.high_sim_threshold
            and top1_votes >= 1
            and frames_scored >= 1  # Mínimo 1 frame scored
        )
        
        if rule_a or rule_b:
            final_sku = top1_sku
        else:
            final_sku = "UNKNOWN"
        
        # Construir top_matches (top-N por votos)
        sorted_skus = sorted(
            votes_track.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5
        
        top_matches = []
        for sku, votes_count in sorted_skus:
            avg_sim_sku = score_sum_track.get(sku, 0.0) / float(votes_count) if votes_count > 0 else 0.0
            top_matches.append({
                "sku": sku,
                "votes": votes_count,
                "avg_sim": round(avg_sim_sku, 4),
                "confidence": round(votes_count / float(frames_scored), 4),
            })
        
        decision = TrackDecision(
            track_id=track_id,
            final_sku=final_sku,
            confidence=round(confidence, 4),
            avg_sim=round(avg_sim, 4),
            frames_validos=frames_seen,  # Mantener compatibilidad
            frames_scored=frames_scored,
            votes=dict(votes_track),
            top_matches=top_matches,
            ended_reason=ended_reason,
        )
        
        # Limpiar memoria del track finalizado
        self._cleanup_track(track_id)
        self.ended_tracks.add(track_id)
        return decision
    
    def _cleanup_track(self, track_id: int) -> None:
        """Limpia toda la memoria asociada a un track."""
        if track_id in self.votes:
            del self.votes[track_id]
        if track_id in self.score_sum:
            del self.score_sum[track_id]
        if track_id in self.frames_seen:
            del self.frames_seen[track_id]
        if track_id in self.frames_scored:
            del self.frames_scored[track_id]
        if track_id in self.last_seen_frame:
            del self.last_seen_frame[track_id]
    
    def get_active_track_ids(self) -> List[int]:
        """Retorna lista de track_ids activos (no finalizados)."""
        all_tracks = set(self.votes.keys()) | set(self.frames_seen.keys())
        return [tid for tid in all_tracks if tid not in self.ended_tracks]
    
    def clear(self) -> None:
        """Limpia el estado (útil para testing o reset)."""
        self.votes.clear()
        self.score_sum.clear()
        self.frames_seen.clear()
        self.frames_scored.clear()
        self.last_seen_frame.clear()
        self.ended_tracks.clear()
