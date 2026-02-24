#!/usr/bin/env python3
"""
Track Exporter — Exporta resultados de tracking y genera inventarios.

Responsabilidades:
- Exportar track_summary.json
- Generar inventario_por_track.csv
- Generar inventario_por_frame.csv (comparación)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import json

from src.tracking.track_vote_accumulator import TrackDecision
from src.pipeline.output.report_generator import ReportGenerator


class TrackExporter:
    """
    Exporta resultados de tracking y genera inventarios.
    """
    
    def __init__(self, output_dir: Path, report: ReportGenerator):
        self.output_dir = output_dir
        self.report = report
    
    def export_track_summary(
        self,
        track_decisions: Dict[int, TrackDecision],
    ) -> Path:
        """
        Exporta track_summary.json con decisiones finales por track.
        
        Args:
            track_decisions: Dict de track_id -> TrackDecision
        
        Returns:
            Path al archivo generado
        """
        if not track_decisions:
            return None
        
        tracks_data = []
        matched_count = 0
        unknown_count = 0
        
        for tid, decision in sorted(track_decisions.items()):
            tracks_data.append({
                "track_id": decision.track_id,
                "final_sku": decision.final_sku,
                "confidence": decision.confidence,
                "avg_sim": decision.avg_sim,
                "frames_validos": decision.frames_validos,
                "frames_scored": getattr(decision, "frames_scored", 0),  # Compatibilidad
                "votes": decision.votes,
                "top_matches": decision.top_matches,
                "ended_reason": decision.ended_reason,
            })
            
            if decision.final_sku != "UNKNOWN":
                matched_count += 1
            else:
                unknown_count += 1
        
        summary_data = {
            "tracks": tracks_data,
            "summary": {
                "total_tracks": len(tracks_data),
                "matched": matched_count,
                "unknown": unknown_count,
                "match_rate": round(matched_count / len(tracks_data), 4) if tracks_data else 0.0,
            },
        }
        
        summary_path = self.output_dir / "track_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"   📊 Track summary exportado: {summary_path}")
        print(f"      Tracks: {len(tracks_data)} | Matched: {matched_count} | Unknown: {unknown_count}")
        
        return summary_path
    
    def generate_inventories(
        self,
        track_decisions: Dict[int, TrackDecision],
        count_per_frame: List[Counter],
    ) -> Tuple[Path, Path]:
        """
        Genera inventarios por tracks y por frames.
        
        Args:
            track_decisions: Decisiones finales por track
            conteo_por_frame: Conteo por frame (para comparación)
        
        Returns:
            Tuple de (csv_path_tracks, csv_path_frame)
        """
        # Inventario por tracks (fuente de verdad)
        print("\n📊 Generando inventario por tracks...")
        inventory_from_tracks = Counter()
        for tid, decision in track_decisions.items():
            if decision.final_sku != "UNKNOWN":
                inventory_from_tracks[decision.final_sku] += 1
        
        csv_path_tracks = self.report.generate_inventory_csv(
            inventory_from_tracks,
            filename="inventario_por_track.csv"
        )
        print(f"   ✅ Inventario por tracks: {csv_path_tracks}")
        
        # Inventario por frame (comparación/debug)
        print("\n📊 Generando inventario por frame (comparación)...")
        dedup_count = ReportGenerator.deduplicate_by_frame(count_per_frame)
        csv_path_frame = self.report.generate_inventory_csv(
            dedup_count,
            filename="inventario_por_frame.csv"
        )
        
        return csv_path_tracks, csv_path_frame
