"""Planification des frames vidéo — pure Python, zéro Qt."""

from __future__ import annotations

from ..models import ExportTimingData, FrameHighlight


def build_frame_schedule(
    timing_data: ExportTimingData,
    fps: int,
) -> list[FrameHighlight]:
    """Pré-calcule l'état de surlignage pour chaque frame de la vidéo."""
    total_ms = timing_data.total_duration_ms
    if total_ms <= 0:
        return []

    total_frames = int(total_ms * fps / 1000) + 1
    schedule: list[FrameHighlight] = []

    for frame_idx in range(total_frames):
        time_ms = frame_idx * 1000.0 / fps
        highlight = find_highlight_at(timing_data, time_ms)
        schedule.append(highlight)

    return schedule


def find_highlight_at(
    timing_data: ExportTimingData,
    time_ms: float,
) -> FrameHighlight:
    """Trouve l'état de surlignage à un instant donné."""
    for gt in timing_data.group_timings:
        if time_ms < gt.start_ms or time_ms >= gt.end_ms:
            continue
        # On est dans ce groupe
        if gt.syllabe_timings:
            for st in gt.syllabe_timings:
                if st.start_ms <= time_ms < st.end_ms:
                    return FrameHighlight(
                        action="syllabe",
                        group_index=gt.group_index,
                        syllabe_index=gt.syllabe_timings.index(st),
                    )
        # Pas de syllabe active, mais groupe actif
        return FrameHighlight(action="group", group_index=gt.group_index)

    return FrameHighlight(action="clear")
