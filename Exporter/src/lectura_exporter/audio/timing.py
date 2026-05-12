"""Export JSON timing."""

from __future__ import annotations

import json
from pathlib import Path

from ..models import ExportTimingData, GroupeTiming, SyllabeTiming


def timing_to_dict(timing_data: ExportTimingData) -> dict:
    """Convertit les données de timing en dict sérialisable JSON.

    Schema v1.0 :
    ```json
    {"version": "1.0", "text": "...", "granularity": "syllabes",
     "sample_rate": 22050, "total_duration_ms": 2450.0,
     "groups": [{"index": 0, "text": "Le chat", "phone": "ləʃa",
       "start_ms": 0.0, "end_ms": 1200.0,
       "syllables": [{"phone": "lə", "ortho": "Le",
                       "start_ms": 0.0, "end_ms": 400.0}]}]}
    ```
    """
    groups = []
    for gt in timing_data.group_timings:
        syllables = [
            {
                "phone": st.phone,
                "ortho": st.ortho,
                "start_ms": round(st.start_ms, 1),
                "end_ms": round(st.end_ms, 1),
            }
            for st in gt.syllabe_timings
        ]
        groups.append({
            "index": gt.group_index,
            "text": gt.text,
            "phone": gt.phone_groupe,
            "start_ms": round(gt.start_ms, 1),
            "end_ms": round(gt.end_ms, 1),
            "syllables": syllables,
        })

    return {
        "version": "1.0",
        "text": timing_data.text,
        "granularity": timing_data.granularity,
        "sample_rate": timing_data.sample_rate,
        "total_duration_ms": round(timing_data.total_duration_ms, 1),
        "groups": groups,
    }


def export_timing_json(
    timing_data: ExportTimingData,
    output: str | Path,
) -> Path:
    """Exporte les données de timing en fichier JSON.

    Parameters
    ----------
    timing_data : ExportTimingData
        Données de timing à exporter.
    output : str | Path
        Chemin du fichier de sortie.

    Returns
    -------
    Path
        Chemin absolu du fichier créé.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = timing_to_dict(timing_data)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output.resolve()
