"""Tests pour l'export JSON timing."""

import json
from pathlib import Path

import pytest

from lectura_exporter.models import (
    ExportTimingData,
    GroupeTiming,
    SyllabeTiming,
)
from lectura_exporter.audio.timing import export_timing_json, timing_to_dict


def _make_timing_data() -> ExportTimingData:
    """Crée des données de timing de test."""
    return ExportTimingData(
        text="Le chat",
        group_timings=[
            GroupeTiming(
                group_index=0,
                text="Le chat",
                phone_groupe="ləʃa",
                start_ms=0.0,
                end_ms=1200.0,
                syllabe_timings=[
                    SyllabeTiming(phone="lə", ortho="Le", start_ms=0.0, end_ms=400.0),
                    SyllabeTiming(phone="ʃa", ortho="chat", start_ms=400.0, end_ms=1200.0),
                ],
            ),
        ],
        total_duration_ms=1200.0,
        sample_rate=22050,
        granularity="syllabes",
    )


def test_timing_to_dict():
    """Vérifie la structure du dict de timing."""
    data = _make_timing_data()
    result = timing_to_dict(data)

    assert result["version"] == "1.0"
    assert result["text"] == "Le chat"
    assert result["granularity"] == "syllabes"
    assert result["sample_rate"] == 22050
    assert result["total_duration_ms"] == 1200.0
    assert len(result["groups"]) == 1

    group = result["groups"][0]
    assert group["index"] == 0
    assert group["text"] == "Le chat"
    assert group["phone"] == "ləʃa"
    assert len(group["syllables"]) == 2

    syll = group["syllables"][0]
    assert syll["phone"] == "lə"
    assert syll["ortho"] == "Le"
    assert syll["start_ms"] == 0.0
    assert syll["end_ms"] == 400.0


def test_export_timing_json_creates_file(tmp_path: Path):
    """Vérifie qu'un fichier JSON valide est créé."""
    data = _make_timing_data()
    output = tmp_path / "timing.json"

    result = export_timing_json(data, output)

    assert result.exists()
    content = json.loads(result.read_text(encoding="utf-8"))
    assert content["version"] == "1.0"
    assert len(content["groups"]) == 1


def test_export_timing_json_creates_parent_dirs(tmp_path: Path):
    """Vérifie que les répertoires parents sont créés."""
    data = _make_timing_data()
    output = tmp_path / "sub" / "dir" / "timing.json"

    result = export_timing_json(data, output)

    assert result.exists()


def test_export_timing_json_utf8(tmp_path: Path):
    """Vérifie que les caractères Unicode sont correctement encodés."""
    data = ExportTimingData(
        text="éàü",
        group_timings=[
            GroupeTiming(
                group_index=0,
                text="éàü",
                phone_groupe="eau",
                start_ms=0.0,
                end_ms=500.0,
                syllabe_timings=[
                    SyllabeTiming(phone="eau", ortho="éàü", start_ms=0.0, end_ms=500.0),
                ],
            ),
        ],
        total_duration_ms=500.0,
        sample_rate=22050,
    )
    output = tmp_path / "timing.json"

    result = export_timing_json(data, output)

    raw = result.read_text(encoding="utf-8")
    assert "éàü" in raw  # ensure_ascii=False
