"""Tests pour training/dataset.py — FormulaDataset + CTCCollator."""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

# Ajouter le dossier parent pour importer training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.dataset import (
    CTCCollator,
    FormulaDataset,
    load_manifest,
    stratified_split,
)


# ──────────────────────────────────────────────
# Helpers : fichiers synthetiques
# ──────────────────────────────────────────────

def _write_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> None:
    """Ecrit un fichier WAV synthetique (sinusoide 440Hz)."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())


def _make_corpus(tmp_path: Path, n_entries: int = 10) -> Path:
    """Cree un mini-corpus avec manifest JSONL et fichiers WAV."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    manifest_path = tmp_path / "manifest.jsonl"

    types = ["nombre", "date", "heure", "sigle", "telephone"]
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i in range(n_entries):
            wav_name = f"sample_{i:03d}.wav"
            wav_path = audio_dir / wav_name
            _write_wav(wav_path, duration_sec=0.5 + (i % 5) * 0.2)
            entry = {
                "audio_path": f"audio/{wav_name}",  # chemin relatif
                "tokens": [2, 4, 1, 10 + i % 5],
                "formula_type": types[i % len(types)],
                "duration": 0.5 + (i % 5) * 0.2,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return manifest_path


# ──────────────────────────────────────────────
# Tests load_manifest
# ──────────────────────────────────────────────

class TestLoadManifest:
    """Tests de chargement du manifest JSONL."""

    def test_load_basic(self, tmp_path):
        """Charge un manifest JSONL et retourne les entrees."""
        manifest_path = _make_corpus(tmp_path, n_entries=5)
        entries = load_manifest(manifest_path)
        assert len(entries) == 5
        assert all("audio_path" in e for e in entries)
        assert all("tokens" in e for e in entries)

    def test_relative_paths_resolved(self, tmp_path):
        """Les chemins relatifs sont resolus par rapport au dossier du manifest."""
        manifest_path = _make_corpus(tmp_path, n_entries=3)
        entries = load_manifest(manifest_path)
        for e in entries:
            p = Path(e["audio_path"])
            assert p.is_absolute(), f"Chemin non resolu : {e['audio_path']}"
            assert p.exists(), f"Fichier inexistant : {e['audio_path']}"

    def test_corpus_dir_override(self, tmp_path):
        """corpus_dir surcharge le dossier de resolution."""
        manifest_path = _make_corpus(tmp_path, n_entries=2)
        entries = load_manifest(manifest_path, corpus_dir=tmp_path)
        for e in entries:
            assert Path(e["audio_path"]).is_absolute()

    def test_empty_lines_ignored(self, tmp_path):
        """Les lignes vides sont ignorees."""
        manifest_path = tmp_path / "manifest.jsonl"
        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path)
        with open(manifest_path, "w") as f:
            f.write(json.dumps({"audio_path": str(wav_path), "tokens": [1, 2]}) + "\n")
            f.write("\n")  # ligne vide
            f.write(json.dumps({"audio_path": str(wav_path), "tokens": [3, 4]}) + "\n")
        entries = load_manifest(manifest_path)
        assert len(entries) == 2


# ──────────────────────────────────────────────
# Tests FormulaDataset
# ──────────────────────────────────────────────

class TestFormulaDataset:
    """Tests du dataset."""

    def test_item_shape(self, tmp_path):
        """Un item a mel (1, 80, T), mel_length, labels."""
        manifest_path = _make_corpus(tmp_path, n_entries=3)
        entries = load_manifest(manifest_path)
        ds = FormulaDataset(entries, augment=False)

        assert len(ds) > 0
        item = ds[0]
        assert "mel" in item
        assert "mel_length" in item
        assert "labels" in item

        mel = item["mel"]
        assert mel.ndim == 3
        assert mel.shape[0] == 1   # canal unique
        assert mel.shape[1] == 80  # n_mels
        assert mel.shape[2] > 0    # frames temporelles
        assert item["mel_length"] == mel.shape[2]

    def test_labels_from_tokens(self, tmp_path):
        """Les labels sont les tokens pre-calcules du manifest."""
        manifest_path = _make_corpus(tmp_path, n_entries=3)
        entries = load_manifest(manifest_path)
        ds = FormulaDataset(entries, augment=False)
        item = ds[0]
        assert item["labels"] == entries[0]["tokens"]

    def test_filter_too_long(self, tmp_path):
        """Les entrees trop longues sont filtrees."""
        manifest_path = _make_corpus(tmp_path, n_entries=5)
        entries = load_manifest(manifest_path)
        # L'entree la plus longue fait ~1.3s, filtrer a 0.6s
        ds = FormulaDataset(entries, max_audio_sec=0.6, augment=False)
        assert len(ds) < len(entries)

    def test_augment_does_not_crash(self, tmp_path):
        """Les augmentations ne plantent pas."""
        manifest_path = _make_corpus(tmp_path, n_entries=5)
        entries = load_manifest(manifest_path)
        ds = FormulaDataset(entries, augment=True)
        # Charger plusieurs items pour exercer toutes les augmentations
        for i in range(min(5, len(ds))):
            item = ds[i]
            assert item["mel"].shape[0] == 1
            assert item["mel"].shape[1] == 80


# ──────────────────────────────────────────────
# Tests CTCCollator
# ──────────────────────────────────────────────

class TestCTCCollator:
    """Tests du collator CTC."""

    def test_batch_shapes(self, tmp_path):
        """Le collator produit des tenseurs correctement paddes."""
        manifest_path = _make_corpus(tmp_path, n_entries=6)
        entries = load_manifest(manifest_path)
        ds = FormulaDataset(entries, augment=False)
        collator = CTCCollator(pad_label_id=0)

        batch_items = [ds[i] for i in range(min(4, len(ds)))]
        batch = collator(batch_items)

        B = len(batch_items)
        assert batch["mel"].shape[0] == B
        assert batch["mel"].shape[1] == 1
        assert batch["mel"].shape[2] == 80
        assert batch["mel_lengths"].shape == (B,)
        assert batch["labels"].shape[0] == B
        assert batch["label_lengths"].shape == (B,)

    def test_mel_padding(self, tmp_path):
        """Les mels sont paddes a la longueur max du batch."""
        manifest_path = _make_corpus(tmp_path, n_entries=6)
        entries = load_manifest(manifest_path)
        ds = FormulaDataset(entries, augment=False)
        collator = CTCCollator()

        batch_items = [ds[i] for i in range(min(3, len(ds)))]
        batch = collator(batch_items)

        # T_max doit etre egal au max des mel_lengths
        max_len = batch["mel_lengths"].max().item()
        assert batch["mel"].shape[3] == max_len

    def test_label_padding(self, tmp_path):
        """Les labels sont paddes avec pad_label_id."""
        manifest_path = _make_corpus(tmp_path, n_entries=4)
        entries = load_manifest(manifest_path)
        ds = FormulaDataset(entries, augment=False)
        collator = CTCCollator(pad_label_id=0)

        batch_items = [ds[i] for i in range(min(3, len(ds)))]
        batch = collator(batch_items)

        # Verifier que les labels apres label_length sont paddes
        for i in range(len(batch_items)):
            lab_len = batch["label_lengths"][i].item()
            max_lab_len = batch["labels"].shape[1]
            if lab_len < max_lab_len:
                padding = batch["labels"][i, lab_len:]
                assert (padding == 0).all()

    def test_label_lengths_correct(self, tmp_path):
        """Les label_lengths correspondent aux longueurs reelles."""
        manifest_path = _make_corpus(tmp_path, n_entries=4)
        entries = load_manifest(manifest_path)
        ds = FormulaDataset(entries, augment=False)
        collator = CTCCollator()

        batch_items = [ds[i] for i in range(min(3, len(ds)))]
        batch = collator(batch_items)

        for i, item in enumerate(batch_items):
            assert batch["label_lengths"][i].item() == len(item["labels"])


# ──────────────────────────────────────────────
# Tests stratified_split
# ──────────────────────────────────────────────

class TestStratifiedSplit:
    """Tests du split stratifie."""

    def test_ratio_respected(self):
        """Le ratio val est approximativement respecte."""
        entries = [
            {"tokens": [1], "formula_type": t}
            for t in ["nombre"] * 50 + ["date"] * 30 + ["heure"] * 20
        ]
        train, val = stratified_split(entries, val_ratio=0.2, seed=42)
        total = len(train) + len(val)
        assert total == len(entries)
        # Ratio val entre 15% et 25% (tolerance pour les petits groupes)
        ratio = len(val) / total
        assert 0.15 <= ratio <= 0.30, f"Ratio val = {ratio:.2f}"

    def test_all_types_covered(self):
        """Tous les types de formules apparaissent dans train et val."""
        types = ["nombre", "date", "heure", "sigle", "telephone"]
        entries = [
            {"tokens": [i], "formula_type": t}
            for i, t in enumerate(types * 10)
        ]
        train, val = stratified_split(entries, val_ratio=0.2, seed=42)

        train_types = {e["formula_type"] for e in train}
        val_types = {e["formula_type"] for e in val}
        assert train_types == set(types)
        assert val_types == set(types)

    def test_deterministic(self):
        """Meme seed -> meme split."""
        entries = [
            {"tokens": [i], "formula_type": "nombre"}
            for i in range(50)
        ]
        train1, val1 = stratified_split(entries, val_ratio=0.2, seed=42)
        train2, val2 = stratified_split(entries, val_ratio=0.2, seed=42)
        assert [e["tokens"] for e in train1] == [e["tokens"] for e in train2]
        assert [e["tokens"] for e in val1] == [e["tokens"] for e in val2]

    def test_no_overlap(self):
        """Pas de recouvrement entre train et val."""
        entries = [
            {"tokens": [i], "formula_type": "nombre", "id": i}
            for i in range(50)
        ]
        train, val = stratified_split(entries, val_ratio=0.2, seed=42)
        train_ids = {e["id"] for e in train}
        val_ids = {e["id"] for e in val}
        assert train_ids.isdisjoint(val_ids)
