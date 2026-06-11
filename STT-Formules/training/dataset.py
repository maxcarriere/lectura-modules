#!/usr/bin/env python3
"""
Dataset CTC pour l'entrainement FormulaCTC.

- Charge un manifest JSONL (genere par generate_corpus.py)
- Labels pre-calcules comme ``tokens: [22, 4, ...]``
- Extraction mel on-the-fly (80 bins, 16kHz)
- SpecAugment (freq + time masking)
- Split stratifie train/val

Adapte de PhoneCTCDataset (stt/dataset.py).
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms
from torch.utils.data import Dataset

# ──────────────────────────────────────────────
# Constantes mel (standard ASR 16kHz)
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000
N_FFT = 512        # 32ms @ 16kHz
HOP_LENGTH = 160   # 10ms -> 100 frames/s
WIN_LENGTH = 400   # 25ms
N_MELS = 80
FMIN = 0
FMAX = 8000        # Nyquist @ 16kHz


# ──────────────────────────────────────────────
# Chargement manifest
# ──────────────────────────────────────────────

def load_manifest(path: str | Path, corpus_dir: str | Path | None = None) -> list[dict]:
    """Charge un manifest JSONL et resout les chemins relatifs.

    Chaque ligne du manifest est un objet JSON avec au minimum :
        audio_path: str  — chemin vers le fichier WAV
        tokens: list[int] — sequence de token IDs
        formula_type: str — type de formule (pour split stratifie)

    Si ``corpus_dir`` est fourni, les chemins relatifs dans ``audio_path``
    sont resolus par rapport a ce dossier.

    Args:
        path: chemin vers le fichier manifest.jsonl
        corpus_dir: dossier racine du corpus (pour resoudre les chemins relatifs)

    Returns:
        Liste d'entrees (dicts)
    """
    path = Path(path)
    if corpus_dir is None:
        corpus_dir = path.parent
    else:
        corpus_dir = Path(corpus_dir)

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Resoudre chemin relatif
            audio_path = Path(entry["audio_path"])
            if not audio_path.is_absolute():
                entry["audio_path"] = str(corpus_dir / audio_path)
            entries.append(entry)

    return entries


# ──────────────────────────────────────────────
# Split stratifie
# ──────────────────────────────────────────────

def stratified_split(
    entries: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
    key: str = "formula_type",
) -> tuple[list[dict], list[dict]]:
    """Split stratifie train/val selon un champ categoriel.

    Args:
        entries: liste d'entrees
        val_ratio: proportion de validation (0.0 - 1.0)
        seed: graine aleatoire
        key: champ utilise pour la stratification

    Returns:
        (train_entries, val_entries)
    """
    rng = random.Random(seed)

    # Grouper par type
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[e.get(key, "unknown")].append(e)

    train_entries = []
    val_entries = []

    for group_key in sorted(groups):
        items = groups[group_key]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_entries.extend(items[:n_val])
        train_entries.extend(items[n_val:])

    # Melanger les resultats
    rng.shuffle(train_entries)
    rng.shuffle(val_entries)

    return train_entries, val_entries


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class FormulaDataset(Dataset):
    """Dataset pour l'entrainement CTC : audio -> tokens semantiques.

    Chaque element retourne un dict :
        mel: (1, n_mels, T)  log-mel spectrogramme
        mel_length: int       nombre de frames mel (avant padding)
        labels: list[int]     sequence de token IDs
    """

    def __init__(
        self,
        entries: list[dict],
        max_audio_sec: float = 10.0,
        augment: bool = False,
        freq_mask_param: int = 15,
        freq_masks: int = 2,
        time_mask_param: int = 30,
        time_masks: int = 2,
    ) -> None:
        self.augment = augment
        self.max_audio_sec = max_audio_sec

        # Filtrer les entrees trop longues et les fichiers manquants
        self.entries = [
            e for e in entries
            if e.get("duration_s", e.get("duration", float("inf"))) <= max_audio_sec
            and Path(e["audio_path"]).exists()
        ]

        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            power=2.0,
        )

        # SpecAugment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.freq_masks = freq_masks
        self.time_masks = time_masks

        # Cache resamplers par sample rate
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        try:
            waveform = self._load_audio(entry)
        except Exception:
            # Fichier audio corrompu/illisible -> retourner un autre exemple
            for offset in range(1, min(100, len(self))):
                try:
                    waveform = self._load_audio(self.entries[(idx + offset) % len(self)])
                    entry = self.entries[(idx + offset) % len(self)]
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"100 entrees consecutives illisibles a partir de l'index {idx}")

        # Waveform augmentations (avant mel, uniquement en entrainement)
        if self.augment:
            if random.random() < 0.5:
                waveform = self._add_gaussian_noise(waveform)
            if random.random() < 0.3:
                waveform = self._speed_perturbation(waveform)
            if random.random() < 0.5:
                waveform = self._volume_perturbation(waveform)

        # Mel spectrogram
        mel = self.mel_transform(waveform)  # (1, n_mels, T)

        # Log mel (epsilon pour stabilite)
        mel = torch.log(mel + 1e-8)

        # SpecAugment (uniquement en entrainement)
        if self.augment:
            for _ in range(self.freq_masks):
                mel = self.freq_mask(mel)
            for _ in range(self.time_masks):
                mel = self.time_mask(mel)

        mel_length = mel.shape[-1]

        # Labels pre-calcules
        labels = entry["tokens"]

        return {
            "mel": mel,               # (1, 80, T)
            "mel_length": mel_length,  # int
            "labels": labels,          # list[int]
        }

    def _load_audio(self, entry: dict) -> torch.Tensor:
        """Charge et resample l'audio en 16kHz mono via soundfile."""
        audio_path = entry["audio_path"]
        data, sr = sf.read(audio_path, dtype="float32")

        # Mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)

        # Resample a 16kHz si necessaire
        if sr != SAMPLE_RATE:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = self._resamplers[sr](waveform)

        # Tronquer si trop long
        max_samples = int(self.max_audio_sec * SAMPLE_RATE)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        return waveform  # (1, samples)

    # -- Waveform augmentations ───────────────────────────

    def _add_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ajoute du bruit gaussien avec un SNR aleatoire dans [15, 40] dB."""
        snr_db = random.uniform(15.0, 40.0)
        signal_power = waveform.var()
        if signal_power < 1e-10:
            return waveform
        noise = torch.randn_like(waveform)
        noise_power = signal_power / (10 ** (snr_db / 10))
        scale = math.sqrt(noise_power / noise.var().item())
        return waveform + noise * scale

    def _speed_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Perturbation de vitesse avec un facteur dans [0.9, 1.1]."""
        factor = random.uniform(0.9, 1.1)
        original_len = waveform.shape[-1]
        new_len = int(original_len / factor)
        if new_len < 1:
            return waveform
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False,
        ).squeeze(0)
        return waveform

    def _volume_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Perturbation de volume avec un gain dans [-6, 6] dB."""
        gain_db = random.uniform(-6.0, 6.0)
        return waveform * (10 ** (gain_db / 20))


# ──────────────────────────────────────────────
# Collator
# ──────────────────────────────────────────────

class CTCCollator:
    """Collator qui pad les mels et labels pour CTC.

    Retourne un dict :
        mel: (B, 1, n_mels, T_max)  padde a droite avec des zeros
        mel_lengths: (B,)           longueurs originales en frames
        labels: (B, L_max)          padde avec pad_id
        label_lengths: (B,)         longueurs des labels
    """

    def __init__(self, pad_label_id: int = 0) -> None:
        self.pad_label_id = pad_label_id

    def __call__(self, batch: list[dict]) -> dict:
        mels = [item["mel"] for item in batch]          # list of (1, 80, T_i)
        mel_lengths = [item["mel_length"] for item in batch]
        labels = [item["labels"] for item in batch]     # list of list[int]

        # Pad mels : trouver T_max et padder a droite
        max_mel_len = max(mel_lengths)
        padded_mels = []
        for mel in mels:
            T = mel.shape[-1]
            if T < max_mel_len:
                pad = torch.zeros(1, N_MELS, max_mel_len - T)
                mel = torch.cat([mel, pad], dim=-1)
            padded_mels.append(mel)

        mel_batch = torch.stack(padded_mels, dim=0)  # (B, 1, 80, T_max)
        mel_lengths_t = torch.tensor(mel_lengths, dtype=torch.long)

        # Pad labels
        max_label_len = max(len(l) for l in labels)
        padded_labels = []
        label_lengths = []
        for l in labels:
            label_lengths.append(len(l))
            padded = l + [self.pad_label_id] * (max_label_len - len(l))
            padded_labels.append(padded)

        labels_t = torch.tensor(padded_labels, dtype=torch.long)
        label_lengths_t = torch.tensor(label_lengths, dtype=torch.long)

        return {
            "mel": mel_batch,                # (B, 1, 80, T_max)
            "mel_lengths": mel_lengths_t,     # (B,)
            "labels": labels_t,              # (B, L_max)
            "label_lengths": label_lengths_t, # (B,)
        }
