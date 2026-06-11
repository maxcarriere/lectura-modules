#!/usr/bin/env python3
"""Generateur de corpus pour le modele CTC formules.

Genere ~48 000 exemples synthetiques (nombres, dates, heures, telephones,
sigles, ordinaux, fractions, monnaies, pourcentages) avec le TTS Lectura
monospeaker, puis sauvegarde les WAV 16 kHz et un manifest JSONL compatible
avec l'entrainement CTC.

Usage:
    python scripts/generate_corpus.py \
        --output-dir /data/voix_ssd/formula_corpus/ \
        --n-base 16000 \
        --n-augmentations 3 \
        --seed 42 \
        --num-workers 4

    # Mode dry-run (pas de synthese audio)
    python scripts/generate_corpus.py \
        --output-dir /tmp/test_corpus \
        --n-base 100 \
        --n-augmentations 1 \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Imports lectura ──────────────────────────────────────────────────────

from lectura_formules import (
    LectureFormuleResult,
    OptionsLecture,
    lire_date,
    lire_fraction,
    lire_heure,
    lire_monnaie,
    lire_nombre,
    lire_ordinal,
    lire_pourcentage,
    lire_sigle,
    lire_telephone,
)
from lectura_stt_formules._tokenizer import events_to_token_sequence
from lectura_stt_formules._vocab import VOCAB, vocab_to_json


# ── Dataclass ────────────────────────────────────────────────────────────

@dataclass
class FormulaExample:
    """Un exemple de formule genere."""
    text: str                    # texte source (ex: "42")
    category: str                # type (ex: "nombre")
    result: LectureFormuleResult | None = None
    token_ids: list[int] = field(default_factory=list)
    token_names: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════
# Generateurs de formules
# ══════════════════════════════════════════════════════════════════════════


def gen_entiers(n: int = 3000) -> list[FormulaExample]:
    """Nombres entiers varies (0 a 999 999 999)."""
    examples: list[FormulaExample] = []
    nums: list[int] = []

    # Petits (0-99) : surrepresentes car briques de base
    nums.extend(range(0, 100))
    nums.extend(random.choices(range(0, 100), k=n // 8))
    # Moyens (100-9999)
    nums.extend(random.randint(100, 9999) for _ in range(n // 3))
    # Grands (10 000 - 999 999)
    nums.extend(random.randint(10_000, 999_999) for _ in range(n // 4))
    # Tres grands (1M - 999M)
    nums.extend(random.randint(1_000_000, 999_000_000) for _ in range(n // 8))
    # Negatifs
    nums.extend(-random.randint(1, 9999) for _ in range(n // 20))

    random.shuffle(nums)
    for num in nums[:n]:
        text = str(num)
        try:
            result = lire_nombre(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "nombre", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip entier %s: %s", text, e)

    return examples[:n]


def gen_decimaux(n: int = 1500) -> list[FormulaExample]:
    """Nombres decimaux."""
    examples: list[FormulaExample] = []
    patterns: list[str] = []

    # X.XX
    for _ in range(n // 3):
        patterns.append(f"{random.randint(0, 99)}.{random.randint(1, 99):02d}")
    # X.XXX
    for _ in range(n // 4):
        patterns.append(f"{random.randint(0, 9)}.{random.randint(1, 999):03d}")
    # XX.X
    for _ in range(n // 4):
        patterns.append(f"{random.randint(10, 999)}.{random.randint(1, 9)}")
    # Cas celebres
    patterns.extend(["3.14", "0.005", "12.75", "2.718", "0.001", "99.99",
                      "1.618", "0.577", "2.302", "0.333", "6.28", "9.81"])

    for text in patterns:
        try:
            result = lire_nombre(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "decimal", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip decimal %s: %s", text, e)

    return examples[:n]


def gen_dates(n: int = 2000) -> list[FormulaExample]:
    """Dates en format JJ/MM/AAAA."""
    examples: list[FormulaExample] = []
    patterns: list[str] = []

    # JJ/MM/AAAA
    for _ in range(n):
        j = random.randint(1, 28)
        m = random.randint(1, 12)
        a = random.randint(1800, 2030)
        patterns.append(f"{j:02d}/{m:02d}/{a}")

    # JJ/MM sans annee
    for _ in range(n // 5):
        j = random.randint(1, 28)
        m = random.randint(1, 12)
        patterns.append(f"{j:02d}/{m:02d}")

    # Dates historiques
    patterns.extend(["14/07/1789", "01/01/2000", "11/11/1918", "08/05/1945",
                      "06/06/1944", "25/12/2024", "14/07/2024"])

    for text in patterns:
        try:
            result = lire_date(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "date", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip date %s: %s", text, e)

    return examples[:n]


def gen_heures(n: int = 1500) -> list[FormulaExample]:
    """Heures variees."""
    examples: list[FormulaExample] = []
    patterns: list[str] = []

    for _ in range(n):
        h = random.randint(0, 23)
        m = random.randint(0, 59)
        if m == 0:
            patterns.append(f"{h}h")
        else:
            patterns.append(f"{h}h{m:02d}")

    patterns.extend(["14h30", "8h15", "23h59", "0h00", "12h", "0h"])

    for text in patterns:
        try:
            result = lire_heure(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "heure", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip heure %s: %s", text, e)

    return examples[:n]


def gen_telephones(n: int = 1000) -> list[FormulaExample]:
    """Numeros de telephone francais (format XX.XX.XX.XX.XX)."""
    examples: list[FormulaExample] = []
    prefixes = ["01", "02", "03", "04", "05", "06", "07", "09"]

    for _ in range(n):
        prefix = random.choice(prefixes)
        rest = [f"{random.randint(0, 99):02d}" for _ in range(4)]
        # Format avec points (le plus courant)
        text = f"{prefix}.{'.'.join(rest)}"
        try:
            result = lire_telephone(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "telephone", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip telephone %s: %s", text, e)

    return examples[:n]


def gen_sigles(n: int = 2000) -> list[FormulaExample]:
    """Sigles epeles lettre par lettre."""
    examples: list[FormulaExample] = []

    # Sigles courants
    sigles_fixes = [
        "SNCF", "RATP", "TGV", "RER", "HLM", "TVA", "PIB", "ONG",
        "ADN", "ARN", "QI", "CV", "PDF", "HTML", "CSS", "PHP",
        "SQL", "API", "USB", "GPU", "CPU", "RAM", "SSD", "IP",
        "HTTP", "FTP", "SMS", "MMS", "GPS", "DVD", "CD",
        "ONU", "UE", "FMI", "OMC", "BCE", "OTAN", "NASA",
        "CNRS", "CEA", "EDF", "GDF", "PME", "PMI", "TPE",
        "BTS", "DUT", "CAP", "BEP", "BAC", "DEA",
        "ORL", "CHU", "IRM", "ECG", "PCR",
        "RSA", "CMU", "CSG", "RMI",
        "TF", "FR", "BFM", "LCI", "RFI", "RTL", "RMC",
        "IBM", "HP", "BMW", "VW", "PSA",
    ]

    # Generer des sigles aleatoires
    for _ in range(n - len(sigles_fixes)):
        length = random.choice([2, 3, 3, 3, 4, 4, 4, 5])
        sigle = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=length))
        sigles_fixes.append(sigle)

    for text in sigles_fixes[:n]:
        try:
            result = lire_sigle(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "sigle", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip sigle %s: %s", text, e)

    return examples[:n]


def gen_ordinaux(n: int = 1500) -> list[FormulaExample]:
    """Ordinaux (1er, 2e, 21e, etc.)."""
    examples: list[FormulaExample] = []
    patterns = ["1er", "1re"]
    for i in range(2, 100):
        patterns.append(f"{i}e")
    for i in [100, 150, 200, 300, 500, 1000]:
        patterns.append(f"{i}e")
    # Aleatoires
    for _ in range(n):
        patterns.append(f"{random.randint(2, 999)}e")

    seen: set[str] = set()
    for text in patterns:
        if text in seen:
            continue
        seen.add(text)
        try:
            result = lire_ordinal(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "ordinal", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip ordinal %s: %s", text, e)

    return examples[:n]


def gen_fractions(n: int = 1000) -> list[FormulaExample]:
    """Fractions simples (1/2, 3/4, etc.)."""
    examples: list[FormulaExample] = []

    common = ["1/2", "1/3", "2/3", "1/4", "3/4", "1/5", "2/5", "3/5", "4/5",
              "1/6", "5/6", "1/8", "3/8", "5/8", "7/8", "1/10", "3/10", "7/10",
              "1/7", "2/7", "3/7", "1/9", "2/9", "4/9", "7/9",
              "1/12", "5/12", "7/12", "11/12",
              "1/100", "3/100", "1/1000",
              "2/3", "5/4", "7/3", "9/5", "11/8", "13/7"]

    generated = []
    for _ in range(n * 2):
        num = random.randint(1, 50)
        den = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20, 100])
        generated.append(f"{num}/{den}")

    seen: set[str] = set()
    for text in common + generated:
        if text in seen:
            continue
        seen.add(text)
        try:
            result = lire_fraction(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "fraction", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip fraction %s: %s", text, e)

    return examples[:n]


def gen_monnaie(n: int = 1500) -> list[FormulaExample]:
    """Montants en devises (euros, dollars, livres)."""
    examples: list[FormulaExample] = []
    devises = [("€", 0.6), ("$", 0.2), ("£", 0.2)]

    patterns: list[str] = []
    for _ in range(n):
        r = random.random()
        if r < devises[0][1]:
            sym = "€"
        elif r < devises[0][1] + devises[1][1]:
            sym = "$"
        else:
            sym = "£"

        montant = round(random.uniform(0.5, 10000), 2)
        if montant == int(montant):
            montant_str = str(int(montant))
        else:
            montant_str = f"{montant:.2f}"
        patterns.append(f"{montant_str}{sym}")

    patterns.extend(["42.50€", "1000$", "3.20£", "100€", "9.99€", "0.50€"])

    for text in patterns:
        try:
            result = lire_monnaie(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "monnaie", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip monnaie %s: %s", text, e)

    return examples[:n]


def gen_pourcentages(n: int = 1000) -> list[FormulaExample]:
    """Pourcentages et pour-mille."""
    examples: list[FormulaExample] = []
    patterns: list[str] = []

    # Pourcentages entiers
    for _ in range(n // 2):
        patterns.append(f"{random.randint(1, 100)}%")
    # Pourcentages decimaux
    for _ in range(n // 3):
        patterns.append(f"{random.randint(0, 99)}.{random.randint(1, 9)}%")
    # Pour-mille
    for _ in range(n // 5):
        patterns.append(f"{random.randint(1, 50)}\u2030")
    # Cas specifiques
    patterns.extend(["42%", "3.5%", "0.1\u2030", "100%", "50%", "99.9%", "0.5%", "75%"])

    for text in patterns:
        try:
            result = lire_pourcentage(text)
            if result and result.events:
                tids = events_to_token_sequence(result)
                tnames = [VOCAB.get(t, f"?{t}") for t in tids]
                examples.append(FormulaExample(text, "pourcentage", result, tids, tnames))
        except (ValueError, Exception) as e:
            log.debug("Skip pourcentage %s: %s", text, e)

    return examples[:n]


# ══════════════════════════════════════════════════════════════════════════
# Distribution du corpus
# ══════════════════════════════════════════════════════════════════════════

GENERATORS = [
    ("nombre",      gen_entiers,      3000),
    ("decimal",     gen_decimaux,     1500),
    ("date",        gen_dates,        2000),
    ("heure",       gen_heures,       1500),
    ("telephone",   gen_telephones,   1000),
    ("sigle",       gen_sigles,       2000),
    ("ordinal",     gen_ordinaux,     1500),
    ("fraction",    gen_fractions,    1000),
    ("monnaie",     gen_monnaie,      1500),
    ("pourcentage", gen_pourcentages, 1000),
]

TOTAL_BASE = sum(n for _, _, n in GENERATORS)  # 16000


def generate_all_formulas(n_base: int, seed: int) -> list[FormulaExample]:
    """Genere les formules de base (avant augmentation)."""
    random.seed(seed)
    np.random.seed(seed)

    # Ajuster les quotas proportionnellement si n_base != 16000
    ratio = n_base / TOTAL_BASE

    all_examples: list[FormulaExample] = []
    for cat_name, gen_func, default_n in GENERATORS:
        target_n = max(1, int(default_n * ratio))
        raw = gen_func(target_n)
        log.info("  %-14s : %4d / %4d exemples generes", cat_name, len(raw), target_n)
        all_examples.extend(raw)

    random.shuffle(all_examples)
    log.info("Total base: %d exemples", len(all_examples))
    return all_examples


# ══════════════════════════════════════════════════════════════════════════
# Augmentation audio
# ══════════════════════════════════════════════════════════════════════════

def add_white_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Ajoute du bruit blanc gaussien a un signal audio."""
    rms_signal = np.sqrt(np.mean(audio ** 2))
    if rms_signal < 1e-8:
        return audio
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, audio.shape).astype(audio.dtype)
    return audio + noise


def speed_change(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Change la vitesse de l'audio par interpolation lineaire."""
    if abs(factor - 1.0) < 0.01:
        return audio
    indices = np.arange(0, len(audio), factor)
    indices = indices[indices < len(audio) - 1]
    idx_floor = indices.astype(int)
    frac = indices - idx_floor
    return (audio[idx_floor] * (1 - frac) + audio[idx_floor + 1] * frac).astype(audio.dtype)


def resample_22050_to_16000(audio: np.ndarray) -> np.ndarray:
    """Resample 22050 Hz → 16000 Hz avec filtre anti-aliasing.

    Utilise scipy.signal.resample_poly (filtre FIR polyphase) pour
    eviter l'aliasing. 22050 → 16000 = facteur 3200/4410 = 320/441.
    """
    from scipy.signal import resample_poly
    # 16000/22050 = 320/441 (deja irreductible)
    return resample_poly(audio, up=320, down=441).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# Synthese TTS
# ══════════════════════════════════════════════════════════════════════════

SPEAKERS = ["siwis", "ezwa", "nadine", "bernard", "gilles", "zeckou"]


class TTSEngine:
    """Wrapper autour de lectura-tts-multispeaker."""

    def __init__(self):
        self._engine = None

    def load(self):
        if self._engine is not None:
            return
        from lectura_tts_multispeaker import creer_engine
        self._engine = creer_engine(mode="local")
        log.info("TTS engine charge (multispeaker, %d voix)", len(SPEAKERS))

    def synthesize(
        self,
        text: str,
        speaker: str = "siwis",
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
    ) -> np.ndarray:
        """Synthetise audio float32 mono @ 22050 Hz."""
        self._engine.set_speaker(speaker)
        result = self._engine.synthesize(
            text,
            duration_scale=duration_scale,
            pitch_shift=pitch_shift,
            pitch_range=pitch_range,
            energy_scale=energy_scale,
        )
        return result.samples


# ══════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AugmentationConfig:
    """Configuration d'une augmentation audio."""
    name: str
    speaker: str = "siwis"
    duration_scale: float = 1.0
    pitch_shift: float = 0.0
    pitch_range: float = 1.3
    energy_scale: float = 1.0
    add_noise: bool = False
    noise_snr_db: float = 20.0
    speed_factor: float = 1.0


def make_augmentations(n_aug: int, rng: random.Random) -> list[AugmentationConfig]:
    """Genere la liste des augmentations a appliquer.

    Chaque variante utilise une voix differente parmi les 6 disponibles.
    """
    voices = rng.sample(SPEAKERS, min(n_aug, len(SPEAKERS)))
    # Si n_aug > 6, completer avec des tirages aleatoires
    while len(voices) < n_aug:
        voices.append(rng.choice(SPEAKERS))

    augs = [AugmentationConfig(name="clean", speaker=voices[0])]

    if n_aug >= 2:
        augs.append(AugmentationConfig(
            name="prosody",
            speaker=voices[1],
            duration_scale=rng.uniform(0.85, 1.15),
            pitch_shift=rng.uniform(-1.0, 1.0),
            pitch_range=rng.uniform(1.0, 1.5),
            energy_scale=rng.uniform(0.8, 1.2),
        ))

    if n_aug >= 3:
        augs.append(AugmentationConfig(
            name="voice_noise",
            speaker=voices[2],
            add_noise=True,
            noise_snr_db=rng.uniform(15.0, 30.0),
            speed_factor=rng.uniform(0.9, 1.1),
        ))

    # Si n_aug > 3, ajouter des variations supplementaires
    for k in range(max(0, n_aug - 3)):
        augs.append(AugmentationConfig(
            name="extra",
            speaker=voices[3 + k],
            duration_scale=rng.uniform(0.8, 1.2),
            pitch_shift=rng.uniform(-1.5, 1.5),
            pitch_range=rng.uniform(0.9, 1.5),
            energy_scale=rng.uniform(0.7, 1.3),
            add_noise=rng.random() < 0.5,
            noise_snr_db=rng.uniform(12.0, 35.0),
            speed_factor=rng.uniform(0.85, 1.15),
        ))

    return augs


def run_pipeline(
    examples: list[FormulaExample],
    output_dir: Path,
    n_augmentations: int,
    seed: int,
    dry_run: bool = False,
    num_workers: int = 1,
) -> None:
    """Pipeline complet : TTS + augmentation + sauvegarde."""
    wav_dir = output_dir / "wavs"
    manifest_path = output_dir / "manifest.jsonl"
    vocab_path = output_dir / "vocab.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        wav_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le vocabulaire
    if not dry_run:
        with open(vocab_path, "w") as f:
            json.dump(vocab_to_json(), f, indent=2, ensure_ascii=False)
        log.info("Vocabulaire sauvegarde: %s", vocab_path)

    # Initialiser TTS
    tts: TTSEngine | None = None
    if not dry_run:
        tts = TTSEngine()
        tts.load()

    rng = random.Random(seed)
    entries: list[dict] = []
    errors = 0
    t0 = time.time()
    total_expected = len(examples) * n_augmentations

    for i, ex in enumerate(examples):
        augs = make_augmentations(n_augmentations, rng)

        for aug_idx, aug in enumerate(augs):
            idx = i * n_augmentations + aug_idx
            wav_name = f"{ex.category}_{i:05d}_{aug.name}.wav"
            wav_path = wav_dir / wav_name
            rel_wav_path = f"wavs/{wav_name}"

            duration_s = 0.0

            if not dry_run and tts is not None:
                try:
                    # Synthetiser
                    audio = tts.synthesize(
                        ex.result.display_fr,
                        speaker=aug.speaker,
                        duration_scale=aug.duration_scale,
                        pitch_shift=aug.pitch_shift,
                        pitch_range=aug.pitch_range,
                        energy_scale=aug.energy_scale,
                    )

                    # Resample 22050 → 16000
                    audio_16k = resample_22050_to_16000(audio)

                    # Augmentation vitesse
                    if abs(aug.speed_factor - 1.0) > 0.01:
                        audio_16k = speed_change(audio_16k, 16000, aug.speed_factor)

                    # Augmentation bruit
                    if aug.add_noise:
                        audio_16k = add_white_noise(audio_16k, aug.noise_snr_db)

                    duration_s = len(audio_16k) / 16000.0

                    # Verification duree raisonnable
                    if duration_s < 0.2 or duration_s > 20.0:
                        log.warning("Duree anormale %.1fs pour '%s', skip", duration_s, ex.text)
                        errors += 1
                        continue

                    # Sauvegarder WAV
                    import soundfile as sf
                    sf.write(str(wav_path), audio_16k, 16000)

                except Exception as e:
                    log.error("Erreur TTS pour '%s' (aug=%s): %s", ex.text, aug.name, e)
                    errors += 1
                    continue
            else:
                # Mode dry-run : estimer la duree
                duration_s = len(ex.token_ids) * 0.15

            entry = {
                "audio_path": rel_wav_path,
                "tokens": ex.token_ids,
                "tokens_str": " ".join(ex.token_names),
                "formula_type": ex.category,
                "formula_text": ex.text,
                "display_fr": ex.result.display_fr if ex.result else "",
                "duration_s": round(duration_s, 3),
                "augmentation": aug.name,
                "speaker": aug.speaker,
            }
            entries.append(entry)

        # Log progression
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            done = len(entries)
            rate = done / elapsed if elapsed > 0 else 0
            log.info("  %d/%d formules (%d entrees, %.1f/s, %d erreurs)",
                     i + 1, len(examples), done, rate, errors)

    # Sauvegarder le manifest JSONL
    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    total_duration = sum(e["duration_s"] for e in entries)

    log.info("=== Termine ===")
    log.info("  %d entrees generees, %d erreurs", len(entries), errors)
    log.info("  Duree audio totale: %.1f h", total_duration / 3600)
    log.info("  Manifest: %s", manifest_path)
    if not dry_run:
        log.info("  WAVs: %s", wav_dir)
        log.info("  Vocab: %s", vocab_path)
    log.info("  Temps de generation: %.1fs", elapsed)

    # Stats par categorie
    log.info("--- Distribution par categorie ---")
    cat_counts: dict[str, int] = {}
    for e in entries:
        cat_counts[e["formula_type"]] = cat_counts.get(e["formula_type"], 0) + 1
    for cat, count in sorted(cat_counts.items()):
        log.info("  %-14s %5d", cat, count)

    # Stats par augmentation
    log.info("--- Distribution par augmentation ---")
    aug_counts: dict[str, int] = {}
    for e in entries:
        aug_counts[e["augmentation"]] = aug_counts.get(e["augmentation"], 0) + 1
    for aug_name, count in sorted(aug_counts.items()):
        log.info("  %-14s %5d", aug_name, count)

    # Afficher quelques exemples
    log.info("--- Exemples ---")
    for e in entries[:10]:
        log.info("  %-25s → %s", e["formula_text"], e["tokens_str"])


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generateur de corpus pour le modele CTC formules"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Repertoire de sortie pour les WAVs et le manifest JSONL"
    )
    parser.add_argument(
        "--n-base", type=int, default=16000,
        help="Nombre d'exemples de base (default: 16000)"
    )
    parser.add_argument(
        "--n-augmentations", type=int, default=3,
        help="Nombre de variantes par exemple (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Nombre de workers paralleles (default: 1, pas encore implemente)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Genere les formules et tokens sans synthetiser l'audio"
    )
    args = parser.parse_args()

    log.info("Generation du corpus STT-Formules")
    log.info("  output-dir: %s", args.output_dir)
    log.info("  n-base: %d", args.n_base)
    log.info("  n-augmentations: %d", args.n_augmentations)
    log.info("  seed: %d", args.seed)
    log.info("  dry-run: %s", args.dry_run)

    # Phase 1 : generer les formules
    examples = generate_all_formulas(n_base=args.n_base, seed=args.seed)

    # Phase 2 : TTS + augmentation + sauvegarde
    run_pipeline(
        examples=examples,
        output_dir=args.output_dir,
        n_augmentations=args.n_augmentations,
        seed=args.seed,
        dry_run=args.dry_run,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
