"""DiphoneEngine — synthese par concatenation de diphones WORLD.

Pipeline v3_world :
  phones → diphone chain → WORLD params → stretch/concat → pw.synthesize → audio 44.1 kHz

Modes : FLUIDE, MOT_A_MOT, SYLLABES
"""

from __future__ import annotations

import logging
import pickle
import random
import time
from enum import Enum
from pathlib import Path

import numpy as np

from lectura_tts_diphone._compression import load_compressed
from lectura_tts_diphone._world import (
    FRAME_PERIOD, OVERLAP_FRAMES, SIWIS_SR,
    apply_timbre, compress_aperiodicity, concat_diphones,
    ensure_full_spectrum, sharpen_formants, stretch_params, synthesize,
    warp_vtln,
)

log = logging.getLogger(__name__)

MIN_STATS_N = 5  # minimum observations for corpus stats
_BASE_RATE = 1.2  # facteur interne : vitesse naturelle (applique avant duration_scale)

_VOWELS = set("aeiouyøœɑɔəɛɛ̃ɑ̃ɔ̃")

# Classification phonetique pour la microprosodie
_VOICELESS_STOPS = {"p", "t", "k"}
_VOICED_STOPS = {"b", "d", "g", "ɡ"}
_VOICELESS_FRICATIVES = {"f", "s", "ʃ", "x"}
_VOICED_FRICATIVES = {"v", "z", "ʒ", "ɣ"}
_NASALS = {"m", "n", "ɲ", "ŋ"}
_LIQUIDS = {"l", "ʁ", "ɹ"}
_GLIDES = {"j", "w", "ɥ"}

# Perturbation F0 (Hz) sur la voyelle suivant une consonne
_ONSET_F0_PERTURB: dict[str, int] = {
    "p": +12, "t": +12, "k": +12,           # occlusives sourdes
    "f": +10, "s": +10, "ʃ": +10, "x": +10, # fricatives sourdes
    "b": -6, "d": -6, "ɡ": -6, "g": -6,     # occlusives voisees
    "v": -5, "z": -5, "ʒ": -5, "ɣ": -5,     # fricatives voisees
    "m": -3, "n": -3, "ɲ": -3, "ŋ": -3,     # nasales
    "l": -2, "ʁ": -2, "ɹ": -2,              # liquides
    "j": 0, "w": 0, "ɥ": 0,                  # glides
}

# F0 intrinseque des voyelles (Hz, relatif a /a/)
_VOWEL_INTRINSIC_F0: dict[str, int] = {
    "i": +10, "y": +10, "u": +10,            # hautes
    "e": +5, "o": +5, "ø": +5,               # mi-hautes
    "ɛ": +3, "œ": +3, "ɔ": +3,               # mi-basses
    "ɛ̃": +3, "ɔ̃": +3, "ɑ̃": 0, "œ̃": +3,      # nasales
    "a": 0, "ɑ": 0, "ə": +2,                 # basses / schwa
}

# Normalisation des phones en entree (banque sans certains phones rares)
_PHONE_NORMALIZE = {"\u0153\u0303": "\u025b\u0303"}  # œ̃ → ɛ̃

# Chute F0 declarative — parametres absolus (en nombre de phones)
_DECL_FALL_PHONES = 7      # derniers phones avec chute acceleree
_DECL_SLOPE_HZ = 0.8       # pente douce par phone avant la zone de chute
_DECL_MAX_FALL_HZ = 28.0   # chute max dans la zone finale (avant macro_k)


def _smooth_noise(n: int, amplitude: float, sigma: float = 3.5) -> np.ndarray:
    """Bruit continu lisse pour micro-prosodie."""
    from scipy.ndimage import gaussian_filter1d

    if n <= 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([np.random.randn() * amplitude * 0.3])
    raw = np.random.randn(n) * amplitude
    return gaussian_filter1d(raw, sigma=min(sigma, max(1.0, n / 2)))


def _phones_to_syllables(
    phones: list[str], vowel_positions: list[int],
) -> list[tuple[int, int]]:
    """Decouper les phones en syllabes (une voyelle = un noyau).

    Chaque consonne inter-vocalique va en onset de la syllabe suivante
    (preference syllabe ouverte, typique du francais).
    """
    n = len(phones)
    if not vowel_positions:
        return [(0, n)]
    spans: list[tuple[int, int]] = []
    for si, vi in enumerate(vowel_positions):
        if si == 0:
            start = 0
        else:
            prev_vi = vowel_positions[si - 1]
            # Consonants entre deux voyelles : split au milieu
            # avec preference onset (ceil)
            gap = vi - prev_vi - 1
            split = prev_vi + 1 + gap // 2
            start = split
        if si == len(vowel_positions) - 1:
            end = n
        else:
            next_vi = vowel_positions[si + 1]
            gap = next_vi - vi - 1
            split = vi + 1 + gap // 2
            end = split
        spans.append((start, end))
    return spans


class SynthMode(str, Enum):
    SYLLABES = "SYLLABES"
    MOT_A_MOT = "MOT_A_MOT"
    FLUIDE = "FLUIDE"


class DiphoneEngine:
    """Synthese par concatenation de diphones dans le domaine WORLD."""

    def __init__(self) -> None:
        self.diphones: dict[str, dict] = {}
        self.diphone_stats: dict[str, dict] = {}
        self.pause_stats: dict[str, dict] = {}
        self.loaded = False

    def load(self, models_dir: str | Path | None = None) -> None:
        """Charge les diphones depuis le repertoire de modeles.

        Cherche diphones.dpk.gz (compresse) ou diphone_averaged.pkl (brut).
        """
        from lectura_tts_diphone._chargeur import find_models_dir

        resolved = find_models_dir(models_dir)
        if resolved is None:
            raise FileNotFoundError(
                "Modeles diphone introuvables. Verifiez l'installation ou "
                "specifiez models_dir."
            )

        t0 = time.time()

        # Essayer le format compresse d'abord
        dpk_path = resolved / "diphones.dpk.gz"
        dpk_enc_path = resolved / "diphones.dpk.gz.enc"
        pkl_path = resolved / "diphone_averaged.pkl"

        if dpk_path.exists():
            self.diphones = load_compressed(dpk_path)
        elif dpk_enc_path.exists():
            from lectura_tts_diphone._crypto import load_encrypted_model
            import gzip
            import io
            raw = load_encrypted_model(dpk_enc_path)
            with gzip.open(io.BytesIO(raw), "rb") as f:
                payload = pickle.load(f)
            # Decompresser comme load_compressed
            compressed = payload["diphones"]
            diphones = {}
            for di_key, entry in compressed.items():
                sp = np.exp(entry["log_sp"].astype(np.float64))
                ap = entry["ap_u8"].astype(np.float64) / 255.0
                f0 = entry["f0"].astype(np.float64)
                diphones[di_key] = {
                    "f0": f0, "sp": sp, "ap": ap,
                    "sr": entry.get("sr", SIWIS_SR),
                    "frame_period": entry.get("frame_period", FRAME_PERIOD),
                    "n_frames": len(f0),
                }
            self.diphones = diphones
        elif pkl_path.exists():
            with open(pkl_path, "rb") as f:
                self.diphones = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Aucun fichier diphone trouve dans {resolved}"
            )

        log.info("Charge %d diphones (%.1fs)", len(self.diphones), time.time() - t0)

        # Charger les statistiques prosodiques (optionnel)
        stats_path = resolved / "diphone_statistics.pkl"
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                stats_data = pickle.load(f)
            self.diphone_stats = stats_data.get("diphone_stats", {})
            self.pause_stats = stats_data.get("pause_stats", {})
            log.info("Charge stats corpus: %d diphones", len(self.diphone_stats))

        self.loaded = True

    # ── Diphone chain ──────────────────────────────────────────────────

    @staticmethod
    def build_diphone_chain(phones: list[str]) -> list[str]:
        """Build diphone key chain: ['b','a'] → ['#-b', 'b-a', 'a-#']."""
        if not phones:
            return []
        chain = [f"#-{phones[0]}"]
        for i in range(len(phones) - 1):
            chain.append(f"{phones[i]}-{phones[i+1]}")
        chain.append(f"{phones[-1]}-#")
        return chain

    @staticmethod
    def _segment_aps(phones: list[str],
                     word_boundaries: list[int]) -> list[tuple[int, int]]:
        """Segmenter les phones en Phrases Accentuelles (AP).

        Retourne une liste de (start, end) indices dans phones[].
        Chaque AP contient 1+ mots. Les mots courts (<=2 phones)
        fusionnent avec le mot suivant.
        """
        n = len(phones)
        if n == 0:
            return []

        # Decouper en mots d'apres word_boundaries
        sorted_wb = sorted(set(word_boundaries))
        # word_boundaries = indices de debut de mot (sauf le premier mot implicite)
        word_starts = [0] + [wb for wb in sorted_wb if 0 < wb < n]
        words: list[tuple[int, int]] = []
        for wi in range(len(word_starts)):
            start = word_starts[wi]
            end = word_starts[wi + 1] if wi + 1 < len(word_starts) else n
            words.append((start, end))

        if not words:
            return [(0, n)]

        # Fusionner les mots courts (<=2 phones) avec le mot suivant
        aps: list[tuple[int, int]] = []
        i = 0
        while i < len(words):
            ap_start = words[i][0]
            ap_end = words[i][1]
            word_len = ap_end - ap_start
            # Mot court: fusionner avec le(s) suivant(s)
            while word_len <= 2 and i + 1 < len(words):
                i += 1
                ap_end = words[i][1]
                word_len = ap_end - ap_start
            aps.append((ap_start, ap_end))
            i += 1

        return aps

    @staticmethod
    def _st_to_hz(base_hz: float, semitones: float) -> float:
        """Convertir un ecart en demi-tons en Hz."""
        return base_hz * (2.0 ** (semitones / 12.0))

    # ── Duration computation ───────────────────────────────────────────

    def compute_phone_durations(self, phones: list[str], mode: SynthMode,
                                accent_positions: set[int] | None = None,
                                ) -> list[float]:
        """Compute target duration per phone in ms (rule-based fallback).

        Args:
            phones: list of IPA phones
            mode: synthesis mode
            accent_positions: set of phone indices that carry AP-final accent
                (vowels get longer durations at these positions)
        """
        vowels = set("aeiouyøœɑɔəɛɛ̃ɑ̃ɔ̃")

        durations = []
        for i, ph in enumerate(phones):
            base = ph[0] if ph else ""
            is_vowel = base in vowels

            if mode == SynthMode.SYLLABES:
                durations.append(350.0 if is_vowel else 80.0)
            elif mode == SynthMode.MOT_A_MOT:
                durations.append(200.0 if is_vowel else 60.0)
            else:  # FLUIDE
                if is_vowel:
                    if accent_positions and i in accent_positions:
                        durations.append(130.0)  # voyelle accentuee AP-finale
                    else:
                        durations.append(90.0)   # voyelle non accentuee
                else:
                    durations.append(60.0)

        # Final lengthening
        if durations and mode != SynthMode.SYLLABES:
            factor = 1.5 if mode == SynthMode.MOT_A_MOT else 1.3
            durations[-1] *= factor

        return durations

    def _phone_durs_to_diphone_durs(self, chain: list[str],
                                     phone_durs: list[float]) -> list[float]:
        """Convert per-phone durations to per-diphone durations."""
        di_durs = []
        for di_idx, di_key in enumerate(chain):
            if di_key.startswith("#-"):
                di_durs.append(phone_durs[0] / 2)
            elif di_key.endswith("-#"):
                # Plancher 80ms pour laisser la consonne finale resonner
                di_durs.append(max(phone_durs[-1] / 2, 80.0))
            else:
                a_idx = di_idx - 1
                b_idx = di_idx
                dur_a = phone_durs[a_idx] if a_idx < len(phone_durs) else 60.0
                dur_b = phone_durs[b_idx] if b_idx < len(phone_durs) else 60.0
                di_durs.append(dur_a / 2 + dur_b / 2)
        return di_durs

    def compute_durations_from_stats(self, chain: list[str], phones: list[str],
                                      mode: SynthMode,
                                      variability: float = 0.4) -> list[float]:
        """Compute diphone durations using corpus statistics."""
        if mode == SynthMode.SYLLABES:
            phone_durs = self.compute_phone_durations(phones, mode)
            return self._phone_durs_to_diphone_durs(chain, phone_durs)

        fallback_phone_durs = self.compute_phone_durations(phones, mode)
        fallback_di_durs = self._phone_durs_to_diphone_durs(chain, fallback_phone_durs)

        di_durs = []
        for di_idx, di_key in enumerate(chain):
            stats = self.diphone_stats.get(di_key)
            template = self.diphones.get(di_key)

            if stats and stats["n"] >= MIN_STATS_N and template:
                tpl_frames = template.get("n_frames", len(template["f0"]))
                tpl_ms = tpl_frames * FRAME_PERIOD

                dr_med = stats["dur_ratio_median"]
                dr_p25 = stats["dur_ratio_p25"]
                dr_p75 = stats["dur_ratio_p75"]
                if variability > 0 and dr_p75 > dr_p25:
                    iqr_half = (dr_p75 - dr_p25) / 2 * variability
                    dur_ratio = dr_med + np.random.uniform(-iqr_half, iqr_half)
                else:
                    dur_ratio = dr_med

                dur_ms = tpl_ms * max(0.2, dur_ratio)
                dur_ms = max(15.0, min(dur_ms, 500.0))
                di_durs.append(dur_ms)
            else:
                di_durs.append(fallback_di_durs[di_idx])

        return di_durs

    # ── F0 contour ─────────────────────────────────────────────────────

    def compute_f0_targets(self, phones: list[str], mode: SynthMode,
                            group_info: dict | None = None,
                            word_boundaries: list[int] | None = None,
                            ) -> list[float]:
        """Compute F0 target per phone."""
        n = len(phones)
        if n == 0:
            return []

        if mode == SynthMode.SYLLABES:
            return [175.0] * n

        if group_info is not None:
            return self._group_f0_contour(
                phones, mode, group_info,
                word_boundaries=word_boundaries or [])

        # Fallback: simple declination
        f0s = []
        for i in range(n):
            pos = i / max(1, n - 1)
            if mode == SynthMode.MOT_A_MOT:
                f0s.append(175.0 - 15.0 * pos)
            else:
                f0 = 175.0 - 30.0 * pos
                if i == n - 1:
                    f0 = 160.0
                f0s.append(f0)
        return f0s

    @classmethod
    def _group_f0_contour(cls, phones: list[str], mode: SynthMode,
                           info: dict,
                           word_boundaries: list[int] | None = None,
                           ) -> list[float]:
        """Dispatch vers le style prosodique demande."""
        style = info.get("prosody_style", "regles")
        if style == "corpus":
            return cls._group_f0_contour_corpus(phones, mode, info)
        return cls._group_f0_contour_regles(phones, mode, info, word_boundaries)

    @classmethod
    def _group_f0_contour_regles(cls, phones: list[str], mode: SynthMode,
                                  info: dict,
                                  word_boundaries: list[int] | None = None,
                                  ) -> list[float]:
        """French prosodic F0 contour based on Accentual Phrases (AP).

        Uses the LHiLH* model: each AP gets a tonal pattern with
        low onset and high AP-final accent. The last AP in the group
        receives the terminal contour (declarative fall, question rise, etc.).

        macro_expressivity controls amplitude:
            0 = flat (no excursions)
            1 = normal (default, ~3 st range)
            2 = exaggerated (~6 st range)
        """
        n = len(phones)
        gi = info.get("group_idx", 0)
        n_groups = info.get("n_groups", 1)
        boundary = info.get("boundary", "none")
        base_f0 = info.get("base_f0", 175.0)
        k = info.get("macro_expressivity", 1.0)

        is_sentence_final = (gi == n_groups - 1) or boundary in (
            "period", "question", "exclamation", "suspensive",
        )
        is_question = boundary == "question"
        is_exclamation = boundary == "exclamation"
        is_suspensive = boundary == "suspensive"

        # Segment into APs
        wb = word_boundaries if word_boundaries else []
        aps = cls._segment_aps(phones, wb)
        if not aps:
            return [base_f0] * n

        n_aps = len(aps)
        f0s = [base_f0] * n

        for ap_idx, (ap_start, ap_end) in enumerate(aps):
            is_last_ap = (ap_idx == n_aps - 1)

            # Global declination: -0.5 semitone per AP, capped at -2.0 st
            decl_st = max(-2.0, -0.5 * ap_idx)
            ap_base = cls._st_to_hz(base_f0, decl_st)

            # Count syllables in this AP (each vowel = 1 nucleus)
            vowel_positions = []
            for j in range(ap_start, ap_end):
                base_ch = phones[j][0] if phones[j] else ""
                if base_ch in _VOWELS:
                    vowel_positions.append(j)
            n_syl = len(vowel_positions)

            if n_syl == 0:
                for j in range(ap_start, ap_end):
                    f0s[j] = ap_base
                continue

            # Build tonal targets per syllable (in semitones relative to ap_base)
            syl_targets_st: list[float] = []
            if n_syl <= 2:
                # LH*: L=-1.5st, H*=+2st
                for si in range(n_syl):
                    t = si / max(1, n_syl - 1)
                    syl_targets_st.append((-1.5 + 3.5 * t) * k)
            elif n_syl <= 4:
                # LLH*: low body, rise on last
                for si in range(n_syl):
                    if si < n_syl - 1:
                        t = si / max(1, n_syl - 2)
                        syl_targets_st.append((-1.5 + 0.5 * t) * k)
                    else:
                        syl_targets_st.append(2.5 * k)
            else:
                # LHiLH*
                for si in range(n_syl):
                    if si == 0:
                        syl_targets_st.append(-1.0 * k)
                    elif si == 1:
                        syl_targets_st.append(1.5 * k)  # Hi
                    elif si < n_syl - 1:
                        t = (si - 1) / max(1, n_syl - 3)
                        syl_targets_st.append((-0.5 + 0.3 * t) * k)
                    else:
                        syl_targets_st.append(2.5 * k)  # H*

            # Terminal contour on the last AP
            if is_last_ap and is_sentence_final:
                if is_question:
                    syl_targets_st[-1] = 4.0 * k
                    if n_syl >= 2:
                        syl_targets_st[-2] = min(syl_targets_st[-2], -0.5 * k)
                elif is_exclamation:
                    if n_syl >= 2:
                        syl_targets_st[0] = max(syl_targets_st[0], 2.0 * k)
                    syl_targets_st[-1] = -3.0 * k
                elif is_suspensive:
                    syl_targets_st[-1] = -0.5 * k
                else:
                    # Declarative: fall on last syllable(s)
                    syl_targets_st[-1] = -3.0 * k
                    if n_syl >= 2:
                        syl_targets_st[-2] = min(syl_targets_st[-2], -0.5 * k)

            # Map syllable targets to phone positions
            for j in range(ap_start, ap_end):
                base_ch = phones[j][0] if phones[j] else ""
                is_vowel = base_ch in _VOWELS

                if is_vowel and j in vowel_positions:
                    si = vowel_positions.index(j)
                    st = syl_targets_st[si]
                    f0s[j] = cls._st_to_hz(ap_base, st)
                else:
                    # Consonant: interpolate from surrounding vowels
                    prev_vi = None
                    next_vi = None
                    for vi_idx, vp in enumerate(vowel_positions):
                        if vp <= j:
                            prev_vi = vi_idx
                        if vp > j and next_vi is None:
                            next_vi = vi_idx
                    if prev_vi is not None and next_vi is not None:
                        vp_prev = vowel_positions[prev_vi]
                        vp_next = vowel_positions[next_vi]
                        t = (j - vp_prev) / max(1, vp_next - vp_prev)
                        st = (syl_targets_st[prev_vi] * (1 - t)
                              + syl_targets_st[next_vi] * t)
                    elif next_vi is not None:
                        st = syl_targets_st[next_vi]
                    elif prev_vi is not None:
                        st = syl_targets_st[prev_vi]
                    else:
                        st = 0.0
                    f0s[j] = cls._st_to_hz(ap_base, st)

        return f0s

    @classmethod
    def _group_f0_contour_corpus(cls, phones: list[str], mode: SynthMode,
                                  info: dict) -> list[float]:
        """F0 contour depuis les clusters corpus SIWIS."""
        from lectura_tts_diphone._prosody_corpus import generate_corpus_prosody

        n = len(phones)
        base_f0 = info.get("base_f0", 200.0)
        k = info.get("macro_expressivity", 1.0)
        boundary = info.get("boundary", "none")
        gi = info.get("group_idx", 0)
        n_groups = info.get("n_groups", 1)

        # Determiner role et mode
        mode_map = {"period": "declaratif", "question": "question",
                    "exclamation": "exclamation", "suspensive": "suspensif",
                    "comma": "declaratif", "none": "declaratif"}
        mode_str = mode_map.get(boundary, "declaratif")

        if n_groups == 1:
            group_role = "seul"
        elif gi == 0:
            group_role = "initial"
        elif gi == n_groups - 1:
            group_role = "terminal"
        else:
            group_role = "medial"

        # Compter syllabes
        vowel_positions = [j for j in range(n)
                           if phones[j] and phones[j][0] in _VOWELS]
        n_syl = len(vowel_positions)
        if n_syl == 0:
            return [base_f0] * n

        rng = random.Random(hash((gi, n_groups, n)))
        prosody = generate_corpus_prosody(
            n_syl, mode_str, rng, base_f0,
            group_role=group_role, expressivity=k,
        )

        # Stocker pour reutiliser dans synthesize_phones (dur_ratio, energy)
        info["_corpus_prosody"] = prosody

        # Assigner F0 par syllabe (plat par syllabe)
        syllable_spans = _phones_to_syllables(phones, vowel_positions)
        f0s = [base_f0] * n
        for si, (syl_start, syl_end) in enumerate(syllable_spans):
            if si < len(prosody):
                for j in range(syl_start, syl_end):
                    f0s[j] = prosody[si]["f0_hz"]
        return f0s

    def apply_f0_contour(self, f0_s: np.ndarray, di_key: str,
                          f0_target: float,
                          variability: float = 0.6) -> np.ndarray:
        """Apply corpus-driven F0 modulation to a diphone segment."""
        voiced = f0_s > 0
        if not np.any(voiced) or f0_target <= 0:
            return f0_s

        f0_out = f0_s.copy()
        current_mean = np.mean(f0_out[voiced])
        if current_mean <= 0:
            return f0_out

        stats = self.diphone_stats.get(di_key)
        if stats and stats["n"] >= MIN_STATS_N:
            f0_ratio_med = stats["f0_ratio_median"]
            f0_ratio_p25 = stats.get("f0_ratio_p25", f0_ratio_med)
            f0_ratio_p75 = stats.get("f0_ratio_p75", f0_ratio_med)

            if variability > 0 and f0_ratio_p75 > f0_ratio_p25:
                iqr_half = (f0_ratio_p75 - f0_ratio_p25) / 2 * variability
                f0_ratio = f0_ratio_med + np.random.uniform(-iqr_half, iqr_half)
            else:
                f0_ratio = f0_ratio_med

            corpus_f0 = current_mean * f0_ratio
            blended_f0 = 0.45 * corpus_f0 + 0.55 * f0_target
            f0_out[voiced] *= blended_f0 / current_mean

            slope_med = stats["f0_slope_median"]
            slope_p25 = stats.get("f0_slope_p25", slope_med)
            slope_p75 = stats.get("f0_slope_p75", slope_med)

            if variability > 0 and slope_p75 > slope_p25:
                iqr_half = (slope_p75 - slope_p25) / 2 * variability
                slope = slope_med + np.random.uniform(-iqr_half, iqr_half)
            else:
                slope = slope_med

            n_voiced = int(np.sum(voiced))
            if n_voiced > 1 and abs(slope) > 1.0:
                slope = max(-100.0, min(100.0, slope))
                dur_s = len(f0_s) * FRAME_PERIOD / 1000.0
                total_delta = slope * dur_s
                ramp = np.linspace(-total_delta / 2, total_delta / 2, n_voiced)
                f0_out[voiced] += ramp

            f0_out[voiced] = np.maximum(f0_out[voiced], 50.0)
        else:
            f0_out[voiced] *= f0_target / current_mean

        return f0_out

    # ── Pause duration ─────────────────────────────────────────────────

    def _get_pause_ms(self, boundary: str) -> float:
        """Get pause duration in ms for a boundary type."""
        if self.pause_stats and boundary in self.pause_stats:
            return self.pause_stats[boundary]["pause_ms"]
        defaults = {"comma": 50.0, "period": 550.0, "question": 550.0,
                     "exclamation": 450.0, "suspensive": 600.0,
                     "word": 80.0, "none": 60.0}
        return defaults.get(boundary, 60.0)

    # ── Main synthesis ─────────────────────────────────────────────────

    # Styles prosodiques valides pour prosody_style
    _PROSODY_STYLES = {"regles", "corpus"}

    # Frontieres qui marquent une fin de phrase
    _SENTENCE_BOUNDARIES = {"period", "exclamation", "question", "suspensive"}

    def synthesize_groups(
        self,
        groups: list[dict],
        mode: str | SynthMode = SynthMode.FLUIDE,
        prosody: dict | None = None,
        duration_scale: float = 1.0,
        pause_scale: float = 1.0,
        macro_expressivity: float = 1.0,
        micro_expressivity: float = 1.0,
        seed: int | None = None,
        prosody_style: str = "regles",
        spectral_contrast: float = 1.3,
        ap_cleanup: float = 1.5,
        formant_sharpening: float = 1.3,
        vtln_alpha: float = 1.0,
        timbre: str | None = None,
        base_f0: float = 175.0,
        sentence_pause_ms: float = 400.0,
        # -- Retimbre (OpenVoice zero-shot) --
        voix: str | Path | list[str] | dict[str, float] | None = None,
        voix_variante: float = 0.0,
        voix_tau: float = 0.3,
        vc_models_dir: str | Path | None = None,
    ) -> np.ndarray:
        """Synthesize multiple prosodic groups with inter-group pauses.

        Le texte est decoupes en phrases aux frontieres terminales
        (period, exclamation, question, suspensive). Chaque phrase est
        synthetisee independamment avec sa propre prosodie, puis les
        audios sont concatenes avec une pause inter-phrase controlee.

        Args:
            groups: list of dicts with keys:
                - "phones": list[str] — phones in the group
                - "boundary": str — boundary type after this group
            mode: "FLUIDE", "MOT_A_MOT", "SYLLABES" (str or SynthMode)
            prosody: optional global prosody dict
            duration_scale: speed factor (>1 = slower)
            pause_scale: scale factor for inter-group pauses
            macro_expressivity: facteur d'amplification des gestes prosodiques
                (F0 + allongement). 0=plat, 1=normal, 2=exagere.
            micro_expressivity: facteur d'amplification des micro-variations
                (F0, duree, energie). 0=robot, 1=normal, 2=tres expressif.
                Actif en mode FLUIDE uniquement.
            seed: graine aleatoire pour la micro-prosodie. None = aleatoire
                a chaque appel. Meme seed = meme resultat reproductible.
            prosody_style: style prosodique a utiliser.
                "regles" (defaut) : Prosodie a base de regles.
                    Modele LHiLH* avec phrases accentuelles (AP), declination
                    lineaire, contours terminaux par type de phrase.
                    Prosodie stable et previsible.
                "corpus" : Prosodie extraite du corpus SIWIS.
                    Clusters multi-parametres (F0 + duree + energie) issus de
                    9750 phrases. Gestion des groupes prosodiques (virgules).
                    Prosodie plus variee et naturelle.
            spectral_contrast: compensation de variance spectrale (GV).
                1.0=pas de changement, 1.3=naturel, 2.0=fort. Restaure le
                detail spectral perdu par le moyennage des diphones.
            ap_cleanup: compression power-law de l'aperiodicite (1.0=off,
                1.5=defaut, max 3.0). Reduit la raucite du moyennage.
            formant_sharpening: affutage cepstral des formants (1.0=off,
                1.3=defaut, max 2.0). Restaure la nettete spectrale.
            vtln_alpha: warping VTLN du tract vocal (0.8=grave/sombre,
                1.0=neutre, 1.2=aigu/brillant).
            timbre: nom de signature de timbre (ex: "homme", "enfant") ou
                chemin vers un fichier .json. None = pas de transfert de timbre.
            base_f0: pitch de base en Hz (defaut 175.0). Ajuste le F0 de
                reference pour toute la synthese (homme ~120, femme ~200,
                enfant ~280).
            sentence_pause_ms: pause inter-phrase en ms (defaut 400).
                Controle la duree du silence entre deux phrases.
                Les pauses intra-phrase (virgules, etc.) restent gerees
                par pause_scale.
            voix: voix cible pour retimbre OpenVoice. Polymorphe :
                - str : nom de preset ("siwis") ou chemin fichier audio.
                - list[str] : plusieurs references (poids egaux).
                - dict[str, float] : blend pondere.
                  Ex: {"siwis": 0.5, "nadine": 0.3, "ezwa": 0.2}
                None = pas de retimbre.
                Requires: pip install 'lectura-tts-diphone[vc]'
            voix_variante: curseur de variante vocale (-1 a +1).
                -1 = grave/masculin, 0 = neutre, +1 = aigu/enfant.
                Decale les formants via le trick SR OpenVoice.
            voix_tau: parametre tau d'OpenVoice (0 = deterministe).
            vc_models_dir: repertoire des modeles VC (defaut: auto-detection).

        Returns:
            np.float32 audio array at 44100 Hz
        """
        if seed is not None:
            np.random.seed(seed)

        if not self.loaded:
            self.load()

        if isinstance(mode, str):
            mode = SynthMode(mode)

        if prosody_style not in self._PROSODY_STYLES:
            raise ValueError(
                f"prosody_style invalide: {prosody_style!r}. "
                f"Valeurs possibles: {sorted(self._PROSODY_STYLES)}"
            )

        # Charger la signature de timbre si demandee
        timbre_signature = None
        if timbre is not None:
            from lectura_tts_diphone._timbre import load_signature
            timbre_signature = load_signature(timbre)

        if not groups:
            return np.array([], dtype=np.float32)

        # -- Decouper les groupes en phrases --
        sentences = self._split_into_sentences(groups)

        synth_kwargs = dict(
            mode=mode, prosody=prosody, duration_scale=duration_scale,
            pause_scale=pause_scale, macro_expressivity=macro_expressivity,
            micro_expressivity=micro_expressivity,
            prosody_style=prosody_style,
            spectral_contrast=spectral_contrast, ap_cleanup=ap_cleanup,
            formant_sharpening=formant_sharpening, vtln_alpha=vtln_alpha,
            timbre_signature=timbre_signature, base_f0=base_f0,
        )

        sentence_audios = []
        for sent_groups in sentences:
            audio = self._synthesize_sentence(sent_groups, **synth_kwargs)
            if len(audio) > 0:
                sentence_audios.append(audio)

        if not sentence_audios:
            return np.array([], dtype=np.float32)

        # -- Concatener les phrases avec pause inter-phrase --
        n_pause = int(sentence_pause_ms * pause_scale / 1000.0 * SIWIS_SR)
        pause = np.zeros(n_pause, dtype=np.float32) if n_pause > 0 else None

        parts = []
        for i, audio in enumerate(sentence_audios):
            parts.append(audio)
            if i < len(sentence_audios) - 1 and pause is not None:
                parts.append(pause)

        combined = np.concatenate(parts)
        combined = np.clip(combined, -0.95, 0.95)

        # -- Retimbre optionnel (OpenVoice zero-shot) --
        if voix is not None:
            combined = self._apply_retimbre(
                combined, SIWIS_SR, voix, voix_variante, voix_tau,
                vc_models_dir,
            )

        return combined

    def _split_into_sentences(
        self, groups: list[dict],
    ) -> list[list[dict]]:
        """Decoupe les groupes en phrases aux frontieres terminales."""
        sentences: list[list[dict]] = []
        current: list[dict] = []
        for group in groups:
            current.append(group)
            boundary = group.get("boundary", "none")
            if boundary in self._SENTENCE_BOUNDARIES:
                sentences.append(current)
                current = []
        # Groupes restants (phrase sans ponctuation finale)
        if current:
            sentences.append(current)
        return sentences

    def _synthesize_sentence(
        self,
        groups: list[dict],
        mode: SynthMode,
        prosody: dict | None,
        duration_scale: float,
        pause_scale: float,
        macro_expressivity: float,
        micro_expressivity: float,
        prosody_style: str,
        spectral_contrast: float,
        ap_cleanup: float,
        formant_sharpening: float,
        vtln_alpha: float,
        timbre_signature,
        base_f0: float,
    ) -> np.ndarray:
        """Synthetise une phrase (liste de groupes prosodiques)."""
        n_groups = len(groups)
        base_f0_start = base_f0

        audio_parts = []
        for gi, group in enumerate(groups):
            phones = group["phones"]
            if not phones:
                continue

            boundary = group.get("boundary", "none")

            group_pos = gi / max(1, n_groups - 1)
            cur_f0 = base_f0_start - 10.0 * group_pos

            # Exclamatives : base F0 relevee (+20%) + boost micro (+50%)
            if boundary == "exclamation":
                cur_f0 *= 1.20
                group_micro = micro_expressivity * 1.5
            else:
                group_micro = micro_expressivity

            group_info = {
                "group_idx": gi,
                "n_groups": n_groups,
                "boundary": boundary,
                "base_f0": cur_f0,
                "macro_expressivity": macro_expressivity,
                "micro_expressivity": group_micro,
                "prosody_style": prosody_style,
            }

            word_boundaries = group.get("word_boundaries", [])
            audio = self.synthesize_phones(
                phones, mode=mode, prosody=prosody,
                use_corpus_stats=True,
                group_info=group_info,
                duration_scale=duration_scale,
                word_boundaries=word_boundaries,
                spectral_contrast=spectral_contrast,
                ap_cleanup=ap_cleanup,
                formant_sharpening=formant_sharpening,
                vtln_alpha=vtln_alpha,
                timbre_signature=timbre_signature,
            )
            audio_parts.append(audio)

            # Pauses intra-phrase (virgules, etc.) — pas aux frontieres
            # de fin de phrase (gerees par sentence_pause_ms)
            if gi < n_groups - 1:
                if boundary not in self._SENTENCE_BOUNDARIES:
                    pause_ms = self._get_pause_ms(boundary) * pause_scale
                    phrase_pos = gi / max(1, n_groups - 1)
                    pause_ms *= 1.0 + 0.15 * phrase_pos * macro_expressivity
                    n_silence = int(pause_ms / 1000.0 * SIWIS_SR)
                    if n_silence > 0:
                        audio_parts.append(
                            np.zeros(n_silence, dtype=np.float32))

        if not audio_parts:
            return np.array([], dtype=np.float32)

        # ── Egalisation RMS inter-groupes ──────────────────────────
        speech_indices = [i for i, p in enumerate(audio_parts)
                          if np.any(p != 0)]
        if len(speech_indices) > 1:
            rms_vals = {}
            for idx in speech_indices:
                r = np.sqrt(np.mean(audio_parts[idx] ** 2))
                if r > 0:
                    rms_vals[idx] = r
            if rms_vals:
                target_rms = max(rms_vals.values())
                for idx, rms in rms_vals.items():
                    gain = target_rms / rms
                    if gain > 1.01:
                        audio_parts[idx] = (audio_parts[idx] * gain).astype(
                            np.float32)

        # Silence final pour laisser la derniere consonne resonner
        tail_ms = 80.0
        audio_parts.append(np.zeros(int(tail_ms / 1000.0 * SIWIS_SR),
                                     dtype=np.float32))

        combined = np.concatenate(audio_parts)
        return combined

    # -- Retimbre helper ─────────────────────────────────────────────────

    _retimbre = None  # RetimbreEngine (lazy, instance-level)

    def _apply_retimbre(
        self,
        audio: np.ndarray,
        sr: int,
        voix: str | Path | list[str] | dict[str, float],
        voix_variante: float,
        voix_tau: float,
        vc_models_dir: str | Path | None,
    ) -> np.ndarray:
        """Applique le retimbre OpenVoice puis resample vers SIWIS_SR."""
        from lectura_tts_diphone._retimbre import RetimbreEngine
        import librosa

        if self._retimbre is None:
            self._retimbre = RetimbreEngine(vc_models_dir=vc_models_dir)

        retimbre_audio, retimbre_sr = self._retimbre.retimbre(
            audio, sr, voix,
            variante=voix_variante, tau=voix_tau,
        )

        # Resample de OV_SR (22050) vers SIWIS_SR (44100)
        if retimbre_sr != SIWIS_SR:
            retimbre_audio = librosa.resample(
                retimbre_audio, orig_sr=retimbre_sr, target_sr=SIWIS_SR,
            )

        # Re-normalize peak a 0.9
        peak = np.max(np.abs(retimbre_audio))
        if peak > 0:
            retimbre_audio = (retimbre_audio * 0.9 / peak).astype(np.float32)

        return retimbre_audio

    def synthesize_phones(
        self,
        phones: list[str],
        mode: str | SynthMode = SynthMode.SYLLABES,
        prosody: dict | None = None,
        use_corpus_stats: bool = False,
        group_info: dict | None = None,
        duration_scale: float = 1.0,
        word_boundaries: list[int] | None = None,
        spectral_contrast: float = 1.0,
        ap_cleanup: float = 1.0,
        formant_sharpening: float = 1.0,
        vtln_alpha: float = 1.0,
        timbre_signature: np.ndarray | None = None,
    ) -> np.ndarray:
        """Synthesize from a phone list using diphone chain.

        Args:
            phones: list of IPA phones
            mode: synthesis mode
            prosody: optional global prosody dict
            use_corpus_stats: use corpus statistics for duration/F0
            group_info: group position for phrase-level F0 contour
            duration_scale: speed factor (>1 = slower)
            word_boundaries: phone indices for word starts (micro-pauses)
            spectral_contrast: GV compensation factor (1.0=off, 1.3=default)
            ap_cleanup: AP compression factor (1.0=off, 1.5=default)
            formant_sharpening: cepstral formant sharpening (1.0=off, 1.3=default)
            vtln_alpha: VTLN warping factor (0.8=grave, 1.0=neutral, 1.2=aigu)
            timbre_signature: vecteur cepstral de timbre cible (None = pas de transfert)

        Returns:
            np.float32 audio array at 44100 Hz
        """
        if not self.loaded:
            self.load()

        if isinstance(mode, str):
            mode = SynthMode(mode)

        # Normaliser les phones (ex: œ̃ → ɛ̃)
        if _PHONE_NORMALIZE:
            phones = [_PHONE_NORMALIZE.get(p, p) for p in phones]

        chain = self.build_diphone_chain(phones)
        n = len(phones)

        # Expressivity from group_info
        micro_k = 0.0
        macro_k = 1.0
        if group_info is not None:
            micro_k = group_info.get("micro_expressivity", 0.0)
            macro_k = group_info.get("macro_expressivity", 1.0)
            # macro_k est utilise directement (1.0 = normal)

        # ── AP segmentation + accent positions ──
        wb = word_boundaries if word_boundaries else []
        aps = self._segment_aps(phones, wb)
        accent_positions: set[int] = set()
        for ap_start, ap_end in aps:
            # Derniere voyelle de chaque AP = position accentuee
            for j in range(ap_end - 1, ap_start - 1, -1):
                base_ch = phones[j][0] if phones[j] else ""
                if base_ch in _VOWELS:
                    accent_positions.add(j)
                    break

        # ── Durations ──
        if use_corpus_stats and self.diphone_stats and mode != SynthMode.SYLLABES:
            diphone_durations = self.compute_durations_from_stats(chain, phones, mode)
        else:
            phone_durations = self.compute_phone_durations(
                phones, mode, accent_positions=accent_positions)
            # Micro jitter duree
            if mode == SynthMode.FLUIDE and n > 1 and micro_k > 0:
                dur_jitter = _smooth_noise(n, 1.0 * micro_k, sigma=2.0)
                phone_durations = [max(20.0, d * (1.0 + j))
                                   for d, j in zip(phone_durations, dur_jitter)]
            diphone_durations = self._phone_durs_to_diphone_durs(chain, phone_durations)

        # Facteur de base pour vitesse naturelle + facteur utilisateur
        rate = _BASE_RATE * duration_scale
        if rate != 1.0:
            diphone_durations = [d * rate for d in diphone_durations]

        # Pre-boundary lengthening
        if group_info is not None and len(diphone_durations) > 2:
            n_di = len(diphone_durations)
            boundary = group_info.get("boundary", "none")
            if boundary == "suspensive":
                lengthen_base = 1.4
                start_frac = 0.6
            elif boundary in ("period", "question", "exclamation"):
                lengthen_base = 1.15
                start_frac = 0.75
            elif boundary == "comma":
                lengthen_base = 1.10
                start_frac = 0.7
            else:
                lengthen_base = 1.1
                start_frac = 0.7
            lengthen_factor = 1.0 + (lengthen_base - 1.0) * macro_k
            start_idx = max(1, int(n_di * start_frac))
            for di in range(start_idx, n_di):
                progress = (di - start_idx) / max(1, n_di - 1 - start_idx)
                factor = 1.0 + (lengthen_factor - 1.0) * progress
                diphone_durations[di] *= factor

        # ── F0 targets ──
        f0_targets = self.compute_f0_targets(
            phones, mode, group_info=group_info, word_boundaries=wb)

        # Deterministic microprosody
        if mode == SynthMode.FLUIDE and n > 1 and micro_k > 0:
            # micro_k est utilise directement (1.0 = normal)
            perturb_scale = micro_k

            for i in range(n):
                base_ch = phones[i][0] if phones[i] else ""
                is_vowel = base_ch in _VOWELS

                if is_vowel:
                    # 1. Consonant onset perturbation
                    if i > 0:
                        prev_ph = phones[i - 1]
                        onset_perturb = _ONSET_F0_PERTURB.get(prev_ph, 0)
                        f0_targets[i] += onset_perturb * perturb_scale

                    # 2. Intrinsic vowel F0
                    intrinsic = _VOWEL_INTRINSIC_F0.get(phones[i], 0)
                    f0_targets[i] += intrinsic * perturb_scale

            # 3. Frame-level jitter (0.5%, unsmoothed)
            # Applied later at diphone level — here we just add per-phone jitter
            f0_jitter = np.random.randn(n) * 0.005
            f0_targets = [max(80.0, f * (1.0 + j * perturb_scale))
                          for f, j in zip(f0_targets, f0_jitter)]

        # Apply global prosody scaling
        if prosody is not None:
            if "f0_hz" in prosody and prosody["f0_hz"] > 0:
                avg_f0 = np.mean(f0_targets) if f0_targets else 190.0
                if avg_f0 > 0:
                    f0_scale = prosody["f0_hz"] / avg_f0
                    f0_targets = [f * f0_scale for f in f0_targets]
            if "duration_scale" in prosody and prosody["duration_scale"] > 0:
                dur_s = prosody["duration_scale"]
                diphone_durations = [d * dur_s for d in diphone_durations]

        # ── Corpus prosody — duration + energy ──
        corpus_prosody = group_info.get("_corpus_prosody") if group_info else None
        if corpus_prosody is not None and mode == SynthMode.FLUIDE:
            vowel_positions = [j for j in range(n)
                               if phones[j] and phones[j][0] in _VOWELS]
            n_syl = len(vowel_positions)
            if n_syl > 0:
                syllable_spans = _phones_to_syllables(phones, vowel_positions)
                # Duration: apply corpus dur_ratio per syllable
                phone_dur_factors = [1.0] * n
                for si, (syl_start, syl_end) in enumerate(syllable_spans):
                    if si < len(corpus_prosody):
                        dr = corpus_prosody[si].get("dur_ratio", 1.0)
                        for j in range(syl_start, syl_end):
                            phone_dur_factors[j] = dr
                # Map phone factors to diphone durations
                for di_idx, di_key in enumerate(chain):
                    if di_key.startswith("#-"):
                        diphone_durations[di_idx] *= phone_dur_factors[0]
                    elif di_key.endswith("-#"):
                        diphone_durations[di_idx] *= phone_dur_factors[-1]
                    else:
                        a_idx = di_idx - 1
                        b_idx = di_idx
                        fa = phone_dur_factors[a_idx] if a_idx < n else 1.0
                        fb = phone_dur_factors[b_idx] if b_idx < n else 1.0
                        diphone_durations[di_idx] *= (fa + fb) / 2
                # Energy: apply corpus energy per syllable
                energy_factors = [1.0] * n
                for si, (syl_start, syl_end) in enumerate(syllable_spans):
                    if si < len(corpus_prosody):
                        e = corpus_prosody[si].get("energy", 1.0)
                        for j in range(syl_start, syl_end):
                            energy_factors[j] = e
            else:
                energy_factors = [1.0] * n
        elif mode == SynthMode.FLUIDE and n > 1 and micro_k > 0:
            energy_noise = _smooth_noise(n, 0.5 * micro_k, sigma=2.5)
            energy_factors = [max(0.6, min(1.4, 1.0 + e)) for e in energy_noise]
        else:
            energy_factors = [1.0] * n

        # ── Build diphone segments ──
        import pyworld as pw
        diphone_segments = []

        for di_idx, di_key in enumerate(chain):
            diphone = self.diphones.get(di_key)

            if diphone is None:
                log.debug("  Missing diphone: %s", di_key)
                fft_size = pw.get_cheaptrick_fft_size(SIWIS_SR)
                n_bins = fft_size // 2 + 1
                n_silence = 4
                diphone_segments.append({
                    "f0": np.zeros(n_silence, dtype=np.float64),
                    "sp": np.full((n_silence, n_bins), 1e-10, dtype=np.float64),
                    "ap": np.ones((n_silence, n_bins), dtype=np.float64),
                    "key": di_key,
                })
                continue

            f0 = diphone["f0"].astype(np.float64)
            sp = diphone["sp"].astype(np.float64)
            ap = diphone["ap"].astype(np.float64)
            sr = diphone.get("sr", SIWIS_SR)

            sp, ap = ensure_full_spectrum(sp, ap, sr)

            target_ms = diphone_durations[di_idx]

            if di_key.startswith("#-"):
                target_f0 = f0_targets[0]
                e_factor = energy_factors[0]
            elif di_key.endswith("-#"):
                target_f0 = f0_targets[-1]
                e_factor = energy_factors[-1]
            else:
                a_idx = di_idx - 1
                b_idx = di_idx
                f0_a = f0_targets[a_idx] if a_idx < len(f0_targets) else 190.0
                f0_b = f0_targets[b_idx] if b_idx < len(f0_targets) else 190.0
                target_f0 = (f0_a + f0_b) / 2
                e_a = energy_factors[a_idx] if a_idx < len(energy_factors) else 1.0
                e_b = energy_factors[b_idx] if b_idx < len(energy_factors) else 1.0
                e_factor = (e_a + e_b) / 2

            n_target = max(4, int(target_ms / FRAME_PERIOD))
            f0_s, sp_s, ap_s = stretch_params(f0, sp, ap, n_target)

            # F0 scaling
            voiced = f0_s > 0
            if np.any(voiced) and target_f0 > 0:
                current_mean = np.mean(f0_s[voiced])
                if current_mean > 0:
                    f0_s[voiced] *= target_f0 / current_mean

            # Energy scaling
            if abs(e_factor - 1.0) > 0.001:
                sp_s = sp_s * e_factor

            diphone_segments.append({
                "f0": f0_s,
                "sp": sp_s,
                "ap": ap_s,
                "key": di_key,
            })

        if not diphone_segments:
            return np.array([], dtype=np.float32)

        f0_cat, sp_cat, ap_cat, boundaries = concat_diphones(diphone_segments)

        # ── Pipeline timbre ──────────────────────────────────────
        # Ordre : AP cleanup → formant sharpening → GV → timbre transfer → VTLN

        # 1. Compression AP : reduire le bruit d'aperiodicite
        if ap_cleanup > 1.0:
            ap_cat = compress_aperiodicity(ap_cat, gamma=ap_cleanup, sr=SIWIS_SR)

        # 2. Affutage formants : restaurer les pics spectraux
        if formant_sharpening > 1.0:
            sp_cat = sharpen_formants(sp_cat, gain=formant_sharpening)

        # 3. GV compensation : contraste spectral frame-a-frame
        if spectral_contrast > 1.0:
            log_sp = np.log(np.maximum(sp_cat, 1e-10))
            mean_log = np.mean(log_sp, axis=0, keepdims=True)
            sp_cat = np.exp(mean_log + (log_sp - mean_log) * spectral_contrast)

        # 4. Transfert de timbre : appliquer la signature cepstrale cible
        if timbre_signature is not None:
            # Adapter la taille de la signature aux bins spectraux
            n_bins = sp_cat.shape[1]
            if len(timbre_signature) >= n_bins:
                sig = timbre_signature[:n_bins]
            else:
                sig = np.zeros(n_bins, dtype=np.float64)
                sig[:len(timbre_signature)] = timbre_signature
            sp_cat = apply_timbre(sp_cat, sig)

        # 5. VTLN : ajustement fin de l'identite vocale
        if abs(vtln_alpha - 1.0) > 0.001:
            sp_cat = warp_vtln(sp_cat, alpha=vtln_alpha, sr=SIWIS_SR)

        audio = synthesize(f0_cat, sp_cat, ap_cat, SIWIS_SR, FRAME_PERIOD)

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = (audio * 0.9 / peak).astype(np.float32)

        return audio
