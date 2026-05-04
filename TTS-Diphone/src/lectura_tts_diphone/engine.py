"""DiphoneEngine — synthese par concatenation de diphones WORLD.

Pipeline v3_world :
  phones → diphone chain → WORLD params → stretch/concat → pw.synthesize → audio 44.1 kHz

Modes : FLUIDE, MOT_A_MOT, SYLLABES
"""

from __future__ import annotations

import logging
import pickle
import time
from enum import Enum
from pathlib import Path

import numpy as np

from lectura_tts_diphone._compression import load_compressed
from lectura_tts_diphone._world import (
    FRAME_PERIOD, OVERLAP_FRAMES, SIWIS_SR,
    concat_diphones, ensure_full_spectrum, stretch_params, synthesize,
)

log = logging.getLogger(__name__)

MIN_STATS_N = 5  # minimum observations for corpus stats


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

    # ── Duration computation ───────────────────────────────────────────

    def compute_phone_durations(self, phones: list[str], mode: SynthMode
                                ) -> list[float]:
        """Compute target duration per phone in ms (rule-based fallback)."""
        vowels = set("aeiouyøœɑɔəɛɛ̃ɑ̃ɔ̃")

        durations = []
        for ph in phones:
            base = ph[0] if ph else ""
            is_vowel = base in vowels

            if mode == SynthMode.SYLLABES:
                durations.append(350.0 if is_vowel else 80.0)
            elif mode == SynthMode.MOT_A_MOT:
                durations.append(200.0 if is_vowel else 60.0)
            else:  # FLUIDE
                durations.append(120.0 if is_vowel else 50.0)

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
                di_durs.append(phone_durs[-1] / 2)
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
                            group_info: dict | None = None) -> list[float]:
        """Compute F0 target per phone."""
        n = len(phones)
        if n == 0:
            return []

        if mode == SynthMode.SYLLABES:
            return [190.0] * n

        if group_info is not None:
            return self._group_f0_contour(phones, mode, group_info)

        # Fallback: simple declination
        f0s = []
        for i in range(n):
            pos = i / max(1, n - 1)
            if mode == SynthMode.MOT_A_MOT:
                f0s.append(195.0 - 20.0 * pos)
            else:
                f0 = 200.0 - 35.0 * pos
                if i == n - 1:
                    f0 = 160.0
                f0s.append(f0)
        return f0s

    @staticmethod
    def _group_f0_contour(phones: list[str], mode: SynthMode,
                           info: dict) -> list[float]:
        """French prosodic F0 contour for a prosodic group."""
        n = len(phones)
        gi = info.get("group_idx", 0)
        n_groups = info.get("n_groups", 1)
        boundary = info.get("boundary", "none")
        base_f0 = info.get("base_f0", 195.0)

        is_last = (gi == n_groups - 1)
        is_question = boundary == "question"
        is_exclamation = boundary == "exclamation"

        f0s = []
        for i in range(n):
            pos = i / max(1, n - 1)

            if is_last and is_question:
                if pos < 0.65:
                    offset = -5.0 * pos
                else:
                    offset = -3.0 + 30.0 * ((pos - 0.65) / 0.35)
            elif is_last and is_exclamation:
                offset = 10.0 - 30.0 * pos
            elif is_last:
                offset = -20.0 * pos
            else:
                offset = -3.0 + 12.0 * pos

            f0s.append(base_f0 + offset)

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
            blended_f0 = 0.6 * corpus_f0 + 0.4 * f0_target
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
                     "exclamation": 450.0, "word": 80.0, "none": 60.0}
        return defaults.get(boundary, 60.0)

    # ── Main synthesis ─────────────────────────────────────────────────

    def synthesize_groups(
        self,
        groups: list[dict],
        mode: str | SynthMode = SynthMode.FLUIDE,
        prosody: dict | None = None,
        duration_scale: float = 1.2,
        pause_scale: float = 1.2,
    ) -> np.ndarray:
        """Synthesize multiple prosodic groups with inter-group pauses.

        Args:
            groups: list of dicts with keys:
                - "phones": list[str] — phones in the group
                - "boundary": str — boundary type after this group
            mode: "FLUIDE", "MOT_A_MOT", "SYLLABES" (str or SynthMode)
            prosody: optional global prosody dict
            duration_scale: speed factor (>1 = slower)
            pause_scale: scale factor for inter-group pauses

        Returns:
            np.float32 audio array at 44100 Hz
        """
        if not self.loaded:
            self.load()

        if isinstance(mode, str):
            mode = SynthMode(mode)

        if not groups:
            return np.array([], dtype=np.float32)

        n_groups = len(groups)
        base_f0_start = 200.0

        audio_parts = []
        for gi, group in enumerate(groups):
            phones = group["phones"]
            if not phones:
                continue

            boundary = group.get("boundary", "none")
            group_pos = gi / max(1, n_groups - 1)
            base_f0 = base_f0_start - 20.0 * group_pos

            group_info = {
                "group_idx": gi,
                "n_groups": n_groups,
                "boundary": boundary,
                "base_f0": base_f0,
            }

            word_boundaries = group.get("word_boundaries", [])
            audio = self.synthesize_phones(
                phones, mode=mode, prosody=prosody,
                use_corpus_stats=True,
                group_info=group_info,
                duration_scale=duration_scale,
                word_boundaries=word_boundaries,
            )
            audio_parts.append(audio)

            if gi < n_groups - 1:
                pause_ms = self._get_pause_ms(boundary) * pause_scale
                n_silence = int(pause_ms / 1000.0 * SIWIS_SR)
                if n_silence > 0:
                    audio_parts.append(np.zeros(n_silence, dtype=np.float32))

        if not audio_parts:
            return np.array([], dtype=np.float32)

        combined = np.concatenate(audio_parts)
        peak = np.max(np.abs(combined))
        if peak > 0:
            combined = (combined * 0.9 / peak).astype(np.float32)

        return combined

    def synthesize_phones(
        self,
        phones: list[str],
        mode: str | SynthMode = SynthMode.SYLLABES,
        prosody: dict | None = None,
        use_corpus_stats: bool = False,
        group_info: dict | None = None,
        duration_scale: float = 1.0,
        word_boundaries: list[int] | None = None,
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

        Returns:
            np.float32 audio array at 44100 Hz
        """
        if not self.loaded:
            self.load()

        if isinstance(mode, str):
            mode = SynthMode(mode)

        chain = self.build_diphone_chain(phones)

        # Duration computation
        if use_corpus_stats and self.diphone_stats and mode != SynthMode.SYLLABES:
            diphone_durations = self.compute_durations_from_stats(chain, phones, mode)
        else:
            phone_durations = self.compute_phone_durations(phones, mode)
            diphone_durations = self._phone_durs_to_diphone_durs(chain, phone_durations)

        if duration_scale != 1.0:
            diphone_durations = [d * duration_scale for d in diphone_durations]

        # Pre-boundary lengthening
        if group_info is not None and len(diphone_durations) > 2:
            n_di = len(diphone_durations)
            boundary = group_info.get("boundary", "none")
            if boundary in ("period", "question", "exclamation"):
                lengthen_factor = 1.4
            elif boundary == "comma":
                lengthen_factor = 1.25
            else:
                lengthen_factor = 1.1
            start_idx = max(1, int(n_di * 0.7))
            for di in range(start_idx, n_di):
                progress = (di - start_idx) / max(1, n_di - 1 - start_idx)
                factor = 1.0 + (lengthen_factor - 1.0) * progress
                diphone_durations[di] *= factor

        f0_targets = self.compute_f0_targets(phones, mode, group_info=group_info)

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

        # Build diphone segments
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
            elif di_key.endswith("-#"):
                target_f0 = f0_targets[-1]
            else:
                a_idx = di_idx - 1
                b_idx = di_idx
                f0_a = f0_targets[a_idx] if a_idx < len(f0_targets) else 190.0
                f0_b = f0_targets[b_idx] if b_idx < len(f0_targets) else 190.0
                target_f0 = (f0_a + f0_b) / 2

            n_target = max(4, int(target_ms / FRAME_PERIOD))
            f0_s, sp_s, ap_s = stretch_params(f0, sp, ap, n_target)

            # F0 scaling
            voiced = f0_s > 0
            if np.any(voiced) and target_f0 > 0:
                current_mean = np.mean(f0_s[voiced])
                if current_mean > 0:
                    f0_s[voiced] *= target_f0 / current_mean

            diphone_segments.append({
                "f0": f0_s,
                "sp": sp_s,
                "ap": ap_s,
                "key": di_key,
            })

        if not diphone_segments:
            return np.array([], dtype=np.float32)

        f0_cat, sp_cat, ap_cat = concat_diphones(diphone_segments)
        audio = synthesize(f0_cat, sp_cat, ap_cat, SIWIS_SR, FRAME_PERIOD)

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = (audio * 0.9 / peak).astype(np.float32)

        return audio
