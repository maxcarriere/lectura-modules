#!/usr/bin/env python3
"""Test du generateur prosodique v2 sur nouvelles phrases.

Le generateur produit (duree_ms, f0_rel_st, pause_after_ms) par syllabe
a partir des distributions corpus SIWIS par zone × mode.

Sortie : /tmp/tts_prosody_v2/
"""

import sys
import json
import math

import numpy as np
import numpy.core.numeric

if not hasattr(np, '_core'):
    class _core: pass
    nc = _core()
    nc.numeric = numpy.core.numeric
    sys.modules['numpy._core'] = nc
    sys.modules['numpy._core.numeric'] = numpy.core.numeric

import soundfile as sf
from pathlib import Path
from random import Random

from lectura_tts_diphone import creer_engine
from lectura_tts_diphone.engine import (
    DiphoneEngine, SynthMode, _VOWELS, _phones_to_syllables,
    _PHONE_NORMALIZE, _VOICELESS_STOPS, _VOICED_STOPS,
    _VOICELESS_FRICATIVES, _VOICED_FRICATIVES, _NASALS, _LIQUIDS, _GLIDES,
)
from lectura_tts_diphone._world import (
    FRAME_PERIOD, OVERLAP_FRAMES, SIWIS_SR,
    compress_aperiodicity, concat_diphones, ensure_full_spectrum,
    sharpen_formants, stretch_params, synthesize,
)

OUT = Path("/tmp/tts_prosody_v2")
OUT.mkdir(exist_ok=True)

# ── Charger les modeles ───────────────────────────────────────────
with open("/tmp/prosody_model_v2.json") as f:
    MODEL = json.load(f)

with open("/tmp/group_contour_bank.json") as f:
    GROUP_BANK = json.load(f)

ZONES_ORDER = ["initial", "early", "mid", "late", "prefinal", "final"]


# ══════════════════════════════════════════════════════════════════
# Generateur prosodique v2
# ══════════════════════════════════════════════════════════════════

def _pos_to_zone(syl_idx: int, n_syl: int) -> str:
    """Determine la zone pour une syllabe."""
    if syl_idx == 0:
        return "initial"
    if syl_idx == n_syl - 1:
        return "final"
    if syl_idx == n_syl - 2:
        return "prefinal"
    pos = syl_idx / max(1, n_syl - 1)
    if pos <= 0.35:
        return "early"
    if pos <= 0.65:
        return "mid"
    return "late"


def _dur_cat(dur_ms: float) -> str:
    if dur_ms < 130:
        return "short"
    if dur_ms < 220:
        return "medium"
    return "long"


def _sample_from_transition(prev_cat: str, rng: Random) -> str:
    """Tire une categorie de duree depuis la matrice de transition."""
    trans = MODEL.get("dur_transitions", {}).get(prev_cat)
    if not trans:
        return "medium"
    r = rng.random()
    cumul = 0.0
    for cat, prob in trans.items():
        cumul += prob
        if r <= cumul:
            return cat
    return "medium"


def _sample_dur_in_cat(cat: str, zone_stats: dict, rng: Random) -> float:
    """Echantillonne une duree dans une categorie."""
    bounds = {"short": (60, 130), "medium": (130, 220), "long": (220, 500)}
    lo, hi = bounds[cat]

    # Utiliser la mediane de la zone comme ancre
    median = zone_stats["dur_median"]
    # Tirer autour de la mediane avec contrainte de categorie
    dur = median + rng.gauss(0, zone_stats["dur_std"] * 0.4)
    dur = max(lo, min(hi, dur))
    return dur


def _syl_bucket(n_syl: int) -> str:
    """Map syllable count to contour bank key suffix."""
    if n_syl <= 1:
        return "2"
    if n_syl <= 10:
        return str(n_syl)
    if n_syl <= 13:
        return "11-13"
    if n_syl <= 17:
        return "14-17"
    return "18+"


def _interp_contour(contour: list[float], n_syl: int) -> list[float]:
    """Interpole un contour a n_syl points si necessaire."""
    if len(contour) == n_syl:
        return list(contour)
    x_src = np.linspace(0, 1, len(contour))
    x_dst = np.linspace(0, 1, n_syl)
    return list(np.interp(x_dst, x_src, contour))


def _select_cluster(mode: str, n_syl: int, rng: Random,
                    group_role: str = "seul") -> dict:
    """Selectionne un cluster complet (f0 + dur + energy) depuis la banque.

    group_role: "seul" | "initial" | "medial" | "terminal"
      - initial/medial → cherche dans "continuation_N"
      - terminal/seul → cherche dans "{mode}_N"

    Returns: dict avec f0, dur, energy interpoles a n_syl points.
    """
    bucket = _syl_bucket(n_syl)

    # Determiner la cle de lookup selon le role du groupe
    if group_role in ("initial", "medial"):
        key = f"continuation_{bucket}"
    else:
        key = f"{mode}_{bucket}"

    clusters = GROUP_BANK.get(key)
    if not clusters:
        clusters = GROUP_BANK.get(f"declaratif_{bucket}")
    if not clusters:
        clusters = GROUP_BANK.get(f"continuation_{bucket}")
    if not clusters:
        return {
            "f0": [0.0] * max(1, n_syl),
            "dur_ratio": [1.0] * max(1, n_syl),
            "energy": [1.0] * max(1, n_syl),
        }

    # Tirage pondere par n (nombre de groupes dans le cluster)
    weights = [c["n"] for c in clusters]
    total = sum(weights)
    r = rng.random() * total
    cumul = 0.0
    selected = clusters[-1]
    for c in clusters:
        cumul += c["n"]
        if r <= cumul:
            selected = c
            break

    # Utiliser dur_ratio (pattern relatif) plutot que dur absolu
    dur_ratio = selected.get("dur_ratio", [1.0] * len(selected.get("f0", [])))
    return {
        "f0": _interp_contour(selected["f0"], n_syl),
        "dur_ratio": _interp_contour(dur_ratio, n_syl),
        "energy": _interp_contour(selected["energy"], n_syl),
    }


def _add_jitter(contour: list[float], rng: Random,
                jitter_st: float = 0.6) -> list[float]:
    """Ajoute du jitter local. Moins de jitter sur la derniere syllabe."""
    result = []
    n = len(contour)
    for i, val in enumerate(contour):
        if i == n - 1:
            j = rng.gauss(0, jitter_st * 0.3)
        else:
            j = rng.gauss(0, jitter_st)
        result.append(val + j)
    return result


def generate_prosody_v2(
    n_syl: int,
    mode: str,
    rng: Random,
    base_f0: float = 200.0,
    group_role: str = "seul",
) -> list[dict]:
    """Genere la prosodie pour N syllabes d'un groupe.

    Selectionne un cluster complet (F0 + duree + energie) puis
    ajoute du jitter local pour la variation.

    group_role: "seul" (phrase sans virgule), "initial", "medial", "terminal"

    Returns:
        Liste de dicts {dur_ms, f0_hz, f0_rel_st, energy, pause_after_ms}.
    """
    if n_syl <= 0:
        return []

    # ── Selectionner un cluster complet (F0 + dur_ratio + energy) ──
    cluster = _select_cluster(mode, n_syl, rng, group_role=group_role)
    f0_contour = cluster["f0"]
    dur_ratio_contour = cluster["dur_ratio"]
    energy_contour = cluster["energy"]

    # Duree de base par syllabe (ms) — tempo neutre
    base_syl_dur = 160.0

    # Ajouter du jitter
    f0_per_syl_st = _add_jitter(f0_contour, rng, jitter_st=0.5)

    result = []
    for si in range(n_syl):
        # ── F0 ──
        f0_rel_st = f0_per_syl_st[si] if si < len(f0_per_syl_st) else 0.0
        f0_rel_st = max(-12.0, min(10.0, f0_rel_st))
        f0_hz = base_f0 * (2.0 ** (f0_rel_st / 12.0))
        f0_hz = max(80.0, f0_hz)

        # ── Duree : base × dur_ratio du cluster + jitter ──
        dr = dur_ratio_contour[si] if si < len(dur_ratio_contour) else 1.0
        dur = base_syl_dur * dr
        dur += rng.gauss(0, dur * 0.08)  # ±8% jitter
        dur = max(60, min(600, dur))

        # ── Energie : depuis le cluster ──
        energy = energy_contour[si] if si < len(energy_contour) else 1.0
        energy += rng.gauss(0, 0.05)  # petit jitter
        energy = max(0.2, min(2.0, energy))

        result.append({
            "dur_ms": dur,
            "f0_hz": f0_hz,
            "f0_rel_st": f0_rel_st,
            "energy": energy,
            "pause_after_ms": 0.0,
        })

    return result


# ══════════════════════════════════════════════════════════════════
# Poids de duree intrinseque par type de phone
# ══════════════════════════════════════════════════════════════════

def _phone_dur_weight(phone: str) -> float:
    """Poids de duree intrinseque par type de phone.

    Les voyelles sont les plus longues (~50-60% de la syllabe),
    les plosives les plus courtes.
    """
    if not phone:
        return 1.0
    base = phone[0]
    if base in _VOWELS:
        return 3.0
    if base in _VOICELESS_FRICATIVES or base in _VOICED_FRICATIVES:
        return 1.8
    if base in _NASALS:
        return 1.5
    if base in _LIQUIDS:
        return 1.4
    if base in _GLIDES:
        return 0.7
    if base in _VOICELESS_STOPS or base in _VOICED_STOPS:
        return 0.8
    return 1.0


# ══════════════════════════════════════════════════════════════════
# Synthese avec prosodie v2
# ══════════════════════════════════════════════════════════════════

def _synthesize_prosody_v2(phones, syl_prosody, base_f0=200.0,
                           use_dur_weights=True, use_f0_interp=True,
                           use_energy=True):
    """Synthese avec prosodie v2 : durees absolues + pauses.

    Flags pour activer/desactiver les fonctionnalites :
      use_dur_weights : poids de duree par type de phone (vs egal)
      use_f0_interp   : interpolation lisse du F0 entre voyelles (vs palier plat)
      use_energy      : appliquer l'energie du cluster (vs constant 1.0)
    """
    import pyworld as pw

    n = len(phones)
    chain = engine.build_diphone_chain(phones)
    vowel_positions = [j for j in range(n) if phones[j] and phones[j][0] in _VOWELS]
    n_syl = len(vowel_positions)

    if n_syl == 0 or not syl_prosody:
        return np.array([], dtype=np.float32)

    syllable_spans = _phones_to_syllables(phones, vowel_positions)

    # ── Construire phone durations ──
    phone_durs = [80.0] * n

    for si, (syl_start, syl_end) in enumerate(syllable_spans):
        if si >= len(syl_prosody):
            break
        n_phones_syl = syl_end - syl_start
        if n_phones_syl <= 0:
            continue
        target_dur = syl_prosody[si]["dur_ms"]

        if use_dur_weights:
            # Proportions naturelles par type de phone
            weights = []
            for j in range(syl_start, syl_end):
                weights.append(_phone_dur_weight(phones[j]))
            total_weight = sum(weights)
            for j in range(syl_start, syl_end):
                phone_durs[j] = target_dur * weights[j - syl_start] / total_weight
        else:
            # Repartition egale
            per_phone = target_dur / n_phones_syl
            for j in range(syl_start, syl_end):
                phone_durs[j] = per_phone

    # ── F0 par phone ──
    if use_f0_interp:
        # Interpolation lisse entre cibles ancrees sur les voyelles
        f0_anchors = []
        for si, vpos in enumerate(vowel_positions):
            if si < len(syl_prosody):
                f0_anchors.append((vpos, syl_prosody[si]["f0_hz"]))

        if len(f0_anchors) >= 2:
            anchor_x = [a[0] for a in f0_anchors]
            anchor_y = [a[1] for a in f0_anchors]
            if anchor_x[0] > 0:
                anchor_x.insert(0, 0)
                anchor_y.insert(0, anchor_y[0])
            if anchor_x[-1] < n - 1:
                anchor_x.append(n - 1)
                anchor_y.append(anchor_y[-1])
            all_x = list(range(n))
            f0_targets = list(np.interp(all_x, anchor_x, anchor_y))
        elif len(f0_anchors) == 1:
            f0_targets = [f0_anchors[0][1]] * n
        else:
            f0_targets = [base_f0] * n
    else:
        # Palier plat par syllabe
        f0_targets = [base_f0] * n
        for si, (syl_start, syl_end) in enumerate(syllable_spans):
            if si >= len(syl_prosody):
                break
            for j in range(syl_start, syl_end):
                f0_targets[j] = syl_prosody[si]["f0_hz"]

    # ── Energie par phone ──
    energy_factors = [1.0] * n
    if use_energy:
        for si, (syl_start, syl_end) in enumerate(syllable_spans):
            if si >= len(syl_prosody):
                break
            en = syl_prosody[si].get("energy", 1.0)
            for j in range(syl_start, syl_end):
                energy_factors[j] = en

    # ── Detecter les pauses et decouper ──
    pause_positions = []
    for si, (syl_start, syl_end) in enumerate(syllable_spans):
        if si >= len(syl_prosody):
            break
        pause = syl_prosody[si]["pause_after_ms"]
        if pause > 0:
            pause_positions.append((syl_end - 1, pause))

    if not pause_positions:
        return _synth_segment(phones, phone_durs, f0_targets, chain,
                              energy_factors=energy_factors)

    # Decouper aux pauses
    audio_parts = []
    prev_end = 0
    for phone_idx, pause_ms in pause_positions:
        seg_end = phone_idx + 1
        if seg_end > prev_end:
            seg_phones = phones[prev_end:seg_end]
            seg_durs = phone_durs[prev_end:seg_end]
            seg_f0 = f0_targets[prev_end:seg_end]
            seg_en = energy_factors[prev_end:seg_end]
            seg_chain = engine.build_diphone_chain(seg_phones)
            audio = _synth_segment(seg_phones, seg_durs, seg_f0, seg_chain,
                                    energy_factors=seg_en)
            audio_parts.append(audio)
        # Pause
        n_silence = int(pause_ms / 1000.0 * SIWIS_SR)
        audio_parts.append(np.zeros(n_silence, dtype=np.float32))
        prev_end = seg_end

    # Dernier segment
    if prev_end < n:
        seg_phones = phones[prev_end:]
        seg_durs = phone_durs[prev_end:]
        seg_f0 = f0_targets[prev_end:]
        seg_en = energy_factors[prev_end:]
        seg_chain = engine.build_diphone_chain(seg_phones)
        audio = _synth_segment(seg_phones, seg_durs, seg_f0, seg_chain,
                                energy_factors=seg_en)
        audio_parts.append(audio)

    if not audio_parts:
        return np.array([], dtype=np.float32)

    combined = np.concatenate(audio_parts)
    combined = np.clip(combined, -0.95, 0.95)
    return combined


def _synth_segment(phones, phone_durs, f0_targets, chain,
                   energy_factors=None):
    """Synthese d'un segment (sans pauses internes)."""
    import pyworld as pw

    n = len(phones)
    if energy_factors is None:
        energy_factors = [1.0] * n
    overlap_comp = OVERLAP_FRAMES * FRAME_PERIOD

    # Phone durs -> diphone durs
    diphone_durs = []
    for di_idx, di_key in enumerate(chain):
        if di_key.startswith("#-"):
            diphone_durs.append(phone_durs[0] * 0.5 + overlap_comp * 0.5)
        elif di_key.endswith("-#"):
            diphone_durs.append(phone_durs[-1] * 0.5 + overlap_comp * 0.5)
        else:
            a_idx = di_idx - 1
            b_idx = di_idx
            da = phone_durs[a_idx] if a_idx < n else 60.0
            db = phone_durs[b_idx] if b_idx < n else 60.0
            diphone_durs.append((da + db) * 0.5 + overlap_comp)

    # Build diphone segments
    diphone_segments = []
    for di_idx, di_key in enumerate(chain):
        diphone = engine.diphones.get(di_key)
        if diphone is None:
            fft_size = pw.get_cheaptrick_fft_size(SIWIS_SR)
            n_bins = fft_size // 2 + 1
            diphone_segments.append({
                "f0": np.zeros(4, dtype=np.float64),
                "sp": np.full((4, n_bins), 1e-10, dtype=np.float64),
                "ap": np.ones((4, n_bins), dtype=np.float64),
                "key": di_key,
            })
            continue

        f0 = diphone["f0"].astype(np.float64)
        sp = diphone["sp"].astype(np.float64)
        ap = diphone["ap"].astype(np.float64)
        sr = diphone.get("sr", SIWIS_SR)
        sp, ap = ensure_full_spectrum(sp, ap, sr)

        target_ms = diphone_durs[di_idx]
        if di_key.startswith("#-"):
            tf0 = f0_targets[0]
        elif di_key.endswith("-#"):
            tf0 = f0_targets[-1]
        else:
            a_idx = di_idx - 1
            b_idx = di_idx
            fa = f0_targets[a_idx] if a_idx < len(f0_targets) else 190.0
            fb = f0_targets[b_idx] if b_idx < len(f0_targets) else 190.0
            tf0 = (fa + fb) / 2

        n_target = max(4, int(target_ms / FRAME_PERIOD))
        f0_s, sp_s, ap_s = stretch_params(f0, sp, ap, n_target)

        voiced = f0_s > 0
        if np.any(voiced) and tf0 > 0:
            current_mean = np.mean(f0_s[voiced])
            if current_mean > 0:
                f0_s[voiced] *= tf0 / current_mean

        # Appliquer le facteur d'energie (gain spectral)
        if di_key.startswith("#-"):
            en = energy_factors[0]
        elif di_key.endswith("-#"):
            en = energy_factors[-1]
        else:
            a_idx = di_idx - 1
            b_idx = di_idx
            ea = energy_factors[a_idx] if a_idx < len(energy_factors) else 1.0
            eb = energy_factors[b_idx] if b_idx < len(energy_factors) else 1.0
            en = (ea + eb) / 2
        if abs(en - 1.0) > 0.01:
            sp_s = sp_s * en

        diphone_segments.append({"f0": f0_s, "sp": sp_s, "ap": ap_s, "key": di_key})

    if not diphone_segments:
        return np.array([], dtype=np.float32)

    f0_cat, sp_cat, ap_cat, boundaries = concat_diphones(diphone_segments)

    if 1.5 > 1.0:
        ap_cat = compress_aperiodicity(ap_cat, gamma=1.5, sr=SIWIS_SR)
    if 1.3 > 1.0:
        sp_cat = sharpen_formants(sp_cat, gain=1.3)
    if 1.5 > 1.0:
        log_sp = np.log(np.maximum(sp_cat, 1e-10))
        mean_log = np.mean(log_sp, axis=0, keepdims=True)
        sp_cat = np.exp(mean_log + (log_sp - mean_log) * 1.5)

    audio = synthesize(f0_cat, sp_cat, ap_cat, sr=SIWIS_SR)
    if len(audio) > 0:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.9 / peak)

    return audio.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# Test etendu : variante A (corpus clusters, F0 plat, dur egales)
# ══════════════════════════════════════════════════════════════════

engine = creer_engine()
print(f"Engine charge: {len(engine.diphones)} diphones\n")

PHRASES = {
    # ── Declaratif ──
    "d_court1":    ("Le chat dort.", "declaratif"),
    "d_court2":    ("Bonjour.", "declaratif"),
    "d_simple":    ("Le petit chat est mort.", "declaratif"),
    "d_moyen":     ("La pluie tombe sur les toits de la ville.", "declaratif"),
    "d_long":      ("Les enfants jouent dans le jardin pendant que les parents discutent.", "declaratif"),
    # ── Virgules ──
    "v_simple":    ("Le matin, je prends un café.", "declaratif"),
    "v_double":    ("Il ouvrit la porte, regarda dehors, et ne vit personne.", "declaratif"),
    "v_longue":    ("Après avoir longuement réfléchi, il décida de partir, malgré la pluie.", "declaratif"),
    "v_enum":      ("Il acheta du pain, du beurre, du lait, et du fromage.", "declaratif"),
    # ── Questions ──
    "q_court":     ("Comment allez-vous ?", "question"),
    "q_moyen":     ("Est-ce que vous avez vu le dernier film ?", "question"),
    "q_long":      ("Pourquoi les oiseaux chantent-ils au lever du soleil ?", "question"),
    # ── Exclamations ──
    "e_court":     ("Quel bonheur !", "exclamation"),
    "e_moyen":     ("Il fait vraiment beau aujourd'hui !", "exclamation"),
    "e_long":      ("Je n'aurais jamais imaginé une chose pareille !", "exclamation"),
    # ── Suspensif ──
    "s_court":     ("Je ne sais pas...", "declaratif"),
    "s_moyen":     ("Il y avait quelque chose de bizarre...", "declaratif"),
    # ── Phrases naturelles variees ──
    "n_dialogue":  ("Oui, c'est une bonne idée.", "declaratif"),
    "n_complexe":  ("Le professeur, qui avait beaucoup voyagé, racontait ses aventures.", "declaratif"),
    "n_lecture":   ("La nuit était calme, les étoiles brillaient dans le ciel.", "declaratif"),
}

BASE_F0 = 200.0
PAUSE_MS = {"comma": 150, "period": 400, "question": 400,
            "exclamation": 350, "suspensive": 500, "none": 0}
mode_from_boundary = {"period": "declaratif", "question": "question",
                      "exclamation": "exclamation", "suspensive": "declaratif",
                      "comma": "declaratif", "none": "declaratif"}

print("=" * 80)
print("  Test etendu variante A : clusters corpus, F0 plat, dur egales")
print("=" * 80)


def _generate_phrase_audio(groups, phrase_mode):
    """Genere l'audio variante A pour une phrase."""
    n_groups = len(groups)
    audio_parts = []
    rng = Random(42)

    for gi, group in enumerate(groups):
        phones = group["phones"]
        if _PHONE_NORMALIZE:
            phones = [_PHONE_NORMALIZE.get(p, p) for p in phones]
        boundary = group.get("boundary", "none")

        if n_groups == 1:
            group_role = "seul"
        elif gi == 0:
            group_role = "initial"
        elif gi == n_groups - 1:
            group_role = "terminal"
        else:
            group_role = "medial"

        n_vowels = sum(1 for p in phones if p and p[0] in _VOWELS)
        prosody = generate_prosody_v2(n_vowels, phrase_mode, rng, BASE_F0,
                                       group_role=group_role)

        audio = _synthesize_prosody_v2(phones, prosody, BASE_F0,
                                        use_dur_weights=False,
                                        use_f0_interp=False,
                                        use_energy=False)
        audio_parts.append(audio)

        if gi < n_groups - 1:
            pause = PAUSE_MS.get(boundary, 60)
            if pause > 0:
                audio_parts.append(np.zeros(int(pause / 1000.0 * SIWIS_SR),
                                            dtype=np.float32))

    if not audio_parts:
        return np.array([], dtype=np.float32)
    combined = np.concatenate(audio_parts)
    return np.clip(combined, -0.95, 0.95)


for name, (text, mode_hint) in PHRASES.items():
    groups = engine._g2p_backend.phonemize(text)
    n_groups = len(groups)

    print(f"\n{'─' * 80}")
    print(f"[{name}] \"{text}\"  ({n_groups} groupes)")

    # Determiner le mode
    phrase_mode = mode_hint
    for g in reversed(groups):
        b = g.get("boundary", "none")
        if b in ("period", "question", "exclamation", "suspensive"):
            phrase_mode = mode_from_boundary.get(b, "declaratif")
            break

    # Generer prosodie et afficher
    rng_display = Random(42)
    for gi, group in enumerate(groups):
        phones = group["phones"]
        if _PHONE_NORMALIZE:
            phones = [_PHONE_NORMALIZE.get(p, p) for p in phones]
        boundary = group.get("boundary", "none")

        if n_groups == 1:
            group_role = "seul"
        elif gi == 0:
            group_role = "initial"
        elif gi == n_groups - 1:
            group_role = "terminal"
        else:
            group_role = "medial"

        n_vowels = sum(1 for p in phones if p and p[0] in _VOWELS)
        prosody = generate_prosody_v2(n_vowels, phrase_mode, rng_display, BASE_F0,
                                       group_role=group_role)

        dur_str = " ".join(f"{sp['dur_ms']:.0f}" for sp in prosody)
        f0_str = " ".join(f"{sp['f0_rel_st']:+.1f}" for sp in prosody)
        print(f"  Gr{gi} [{group_role:8s}] ({boundary:5s}, {n_vowels}syl): "
              f"F0=[{f0_str}]  Dur=[{dur_str}]")

    # Audio v2 (variante A)
    audio_v2 = _generate_phrase_audio(groups, phrase_mode)
    sf.write(str(OUT / f"v2_{name}.wav"), audio_v2, 44100)

    # Reference moteur v11
    audio_ref = engine.synthesize_groups(
        groups,
        mode=SynthMode.FLUIDE,
        duration_scale=1.0, pause_scale=1.0,
        macro_expressivity=1.0, micro_expressivity=1.0,
        seed=42,
        spectral_contrast=1.5, ap_cleanup=1.5, formant_sharpening=1.3,
        base_f0=BASE_F0,
    )
    sf.write(str(OUT / f"ref_{name}.wav"), audio_ref, 44100)

    print(f"  v2={len(audio_v2)/44100:.2f}s  ref={len(audio_ref)/44100:.2f}s")


# ── Recapitulatif ──
print(f"\n{'=' * 80}")
print(f"  {len(PHRASES) * 2} fichiers generes dans {OUT}/")
print(f"{'=' * 80}")

print(f"\nEcoute (v2 vs ref) :")
for name, (text, _) in PHRASES.items():
    print(f"  aplay {OUT}/v2_{name}.wav && aplay {OUT}/ref_{name}.wav  # {text[:50]}")
