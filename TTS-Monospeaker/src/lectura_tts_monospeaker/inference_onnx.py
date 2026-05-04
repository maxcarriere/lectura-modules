"""Engine TTS ONNX local — inference FastPitch + HiFi-GAN via onnxruntime.

Charge 3 sessions ONNX (lazy) :
- fastpitch_encoder.onnx
- fastpitch_decoder.onnx
- hifigan.onnx

Glue numpy entre encoder et decoder (length regulation + embeddings).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SEMITONE = 0.0577622  # log(2) / 12

# Cache des mots avec correction G2P (evite de splitter les composés corrigés)
_G2P_CORRECTIONS_KEYS: set[str] | None = None


def _get_g2p_corrections() -> set[str]:
    """Charge les clefs de la table de corrections G2P (singleton).

    Permet de savoir si un mot compose (avec tiret) a une correction
    dediee, auquel cas on ne le scinde pas avant le G2P.
    """
    global _G2P_CORRECTIONS_KEYS
    if _G2P_CORRECTIONS_KEYS is not None:
        return _G2P_CORRECTIONS_KEYS
    _G2P_CORRECTIONS_KEYS = set()
    try:
        import lectura_nlp
        corr_path = Path(lectura_nlp.__file__).parent / "data" / "g2p_corrections_unifie.json"
        if corr_path.exists():
            with open(corr_path, encoding="utf-8") as f:
                _G2P_CORRECTIONS_KEYS = set(json.load(f).keys())
    except (ImportError, Exception):
        pass
    return _G2P_CORRECTIONS_KEYS

# Ponctuation reconnue par le modele TTS
_PUNCT_MAP = {",": ",", ";": ",", ":": ",", ".": ".", "!": "!", "?": "?",
              "\u2026": "\u2026", "...": "\u2026"}

# Ponctuation terminale (decoupe en phrases)
_SENTENCE_PUNCT = {".", "?", "!", "\u2026", "..."}

# Durees minimales en frames pour la ponctuation (1 frame ≈ 11.6 ms)
_PUNCT_MIN_FRAMES = {",": 10, ".": 20, "?": 15, "!": 15, "\u2026": 25}

# Phones correspondant a des silences (zeroes dans l'audio)
_SILENCE_PHONES = {"#", ",", ".", "?", "!", "\u2026"}


def _zero_silence_regions(
    audio: np.ndarray,
    phones: list[str],
    durations: np.ndarray,
    hop_length: int,
    fade_samples: int = 128,
) -> np.ndarray:
    """Remplace les zones SIL/ponctuation par du vrai silence.

    Utilise les durees predites pour identifier precisement les regions
    de silence et de ponctuation, puis les met a zero avec un court
    fondu aux transitions pour eviter les clics.

    Args:
        audio: Signal float32 mono.
        phones: Liste des phones (sans les SIL aux extremites).
        durations: Durees en frames [SIL, phone1, ..., phoneN, SIL].
        hop_length: Nombre d'echantillons par frame.
        fade_samples: Longueur du fondu aux transitions.
    """
    if len(audio) == 0:
        return audio

    # Construire le masque : 1.0 = parole, 0.0 = silence
    all_phones = ["#"] + list(phones) + ["#"]
    mask = np.ones(len(audio), dtype=np.float32)

    offset = 0
    for i, phone in enumerate(all_phones):
        dur_samples = int(durations[i]) * hop_length
        if phone in _SILENCE_PHONES:
            s = max(0, offset)
            e = min(len(audio), offset + dur_samples)
            mask[s:e] = 0.0
        offset += dur_samples

    # Lisser les transitions (fondu dans la zone de silence)
    fade = fade_samples
    diff = np.diff(mask, prepend=mask[0])

    # Transitions 1→0 (parole → silence) : fondu au debut du silence
    for idx in np.where(diff < -0.5)[0]:
        e = min(len(audio), idx + fade)
        n = e - idx
        if n > 0:
            mask[idx:e] = np.linspace(1.0, 0.0, n)

    # Transitions 0→1 (silence → parole) : fondu a la fin du silence
    for idx in np.where(diff > 0.5)[0]:
        s = max(0, idx - fade)
        n = idx - s
        if n > 0:
            mask[s:idx] = np.linspace(0.0, 1.0, n)

    return audio * mask


def _text_to_sentences(text: str, g2p) -> list[tuple[str, int]]:
    """Decoupe le texte en phrases, chacune avec IPA + phrase_type.

    Pipeline : Tokeniseur → (Formules) → decoupage en phrases → G2P
               → (Aligneur/liaisons) → IPA par phrase.

    Modules optionnels :
        - lectura-formules : lecture des nombres/formules (display_fr)
        - lectura-aligneur : liaisons et groupes de lecture

    Args:
        text: Texte francais.
        g2p: Engine G2P (lectura_nlp).

    Returns:
        Liste de (ipa_string, phrase_type) par phrase.
    """
    try:
        from lectura_tokeniseur import tokenise
    except ImportError:
        log.warning("lectura-tokeniseur non installe — ponctuation ignoree")
        words = text.split()
        if not words:
            return []
        result = g2p.analyser(words)
        return [("".join(result.get("g2p", [])), 0)]

    all_tokens = tokenise(text)

    # Enrichir les formules si le module est disponible
    try:
        from lectura_formules import enrichir_formules
        enrichir_formules(all_tokens)
    except ImportError:
        pass

    # Grouper les tokens en phrases (decoupage a la ponctuation terminale)
    sentences: list[list] = []
    current: list = []
    for token in all_tokens:
        current.append(token)
        if token.type.name == "PONCTUATION" and token.text.strip() in _SENTENCE_PUNCT:
            sentences.append(current)
            current = []
    if current and any(t.type.name in ("MOT", "FORMULE") for t in current):
        sentences.append(current)

    if not sentences:
        return []

    # Collecter uniquement les MOT pour G2P (les FORMULE ont deja leur IPA)
    # Les mots composes (tirets) sont scindes en sous-mots pour le G2P,
    # SAUF si le mot complet a une correction dediee dans la table G2P.
    corrections = _get_g2p_corrections()
    all_mot_words: list[str] = []
    mot_counts: list[int] = []
    for sent_tokens in sentences:
        n = 0
        for t in sent_tokens:
            if t.type.name == "MOT":
                if "-" in t.text and t.text.lower() in corrections:
                    # Correction dediee : passer le mot entier
                    all_mot_words.append(t.text)
                    n += 1
                else:
                    parts = [p for p in t.text.split("-") if p]
                    all_mot_words.extend(parts)
                    n += len(parts)
        mot_counts.append(n)

    # Un seul appel G2P pour tous les mots (pas les formules)
    if all_mot_words:
        g2p_result = g2p.analyser(all_mot_words)
        all_mot_ipa = g2p_result.get("g2p", [])
        all_mot_liaison = g2p_result.get("liaison", [])
    else:
        all_mot_ipa = []
        all_mot_liaison = []

    # Importer l'aligneur si disponible (pour les liaisons)
    _cg = None
    _MA = None
    try:
        from lectura_aligneur import construire_groupes as _cg_fn
        from lectura_aligneur import MotAnalyse as _MA_cls
        _cg = _cg_fn
        _MA = _MA_cls
    except ImportError:
        pass

    # Construire l'IPA par phrase
    results: list[tuple[str, int]] = []
    mot_idx = 0

    for sent_tokens, n_mots in zip(sentences, mot_counts):
        # Distribuer les resultats G2P pour cette phrase
        sent_mot_ipa = all_mot_ipa[mot_idx:mot_idx + n_mots]
        sent_mot_liaison = (
            all_mot_liaison[mot_idx:mot_idx + n_mots]
            if all_mot_liaison else ["none"] * n_mots
        )
        mot_idx += n_mots

        # Detecter phrase_type depuis la ponctuation terminale
        phrase_type = 0
        for tok in reversed(sent_tokens):
            if tok.type.name == "PONCTUATION":
                p = tok.text.strip()
                if p == "?":
                    phrase_type = 1
                elif p == "!":
                    phrase_type = 2
                elif p in ("\u2026", "..."):
                    phrase_type = 3
                break

        # Construire la sequence IPA/liaison a partir des tokens
        # Les MOT prennent leur IPA du G2P, les FORMULE de lecture.phone
        word_entries = _build_word_entries(
            sent_tokens, sent_mot_ipa, sent_mot_liaison, corrections,
        )

        if not word_entries:
            continue

        if _cg is not None and _MA is not None:
            ipa = _build_ipa_groupes(word_entries, _cg, _MA)
        else:
            ipa = _build_ipa_simple(word_entries)

        if ipa:
            results.append((ipa, phrase_type))

    return results


def _build_word_entries(
    sent_tokens: list,
    mot_ipa: list[str],
    mot_liaison: list[str],
    corrections: set[str] | None = None,
) -> list[dict]:
    """Construit la liste des entrees mot/formule/ponctuation pour une phrase.

    Chaque entree est un dict avec : type, ipa, liaison, punct_before, punct_after.
    Les FORMULE utilisent lecture.phone directement (construit composant par
    composant), les MOT utilisent le G2P.
    """
    entries: list[dict] = []
    pending_punct: str | None = None
    mi = 0  # index dans mot_ipa
    corr = corrections or set()

    for ti, tok in enumerate(sent_tokens):
        if tok.type.name == "MOT":
            # Elision : si le mot suivant est un SEPARATEUR apostrophe,
            # forcer la liaison a "none" (la consonne elidee n'/l'/j'/etc.
            # est deja le lien phonetique, pas besoin de liaison supplementaire).
            is_elision = (
                ti + 1 < len(sent_tokens)
                and sent_tokens[ti + 1].type.name == "SEPARATEUR"
                and "'" in sent_tokens[ti + 1].text
            )

            if "-" in tok.text and tok.text.lower() in corr:
                # Mot compose avec correction dediee — passe entier au G2P
                if mi < len(mot_ipa):
                    entries.append({
                        "ipa": mot_ipa[mi],
                        "liaison": "none" if is_elision else (mot_liaison[mi] if mi < len(mot_liaison) else "none"),
                        "punct_before": pending_punct,
                        "punct_after": None,
                    })
                    mi += 1
                    pending_punct = None
            else:
                # Scinder les mots composes (tirets) — chaque partie a son IPA G2P
                parts = [p for p in tok.text.split("-") if p]
                for j, _part in enumerate(parts):
                    if mi < len(mot_ipa):
                        # Elision s'applique au dernier sous-mot du compose
                        force_none = is_elision and j == len(parts) - 1
                        entries.append({
                            "ipa": mot_ipa[mi],
                            "liaison": "none" if force_none else (mot_liaison[mi] if mi < len(mot_liaison) else "none"),
                            "punct_before": pending_punct if j == 0 else None,
                            "punct_after": None,
                        })
                        mi += 1
                        pending_punct = None

        elif tok.type.name == "FORMULE":
            # Utiliser le phone des formules (construit morceau par morceau)
            lecture = getattr(tok, "lecture", None)
            if lecture and getattr(lecture, "phone", ""):
                # phone contient des espaces entre composants : "kaʁɑ̃t dø"
                components = lecture.phone.split()
            else:
                # Fallback : texte brut comme un seul mot
                components = [tok.text]

            for j, comp_ipa in enumerate(components):
                entries.append({
                    "ipa": comp_ipa,
                    "liaison": "none",
                    "punct_before": pending_punct if j == 0 else None,
                    "punct_after": None,
                })
                pending_punct = None

        elif tok.type.name == "PONCTUATION":
            p = _PUNCT_MAP.get(tok.text.strip())
            if p:
                if entries:
                    entries[-1]["punct_after"] = p
                pending_punct = p

    return entries


def _build_ipa_simple(word_entries: list[dict]) -> str:
    """Construit l'IPA sans liaisons (fallback sans aligneur).

    Insere des espaces entre les mots pour marquer les frontieres
    (utile pour les phoneme_timings et le DTW d'alignement).
    """
    parts: list[str] = []
    for i, entry in enumerate(word_entries):
        if i > 0 and not word_entries[i - 1].get("punct_after"):
            parts.append(" ")
        parts.append(entry["ipa"])
        if entry["punct_after"]:
            parts.append(entry["punct_after"])
    return "".join(parts)


def _build_ipa_groupes(
    word_entries: list[dict],
    construire_groupes,
    MotAnalyse,
) -> str:
    """Construit l'IPA avec liaisons via l'aligneur.

    Cree des MotAnalyse a partir des entrees, applique
    construire_groupes() pour les liaisons et enchainements,
    puis reassemble avec la ponctuation.
    """
    if not word_entries:
        return ""

    # Construire les MotAnalyse
    mots = [
        MotAnalyse(
            phone=e["ipa"],
            liaison=e["liaison"],
            ponctuation_avant=e["punct_before"] is not None,
        )
        for e in word_entries
    ]

    # Construire les groupes de lecture (applique liaisons + enchainements)
    groupes = construire_groupes(mots)

    # Assembler l'IPA depuis les groupes + ponctuation inter-groupes
    # NB: phone_groupe ne contient PAS les consonnes de liaison,
    # elles sont indiquees dans jonctions ("liaison_z", "liaison_t", etc.)
    # On insere des espaces entre les groupes pour marquer les frontieres.
    parts: list[str] = []
    wd_offset = 0
    for gi, grp in enumerate(groupes):
        # Espace entre groupes (sauf avant le premier, et sauf apres ponctuation)
        if gi > 0 and parts and parts[-1] not in (",", ".", "?", "!", "\u2026"):
            parts.append(" ")

        # Reconstituer l'IPA du groupe avec insertions de liaison
        grp_phones = [m.phone for m in grp.mots]
        if grp.jonctions:
            grp_parts = [grp_phones[0]]
            for j, jonction in enumerate(grp.jonctions):
                if jonction.startswith("liaison_"):
                    grp_parts.append(jonction[len("liaison_"):])
                grp_parts.append(grp_phones[j + 1])
            parts.append("".join(grp_parts))
        else:
            parts.append(grp.phone_groupe)

        last_idx = wd_offset + len(grp.mots) - 1
        if last_idx < len(word_entries) and word_entries[last_idx]["punct_after"]:
            parts.append(word_entries[last_idx]["punct_after"])
        wd_offset += len(grp.mots)

    return "".join(parts)


@dataclass
class PhonemeTiming:
    """Timing d'un phoneme dans l'audio synthetise."""
    ipa: str
    start_ms: float
    end_ms: float


@dataclass
class TTSResult:
    """Resultat d'une synthese TTS."""
    samples: np.ndarray  # float32, mono
    sample_rate: int
    phoneme_timings: list[PhonemeTiming] = field(default_factory=list)


class OnnxTTSEngine:
    """Engine TTS ONNX local.

    Parameters
    ----------
    models_dir : Path
        Repertoire contenant les 3 ONNX + config.json + phoneme_vocab.json
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._encoder = None
        self._decoder = None
        self._hifigan = None
        self._config: dict[str, Any] | None = None
        self._phone2id: dict[str, int] | None = None
        self._sample_rate = 22050
        self._g2p = None

    def _ensure_loaded(self) -> None:
        """Charge les sessions ONNX (lazy)."""
        if self._encoder is not None:
            return

        import onnxruntime as ort

        from lectura_tts_monospeaker._chargeur import load_model_bytes

        # Charger config
        config_path = self._models_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
            self._config = json.load(f)

        # Charger vocabulaire
        vocab_path = self._models_dir / "phoneme_vocab.json"
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)
        self._phone2id = vocab_data["phone2id"]

        # Options ONNX
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        providers = ["CPUExecutionProvider"]

        # Charger les 3 modeles (supporte .onnx ou .enc)
        enc_bytes = load_model_bytes(self._models_dir, "fastpitch_encoder.onnx")
        dec_bytes = load_model_bytes(self._models_dir, "fastpitch_decoder.onnx")
        hifi_bytes = load_model_bytes(self._models_dir, "hifigan.onnx")

        if enc_bytes is None or dec_bytes is None or hifi_bytes is None:
            raise FileNotFoundError(
                f"Modeles ONNX incomplets dans {self._models_dir}"
            )

        self._encoder = ort.InferenceSession(enc_bytes, sess_options=opts, providers=providers)
        self._decoder = ort.InferenceSession(dec_bytes, sess_options=opts, providers=providers)
        self._hifigan = ort.InferenceSession(hifi_bytes, sess_options=opts, providers=providers)

        self._sample_rate = self._config.get("audio", {}).get("sample_rate", 22050)
        log.info("OnnxTTSEngine charge depuis %s", self._models_dir)

    def synthesize(
        self,
        text: str,
        phrase_type: int | None = None,
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
        pause_scale: float = 1.0,
        variability: bool = False,
    ) -> TTSResult:
        """Synthetise du texte (necessite lectura-g2p).

        Pipeline : Tokeniseur → decoupage en phrases → G2P → IPA → synthese
        par phrase avec phrase_type individuel → concatenation.

        Args:
            text: Texte francais a synthetiser.
            phrase_type: 0=decl, 1=inter, 2=excl, 3=susp (None=auto-detect).
            duration_scale: Multiplicateur de duree globale.
            pitch_shift: Decalage F0 en demi-tons.
            pitch_range: Echelle de variation F0.
            energy_scale: Multiplicateur d'energie.
            pause_scale: Multiplicateur pour les pauses.
        """
        try:
            from lectura_nlp import creer_engine as creer_g2p
        except ImportError:
            raise ImportError(
                "lectura-g2p requis pour synthesize(text). "
                "Installer avec : pip install lectura-tts-monospeaker[g2p]"
            )

        if self._g2p is None:
            self._g2p = creer_g2p(mode="auto")

        sentences = _text_to_sentences(text, self._g2p)

        if not sentences:
            return TTSResult(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                phoneme_timings=[],
            )

        prosody = dict(duration_scale=duration_scale, pitch_shift=pitch_shift,
                       pitch_range=pitch_range, energy_scale=energy_scale,
                       pause_scale=pause_scale, variability=variability)

        # Phrase unique — pas de concatenation
        if len(sentences) == 1:
            ipa, auto_pt = sentences[0]
            return self.synthesize_phonemes(
                ipa,
                phrase_type=phrase_type if phrase_type is not None else auto_pt,
                **prosody,
            )

        # Multi-phrases : synthetiser chacune et concatener
        all_samples: list[np.ndarray] = []
        all_timings: list[PhonemeTiming] = []
        time_offset_ms = 0.0

        for i, (ipa, auto_pt) in enumerate(sentences):
            result = self.synthesize_phonemes(
                ipa,
                phrase_type=phrase_type if phrase_type is not None else auto_pt,
                **prosody,
            )

            # Decaler les timings
            for t in result.phoneme_timings:
                all_timings.append(PhonemeTiming(
                    ipa=t.ipa,
                    start_ms=t.start_ms + time_offset_ms,
                    end_ms=t.end_ms + time_offset_ms,
                ))

            all_samples.append(result.samples)
            time_offset_ms += len(result.samples) / self._sample_rate * 1000

        combined = np.concatenate(all_samples)
        return TTSResult(
            samples=combined,
            sample_rate=self._sample_rate,
            phoneme_timings=all_timings,
        )

    def synthesize_phonemes(
        self,
        phonemes_ipa: str,
        phrase_type: int = 0,
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
        pause_scale: float = 1.0,
        variability: bool = False,
    ) -> TTSResult:
        """Synthetise une sequence de phonemes IPA.

        Args:
            phonemes_ipa: Chaine IPA (ex: "bɔ̃ʒuʁ")
            phrase_type: 0=decl, 1=inter, 2=excl, 3=susp
            duration_scale: Multiplicateur de duree globale
            pitch_shift: Decalage F0 en demi-tons
            pitch_range: Echelle de variation F0
            energy_scale: Multiplicateur d'energie
            pause_scale: Multiplicateur pour les pauses

        Returns:
            TTSResult avec samples (float32), sample_rate, phoneme_timings
        """
        self._ensure_loaded()

        from lectura_tts_monospeaker.phonemes import ipa_to_phones
        from lectura_tts_monospeaker._enhance import enhance_mel, noise_gate, fade_out

        # Convertir IPA → phone IDs
        phones = ipa_to_phones(phonemes_ipa)

        # Reperer les frontieres de mots (espaces dans l'IPA) pour les timings
        space_after: list[int] = []
        segments = phonemes_ipa.split(" ")
        phone_count = 0
        for seg_idx, segment in enumerate(segments):
            if segment:
                phone_count += len(ipa_to_phones(segment))
            if seg_idx < len(segments) - 1 and phone_count > 0:
                space_after.append(phone_count - 1)

        sil_id = self._phone2id["#"]
        unk_id = self._phone2id.get("<UNK>", 1)

        phone_ids = [sil_id] + [self._phone2id.get(p, unk_id) for p in phones] + [sil_id]
        phone_ids_np = np.array([phone_ids], dtype=np.int64)
        phrase_type_np = np.array([phrase_type], dtype=np.int64)

        # 1. Encoder
        enc_out, dur_pred, pitch_pred, energy_pred = self._encoder.run(None, {
            "phone_ids": phone_ids_np,
            "phrase_type": phrase_type_np,
        })

        # 2. Process predictions (numpy glue)
        dur_raw = np.exp(dur_pred[0])

        # Apply pause_scale aux SIL + ponctuation
        punct_ids = {self._phone2id.get(p, -1) for p in _PUNCT_MIN_FRAMES}
        pause_mask = np.zeros(len(phone_ids), dtype=bool)
        for idx, pid in enumerate(phone_ids):
            if pid == sil_id or pid in punct_ids:
                pause_mask[idx] = True
        if pause_scale != 1.0:
            dur_raw[pause_mask] *= pause_scale

        durations = np.maximum(1, np.round(dur_raw * duration_scale)).astype(np.int64)

        if variability:
            rng = np.random.default_rng()
            dur_noise = rng.normal(1.0, 0.10, size=durations.shape)
            dur_noise[pause_mask] = 1.0
            durations = np.maximum(1, np.round(durations * dur_noise)).astype(np.int64)

        # Durees minimales pour la ponctuation
        for idx, phone in enumerate(phones):
            min_frames = _PUNCT_MIN_FRAMES.get(phone)
            if min_frames is not None:
                durations[idx + 1] = max(durations[idx + 1], min_frames)  # +1 pour SIL

        # Pitch avec shift et range
        pitch_mean = pitch_pred[0].mean()
        pitch_values = (
            pitch_mean
            + (pitch_pred[0] - pitch_mean) * pitch_range
            + pitch_shift * SEMITONE
        )

        if variability:
            pitch_values *= rng.normal(1.0, 0.03, size=pitch_values.shape)

        # Energy
        energy_values = energy_pred[0] * energy_scale

        if variability:
            energy_values *= rng.normal(1.0, 0.02, size=energy_values.shape)

        # 3. Embeddings (matmul simple — poids Conv1d(1, D, 1))
        emb = self._config["embeddings"]
        pitch_w = np.array(emb["pitch_emb_weight"], dtype=np.float32)
        pitch_b = np.array(emb["pitch_emb_bias"], dtype=np.float32)
        energy_w = np.array(emb["energy_emb_weight"], dtype=np.float32)
        energy_b = np.array(emb["energy_emb_bias"], dtype=np.float32)

        pitch_emb = np.outer(pitch_values, pitch_w) + pitch_b[np.newaxis, :]
        energy_emb = np.outer(energy_values, energy_w) + energy_b[np.newaxis, :]

        # 4. Enrichir + expand
        enriched = enc_out[0] + pitch_emb + energy_emb  # [T, D]
        expanded = np.repeat(enriched, durations, axis=0)  # [T_mel, D]
        expanded = expanded[np.newaxis].astype(np.float32)  # [1, T_mel, D]

        # 5. Decoder
        mel = self._decoder.run(None, {"decoder_in": expanded})[0]  # [1, 80, T_mel]
        mel_np = mel[0]  # [80, T_mel]

        # 6. Enhancement
        enh = self._config.get("enhance", {})
        mel_np = enhance_mel(
            mel_np,
            spectral_alpha=enh.get("spectral_alpha", 0.20),
            temporal_alpha=enh.get("temporal_alpha", 0.20),
        )
        mel_np = noise_gate(
            mel_np,
            threshold=enh.get("noise_gate_threshold", -8.0),
            silence_val=enh.get("silence_val", -11.5),
        )
        mel_np = fade_out(
            mel_np,
            n_frames=enh.get("fade_frames", 5),
            silence_val=enh.get("silence_val", -11.5),
        )

        # 7. Vocoder
        mel_input = mel_np[np.newaxis].astype(np.float32)
        audio = self._hifigan.run(None, {"mel": mel_input})[0]  # [1, 1, T_audio]
        audio = audio.squeeze()

        # 8. Post-traitement audio
        hop_length = self._config.get("audio", {}).get("hop_length", 256)

        # Zeroing des silences/ponctuation (base sur les durees predites)
        audio = _zero_silence_regions(audio, phones, durations, hop_length)

        # Normaliser
        max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
        audio = np.clip(audio / max_val, -1.0, 1.0).astype(np.float32)

        # 9. Construire les timings phonemes (avec espaces aux frontieres de mots)
        timings = self._build_timings(phones, durations, hop_length, space_after)

        return TTSResult(
            samples=audio,
            sample_rate=self._sample_rate,
            phoneme_timings=timings,
        )

    def _build_timings(
        self,
        phones: list[str],
        durations: np.ndarray,
        hop_length: int,
        space_after: list[int] | None = None,
    ) -> list[PhonemeTiming]:
        """Construit les timings phonemes depuis les durees predites.

        Si space_after est fourni, insere des PhonemeTiming(ipa=" ")
        aux frontieres de mots pour le DTW d'alignement.
        """
        timings: list[PhonemeTiming] = []

        # durations correspond a [SIL, ...phones..., SIL]
        # on skip le premier SIL
        offset = int(durations[0]) * hop_length

        for i, phone in enumerate(phones):
            dur_frames = int(durations[i + 1])  # +1 pour skip SIL initial
            dur_samples = dur_frames * hop_length
            start_ms = offset / self._sample_rate * 1000
            end_ms = (offset + dur_samples) / self._sample_rate * 1000
            timings.append(PhonemeTiming(ipa=phone, start_ms=start_ms, end_ms=end_ms))
            offset += dur_samples

        # Inserer les espaces aux frontieres de mots (du dernier au premier)
        if space_after:
            sa = set(space_after)
            for idx in sorted(sa, reverse=True):
                if 0 <= idx < len(timings):
                    boundary_ms = timings[idx].end_ms
                    timings.insert(idx + 1, PhonemeTiming(
                        ipa=" ", start_ms=boundary_ms, end_ms=boundary_ms,
                    ))

        return timings
