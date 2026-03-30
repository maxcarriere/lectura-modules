"""Lectura G2P — Convertisseur graphème-phonème pour le français.

Fichier unique, autonome, zéro dépendance externe (mode CRF).
Support optionnel BiLSTM et Seq2Seq si ``onnxruntime`` est installé.

Usage rapide :
    from lectura_g2p import LecturaG2P

    g2p = LecturaG2P("modele/g2p_model_crf.json",
                       corrections_path="modele/g2p_corrections_crf.json")
    phone = g2p.predict("bonjour")   # → "bɔ̃ʒuʁ"
    phone = g2p.predict("maison")    # → "mɛzɔ̃"

Copyright (c) 2025 Lectura — Licence CC BY-SA 4.0.
Voir LICENCE.txt et ATTRIBUTION.md.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

__version__ = "1.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires IPA
# ══════════════════════════════════════════════════════════════════════════════

_VOYELLES: set[str] = {
    "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə",
}

_CONSONNES: set[str] = {
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
    "ʃ", "ʒ", "m", "n", "ɲ", "ŋ", "l", "ʁ",
}

_SEMI_VOYELLES: set[str] = {"j", "w", "ɥ"}

_OBSTRUENTS: set[str] = {"p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ"}


def iter_phonemes(ipa: str) -> list[str]:
    """Itere sur les phonemes d'une chaine IPA, en regroupant les combining marks.

    Exemples :
        >>> iter_phonemes("ʃa")
        ['ʃ', 'a']
        >>> iter_phonemes("ɑ̃")
        ['ɑ̃']
    """
    if not ipa:
        return []
    phonemes: list[str] = []
    current = ""
    for ch in ipa:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            current += ch
        else:
            if current:
                phonemes.append(current)
            current = ch
    if current:
        phonemes.append(current)
    return phonemes


def est_voyelle(phoneme: str) -> bool:
    """Vrai si le phoneme est une voyelle IPA (orale ou nasale)."""
    if not phoneme:
        return False
    if phoneme in _VOYELLES:
        return True
    return bool(phoneme[0] in _VOYELLES)


def est_consonne(phoneme: str) -> bool:
    """Vrai si le phoneme est une consonne IPA."""
    return bool(phoneme and phoneme in _CONSONNES)


def est_semi_voyelle(phoneme: str) -> bool:
    """Vrai si le phoneme est une semi-voyelle IPA."""
    return bool(phoneme and phoneme in _SEMI_VOYELLES)


# ══════════════════════════════════════════════════════════════════════════════
# Features CRF (extraction de features caractere)
# ══════════════════════════════════════════════════════════════════════════════

_CONT = "_CONT"
_ORTHO_VOWELS = set("aeiouyàâéèêëïîôûùüœæ")


def _extract_char_features(word: str, idx: int) -> dict[str, str | float]:
    """Extrait les features pour le caractere word[idx]."""
    n = len(word)
    ch = word[idx]

    feats: dict[str, str | float] = {
        "bias": 1.0,
        "char": ch,
        "is_vowel": 1.0 if ch in _ORTHO_VOWELS else 0.0,
        "is_first": 1.0 if idx == 0 else 0.0,
        "is_last": 1.0 if idx == n - 1 else 0.0,
        "pos_ratio": round(idx / max(n - 1, 1), 2),
    }

    # Word length bucket
    if n <= 4:
        feats["word_len_bucket"] = "short"
    elif n <= 8:
        feats["word_len_bucket"] = "medium"
    else:
        feats["word_len_bucket"] = "long"

    # Contexte caractere
    feats["char-1"] = word[idx - 1] if idx > 0 else "__BOS__"
    feats["char-2"] = word[idx - 2] if idx > 1 else "__BOS__"
    feats["char+1"] = word[idx + 1] if idx < n - 1 else "__EOS__"
    feats["char+2"] = word[idx + 2] if idx < n - 2 else "__EOS__"

    # N-grammes combines
    feats["bigram-1"] = (feats["char-1"] + ch) if idx > 0 else "__BOS__" + ch
    feats["bigram+1"] = (ch + feats["char+1"]) if idx < n - 1 else ch + "__EOS__"
    if idx > 0 and idx < n - 1:
        feats["trigram"] = word[idx - 1] + ch + word[idx + 1]
    else:
        feats["trigram"] = feats["bigram-1"] if idx == n - 1 else feats["bigram+1"]

    return feats


def _extract_word_char_features(word: str) -> list[dict[str, str | float]]:
    """Extrait les features pour tous les caracteres du mot."""
    return [_extract_char_features(word, i) for i in range(len(word))]


def _reconstruct_ipa(labels: list[str]) -> str:
    """Reconstruit la chaine IPA depuis les labels caractere."""
    return "".join(label for label in labels if label != _CONT)


# ══════════════════════════════════════════════════════════════════════════════
# Modele CRF + Viterbi
# ══════════════════════════════════════════════════════════════════════════════

class CrfG2pModel:
    """Modele CRF G2P avec decodage Viterbi pur Python.

    Le modele est charge depuis un fichier JSON contenant :
      - state_features: dict[str, dict[str, float]]
      - transitions: dict[str, dict[str, float]]
      - tags: list[str]
    """

    __slots__ = ("state_features", "transitions", "tags", "_tag_idx")

    def __init__(
        self,
        state_features: dict[str, dict[str, float]],
        transitions: dict[str, dict[str, float]],
        tags: list[str],
    ) -> None:
        self.state_features = state_features
        self.transitions = transitions
        self.tags = tags
        self._tag_idx = {t: i for i, t in enumerate(tags)}

    @classmethod
    def load(cls, path: Path | str) -> CrfG2pModel:
        """Charge un modele depuis un fichier JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            state_features=data["state_features"],
            transitions=data["transitions"],
            tags=data["tags"],
        )

    def _score_state(self, feats: dict[str, str | float], tag: str) -> float:
        score = 0.0
        for feat_key, feat_val in feats.items():
            lookup = f"{feat_key}:{feat_val}"
            weights = self.state_features.get(lookup)
            if weights and tag in weights:
                score += weights[tag]
        return score

    def _viterbi(self, observations: list[dict[str, str | float]]) -> list[str]:
        n = len(observations)
        if n == 0:
            return []

        tags = self.tags
        n_tags = len(tags)
        NEG_INF = -1e30

        # Initialisation (t=0)
        scores_0: list[float] = []
        bos_trans = self.transitions.get("__BOS__", {})
        for tag in tags:
            s = self._score_state(observations[0], tag)
            s += bos_trans.get(tag, 0.0)
            scores_0.append(s)

        viterbi: list[list[float]] = [scores_0]
        backptr: list[list[int]] = [[0] * n_tags]

        # Recurrence
        for t in range(1, n):
            scores_t: list[float] = []
            bp_t: list[int] = []
            for j, tag_j in enumerate(tags):
                state_score = self._score_state(observations[t], tag_j)
                best_score = NEG_INF
                best_prev = 0
                for i, tag_i in enumerate(tags):
                    prev = viterbi[t - 1][i]
                    if prev <= NEG_INF:
                        continue
                    trans = self.transitions.get(tag_i, {}).get(tag_j, 0.0)
                    score = prev + trans + state_score
                    if score > best_score:
                        best_score = score
                        best_prev = i
                scores_t.append(best_score)
                bp_t.append(best_prev)
            viterbi.append(scores_t)
            backptr.append(bp_t)

        # Backtrack
        best_last = max(range(n_tags), key=lambda j: viterbi[n - 1][j])
        result = [0] * n
        result[n - 1] = best_last
        for t in range(n - 2, -1, -1):
            result[t] = backptr[t + 1][result[t + 1]]

        return [tags[idx] for idx in result]

    def predict(self, word: str) -> str:
        """Predit la transcription IPA pour un mot."""
        if not word:
            return ""
        word_lower = word.lower()
        features = _extract_word_char_features(word_lower)
        labels = self._viterbi(features)
        return _reconstruct_ipa(labels)


# ══════════════════════════════════════════════════════════════════════════════
# Modele BiLSTM ONNX (optionnel — necessite onnxruntime + numpy)
# ══════════════════════════════════════════════════════════════════════════════

class OnnxG2pModel:
    """Modele G2P BiLSTM via ONNX Runtime.

    Charge un modele ONNX + vocabulaire JSON (char->idx, idx->phoneme).
    Meme cadrage que le CRF : sequence-labeling caractere par caractere.

    Necessite : pip install onnxruntime numpy
    """

    def __init__(self, onnx_path: Path | str, vocab_path: Path | str) -> None:
        import numpy as np
        import onnxruntime as ort

        self._np = np
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)
        self._char2idx: dict[str, int] = vocab_data["char2idx"]
        self._idx2label: dict[int, str] = {
            int(k): v for k, v in vocab_data["idx2label"].items()
        }
        self._pad_idx: int = vocab_data.get("pad_idx", 0)
        self._unk_idx: int = vocab_data.get("unk_idx", 1)

    def predict(self, word: str) -> str:
        if not word:
            return ""
        word_lower = word.lower()
        np = self._np

        char_ids = [self._char2idx.get(ch, self._unk_idx) for ch in word_lower]
        input_array = np.array([char_ids], dtype=np.int64)
        lengths = np.array([len(char_ids)], dtype=np.int64)

        input_name = self._session.get_inputs()[0].name
        inputs = {input_name: input_array}
        if len(self._session.get_inputs()) > 1:
            len_name = self._session.get_inputs()[1].name
            inputs[len_name] = lengths

        outputs = self._session.run(None, inputs)
        logits = outputs[0]

        pred_ids = np.argmax(logits[0], axis=-1)
        labels = [self._idx2label.get(int(idx), _CONT) for idx in pred_ids]
        return _reconstruct_ipa(labels)


# ══════════════════════════════════════════════════════════════════════════════
# Modele Seq2Seq ONNX (optionnel — necessite onnxruntime + numpy)
# ══════════════════════════════════════════════════════════════════════════════

class OnnxG2pSeq2SeqModel:
    """Modele G2P Seq2Seq (encoder-decoder + attention) via ONNX Runtime.

    Charge un encoder ONNX + decoder ONNX + vocabulaire JSON.
    Decodage greedy autoregressif.

    Necessite : pip install onnxruntime numpy
    """

    def __init__(
        self,
        encoder_path: Path | str,
        decoder_path: Path | str,
        vocab_path: Path | str,
    ) -> None:
        import numpy as np
        import onnxruntime as ort

        self._np = np
        self._encoder = ort.InferenceSession(
            str(encoder_path),
            providers=["CPUExecutionProvider"],
        )
        self._decoder = ort.InferenceSession(
            str(decoder_path),
            providers=["CPUExecutionProvider"],
        )
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)
        self._char2idx: dict[str, int] = vocab_data["char2idx"]
        self._idx2phone: dict[int, str] = {
            int(k): v for k, v in vocab_data["idx2phone"].items()
        }
        self._sos_idx: int = vocab_data["sos_idx"]
        self._eos_idx: int = vocab_data["eos_idx"]
        self._max_len: int = vocab_data.get("max_len", 50)

    def predict(self, word: str) -> str:
        """Predit la transcription IPA pour un mot."""
        if not word:
            return ""

        np = self._np
        word_lower = word.lower()

        # Encoder les caracteres
        char_ids = [self._char2idx.get(ch, 1) for ch in word_lower]
        src = np.array([char_ids], dtype=np.int64)

        # Encoder
        enc_out, h, c = self._encoder.run(None, {"src": src})

        # Greedy decode
        inp = np.array([self._sos_idx], dtype=np.int64)
        phones: list[str] = []

        for _ in range(self._max_len):
            logits, h, c = self._decoder.run(
                None,
                {
                    "input_token": inp,
                    "h": h,
                    "c": c,
                    "encoder_outputs": enc_out,
                },
            )
            idx = int(np.argmax(logits[0]))
            if idx == self._eos_idx:
                break
            ph = self._idx2phone.get(idx, "")
            if ph not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
                phones.append(ph)
            inp = np.array([idx], dtype=np.int64)

        return "".join(phones)


# ══════════════════════════════════════════════════════════════════════════════
# Corrections (table de lookup ortho -> phone)
# ══════════════════════════════════════════════════════════════════════════════

class G2pCorrections:
    """Table de corrections G2P : lookup rapide ortho -> phone.

    Supporte deux sections :
      - ``g2p``     : dict[mot, phone]  (corrections simples)
      - ``g2p_pos`` : dict[mot, dict[POS, phone]]  (POS-aware)
    """

    def __init__(
        self,
        g2p: dict[str, str] | None = None,
        g2p_pos: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self._g2p: dict[str, str] = g2p or {}
        self._g2p_pos: dict[str, dict[str, str]] = g2p_pos or {}

    @classmethod
    def load(cls, path: Path | str) -> G2pCorrections:
        """Charge les corrections depuis un fichier JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            g2p={k.lower(): v for k, v in data.get("g2p", {}).items()},
            g2p_pos={k.lower(): v for k, v in data.get("g2p_pos", {}).items()},
        )

    def get_phone(self, word: str, pos: str | None = None) -> str | None:
        """Retourne la phonemisation corrigee, ou None.

        Ordre :
        1. g2p_pos[word][POS] (match exact puis prefixe)
        2. g2p_pos[word][_default]
        3. g2p[word]
        """
        key = word.lower()

        if key in self._g2p_pos:
            pos_map = self._g2p_pos[key]
            if pos:
                pos_upper = pos.upper()
                if pos_upper in pos_map:
                    return pos_map[pos_upper]
                for tag, phone in pos_map.items():
                    if tag != "_default" and pos_upper.startswith(tag):
                        return phone
            if "_default" in pos_map:
                return pos_map["_default"]

        return self._g2p.get(key)

    @property
    def n_entries(self) -> int:
        """Nombre total d'entrees (g2p + g2p_pos)."""
        return len(self._g2p) + len(self._g2p_pos)


# ══════════════════════════════════════════════════════════════════════════════
# Post-traitement : regles R1-R13
# ══════════════════════════════════════════════════════════════════════════════

_VOWELS_ORTHO = set("aeiouyàâäéèêëïîôùûüœæ")

_RE_ENNE = re.compile(r"i?ennes?$", re.IGNORECASE)
_RE_ISME = re.compile(r"ismes?$", re.IGNORECASE)
_RE_TION = re.compile(r"tions?$", re.IGNORECASE)
_RE_ENT_FINAL = re.compile(r"ent$", re.IGNORECASE)

_HIATUS_CLUSTERS = {
    "bl", "pl", "cl", "gl", "fl",
    "tr", "br", "pr", "dr", "cr", "gr", "fr", "vr",
}
_RE_HIATUS = re.compile(
    r"(" + "|".join(_HIATUS_CLUSTERS) + r")i[eéèêaâoôuû]",
    re.IGNORECASE,
)
_RE_G_INITIAL = re.compile(r"^g[a-zéèêëàâäïîôùûüœæ]", re.IGNORECASE)


def postprocess(word: str, phone: str, pos: str | None = None) -> str:
    """Applique des corrections systematiques sur la sortie G2P.

    Regles appliquees dans l'ordre :
      R1 : x -> ks/ɡz
      R2 : -enne/-ienne -> ɛn (pas ɛ̃n)
      R3 : -isme -> izm (pas ism)
      R4 : -tion -> sjɔ̃ (POS-aware : pas pour VER)
      R5 : -ent verbal -> strip ɑ̃ si VER 3pp
      R7 : Prefixe ex- -> ɛɡz/ɛks
      R8 : Hiatus /ij/ manquant apres cluster consonantique
      R9 : Assimilation bs->ps, bt->pt
      R10 : /ɡ/ initial disparu
      R11 : Digraphe oe/oeu -> œ/ø
      R13 : ø -> œ en syllabe fermee
    """
    lower = word.lower()

    # R1 : x -> ks ou ɡz
    if "x" in lower and "ks" not in phone and "ɡz" not in phone:
        phone = _fix_x_phoneme(lower, phone)

    # R2 : -enne/-ienne -> ɛn (denasalisation)
    if _RE_ENNE.search(lower) and "ɛ\u0303n" in phone:
        phone = phone.replace("ɛ\u0303n", "ɛn")

    # R3 : -isme -> izm (voisement s->z avant m)
    if _RE_ISME.search(lower) and "ism" in phone:
        phone = phone.replace("ism", "izm")

    # R4 : -tion -> sjɔ̃ (POS-aware)
    if pos and _RE_TION.search(lower) and "tjɔ\u0303" in phone:
        is_verb = pos.upper().startswith("VER") or pos.upper().startswith("AUX")
        has_st = lower.endswith("stion") or lower.endswith("stions")
        has_xt = lower.endswith("xtion") or lower.endswith("xtions")
        if not is_verb and not has_st and not has_xt:
            phone = phone.replace("tjɔ\u0303", "sjɔ\u0303")

    # R5 : -ent verbal -> strip ɑ̃ final si VER 3pp
    if (
        len(lower) > 4
        and _RE_ENT_FINAL.search(lower)
        and pos
        and (pos.upper().startswith("VER") or pos.upper().startswith("AUX"))
        and phone.endswith("ɑ\u0303")
    ):
        phone = phone[:-2]

    # R7 : Prefixe ex- -> ɛɡz/ɛks
    phone = _fix_ex_prefix(lower, phone)

    # R8 : Hiatus /ij/ manquant apres cluster consonantique
    phone = _fix_hiatus_ij(lower, phone)

    # R9 : Assimilation bs->ps, bt->pt (devoisement)
    phone = phone.replace("bs", "ps").replace("bt", "pt")

    # R10 : /ɡ/ initial disparu
    phone = _fix_g_initial(lower, phone)

    # R11 : Digraphe oe/oeu -> œ/ø
    phone = phone.replace("oœ", "œ").replace("oø", "ø")

    # R13 : ø -> œ en syllabe fermee
    if "ø" in phone:
        phone = _fix_mid_rounded_vowel(phone)

    return phone


def _fix_x_phoneme(word: str, phone: str) -> str:
    """Corrige la phonemisation du 'x'."""
    replacement = None
    for i, ch in enumerate(word):
        if ch != "x":
            continue
        if i == len(word) - 1:
            continue
        if i + 1 < len(word) and word[i + 1] == "-":
            continue
        if word[i + 1:].startswith("ième") or word[i + 1:].startswith("iém"):
            continue
        is_ex_prefix = (
            i >= 1
            and word[i - 1] == "e"
            and i + 1 < len(word)
            and word[i + 1] in _VOWELS_ORTHO
        )
        replacement = "ɡz" if is_ex_prefix else "ks"
        break

    if replacement is None:
        return phone
    if len(phone) < len(word) * 0.4:
        return phone

    if "z" in phone:
        phone = phone.replace("z", replacement, 1)
    elif "s" in phone and replacement == "ks":
        if "s" not in word.replace("x", ""):
            phone = phone.replace("s", "ks", 1)
    return phone


def _fix_ex_prefix(word: str, phone: str) -> str:
    """R7 : Corrige le prefixe ex- (et inex-, reex-, coex-)."""
    ex_pos = -1
    for prefix in ("inex", "réex", "coex", "ex"):
        if word.startswith(prefix):
            ex_pos = len(prefix) - 2
            break
    if ex_pos < 0:
        return phone

    x_pos = ex_pos + 1
    if x_pos + 1 >= len(word):
        return phone
    after_x = word[x_pos + 1]

    if after_x in _VOWELS_ORTHO:
        if "ɛz" in phone and "ɛɡz" not in phone:
            phone = phone.replace("ɛz", "ɛɡz", 1)
    else:
        if "ɛk" in phone and "ɛks" not in phone:
            phone = phone.replace("ɛk", "ɛks", 1)
    return phone


def _fix_hiatus_ij(word: str, phone: str) -> str:
    """R8 : Insere /i/ avant /j/ quand le hiatus est manquant apres cluster."""
    if not _RE_HIATUS.search(word):
        return phone

    phonemes = iter_phonemes(phone)
    result = []
    modified = False

    for idx, ph in enumerate(phonemes):
        if (
            ph == "j"
            and not modified
            and idx > 0
            and est_consonne(phonemes[idx - 1])
            and (idx < 2 or phonemes[idx - 2] != "i")
        ):
            if idx >= 2 and (
                est_consonne(phonemes[idx - 2])
                or est_semi_voyelle(phonemes[idx - 2])
            ):
                result.append("i")
                modified = True
        result.append(ph)

    return "".join(result) if modified else phone


def _fix_g_initial(word: str, phone: str) -> str:
    """R10 : Restaure /ɡ/ initial disparu."""
    if not _RE_G_INITIAL.match(word):
        return phone

    phonemes = iter_phonemes(phone)
    if not phonemes:
        return phone
    if phonemes[0] == "ɡ":
        return phone

    if len(phonemes) >= 2 and phonemes[0] == phonemes[1] and est_consonne(phonemes[0]):
        return "ɡ" + "".join(phonemes[1:])

    if est_voyelle(phonemes[0]) and len(phonemes) < len(word):
        return "ɡ" + phone

    return phone


def _is_legal_onset(consonants: list[str]) -> bool:
    """Verifie si une sequence de consonnes forme une attaque legale."""
    n = len(consonants)
    if n <= 1:
        return True
    if n == 2:
        c1, c2 = consonants
        if c1 in _OBSTRUENTS and c2 in ("l", "ʁ"):
            return True
        if est_semi_voyelle(c2):
            return True
        return False
    if n == 3:
        c1, c2, c3 = consonants
        if c1 in _OBSTRUENTS and c2 in ("l", "ʁ") and est_semi_voyelle(c3):
            return True
        return False
    return False


def _fix_mid_rounded_vowel(phone: str) -> str:
    """R13 : Remplace ø par œ en syllabe fermee."""
    phonemes = iter_phonemes(phone)
    result = list(phonemes)
    modified = False

    for i, ph in enumerate(result):
        if ph != "ø":
            continue

        remaining = result[i + 1:]

        if not remaining:
            continue

        # Exception : ø + z -> suffixe -euse
        if remaining[0] == "z":
            continue

        if all(est_consonne(p) or est_semi_voyelle(p) for p in remaining):
            result[i] = "œ"
            modified = True
            continue

        consonants: list[str] = []
        next_vowel: str | None = None
        for j in range(i + 1, len(result)):
            if est_voyelle(result[j]):
                next_vowel = result[j]
                break
            consonants.append(result[j])

        if not consonants:
            continue

        if next_vowel == "ə":
            result[i] = "œ"
            modified = True
            continue

        if _is_legal_onset(consonants):
            continue

        result[i] = "œ"
        modified = True

    return "".join(result) if modified else phone


# ══════════════════════════════════════════════════════════════════════════════
# API publique
# ══════════════════════════════════════════════════════════════════════════════

class LecturaG2P:
    """Convertisseur grapheme-phoneme pour le francais.

    Supporte trois backends :
      - **CRF** (defaut) : zero dependance, fichier JSON
      - **BiLSTM** : necessite onnxruntime, fichier ONNX + vocab JSON
      - **Seq2Seq** : necessite onnxruntime, encoder ONNX + decoder ONNX + vocab JSON

    Args:
        model_path: Chemin vers le modele (JSON pour CRF, ONNX pour BiLSTM/Seq2Seq).
        corrections_path: Chemin vers les corrections (JSON, optionnel).
        vocab_path: Chemin vers le vocabulaire (JSON, requis pour BiLSTM et Seq2Seq).
        decoder_path: Chemin vers le decoder ONNX (requis pour Seq2Seq).

    Exemple :
        >>> g2p = LecturaG2P("modele/g2p_model_crf.json",
        ...                   corrections_path="modele/g2p_corrections_crf.json")
        >>> g2p.predict("bonjour")
        'bɔ̃ʒuʁ'
        >>> g2p.predict_batch(["le", "chat", "mange"])
        ['lə', 'ʃa', 'mɑ̃ʒ']
    """

    def __init__(
        self,
        model_path: str | Path,
        corrections_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
    ) -> None:
        model_path = Path(model_path)

        if model_path.suffix == ".json":
            self._model = CrfG2pModel.load(model_path)
            self._backend = "crf"
        elif model_path.suffix == ".onnx":
            if vocab_path is None:
                raise ValueError(
                    "vocab_path est requis pour les backends ONNX (fichier JSON)"
                )
            # Detecter le type de modele via le vocab
            is_seq2seq = False
            if decoder_path is not None:
                is_seq2seq = True
            else:
                with open(vocab_path, encoding="utf-8") as f:
                    vocab_data = json.load(f)
                if vocab_data.get("type") == "seq2seq":
                    is_seq2seq = True

            if is_seq2seq:
                if decoder_path is None:
                    raise ValueError(
                        "decoder_path est requis pour le backend Seq2Seq"
                    )
                self._model = OnnxG2pSeq2SeqModel(model_path, decoder_path, vocab_path)
                self._backend = "seq2seq"
            else:
                self._model = OnnxG2pModel(model_path, vocab_path)
                self._backend = "bilstm"
        else:
            raise ValueError(f"Format de modele non supporte : {model_path.suffix}")

        self._corrections: G2pCorrections | None = None
        if corrections_path is not None:
            self._corrections = G2pCorrections.load(corrections_path)

    @property
    def backend(self) -> str:
        """Retourne le type de backend : 'crf', 'bilstm' ou 'seq2seq'."""
        return self._backend

    def predict(self, word: str, pos: str | None = None) -> str:
        """Predit la transcription IPA d'un mot.

        Args:
            word: Mot a transcrire.
            pos: Tag POS optionnel (pour la desambiguisation des homographes).

        Returns:
            Transcription IPA (ex: "bɔ̃ʒuʁ").
        """
        if not word:
            return ""

        # 1) Corrections (priorite maximale)
        if self._corrections is not None:
            phone = self._corrections.get_phone(word, pos=pos)
            if phone is not None:
                return phone

        # 2) Modele G2P + post-traitement
        phone = self._model.predict(word)
        if phone:
            phone = postprocess(word, phone, pos=pos)
        return phone or ""

    def predict_batch(
        self,
        words: list[str],
        pos_tags: list[str | None] | None = None,
    ) -> list[str]:
        """Predit les transcriptions IPA pour une liste de mots.

        Args:
            words: Liste de mots a transcrire.
            pos_tags: Liste optionnelle de tags POS (meme longueur que words).

        Returns:
            Liste de transcriptions IPA.
        """
        if pos_tags is None:
            pos_tags = [None] * len(words)
        return [self.predict(w, pos=p) for w, p in zip(words, pos_tags)]

    def predict_formatted(self, word: str, pos: str | None = None) -> str:
        """Retourne un resultat formate lisible.

        Returns:
            Chaine "mot → /phone/".
        """
        phone = self.predict(word, pos=pos)
        return f"{word} → /{phone}/"


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entree direct
# ══════════════════════════════════════════════════════════════════════════════

def _load_backend(here: Path, backend: str) -> LecturaG2P:
    """Charge un backend depuis son sous-dossier."""
    if backend == "crf":
        d = here / "G2P_CRF" / "modele"
        model_path = d / "g2p_model_crf.json"
        corr_path = d / "g2p_corrections_crf.json"
        if not model_path.exists():
            print(f"ERREUR : modele non trouve : {model_path}", file=sys.stderr)
            sys.exit(1)
        corr = corr_path if corr_path.exists() else None
        return LecturaG2P(model_path, corrections_path=corr)

    elif backend == "bilstm":
        d = here / "G2P_BiLSTM" / "modele"
        model_path = d / "g2p_model_bilstm_int8.onnx"
        vocab_path = d / "g2p_vocab.json"
        corr_path = d / "g2p_corrections_bilstm.json"
        if not model_path.exists():
            print(f"ERREUR : modele non trouve : {model_path}", file=sys.stderr)
            sys.exit(1)
        corr = corr_path if corr_path.exists() else None
        return LecturaG2P(model_path, vocab_path=vocab_path, corrections_path=corr)

    elif backend == "seq2seq":
        d = here / "G2P_Seq2Seq" / "modele"
        enc = d / "g2p_seq2seq_encoder_int8.onnx"
        dec = d / "g2p_seq2seq_decoder_int8.onnx"
        vocab_path = d / "g2p_seq2seq_vocab.json"
        corr_path = d / "g2p_corrections_seq2seq.json"
        if not enc.exists():
            print(f"ERREUR : modele non trouve : {enc}", file=sys.stderr)
            sys.exit(1)
        corr = corr_path if corr_path.exists() else None
        return LecturaG2P(enc, decoder_path=dec, vocab_path=vocab_path, corrections_path=corr)

    else:
        print(f"ERREUR : backend inconnu : {backend}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Point d'entree en ligne de commande (facade multi-backend).

    Options de backend :
        --crf      CRF (defaut) — charge G2P_CRF/modele/
        --bilstm   BiLSTM — charge G2P_BiLSTM/modele/
        --seq2seq  Seq2Seq — charge G2P_Seq2Seq/modele/
        --compare  Charge les 3 et affiche cote a cote
    """
    here = Path(__file__).parent

    # Separer les flags backend des mots a transcrire
    backend_flags = {"--crf", "--bilstm", "--seq2seq", "--compare"}
    args = sys.argv[1:]
    chosen: str | None = None
    words: list[str] = []

    for arg in args:
        if arg in backend_flags:
            chosen = arg.lstrip("-")
        else:
            words.append(arg)

    if chosen is None:
        chosen = "crf"

    if not words:
        print(f"Lectura G2P v{__version__} — facade multi-backend")
        print()
        print("Usage : python lectura_g2p.py [--crf|--bilstm|--seq2seq|--compare] mot1 mot2 ...")
        print()
        print("  --crf      Backend CRF (defaut)")
        print("  --bilstm   Backend BiLSTM (ONNX)")
        print("  --seq2seq  Backend Seq2Seq (ONNX)")
        print("  --compare  Compare les 3 backends cote a cote")
        return

    if chosen == "compare":
        backends = {}
        for name in ("crf", "bilstm", "seq2seq"):
            try:
                backends[name] = _load_backend(here, name)
            except SystemExit:
                print(f"  WARN: {name} non disponible", file=sys.stderr)

        if not backends:
            print("ERREUR : aucun backend disponible", file=sys.stderr)
            sys.exit(1)

        header = f"  {'mot':20}"
        for name in backends:
            header += f" {'/' + name + '/':>22}"
        print(header)
        print("  " + "─" * (20 + 22 * len(backends)))

        for word in words:
            line = f"  {word:20}"
            for name, g2p in backends.items():
                phone = g2p.predict(word)
                line += f" {'/' + phone + '/':>22}"
            print(line)
    else:
        g2p = _load_backend(here, chosen)
        for word in words:
            phone = g2p.predict(word)
            print(f"  {word:20} → /{phone}/")


if __name__ == "__main__":
    main()
