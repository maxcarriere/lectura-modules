"""Lectura G2P — Convertisseur graphème-phonème pour le français (backend Seq2Seq).

Utilise un modèle Seq2Seq (encoder-decoder + attention) via ONNX Runtime.

Usage rapide :
    from lectura_g2p import LecturaG2P

    g2p = LecturaG2P("modele/g2p_seq2seq_encoder_int8.onnx",
                       decoder_path="modele/g2p_seq2seq_decoder_int8.onnx",
                       vocab_path="modele/g2p_seq2seq_vocab.json",
                       corrections_path="modele/g2p_corrections_seq2seq.json")
    phone = g2p.predict("bonjour")   # → "bɔ̃ʒuʁ"
    phone = g2p.predict("maison")    # → "mɛzɔ̃"

Pre-requis : pip install onnxruntime numpy

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
# Modele Seq2Seq ONNX (necessite onnxruntime + numpy)
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
    """Convertisseur graphème-phonème pour le français (backend Seq2Seq).

    Nécessite onnxruntime et numpy. Utilise un modèle Seq2Seq
    (encoder-decoder + attention) exporté en ONNX.

    Args:
        encoder_path: Chemin vers l'encoder ONNX.
        decoder_path: Chemin vers le decoder ONNX.
        vocab_path: Chemin vers le vocabulaire (fichier JSON).
        corrections_path: Chemin vers les corrections (JSON, optionnel).

    Exemple :
        >>> g2p = LecturaG2P("modele/g2p_seq2seq_encoder_int8.onnx",
        ...                   decoder_path="modele/g2p_seq2seq_decoder_int8.onnx",
        ...                   vocab_path="modele/g2p_seq2seq_vocab.json",
        ...                   corrections_path="modele/g2p_corrections_seq2seq.json")
        >>> g2p.predict("bonjour")
        'bɔ̃ʒuʁ'
        >>> g2p.predict_batch(["le", "chat", "mange"])
        ['lə', 'ʃa', 'mɑ̃ʒ']
    """

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        vocab_path: str | Path,
        corrections_path: str | Path | None = None,
    ) -> None:
        self._model = OnnxG2pSeq2SeqModel(encoder_path, decoder_path, vocab_path)
        self._backend = "seq2seq"

        self._corrections: G2pCorrections | None = None
        if corrections_path is not None:
            self._corrections = G2pCorrections.load(corrections_path)

    @property
    def backend(self) -> str:
        """Retourne le type de backend : 'seq2seq'."""
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

def main() -> None:
    """Point d'entree en ligne de commande."""
    here = Path(__file__).parent
    encoder_path = here / "modele" / "g2p_seq2seq_encoder_int8.onnx"
    decoder_path = here / "modele" / "g2p_seq2seq_decoder_int8.onnx"
    vocab_path = here / "modele" / "g2p_seq2seq_vocab.json"
    corrections_path = here / "modele" / "g2p_corrections_seq2seq.json"

    if not encoder_path.exists():
        print(f"ERREUR : modele non trouve : {encoder_path}", file=sys.stderr)
        sys.exit(1)

    corr = corrections_path if corrections_path.exists() else None
    g2p = LecturaG2P(
        encoder_path, decoder_path=decoder_path,
        vocab_path=vocab_path, corrections_path=corr,
    )

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        words = text.split()
        for word in words:
            phone = g2p.predict(word)
            print(f"  {word:20} → /{phone}/")
    else:
        print(f"Lectura G2P v{__version__} — backend: {g2p.backend}")
        if g2p._corrections:
            print(f"  Corrections : {g2p._corrections.n_entries} entrees")
        print()
        print("Usage : python lectura_g2p.py mot1 mot2 ...")


if __name__ == "__main__":
    main()
