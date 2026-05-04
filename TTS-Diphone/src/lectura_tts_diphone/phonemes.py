"""Vocabulaire phonetique et conversion IPA → phones.

Self-contained : charge le vocabulaire depuis phoneme_vocab.json.
Utilise pour le parsing IPA dans le pipeline diphone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "data"
_VOCAB_CACHE: dict[str, Any] | None = None

# Combinants Unicode pour l'IPA (nasalisation, etc.)
_COMBINING_CHARS = {'\u0303', '\u0308', '\u0300', '\u0301', '\u0302',
                    '\u031d', '\u031e', '\u0325', '\u032a', '\u0327', '\u030C'}

# Fallback mapping pour phones non standard
PHONE_FALLBACKS: dict[str, str] = {
    "mʲ": "m",
    "ʎ": "l",
    "nʲ": "n",
    "tʲ": "t",
    "dʲ": "d",
    "r": "ʁ",
    "ɾ": "ʁ",
    "h": "#",
    "ʔ": "#",
}


def _load_vocab() -> dict[str, Any]:
    """Charge le vocabulaire (singleton)."""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is not None:
        return _VOCAB_CACHE

    vocab_path = _DATA_DIR / "phoneme_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"phoneme_vocab.json introuvable : {vocab_path}\n"
            "Assurez-vous que le package est correctement installe."
        )

    with open(vocab_path, encoding="utf-8") as f:
        _VOCAB_CACHE = json.load(f)
    return _VOCAB_CACHE


def get_phone2id() -> dict[str, int]:
    """Retourne le mapping phone → ID."""
    return _load_vocab()["phone2id"]


def ipa_to_phones(ipa: str) -> list[str]:
    """Parse une chaine IPA en liste de phones, gerant les multi-caracteres.

    Exemples :
        "bɔ̃ʒuʁ" → ["b", "ɔ̃", "ʒ", "u", "ʁ"]
        "tʃ" → ["tʃ"]
    """
    phone2id = get_phone2id()
    phones: list[str] = []
    i = 0

    while i < len(ipa):
        found = False

        # Essayer multi-caracteres (3, 2)
        for length in [3, 2]:
            if i + length <= len(ipa):
                candidate = ipa[i:i + length]
                j = i + length
                while j < len(ipa) and ipa[j] in _COMBINING_CHARS:
                    candidate += ipa[j]
                    j += 1
                if candidate in phone2id:
                    phones.append(candidate)
                    i = j
                    found = True
                    break

        if not found:
            # Caractere simple + combinants
            phone = ipa[i]
            j = i + 1
            while j < len(ipa) and ipa[j] in _COMBINING_CHARS:
                phone += ipa[j]
                j += 1

            if phone in phone2id:
                phones.append(phone)
            elif phone.strip():
                # Essayer fallback
                fb = PHONE_FALLBACKS.get(phone)
                if fb and fb in phone2id:
                    phones.append(fb)
                else:
                    phones.append(phone)  # garder pour diagnostic
            i = j

    return phones


def phone_to_id(phone: str) -> int:
    """Convertit un phone en ID vocabulaire, avec fallback."""
    phone2id = get_phone2id()
    unk_id = phone2id.get("<UNK>", 1)

    if phone in phone2id:
        return phone2id[phone]
    fb = PHONE_FALLBACKS.get(phone)
    if fb is not None:
        return phone2id.get(fb, unk_id)
    return unk_id


def syllables_to_phones(syllables_ipa: list[str]) -> list[str]:
    """Aplatit une liste de syllabes IPA en liste de phones."""
    phones = []
    for syl in syllables_ipa:
        phones.extend(ipa_to_phones(syl))
    return phones
