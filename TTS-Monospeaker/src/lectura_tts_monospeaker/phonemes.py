"""Vocabulaire phonetique et conversion IPA → phone_ids.

Self-contained : charge le vocabulaire depuis phoneme_vocab.json.
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


def get_vocab() -> list[str]:
    """Retourne la liste ordonnee du vocabulaire."""
    return _load_vocab()["vocab"]


def ipa_to_phones(ipa: str) -> list[str]:
    """Parse une chaine IPA en liste de phones, gerant les multi-caracteres.

    Exemples :
        "bɔ̃ʒuʁ" → ["b", "ɔ̃", "ʒ", "u", "ʁ"]
        "tʃ" → ["tʃ"]
    """
    # Mapper œ̃ → ɛ̃ (voyelle nasale rare, pas dans le vocabulaire du modele)
    ipa = ipa.replace("œ\u0303", "ɛ\u0303")

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
                phones.append(phone)  # garder pour diagnostic
            i = j

    return phones


# Fallback mapping for phones not in the vocab (robustesse inference)
_PHONE_FALLBACKS: dict[str, str] = {
    "c": "k", "ɟ": "ɡ", "ts": "t",
    "x": "k", "ɣ": "ɡ", "ɹ": "ʁ",
    "mʲ": "m", "ʎ": "l", "r": "ʁ", "ɾ": "ʁ",
    "h": "#", "ʔ": "#",
}

# Voyelles pour le duration floor
_VOWELS = frozenset({
    "a", "e", "i", "o", "u", "y",
    "ɛ", "ɔ", "ø", "œ", "ə", "ɑ",
    "ɛ̃", "ɔ̃", "ɑ̃", "œ̃",
})


def _resolve_phone_id(phone: str, phone2id: dict[str, int], unk_id: int) -> int:
    """Resolve a phone to its ID, applying fallbacks if needed."""
    pid = phone2id.get(phone)
    if pid is not None:
        return pid
    fallback = _PHONE_FALLBACKS.get(phone)
    if fallback is not None:
        return phone2id.get(fallback, unk_id)
    return unk_id


def get_phone_min_frames(phone: str) -> int:
    """Return minimum duration in frames for a phone.

    Vowels: 5 frames (~58 ms), consonants/other: 3 frames (~35 ms).
    """
    if phone in _VOWELS:
        return 5
    return 3


def ipa_to_phone_ids(ipa: str, add_silence: bool = True) -> list[int]:
    """Convertit une chaine IPA en sequence de phone IDs.

    Args:
        ipa: Chaine IPA (ex: "bɔ̃ʒuʁ")
        add_silence: Ajouter SIL (#) au debut et a la fin

    Returns:
        Liste d'entiers (phone IDs)
    """
    phone2id = get_phone2id()
    sil_id = phone2id["#"]
    unk_id = phone2id.get("<UNK>", 1)

    phones = ipa_to_phones(ipa)
    ids = [_resolve_phone_id(p, phone2id, unk_id) for p in phones]

    if add_silence:
        ids = [sil_id] + ids + [sil_id]

    return ids


def phones_to_ids(phones: list[str]) -> list[int]:
    """Convertit une liste de phones en IDs."""
    phone2id = get_phone2id()
    unk_id = phone2id.get("<UNK>", 1)
    return [_resolve_phone_id(p, phone2id, unk_id) for p in phones]
