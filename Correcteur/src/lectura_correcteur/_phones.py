"""Generateur de variantes phonetiques proches + estimateur par regles.

Deplace depuis lectura-lexique (specifique au correcteur).
"""

from __future__ import annotations

import unicodedata


# ======================================================================
# Estimateur phonetique par regles (zero dependance)
# ======================================================================

_VOWELS = set("aeiouyàâäéèêëïîôùûüœæ")

# Trigrams (priorite haute)
_TRIGRAMS: list[tuple[str, str]] = [
    ("eau", "o"),
    ("ain", "ɛ̃"),
    ("ein", "ɛ̃"),
    ("oin", "wɛ̃"),
]

# Digrams
_DIGRAMS: list[tuple[str, str]] = [
    ("ch", "ʃ"),
    ("ph", "f"),
    ("ou", "u"),
    ("ai", "ɛ"),
    ("ei", "ɛ"),
    ("au", "o"),
    ("oi", "wa"),
    ("gn", "ɲ"),
    ("qu", "k"),
]

# Nasales (non suivies d'une voyelle)
_NASALS: list[tuple[str, str]] = [
    ("an", "ɑ̃"),
    ("am", "ɑ̃"),
    ("en", "ɑ̃"),
    ("em", "ɑ̃"),
    ("on", "ɔ̃"),
    ("om", "ɔ̃"),
    ("in", "ɛ̃"),
    ("im", "ɛ̃"),
    ("un", "œ̃"),
    ("um", "œ̃"),
]

# Simples
_SIMPLES: dict[str, str] = {
    "ç": "s",
    "j": "ʒ",
    "y": "i",
    "x": "ks",
    "r": "ʁ",
}

# Consonnes finales muettes (sauf CaReFuL)
_MUETTES_FINALES = set("tdspxz")
# CaReFuL : ces consonnes finales sont prononcees
_CAREFUL: dict[str, str] = {"c": "k", "r": "ʁ", "f": "f", "l": "l"}


def estimer_phone(mot: str) -> str:
    """Estime la prononciation IPA d'un mot francais par regles.

    Ordre : trigrams > digrams > nasales > contextuels > simples > finales muettes.
    Approximatif mais suffisant pour le scoring phonetique du correcteur.
    """
    if not mot:
        return ""
    low = mot.lower()
    result: list[str] = []
    i = 0
    n = len(low)

    while i < n:
        matched = False

        # --- Trigrams ---
        if i + 2 < n:
            tri = low[i:i + 3]
            for pat, ipa in _TRIGRAMS:
                if tri == pat:
                    result.append(ipa)
                    i += 3
                    matched = True
                    break
            if matched:
                continue

        # --- Digrams ---
        if i + 1 < n:
            di = low[i:i + 2]

            # gu + e/i → g (avant les digrams generaux)
            if di == "gu" and i + 2 < n and low[i + 2] in "eiéèêë":
                result.append("g")
                i += 2
                continue

            for pat, ipa in _DIGRAMS:
                if di == pat:
                    result.append(ipa)
                    i += 2
                    matched = True
                    break
            if matched:
                continue

            # --- Nasales (non suivies d'une voyelle) ---
            for pat, ipa in _NASALS:
                if di == pat:
                    # Si suivi d'une voyelle → pas nasal
                    if i + 2 < n and low[i + 2] in _VOWELS:
                        break
                    # Si suivi d'un autre n/m → pas nasal (ex: "pomme")
                    if i + 2 < n and low[i + 2] in "nm":
                        break
                    result.append(ipa)
                    i += 2
                    matched = True
                    break
            if matched:
                continue

        # --- Contextuels ---
        ch = low[i]

        # c + e/i → s, c + autre → k
        if ch == "c":
            if i + 1 < n and low[i + 1] in "eiéèêë":
                result.append("s")
            else:
                result.append("k")
            i += 1
            continue

        # g + e/i → ʒ (gu+e/i deja traite)
        if ch == "g":
            if i + 1 < n and low[i + 1] in "eiéèêë":
                result.append("ʒ")
            else:
                result.append("g")
            i += 1
            continue

        # s intervocalique → z, s final → muet
        if ch == "s":
            if i == n - 1:
                pass  # s final muet
            elif (
                i > 0
                and i + 1 < n
                and low[i - 1] in _VOWELS
                and low[i + 1] in _VOWELS
            ):
                result.append("z")
            else:
                result.append("s")
            i += 1
            continue

        # --- Simples ---
        if ch in _SIMPLES:
            # x final est muet (sauf exceptions rares)
            if ch == "x" and i == n - 1:
                i += 1
                continue
            result.append(_SIMPLES[ch])
            i += 1
            continue

        # --- Voyelles de base ---
        if ch in _VOWELS:
            # e final muet
            if ch == "e" and i == n - 1:
                i += 1
                continue
            if ch in "àâ":
                result.append("a")
            elif ch in "éè":
                result.append("ɛ")
            elif ch == "ê":
                result.append("ɛ")
            elif ch in "ëä":
                result.append(ch.replace("ë", "e").replace("ä", "a"))
            elif ch in "ùû":
                result.append("y")
            elif ch == "ü":
                result.append("y")
            elif ch in "ïî":
                result.append("i")
            elif ch == "ô":
                result.append("o")
            elif ch == "œ":
                result.append("œ")
            elif ch == "æ":
                result.append("e")
            else:
                result.append(ch)
            i += 1
            continue

        # --- Consonnes finales ---
        if i == n - 1 and ch in _MUETTES_FINALES:
            i += 1
            continue

        # CaReFuL finales
        if i == n - 1 and ch in _CAREFUL:
            result.append(_CAREFUL[ch])
            i += 1
            continue

        # Consonnes standards
        if ch.isalpha():
            result.append(ch)

        i += 1

    return "".join(result)


class _RuleBasedG2P:
    """Wrapper autour de estimer_phone() pour satisfaire l'interface prononcer()."""

    def prononcer(self, mot: str) -> str | None:
        r = estimer_phone(mot)
        return r if r else None

# Confusions vocaliques courantes (SMS/informel)
_PHONE_VARIANTES: dict[str, list[str]] = {
    "e": ["e", "\u025b", "\u0259"],       # e <-> ɛ <-> ə
    "\u025b": ["\u025b", "e", "\u0259"],   # ɛ <-> e <-> ə
    "\u0259": ["\u0259", "e", "\u025b"],   # ə <-> e <-> ɛ
    "o": ["o", "\u0254"],        # o <-> ɔ
    "\u0254": ["\u0254", "o"],   # ɔ <-> o
    "\u00f8": ["\u00f8", "\u0153"],  # ø <-> œ
    "\u0153": ["\u0153", "\u00f8"],  # œ <-> ø
}


def _segmenter_ipa(phone: str) -> list[str]:
    """Decoupe une chaine IPA en segments (char + diacritiques combinants)."""
    segments: list[str] = []
    i = 0
    while i < len(phone):
        seg = phone[i]
        i += 1
        while i < len(phone) and unicodedata.category(phone[i]).startswith("M"):
            seg += phone[i]
            i += 1
        segments.append(seg)
    return segments


def generer_phones_proches(phone: str) -> list[str]:
    """Genere les variantes proches d'un phone (e<->ɛ, o<->ɔ, etc.).

    Remplace chaque voyelle ambigue par ses variantes et retourne
    la liste de tous les phones possibles (incluant l'original).
    Limite a 1 substitution a la fois.
    """
    segments = _segmenter_ipa(phone)
    results = {phone}

    for idx, seg in enumerate(segments):
        base = seg[0] if seg else seg
        if base in _PHONE_VARIANTES:
            for v in _PHONE_VARIANTES[base]:
                new_seg = v + seg[1:]  # conserver diacritiques
                new_segs = segments[:idx] + [new_seg] + segments[idx + 1:]
                results.add("".join(new_segs))

    return list(results)


def generer_phones_d1(phone: str) -> list[str]:
    """Genere les variantes a distance phonetique <= 1 (deletions + substitutions vocaliques).

    Combine :
    - Suppression d'un segment IPA (ex: [zanimo] -> [animo])
    - Variantes vocaliques (ex: [dinozɔʁ] -> [dinozoʁ])
    Retourne la liste sans l'original.
    """
    segments = _segmenter_ipa(phone)
    results: set[str] = set()

    # Deletions : supprimer un segment a la fois
    for idx in range(len(segments)):
        variant = "".join(segments[:idx] + segments[idx + 1:])
        if variant and variant != phone:
            results.add(variant)

    # Substitutions vocaliques (1 a la fois)
    for idx, seg in enumerate(segments):
        base = seg[0] if seg else seg
        if base in _PHONE_VARIANTES:
            for v in _PHONE_VARIANTES[base]:
                new_seg = v + seg[1:]
                new_segs = segments[:idx] + [new_seg] + segments[idx + 1:]
                variant = "".join(new_segs)
                if variant != phone:
                    results.add(variant)

    return list(results)
