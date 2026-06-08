"""Lexique phonemique pour le post-traitement STT.

Wrapper leger autour d'une base SQLite (lexique_lectura_v5.db ou
phone_lexicon.db) qui charge les formes IPA en memoire pour des
lookups O(1) rapides.

Le PhoneLexicon est utilise par la segmentation (_segmentation.py)
pour determiner si un mot IPA est connu, trouver son orthographe,
et resoudre les ambiguites.

Copyright (C) 2025-2026 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

import sqlite3
import unicodedata
from pathlib import Path


# Consonnes de liaison (pour strip_liaisons, ipa_with_liaison)
LIAISON_CONSONANTS = frozenset({"z", "t", "n", "p", "k", "ʁ"})

# Consonnes clitiques elidables → grapheme apostrophe
CLITIC_MAP: dict[str, str] = {
    "l": "l'", "d": "d'", "n": "n'", "s": "s'",
    "ʒ": "j'", "k": "qu'", "m": "m'", "t": "t'",
}

# Forme pleine des clitiques (pour rejoin_elisions devant consonne)
CLITIC_FULL: dict[str, str] = {
    "l": "le", "d": "de", "n": "ne", "s": "se",
    "ʒ": "je", "k": "que", "m": "me", "t": "te",
}

# Voyelles IPA (pour detection frontiere)
IPA_VOWELS = frozenset("aeiouyøœəɛɔɑɥ")


# ── Table de confusions CTC (Tier 4) ─────────────────────────
# Paires bidirectionnelles derivees de analyze_errors.py
CONFUSION_PAIRS: list[tuple[str, str]] = [
    ("ɛ", "e"),      # 700 confusions
    ("ɔ", "o"),      # 163
    ("ø", "ə"),      # 129
    ("ø", "o"),      # CTC confond ø/o (peu vs pot)
    ("ɔ̃", "ɑ̃"),     # 131
    ("e", "i"),      # 123
    ("d", "t"),      # 66
    ("s", "z"),      # 65
    ("l", "n"),      # 61
    ("o", "u"),      # 81
    ("ɑ̃", "ɛ̃"),     # 62
    ("a", "ɛ"),      # 88
    ("a", "ɑ"),      # CTC ɑ est utilise seul (sans nasale)
]

# Index rapide : phone → set de substitutions possibles
_CONFUSION_MAP: dict[str, set[str]] = {}
for _a, _b in CONFUSION_PAIRS:
    _CONFUSION_MAP.setdefault(_a, set()).add(_b)
    _CONFUSION_MAP.setdefault(_b, set()).add(_a)

SCHWA = "ə"

# Normalisation IPA : certains lexiques utilisent g (U+0067) vs ɡ (U+0261)
_IPA_NORMALIZE = str.maketrans({"g": "ɡ"})  # ASCII g → IPA script g


def _normalize_ipa(s: str) -> str:
    """Normalise les codepoints IPA ambigus (g→ɡ)."""
    return s.translate(_IPA_NORMALIZE)


def _ipa_grapheme_clusters(s: str) -> list[str]:
    """Segmente une chaine IPA en clusters (base + diacritiques combinants).

    Ex: "ɑ̃fɑ̃" → ["ɑ̃", "f", "ɑ̃"] (chaque voyelle nasale est un cluster).
    """
    s = unicodedata.normalize("NFC", s)
    clusters: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        j = i + 1
        while j < len(s) and unicodedata.category(s[j]).startswith("M"):
            j += 1
        clusters.append(s[i:j])
        i = j
    return clusters


class PhoneLexicon:
    """Encapsule un lexique SQLite avec un cache phone precharge.

    Attributes:
        phone_set:         ensemble de toutes les formes IPA connues
        phone_set_reliable: phones avec freq >= seuil (fiables)
        phone_to_best:     phone → (ortho, freq) la plus frequente
        ipa_with_liaison:  formes IPA+consonne_latente (pour strip_liaisons)
    """

    _RELIABLE_FREQ_MIN = 0.1

    def __init__(self, db_path: str | Path) -> None:
        self.phone_set: set[str] = set()
        self.phone_set_reliable: set[str] = set()
        self.phone_to_best: dict[str, tuple[str, float]] = {}
        self.ipa_with_liaison: set[str] = set()
        self._db_path = str(db_path)
        self._preload()

    def _preload(self) -> None:
        """Charge phone_set et phone_to_best depuis la DB."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Charger les phones depuis la table lexique
        try:
            cursor.execute(
                "SELECT phone, ortho, freq, cgram FROM lexique "
                "WHERE phone != '' AND phone IS NOT NULL"
            )
        except sqlite3.OperationalError:
            conn.close()
            return

        for phone, ortho, freq, cgram in cursor:
            phone = phone.strip()
            if not phone:
                continue
            freq = freq or 0.0
            if freq == 0.0 and cgram == "NOM PROPRE":
                freq = 0.15
            self.phone_set.add(phone)
            if freq >= self._RELIABLE_FREQ_MIN:
                self.phone_set_reliable.add(phone)
            prev = self.phone_to_best.get(phone)
            if prev is None or freq > prev[1]:
                self.phone_to_best[phone] = (ortho, freq)

        # Charger ipa_with_liaison (table formes, si elle existe)
        try:
            cursor.execute(
                "SELECT phone, consonne_latente FROM formes "
                "WHERE phone IS NOT NULL AND phone != '' "
                "AND consonne_latente IS NOT NULL AND consonne_latente != ''"
            )
            for phone, liaison in cursor:
                phone = phone.strip()
                if phone and liaison:
                    self.ipa_with_liaison.add(phone + liaison.strip())
        except sqlite3.OperationalError:
            pass

        conn.close()

    @staticmethod
    def _strip_sep(phone: str) -> str:
        """Retire les separateurs (-/'/\u2019) d'un phone pour le lookup."""
        return phone.replace("-", "").replace("'", "").replace("\u2019", "")

    def exists(self, phone: str) -> bool:
        """Verifie si un phone est dans le lexique."""
        p = self._strip_sep(phone)
        return p in self.phone_set or _normalize_ipa(p) in self.phone_set

    def best_ortho(self, phone: str) -> str | None:
        """Retourne l'orthographe la plus frequente pour un phone."""
        p = self._strip_sep(phone)
        entry = self.phone_to_best.get(p) or self.phone_to_best.get(_normalize_ipa(p))
        return entry[0] if entry else None

    def best_freq(self, phone: str) -> float:
        """Retourne la frequence la plus haute pour un phone."""
        p = self._strip_sep(phone)
        entry = self.phone_to_best.get(p) or self.phone_to_best.get(_normalize_ipa(p))
        return entry[1] if entry else 0.0

    def all_entries(self, phone: str) -> list[dict]:
        """Retourne toutes les entrees lexicales pour un phone."""
        p = self._strip_sep(phone)
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT ortho, freq, cgram, genre, nombre FROM lexique "
                "WHERE phone = ? OR phone = ?", (p, _normalize_ipa(p))
            )
            results = []
            seen: set[str] = set()
            for ortho, freq, cgram, genre, nombre in cursor:
                key = f"{ortho}|{cgram}"
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "ortho": ortho,
                    "freq": freq or 0.0,
                    "cgram": cgram or "",
                    "genre": genre or "",
                    "nombre": nombre or "",
                    "phone": p,
                })
            conn.close()
            return results
        except sqlite3.OperationalError:
            conn.close()
            return []

    def all_entries_with_perturbations(
        self, phone: str, k_max: int = 20,
    ) -> tuple[list[dict], dict[str, str]]:
        """Retourne les entrees lexicales enrichies par perturbations CTC.

        Cherche d'abord les entrees exactes, puis ajoute des candidats via
        les confusions CTC (tier 4) et reductions de doubles (tier 5).

        Returns:
            entries: liste de dicts dedupliques par ortho.lower(), tries
                     exact d'abord puis perturbes, par freq desc.
                     Chaque dict a un champ ``_source`` ("exact" ou le phone variant).
            resolved_map: {ortho.lower(): source_phone} pour tracabilite
        """
        exact_entries = [dict(e, _source="exact") for e in self.all_entries(phone)]

        perturbed_entries: list[dict] = []
        seen_variants: set[str] = set()

        for tier_func in (_tier4_ctc_confusions, _tier5_reduce_doubles):
            variants = tier_func(phone, self)
            for variant_phone, _freq, _n_edits in variants:
                if variant_phone in seen_variants:
                    continue
                seen_variants.add(variant_phone)
                for e in self.all_entries(variant_phone):
                    entry = dict(e)
                    entry["_source"] = variant_phone
                    perturbed_entries.append(entry)

        # Dedupliquer par ortho.lower() : exact prioritaire
        by_ortho: dict[str, dict] = {}
        for e in exact_entries:
            key = e["ortho"].lower()
            if key not in by_ortho or (e.get("freq", 0) or 0) > (by_ortho[key].get("freq", 0) or 0):
                by_ortho[key] = e
        for e in perturbed_entries:
            key = e["ortho"].lower()
            if key not in by_ortho:
                by_ortho[key] = e
            elif by_ortho[key].get("_source") != "exact":
                if (e.get("freq", 0) or 0) > (by_ortho[key].get("freq", 0) or 0):
                    by_ortho[key] = e

        exact_list = sorted(
            [e for e in by_ortho.values() if e.get("_source") == "exact"],
            key=lambda e: -(e.get("freq", 0) or 0),
        )
        perturbed_list = sorted(
            [e for e in by_ortho.values() if e.get("_source") != "exact"],
            key=lambda e: -(e.get("freq", 0) or 0),
        )
        merged = (exact_list + perturbed_list)[:k_max]

        resolved_map: dict[str, str] = {}
        for e in merged:
            resolved_map[e["ortho"].lower()] = e.get("_source", "exact")

        return merged, resolved_map


# ── Perturbations CTC (fonctions tier) ───────────────────────


def _tier4_ctc_confusions(
    word: str, lex: PhoneLexicon,
) -> list[tuple[str, float, int]]:
    """Tier 4 : substitutions basees sur les confusions CTC.

    Applique UNE substitution a la fois depuis _CONFUSION_MAP.
    Inclut aussi schwa insertion/deletion.

    Returns list of (phone, freq, n_edits).
    """
    segments = _ipa_grapheme_clusters(word)
    n = len(segments)
    candidates: list[tuple[str, float, int]] = []
    seen: set[str] = set()

    for i in range(n):
        base = segments[i][0] if segments[i] else segments[i]
        # Substitution par confusion CTC
        if base in _CONFUSION_MAP:
            for replacement in _CONFUSION_MAP[base]:
                new_seg = replacement + segments[i][1:]  # garder diacritiques
                variant = "".join(segments[:i] + [new_seg] + segments[i + 1:])
                if variant not in seen and variant in lex.phone_set:
                    seen.add(variant)
                    candidates.append((variant, lex.best_freq(variant), 1))

        # Suppression d'un schwa
        if segments[i] == SCHWA:
            variant = "".join(segments[:i] + segments[i + 1:])
            if variant and variant not in seen and variant in lex.phone_set:
                seen.add(variant)
                candidates.append((variant, lex.best_freq(variant), 1))

    # Insertion d'un schwa entre deux segments
    for i in range(n + 1):
        variant = "".join(segments[:i] + [SCHWA] + segments[i:])
        if variant not in seen and variant in lex.phone_set:
            seen.add(variant)
            candidates.append((variant, lex.best_freq(variant), 1))

    # Insertion de consonne nasale apres voyelle nasale
    _NASAL_VOWELS = {"ɑ̃", "ɛ̃", "ɔ̃", "œ̃"}
    _LABIALS = {"p", "b", "m"}
    for i in range(n - 1):
        if segments[i] in _NASAL_VOWELS:
            next_base = segments[i + 1][0] if segments[i + 1] else ""
            nasal = "m" if next_base in _LABIALS else "n"
            variant = "".join(segments[:i + 1] + [nasal] + segments[i + 1:])
            if variant not in seen and variant in lex.phone_set:
                seen.add(variant)
                candidates.append((variant, lex.best_freq(variant), 1))

    return candidates


def _tier5_reduce_doubles(
    word: str, lex: PhoneLexicon,
) -> list[tuple[str, float, int]]:
    """Tier 5 : reduction doubles/voyelles consecutives.

    - Doubles consonnes/voyelles : ss→s, ee→e, ll→l, etc.
    - Voyelles consecutives : ea→e ou ea→a

    Returns list of (phone, freq, n_edits).
    """
    segments = _ipa_grapheme_clusters(word)
    n = len(segments)
    candidates: list[tuple[str, float, int]] = []
    seen: set[str] = set()

    for i in range(n - 1):
        # Doubles identiques → reduire a un seul
        if segments[i] == segments[i + 1]:
            variant = "".join(segments[:i] + segments[i + 1:])
            if variant and variant not in seen and variant in lex.phone_set:
                seen.add(variant)
                candidates.append((variant, lex.best_freq(variant), 1))

        # Voyelles consecutives → garder l'une ou l'autre
        base_i = segments[i][0] if segments[i] else ""
        base_j = segments[i + 1][0] if segments[i + 1] else ""
        if base_i in IPA_VOWELS and base_j in IPA_VOWELS and base_i != base_j:
            v1 = "".join(segments[:i + 1] + segments[i + 2:])
            if v1 and v1 not in seen and v1 in lex.phone_set:
                seen.add(v1)
                candidates.append((v1, lex.best_freq(v1), 1))
            v2 = "".join(segments[:i] + segments[i + 1:])
            if v2 and v2 not in seen and v2 in lex.phone_set:
                seen.add(v2)
                candidates.append((v2, lex.best_freq(v2), 1))

    # Expansion : phone simple → double
    for i in range(n):
        variant = "".join(segments[:i] + [segments[i], segments[i]] + segments[i + 1:])
        if variant not in seen and variant in lex.phone_set:
            seen.add(variant)
            candidates.append((variant, lex.best_freq(variant), 1))

    return candidates
