"""Reconnaissance IPA → Formule.

A partir d'une chaine IPA (espace-tolerant), retrouve la formule source
et le texte francais, puis verifie par aller-retour (forward pass).

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

from lectura_formules._chargeur import (
    inv_phone_nombre,
    inv_phone_mois,
    inv_phone_devise,
    inv_phone_lettre,
    inv_phone_ordinal,
    inv_phone_symbole,
    inv_phone_grec,
    symboles as _load_symboles,
    grec as _load_grec,
    heure_words as _load_heure_words,
    pourcent_words as _load_pourcent_words,
    devises as _load_devises,
)
from lectura_formules.lecture_formules import (
    LectureFormuleResult,
    lire_nombre,
    lire_date,
    lire_heure,
    lire_monnaie,
    lire_pourcentage,
    lire_ordinal,
    lire_sigle,
    lire_maths,
)


# ══════════════════════════════════════════════════════════════════════════════
# Types internes
# ══════════════════════════════════════════════════════════════════════════════

FormulaType = Literal[
    "nombre", "date", "heure", "monnaie",
    "pourcentage", "ordinal", "sigle",
]


@dataclass(frozen=True, slots=True)
class IpaToken:
    """Token reconnu dans le flux IPA."""
    category: str   # nombre, mois, heure_word, devise, pourcent, lettre, ordinal, special
    value: object   # int pour nombre, int pour mois, str pour lettre, etc.
    key: str        # cle d'origine dans la table (ex: "40", "et_1", "EUR")
    ipa: str        # IPA original (sans espaces) ayant matche


# ══════════════════════════════════════════════════════════════════════════════
# Construction de la table de lookup unifiee (singleton)
# ══════════════════════════════════════════════════════════════════════════════

_lookup_table: list[tuple[str, IpaToken]] | None = None


def _nfc(s: str) -> str:
    """Normalise en NFC unicode."""
    return unicodedata.normalize("NFC", s)


def _strip_spaces(s: str) -> str:
    """Supprime tous les espaces d'une chaine."""
    return s.replace(" ", "")


def _build_lookup_table() -> list[tuple[str, IpaToken]]:
    """Construit la table unifiee IPA->token, triee par longueur decroissante."""
    global _lookup_table
    if _lookup_table is not None:
        return _lookup_table

    entries: dict[str, IpaToken] = {}

    def _add(ipa_raw: str, token: IpaToken) -> None:
        key = _nfc(_strip_spaces(ipa_raw))
        if not key:
            return
        # Plus long gagne; si meme longueur, garder le premier
        if key not in entries or len(token.ipa) > len(entries[key].ipa):
            entries[key] = token

    # -- Nombres (unites) --
    for phone, (val, unit_key) in inv_phone_nombre().items():
        _add(phone, IpaToken("nombre", val, unit_key, _nfc(_strip_spaces(phone))))

    # -- Mois --
    for phone, num in inv_phone_mois().items():
        _add(phone, IpaToken("mois", num, str(num), _nfc(_strip_spaces(phone))))

    # -- Heure words --
    heure_words = _load_heure_words()
    for word_key, (ortho, phone) in heure_words.items():
        _add(phone, IpaToken("heure_word", word_key, word_key, _nfc(_strip_spaces(phone))))

    # -- Devises --
    for phone, (code, role) in inv_phone_devise().items():
        _add(phone, IpaToken("devise", (code, role), code, _nfc(_strip_spaces(phone))))

    # -- Pourcent --
    pourcent_words = _load_pourcent_words()
    for sym, (ortho, phone) in pourcent_words.items():
        _add(phone, IpaToken("pourcent", sym, sym, _nfc(_strip_spaces(phone))))

    # -- Lettres --
    for phone, letter in inv_phone_lettre().items():
        _add(phone, IpaToken("lettre", letter, letter, _nfc(_strip_spaces(phone))))

    # -- Ordinaux --
    for phone, cardinal in inv_phone_ordinal().items():
        _add(phone, IpaToken("ordinal", cardinal, cardinal, _nfc(_strip_spaces(phone))))

    # -- Mots speciaux --
    # "virgule" pour la separation decimale
    _add("viʁɡyl", IpaToken("special", "virgule", "virgule", _nfc("viʁɡyl")))

    # "premier/premiere" (dates: 1er janvier)
    _add("pʁømje", IpaToken("special", "premier", "premier", _nfc("pʁømje")))
    _add("pʁømjɛʁ", IpaToken("special", "premiere", "premiere", _nfc("pʁømjɛʁ")))

    # "moins" (signe negatif) — deja dans nombres via symboles,
    # mais on s'assure qu'il est present comme "signe"
    _add("mwɛ̃", IpaToken("special", "moins", "moins", _nfc("mwɛ̃")))

    # "et" (connecteur monnaie)
    _add("e", IpaToken("special", "et", "et", _nfc("e")))

    # Trier par longueur decroissante de cle IPA
    sorted_entries = sorted(entries.items(), key=lambda x: len(x[0]), reverse=True)
    _lookup_table = sorted_entries
    return _lookup_table


# ══════════════════════════════════════════════════════════════════════════════
# Tokenisation IPA (greedy longest-match)
# ══════════════════════════════════════════════════════════════════════════════


def _tokenize_ipa(ipa: str) -> list[IpaToken] | None:
    """Tokenise une chaine IPA en liste de IpaToken.

    Retourne None si la tokenisation echoue (caracteres non reconnus restants).
    """
    table = _build_lookup_table()
    normalized = _nfc(_strip_spaces(ipa))

    if not normalized:
        return None

    tokens: list[IpaToken] = []
    pos = 0
    length = len(normalized)

    while pos < length:
        matched = False
        for entry_key, entry_token in table:
            entry_len = len(entry_key)
            if pos + entry_len <= length and normalized[pos:pos + entry_len] == entry_key:
                tokens.append(entry_token)
                pos += entry_len
                matched = True
                break
        if not matched:
            return None

    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Detection du type de formule
# ══════════════════════════════════════════════════════════════════════════════


def _detect_type(tokens: list[IpaToken], type_hint: str | None) -> FormulaType:
    """Detecte le type de formule a partir des tokens."""
    if type_hint is not None:
        return type_hint  # type: ignore[return-value]

    categories = {t.category for t in tokens}
    specials = {t.value for t in tokens if t.category == "special"}

    # 1. Mois present → date
    if "mois" in categories:
        return "date"

    # 2. Heure word present → heure
    if "heure_word" in categories:
        return "heure"

    # 3. Devise present → monnaie
    if "devise" in categories:
        return "monnaie"

    # 4. Pourcent present → pourcentage
    if "pourcent" in categories:
        return "pourcentage"

    # 5. Ordinal present → ordinal
    if "ordinal" in categories:
        return "ordinal"

    # 5b. "premier"/"premiere" seul → ordinal
    if specials & {"premier", "premiere"} and "nombre" not in categories:
        return "ordinal"

    # 6. Tous lettres (+ eventuellement chiffres isoles) → sigle
    non_lettre = {t.category for t in tokens} - {"lettre", "nombre"}
    if not non_lettre and all(
        t.category == "lettre" or (t.category == "nombre" and isinstance(t.value, int) and 0 <= t.value <= 9)
        for t in tokens
    ):
        # Verifier qu'il y a au moins une lettre
        if any(t.category == "lettre" for t in tokens):
            return "sigle"

    # 7. Defaut → nombre
    return "nombre"


# ══════════════════════════════════════════════════════════════════════════════
# Reconstruction des nombres
# ══════════════════════════════════════════════════════════════════════════════


def _reconstruct_number(tokens: list[IpaToken]) -> int:
    """Reconstruit un entier a partir de tokens de type nombre.

    Algorithme inverse des regles de composition francaises.
    Gere le pattern "quatre-vingts" (4 × 20 = 80).

    L'algorithme utilise deux accumulateurs :
    - result : valeur accumulee par les multiplicateurs d'echelle (mille, million...)
    - current : valeur du groupe courant (0-999)
    - sub : sous-accumulateur pour dizaines+unites a l'interieur du groupe

    Pour le pattern quatre-vingts, on detecte la sequence (4, 20) et on
    la remplace par 80 avant le calcul principal.
    """
    if not tokens:
        return 0

    # Extraire uniquement les tokens nombre avec leurs valeurs
    vals: list[int] = []
    for tok in tokens:
        if isinstance(tok.value, int):
            vals.append(tok.value)

    if not vals:
        return 0

    # Pre-traitement : remplacer les sequences (4, 20) par 80
    processed: list[int] = []
    i = 0
    while i < len(vals):
        if vals[i] == 4 and i + 1 < len(vals) and vals[i + 1] == 20:
            processed.append(80)
            i += 2
        else:
            processed.append(vals[i])
            i += 1

    result = 0
    current = 0

    for val in processed:
        if val >= 1_000_000:
            # million, milliard, billion : multiplie le groupe courant
            result += max(current, 1) * val
            current = 0
        elif val == 1000:
            result += max(current, 1) * 1000
            current = 0
        elif val == 100:
            current = max(current, 1) * 100
        else:
            current += val

    return result + current



# ══════════════════════════════════════════════════════════════════════════════
# Reconstructeurs par type
# ══════════════════════════════════════════════════════════════════════════════


def _reconstruct_nombre(tokens: list[IpaToken]) -> str | None:
    """Reconstruit un nombre (entier ou decimal) depuis les tokens."""
    # Verifier s'il y a un signe negatif
    negatif = False
    work_tokens = list(tokens)
    if work_tokens and work_tokens[0].category == "special" and work_tokens[0].value == "moins":
        negatif = True
        work_tokens = work_tokens[1:]
    elif (work_tokens and work_tokens[0].category == "nombre"
          and work_tokens[0].key == "moins"):
        negatif = True
        work_tokens = work_tokens[1:]

    # Verifier s'il y a une virgule (decimal)
    virgule_pos = None
    for i, tok in enumerate(work_tokens):
        if tok.category == "special" and tok.value == "virgule":
            virgule_pos = i
            break

    if virgule_pos is not None:
        # Decimal
        int_tokens = work_tokens[:virgule_pos]
        dec_tokens = work_tokens[virgule_pos + 1:]

        int_part = _reconstruct_number(int_tokens)

        # Partie decimale : les tokens representent les chiffres decimaux
        # Ils peuvent etre un nombre lu comme un entier (ex: "quatorze" = 14 → "14")
        # ou des chiffres individuels
        dec_part = _reconstruct_number(dec_tokens)
        dec_str = str(dec_part)

        sign = "-" if negatif else ""
        return f"{sign}{int_part},{dec_str}"
    else:
        # Entier
        n = _reconstruct_number(work_tokens)
        sign = "-" if negatif else ""
        return f"{sign}{n}"


def _reconstruct_date(tokens: list[IpaToken]) -> str | None:
    """Reconstruit une date depuis les tokens."""
    # Trouver le token mois
    mois_idx = None
    for i, tok in enumerate(tokens):
        if tok.category == "mois":
            mois_idx = i
            break

    if mois_idx is None:
        return None

    # Tokens avant le mois = jour
    jour_tokens = tokens[:mois_idx]
    # Tokens apres le mois = annee
    annee_tokens = tokens[mois_idx + 1:]

    mois_num = tokens[mois_idx].value

    # Jour : peut etre "premier" ou un nombre
    if jour_tokens and jour_tokens[0].category == "special" and jour_tokens[0].value == "premier":
        jour = 1
    else:
        jour_nombre_tokens = [t for t in jour_tokens if t.category == "nombre"]
        jour = _reconstruct_number(jour_nombre_tokens) if jour_nombre_tokens else 1

    # Annee
    annee_nombre_tokens = [t for t in annee_tokens if t.category == "nombre"]
    annee = _reconstruct_number(annee_nombre_tokens) if annee_nombre_tokens else 0

    if jour < 1 or jour > 31 or not isinstance(mois_num, int) or mois_num < 1 or mois_num > 12:
        return None

    return f"{jour:02d}/{mois_num:02d}/{annee}"


def _reconstruct_heure(tokens: list[IpaToken]) -> str | None:
    """Reconstruit une heure depuis les tokens."""
    # Trouver le premier heure_word
    heure_word_idx = None
    for i, tok in enumerate(tokens):
        if tok.category == "heure_word" and tok.key in ("heure", "heures"):
            heure_word_idx = i
            break

    if heure_word_idx is None:
        return None

    # Tokens avant = heures
    h_tokens = [t for t in tokens[:heure_word_idx] if t.category == "nombre"]
    heures = _reconstruct_number(h_tokens) if h_tokens else 0

    # Tokens apres le mot "heure(s)" = minutes (avant "minute(s)")
    remaining = tokens[heure_word_idx + 1:]

    # Filtrer les tokens minutes_word si presents
    min_tokens: list[IpaToken] = []
    for tok in remaining:
        if tok.category == "heure_word" and tok.key in ("minute", "minutes"):
            break
        if tok.category == "nombre":
            min_tokens.append(tok)

    minutes = _reconstruct_number(min_tokens) if min_tokens else 0

    if minutes > 0:
        return f"{heures}h{minutes:02d}"
    else:
        return f"{heures}h00"


def _reconstruct_monnaie(tokens: list[IpaToken]) -> str | None:
    """Reconstruit un montant monetaire depuis les tokens."""
    # Trouver le premier token devise majeur
    devise_idx = None
    devise_code = None
    for i, tok in enumerate(tokens):
        if tok.category == "devise":
            code, role = tok.value
            if role == "major":
                devise_idx = i
                devise_code = code
                break

    if devise_idx is None:
        return None

    devises = _load_devises()
    cur = devises.get(devise_code)
    if cur is None:
        return None
    sym = cur.get("symbole", devise_code)

    # Signe negatif
    negatif = False
    start_idx = 0
    if tokens and tokens[0].category == "special" and tokens[0].value == "moins":
        negatif = True
        start_idx = 1

    # Tokens avant la devise = montant majeur
    major_tokens = [t for t in tokens[start_idx:devise_idx] if t.category == "nombre"]
    major = _reconstruct_number(major_tokens) if major_tokens else 0

    # Chercher la partie mineure apres la devise
    remaining = tokens[devise_idx + 1:]
    minor = 0
    if remaining:
        # Sauter le "et" ou "virgule" si present
        idx = 0
        if remaining[0].category == "special" and remaining[0].value in ("et", "virgule"):
            idx = 1
        # Filtrer jusqu'au token devise minor (ou fin)
        minor_tokens: list[IpaToken] = []
        for tok in remaining[idx:]:
            if tok.category == "devise":
                break
            if tok.category == "nombre":
                minor_tokens.append(tok)
        minor = _reconstruct_number(minor_tokens) if minor_tokens else 0

    sign = "-" if negatif else ""
    if minor > 0:
        return f"{sign}{major},{minor:02d}{sym}"
    else:
        return f"{sign}{major}{sym}"


def _reconstruct_pourcentage(tokens: list[IpaToken]) -> str | None:
    """Reconstruit un pourcentage depuis les tokens."""
    # Trouver le token pourcent
    pct_idx = None
    pct_sym = None
    for i, tok in enumerate(tokens):
        if tok.category == "pourcent":
            pct_idx = i
            pct_sym = tok.value  # "%" ou "‰"
            break

    if pct_idx is None:
        return None

    # Tokens avant = nombre
    nombre_tokens = tokens[:pct_idx]
    nombre_str = _reconstruct_nombre(nombre_tokens)
    if nombre_str is None:
        return None

    return f"{nombre_str}{pct_sym}"


def _reconstruct_ordinal(tokens: list[IpaToken]) -> str | None:
    """Reconstruit un ordinal depuis les tokens."""
    # Trouver le token ordinal
    ordinal_idx = None
    for i, tok in enumerate(tokens):
        if tok.category == "ordinal":
            ordinal_idx = i
            break

    if ordinal_idx is None:
        # Peut etre "premier" / "premiere" (special)
        for i, tok in enumerate(tokens):
            if tok.category == "special" and tok.value in ("premier", "premiere"):
                if tok.value == "premiere":
                    return "1re"
                return "1er"
        return None

    # Le token ordinal contient le cardinal de base
    # Tokens avant l'ordinal contribuent au nombre
    nombre_tokens = [t for t in tokens[:ordinal_idx] if t.category == "nombre"]
    base = _reconstruct_number(nombre_tokens) if nombre_tokens else 0

    # L'ordinal lui-meme peut contenir une valeur numerique
    # via le cardinal de base dans inv_phone_ordinal
    # Le cardinal name est dans tok.value (ex: "douze")
    # On doit retrouver la valeur numerique du cardinal
    ordinal_tok = tokens[ordinal_idx]
    cardinal_name = ordinal_tok.value  # ex: "douze", "cent", "mille"

    # Chercher dans inv_phone_nombre (par texte francais)
    from lectura_formules._chargeur import inv_fr_nombre
    fr_table = inv_fr_nombre()

    ordinal_val = 0
    if cardinal_name in fr_table:
        ordinal_val = fr_table[cardinal_name][0]

    total = base + ordinal_val if base > 0 else ordinal_val
    if total <= 0:
        return None

    if total == 1:
        return "1er"
    return f"{total}e"


def _reconstruct_sigle(tokens: list[IpaToken]) -> str | None:
    """Reconstruit un sigle depuis les tokens."""
    parts: list[str] = []
    for tok in tokens:
        if tok.category == "lettre":
            parts.append(tok.value)
        elif tok.category == "nombre" and isinstance(tok.value, int):
            parts.append(str(tok.value))
        else:
            return None
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Distance d'edition (Levenshtein) — zero dependance
# ══════════════════════════════════════════════════════════════════════════════


def _levenshtein(s1: str, s2: str) -> int:
    """Calcule la distance de Levenshtein entre deux chaines."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,      # deletion
                curr[j] + 1,           # insertion
                prev[j] + (c1 != c2),  # substitution
            ))
        prev = curr
    return prev[-1]


def _stt_max_distance(phone_len: int) -> int:
    """Tolerance de distance en fonction de la longueur de la chaine IPA.

    - < 3 phones : 0 (match exact requis)
    - 3-5 phones : 1 edit
    - 6+ phones : ~30% d'erreur tolere (min 2)

    La tokenisation avec variantes CTC gere les confusions simples ;
    la verification aller-retour rattrape les residus de composition
    multi-mots (2+ confusions empilees, ex: nasale + glide manquante).
    """
    if phone_len < 3:
        return 0
    if phone_len <= 5:
        return 1
    return max(2, phone_len * 30 // 100)


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation IPA pour comparaison
# ══════════════════════════════════════════════════════════════════════════════


def _normalize_ipa_for_compare(ipa: str) -> str:
    """Normalise une chaine IPA pour comparaison : NFC, strip espaces."""
    return _nfc(_strip_spaces(ipa))


import re as _re

# Schwas CTC : ə (U+0259) entre consonnes ou en fin de cluster
_RE_SCHWA = _re.compile("ə")

# ɑ (U+0251) non suivi de combining tilde → normaliser en a (U+0061)
# ɑ̃ (U+0251 + U+0303) est une voyelle nasale, on la garde
_RE_ALPHA_STANDALONE = _re.compile("ɑ(?!\u0303)")

# ɔ (U+0254) non suivi de combining tilde → normaliser en o (U+006F)
# ɔ̃ (U+0254 + U+0303) est une voyelle nasale, on la garde
_RE_OPEN_O_STANDALONE = _re.compile("ɔ(?!\u0303)")

# ɛ (U+025B) non suivi de combining tilde → normaliser en e (U+0065)
# ɛ̃ (U+025B + U+0303) est une voyelle nasale, on la garde
_RE_OPEN_E_STANDALONE = _re.compile("ɛ(?!\u0303)")


def _normalize_ipa_for_stt(ipa: str) -> str:
    """Normalise une chaine IPA pour la reconnaissance STT.

    - NFC
    - Strip espaces
    - Supprime les schwas (ə) inseres par le CTC/lexique
    - Normalise ɑ (U+0251) standalone en a (U+0061)
    - Normalise ɔ (U+0254) standalone en o (U+006F)
    - Normalise ɛ (U+025B) standalone en e (U+0065)
    - Normalise g (U+0067) en ɡ (U+0261)
    """
    s = _nfc(_strip_spaces(ipa))
    s = _RE_SCHWA.sub("", s)
    s = _RE_ALPHA_STANDALONE.sub("a", s)
    s = _RE_OPEN_O_STANDALONE.sub("o", s)
    s = _RE_OPEN_E_STANDALONE.sub("e", s)
    s = s.replace("g", "ɡ")  # ASCII g → IPA script g
    return s


# Regles de confusion CTC : substitutions plausibles (IPA canonique → variante CTC)
_CTC_CONFUSION_RULES: list[tuple[str, str]] = [
    # Nasales : ɛ̃ ↔ œ̃ ↔ ɑ̃
    ("ɛ̃", "ɑ̃"),
    ("ɛ̃", "œ̃"),
    ("œ̃", "ɑ̃"),
    ("ɑ̃", "œ̃"),
    ("ɑ̃", "ɛ̃"),
    ("œ̃", "ɛ̃"),
    # Voisement
    ("k", "ɡ"),
    ("ɡ", "k"),
    ("t", "d"),
    ("d", "t"),
    ("p", "b"),
    ("b", "p"),
    # Glides manquantes
    ("lj", "l"),     # miljɔ̃ → milɔ̃
    ("l", "lj"),     # reverse
    # Consonnes finales
    ("tʁ", "t"),     # katʁ → kat
]


def _generate_variants(canonical_ipa: str, max_variants: int = 12) -> list[str]:
    """Variantes CTC plausibles d'un IPA canonique (1 regle a la fois)."""
    variants: set[str] = set()
    nfc = _nfc(_strip_spaces(canonical_ipa))
    for pattern, replacement in _CTC_CONFUSION_RULES:
        if pattern in nfc:
            v = nfc.replace(pattern, replacement, 1)
            if v != nfc:
                variants.add(v)
    # + versions STT-normalisees
    extra: set[str] = set()
    for v in variants:
        nv = _normalize_ipa_for_stt(v)
        if nv and nv != v and nv != nfc:
            extra.add(nv)
    variants.update(extra)
    return list(variants)[:max_variants]


# Table de variantes CTC supplementaires pour la tokenisation STT
_STT_PHONE_VARIANTS: dict[str, list[tuple[str, int, str]]] | None = None


def _build_stt_variants() -> dict[str, list[tuple[str, int, str]]]:
    """Construit la table de variantes IPA pour la reconnaissance STT.

    Retourne un dict supplementaire {ipa_normalise: [(ortho, valeur, cle)]}
    pour les formes CTC non couvertes par donnees_formules.json.
    """
    global _STT_PHONE_VARIANTS
    if _STT_PHONE_VARIANTS is not None:
        return _STT_PHONE_VARIANTS
    _STT_PHONE_VARIANTS = {}
    # Variantes connues du CTC / lexique
    variants = [
        ("wit", 8, "8"),       # huit: ɥit → wit
        ("ɥi", 8, "8"),        # huit: variante sans t final
        ("diz", 10, "10"),     # dix: liaison diz au lieu de dis
        ("kat", 4, "4"),       # quatre: sans ʁ final
    ]
    for phone, val, key in variants:
        normalized = _nfc(_strip_spaces(phone))
        _STT_PHONE_VARIANTS[normalized] = (val, key)

    # Variantes auto-generees depuis les regles de confusion CTC
    for canonical_phone, (val, unit_key) in inv_phone_nombre().items():
        nfc = _nfc(_strip_spaces(canonical_phone))
        for variant in _generate_variants(nfc):
            if variant not in _STT_PHONE_VARIANTS:
                _STT_PHONE_VARIANTS[variant] = (val, unit_key)

    return _STT_PHONE_VARIANTS


# Lookup table etendue pour STT (avec variantes CTC)
_stt_lookup_table: list[tuple[str, IpaToken]] | None = None


def _build_stt_lookup_table() -> list[tuple[str, IpaToken]]:
    """Table de lookup etendue avec variantes CTC pour la reconnaissance STT."""
    global _stt_lookup_table
    if _stt_lookup_table is not None:
        return _stt_lookup_table

    # Partir de la table standard
    base_table = dict(_build_lookup_table())

    # Ajouter les variantes CTC (seulement si pas deja presentes)
    variants = _build_stt_variants()
    for phone, (val, key) in variants.items():
        nphone = _nfc(phone)
        if nphone not in base_table:
            base_table[nphone] = IpaToken("nombre", val, key, nphone)

    # Variantes CTC auto-generees pour toutes les categories
    variant_entries: dict[str, IpaToken] = {}
    for ipa_key, token in list(base_table.items()):
        for variant in _generate_variants(ipa_key):
            if variant not in base_table and variant not in variant_entries:
                variant_entries[variant] = token
    base_table.update(variant_entries)

    # Re-normaliser les entrees existantes (strip schwas, ɑ→a)
    extended: dict[str, IpaToken] = {}
    for ipa_key, token in base_table.items():
        if ipa_key:  # ignorer les cles vides
            extended[ipa_key] = token
        # Ajouter aussi la version normalisee STT
        norm_key = _normalize_ipa_for_stt(ipa_key)
        if norm_key and norm_key != ipa_key and norm_key not in extended:
            extended[norm_key] = token

    _stt_lookup_table = sorted(extended.items(), key=lambda x: len(x[0]), reverse=True)
    return _stt_lookup_table


def _tokenize_ipa_stt(ipa: str) -> list[IpaToken] | None:
    """Tokenise une chaine IPA avec tolerance aux variantes CTC/STT.

    Comme _tokenize_ipa mais :
    - Pre-normalise l'IPA (schwas, ɑ→a)
    - Utilise la table etendue avec variantes CTC
    """
    table = _build_stt_lookup_table()
    normalized = _normalize_ipa_for_stt(ipa)

    if not normalized:
        return None

    tokens: list[IpaToken] = []
    pos = 0
    length = len(normalized)

    while pos < length:
        matched = False
        for entry_key, entry_token in table:
            entry_len = len(entry_key)
            if pos + entry_len <= length and normalized[pos:pos + entry_len] == entry_key:
                tokens.append(entry_token)
                pos += entry_len
                matched = True
                break
        if not matched:
            return None

    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# API publique
# ══════════════════════════════════════════════════════════════════════════════


def reconnaitre_ipa(
    ipa: str,
    type_hint: str | None = None,
) -> LectureFormuleResult | None:
    """Reconnait une formule a partir de sa transcription IPA.

    Retourne un LectureFormuleResult (display_fr, phone, valeur, display_num)
    ou None si la reconnaissance echoue.

    Args:
        ipa: Chaine IPA (espaces toleres).
        type_hint: Force le type de formule ("nombre", "date", "heure",
                   "monnaie", "pourcentage", "ordinal", "sigle").

    Returns:
        LectureFormuleResult si reconnu et verifie, None sinon.
    """
    # Tokenisation
    tokens = _tokenize_ipa(ipa)
    if not tokens:
        return None

    # Detection du type
    formula_type = _detect_type(tokens, type_hint)

    # Reconstruction
    formula_str: str | None = None

    if formula_type == "nombre":
        formula_str = _reconstruct_nombre(tokens)
    elif formula_type == "date":
        formula_str = _reconstruct_date(tokens)
    elif formula_type == "heure":
        formula_str = _reconstruct_heure(tokens)
    elif formula_type == "monnaie":
        formula_str = _reconstruct_monnaie(tokens)
    elif formula_type == "pourcentage":
        formula_str = _reconstruct_pourcentage(tokens)
    elif formula_type == "ordinal":
        formula_str = _reconstruct_ordinal(tokens)
    elif formula_type == "sigle":
        formula_str = _reconstruct_sigle(tokens)

    if formula_str is None:
        return None

    # Forward pass pour verification aller-retour
    forward_fn = {
        "nombre": lire_nombre,
        "date": lire_date,
        "heure": lire_heure,
        "monnaie": lire_monnaie,
        "pourcentage": lire_pourcentage,
        "ordinal": lire_ordinal,
        "sigle": lire_sigle,
    }.get(formula_type)

    if forward_fn is None:
        return None

    try:
        result = forward_fn(formula_str)
    except Exception:
        return None

    # Verification : l'IPA produit par le forward doit correspondre
    forward_ipa = _normalize_ipa_for_compare(result.phone)
    input_ipa = _normalize_ipa_for_compare(ipa)

    if forward_ipa == input_ipa:
        return result

    # Echec de verification — la reconstruction ne correspond pas
    return None


def reconnaitre_ipa_stt(
    ipa: str,
    type_hint: str | None = None,
) -> LectureFormuleResult | None:
    """Reconnait une formule a partir d'IPA produit par un pipeline STT.

    Version tolerante aux variantes CTC :
    - Schwas (ə) inseres entre consonnes
    - ɑ (U+0251) standalone normalise en a (U+0061)
    - Variantes de phone (wit/ɥi pour huit, diz pour dix, kat pour quatre)
    - Verification aller-retour relaxee (compare apres normalisation STT)

    Args:
        ipa: Chaine IPA (espaces toleres).
        type_hint: Force le type de formule.

    Returns:
        LectureFormuleResult si reconnu, None sinon.
    """
    # Tokenisation avec table etendue STT
    tokens = _tokenize_ipa_stt(ipa)
    if not tokens:
        return None

    # Filtrer les tokens "lettre" parasites causes par le schwa → E
    # Si on a un melange nombre + lettre E, c'est probablement un schwa residuel
    has_nombre = any(t.category == "nombre" for t in tokens)
    has_lettre = any(t.category == "lettre" for t in tokens)
    if has_nombre and has_lettre:
        # Garder seulement les nombres (les lettres E sont des schwas)
        filtered = [t for t in tokens if t.category != "lettre"]
        if filtered:
            tokens = filtered

    # Detection du type
    formula_type = _detect_type(tokens, type_hint)

    # Reconstruction
    formula_str: str | None = None

    if formula_type == "nombre":
        formula_str = _reconstruct_nombre(tokens)
    elif formula_type == "date":
        formula_str = _reconstruct_date(tokens)
    elif formula_type == "heure":
        formula_str = _reconstruct_heure(tokens)
    elif formula_type == "monnaie":
        formula_str = _reconstruct_monnaie(tokens)
    elif formula_type == "pourcentage":
        formula_str = _reconstruct_pourcentage(tokens)
    elif formula_type == "ordinal":
        formula_str = _reconstruct_ordinal(tokens)
    elif formula_type == "sigle":
        formula_str = _reconstruct_sigle(tokens)

    if formula_str is None:
        return None

    # Forward pass pour verification relaxee
    forward_fn = {
        "nombre": lire_nombre,
        "date": lire_date,
        "heure": lire_heure,
        "monnaie": lire_monnaie,
        "pourcentage": lire_pourcentage,
        "ordinal": lire_ordinal,
        "sigle": lire_sigle,
    }.get(formula_type)

    if forward_fn is None:
        return None

    try:
        result = forward_fn(formula_str)
    except Exception:
        return None

    # Verification relaxee : compare apres normalisation STT (schwas, ɑ→a, ɔ→o, ɛ→e)
    forward_norm = _normalize_ipa_for_stt(result.phone)
    input_norm = _normalize_ipa_for_stt(ipa)

    if forward_norm == input_norm:
        return result

    # Tolerance par distance d'edition pour les chaines longues
    # (grands nombres multi-mots ou la probabilite d'erreur STT augmente)
    max_dist = _stt_max_distance(len(input_norm))
    if max_dist > 0 and _levenshtein(forward_norm, input_norm) <= max_dist:
        return result

    return None


def _is_number_token(ipa_word: str) -> bool:
    """Verifie si un mot IPA individuel peut etre un token numerique.

    Utilise la tokenisation STT pour determiner si le mot est entierement
    compose de tokens nombre (potentiellement avec 'et' ou 'special').
    """
    tokens = _tokenize_ipa_stt(ipa_word)
    if not tokens:
        return False

    for tok in tokens:
        if tok.category in ("nombre", "special"):
            continue
        # Lettres isolees ne sont pas des nombres
        # sauf si c'est un schwa (E) qui a survecu a la normalisation
        if tok.category == "lettre" and tok.value == "E":
            continue
        return False
    return True


def _is_letter_token(ipa_word: str) -> bool:
    """Verifie si un mot IPA est exactement un token lettre."""
    tokens = _tokenize_ipa_stt(ipa_word)
    if not tokens:
        return False
    return len(tokens) == 1 and tokens[0].category == "lettre"



# Phones IPA ambigus : homophones nombre/mot courant.
# En mode "auto", les spans de 1 seul mot dont le phone normalise
# (apres _normalize_ipa_for_stt) est dans cette liste sont rejetes.
# Les spans >= 2 mots les contenant sont preserves (ex: "sept cent" → 700).
_AMBIGUOUS_NUMBER_PHONES: frozenset[str] = frozenset()

def _get_ambiguous_number_phones() -> frozenset[str]:
    """Retourne l'ensemble des phones normalises ambigus (lazy init)."""
    global _AMBIGUOUS_NUMBER_PHONES
    if _AMBIGUOUS_NUMBER_PHONES:
        return _AMBIGUOUS_NUMBER_PHONES

    # IPA bruts des mots ambigus (nombre vs mot courant)
    raw_ambiguous = [
        "sɛt",   # sept (7) vs cette
        "sɑ̃",    # cent (100) vs sang/sent/sans
        "vɛ̃",    # vingt (20) vs vain
        "œ̃",     # un (1) vs article un
        "yn",    # une (1) vs article une
    ]
    normalized = set()
    for raw in raw_ambiguous:
        norm = _normalize_ipa_for_stt(raw)
        if norm:
            normalized.add(norm)
    _AMBIGUOUS_NUMBER_PHONES = frozenset(normalized)
    return _AMBIGUOUS_NUMBER_PHONES


def detect_number_spans(
    ipa_words: list[str],
    *,
    min_span: int = 2,
    max_span: int = 8,
    mode: str = "num",
) -> list[tuple[int, int, LectureFormuleResult]]:
    """Detecte les spans de mots IPA formant des nombres/formules.

    Scanne la liste de mots IPA pour trouver des sequences contigues
    qui forment des nombres reconnaissables. Utilise _is_number_token
    comme pre-filtre puis reconnaitre_ipa_stt pour la validation.

    Args:
        ipa_words: Liste de mots IPA.
        min_span: Taille minimum de span (defaut: 2).
        max_span: Taille maximum de span (defaut: 8).
        mode: Mode de detection :
            - ``"num"``   : agressif (min_span respecte tel quel, tous les
              nombres isoles sont convertis) — comportement original.
            - ``"texte"`` : pas de conversion numerique (retourne []).
            - ``"auto"``  : detection intelligente — la tokenisation IPA
              doit produire au moins 2 tokens nombre pour declencher la
              conversion. Un mot isole (trois, cent, vingt) n'est jamais
              converti, meme s'il est non ambigu. Un mot unique qui
              contient 2+ tokens (ex: "kaʁɑ̃tdø" → quarante + deux)
              est accepte.

    Returns:
        Liste de (start, end, LectureFormuleResult) triee par position.
        Les spans ne se chevauchent pas (greedy plus long d'abord).
    """
    if mode == "texte":
        return []

    n = len(ipa_words)
    if n < min_span:
        return []

    # Phase 1 : pre-filtrer les mots qui ressemblent a des tokens numeriques
    is_num = [_is_number_token(w) for w in ipa_words]

    # Phase 2 : trouver les runs contigus de tokens numeriques potentiels
    # (avec tolerance d'un mot non-numerique entre deux numeriques pour
    # gerer les cas comme "sɛt sɑ̃ tʁɑ̃t sis" où un mot intermediaire
    # pourrait ne pas etre reconnu individuellement)
    results: list[tuple[int, int, LectureFormuleResult]] = []
    used: set[int] = set()

    # Essayer les spans du plus long au plus court (greedy)
    for span_len in range(min(max_span, n), min_span - 1, -1):
        for i in range(n - span_len + 1):
            end = i + span_len
            # Verifier que le span n'est pas deja utilise
            if any(j in used for j in range(i, end)):
                continue

            # Tous les mots du span doivent etre des tokens numeriques
            # (evite d'absorber des mots grammaticaux comme "a", "de", "il")
            if not all(is_num[j] for j in range(i, end)):
                continue

            # Concatener et tenter la reconnaissance
            merged_ipa = " ".join(ipa_words[i:end])
            result = reconnaitre_ipa_stt(merged_ipa, type_hint="nombre")
            if result is not None:
                # Mode auto : exiger au moins 2 tokens nombre
                # (jamais de conversion pour un mot isole : trois, cent, vingt)
                if mode == "auto":
                    tokens = _tokenize_ipa_stt(merged_ipa)
                    if tokens is None:
                        continue
                    nombre_count = sum(
                        1 for t in tokens if t.category == "nombre"
                    )
                    if nombre_count < 2:
                        continue

                results.append((i, end, result))
                used.update(range(i, end))
                break  # passer au span suivant (greedy)

    # Trier par position
    results.sort(key=lambda x: x[0])
    return results


def detect_sigle_spans(
    ipa_words: list[str],
    *,
    min_span: int = 2,
    max_span: int = 8,
) -> list[tuple[int, int, LectureFormuleResult]]:
    """Detecte les spans de mots IPA formant des sigles (acronymes epeles).

    Scanne la liste de mots IPA pour trouver des sequences contigues
    de lettres qui forment un sigle reconnaissable. Utilise _is_letter_token
    comme pre-filtre puis reconnaitre_ipa_stt avec type_hint="sigle".

    Args:
        ipa_words: Liste de mots IPA.
        min_span: Taille minimum de span (defaut: 2).
        max_span: Taille maximum de span (defaut: 8).

    Returns:
        Liste de (start, end, LectureFormuleResult) triee par position.
        Les spans ne se chevauchent pas (greedy plus long d'abord).
    """
    n = len(ipa_words)
    if n < min_span:
        return []

    # Phase 1 : pre-filtrer les mots qui sont des tokens lettre
    is_letter = [_is_letter_token(w) for w in ipa_words]

    # Phase 2 : greedy longest-first sur les runs contigus de lettres
    results: list[tuple[int, int, LectureFormuleResult]] = []
    used: set[int] = set()

    for span_len in range(min(max_span, n), min_span - 1, -1):
        for i in range(n - span_len + 1):
            end = i + span_len
            # Verifier que le span n'est pas deja utilise
            if any(j in used for j in range(i, end)):
                continue

            # Tous les mots du span doivent etre des tokens lettre
            if not all(is_letter[j] for j in range(i, end)):
                continue

            # Concatener et tenter la reconnaissance
            merged_ipa = " ".join(ipa_words[i:end])
            result = reconnaitre_ipa_stt(merged_ipa, type_hint="sigle")
            if result is not None:
                results.append((i, end, result))
                used.update(range(i, end))
                break  # passer au span suivant (greedy)

    # Trier par position
    results.sort(key=lambda x: x[0])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Reconnaissance de formules mathematiques (IPA → formule math)
# ══════════════════════════════════════════════════════════════════════════════

# Categories de symboles pour le classement dans la table math
_FUNC_SYMS = frozenset({"sin", "cos", "tan", "exp", "ln", "log", "sqrt", "abs"})
_BRACKET_SYMS = frozenset({"(", ")", "[", "]", "{", "}"})
_UNIT_SYMS = frozenset({
    "kg", "km", "cm", "mm", "mg", "ml", "°C", "°F",
    "g", "m", "s", "h", "L", "min",
})

# Operateurs binaires (attendent operande a gauche et a droite)
_BINARY_OPS = frozenset({
    "+", "-", "−", "±", "=", "≠", "<", ">", "≤", "≥",
    "×", "*", "÷", "/", "^",
    "→", "←", "↔", "⇒", "⇔",
    "≈", "≃", "≡",
    "<=", ">=", "!=", "==",
    "∈", "∉", "⊂",
    "∪", "∩",
})

# Operateurs postfix (attendent operande a gauche)
_POSTFIX_OPS = frozenset({"²", "³", "!", "°", "%", "‰"})

# Operateurs prefix (attendent operande a droite)
_PREFIX_OPS = frozenset({"√", "∑", "∏", "∫", "∂", "∇"})


# ── Table etendue pour les formules math ──────────────────────────────

_math_lookup_table: list[tuple[str, IpaToken]] | None = None


def _build_math_lookup_table() -> list[tuple[str, IpaToken]]:
    """Construit la table etendue IPA->token pour les formules math.

    Herite de la table standard (nombres, lettres, etc.) et ajoute
    les symboles, fonctions, brackets, unites et lettres grecques.
    """
    global _math_lookup_table
    if _math_lookup_table is not None:
        return _math_lookup_table

    # Partir de la table standard
    base = dict(_build_lookup_table())

    # -- Symboles math --
    for sym, (ortho, phone) in _load_symboles().items():
        nphone = _nfc(_strip_spaces(phone))
        if not nphone:
            continue
        # Les tokens existants (nombre, lettre, special) ont priorite
        if nphone in base:
            continue
        # Classifier le symbole
        if sym in _FUNC_SYMS:
            cat = "fonction"
        elif sym in _BRACKET_SYMS:
            cat = "bracket"
        elif sym in _UNIT_SYMS:
            cat = "unite"
        else:
            cat = "symbole"
        base[nphone] = IpaToken(cat, sym, sym, nphone)

    # -- Lettres grecques (preference pour les minuscules) --
    for char, (ortho, phone) in _load_grec().items():
        nphone = _nfc(_strip_spaces(phone))
        if not nphone:
            continue
        if nphone in base:
            # Remplacer par la minuscule si l'existant est grec uppercase
            existing = base[nphone]
            if existing.category == "grec" and isinstance(existing.value, str) and not existing.value.islower() and char.islower():
                base[nphone] = IpaToken("grec", char, char, nphone)
            continue
        base[nphone] = IpaToken("grec", char, char, nphone)

    # -- Connecteur "de" pour smart parens inverses --
    de_phone = _nfc("də")
    if de_phone not in base:
        base[de_phone] = IpaToken("connecteur", "de", "de", de_phone)

    # -- Connecteur "par" pour unites (km/h → "kilometre par heure") --
    par_phone = _nfc("paʁ")
    if par_phone not in base:
        base[par_phone] = IpaToken("connecteur", "par", "par", par_phone)

    # -- "indice" pour subscripts --
    indice_phone = _nfc(_strip_spaces("ɛ̃dis"))
    if indice_phone not in base:
        base[indice_phone] = IpaToken("symbole", "indice", "indice", indice_phone)

    # -- "puissance" pour exposants generaux --
    puissance_phone = _nfc(_strip_spaces("pɥisɑ̃s"))
    if puissance_phone not in base:
        base[puissance_phone] = IpaToken("symbole", "puissance", "puissance", puissance_phone)

    # -- "enne" pour exposant n --
    enne_phone = _nfc("ɛn")
    if enne_phone not in base:
        base[enne_phone] = IpaToken("symbole", "enne", "enne", enne_phone)

    # -- "prime" (′) --
    prime_phone = _nfc("pʁim")
    if prime_phone not in base:
        base[prime_phone] = IpaToken("symbole", "prime", "prime", prime_phone)

    # -- "facteur de" pour paren_facteur --
    facteur_de_phone = _nfc(_strip_spaces("faktœʁ də"))
    if facteur_de_phone not in base:
        base[facteur_de_phone] = IpaToken("connecteur", "facteur_de", "facteur_de", facteur_de_phone)

    # -- "valeur absolue de" pour abs pipes --
    abs_de_phone = _nfc(_strip_spaces("valœʁ apsoly də"))
    if abs_de_phone not in base:
        base[abs_de_phone] = IpaToken("connecteur", "abs_de", "abs_de", abs_de_phone)

    # -- Fragments de tokens multi-mots (pour le pre-filtre _is_math_token) --
    # "kaʁe" seul (fragment de "o kaʁe" = ²) ne matche pas dans la table,
    # mais doit etre reconnu comme token math potentiel pour le pre-filtre.
    # On les ajoute comme "fragment" — ils ne seront jamais matche en greedy
    # (car "okaʁe" est plus long) mais permettent au pre-filtre de fonctionner.
    _math_fragments = {
        "kaʁe": IpaToken("fragment", "carré", "carré", _nfc("kaʁe")),
        "kyb": IpaToken("fragment", "cube", "cube", _nfc("kyb")),
        "ʁasin": IpaToken("fragment", "racine", "racine", _nfc("ʁasin")),
    }
    for frag_phone, frag_token in _math_fragments.items():
        nfrag = _nfc(frag_phone)
        if nfrag not in base:
            base[nfrag] = frag_token

    # Trier par longueur decroissante (greedy longest-match)
    _math_lookup_table = sorted(base.items(), key=lambda x: len(x[0]), reverse=True)
    return _math_lookup_table


# ── Tokenisation math ─────────────────────────────────────────────────

def _tokenize_ipa_math(ipa: str) -> list[IpaToken] | None:
    """Tokenise une chaine IPA avec la table etendue math.

    Retourne None si la tokenisation echoue.
    """
    table = _build_math_lookup_table()
    normalized = _nfc(_strip_spaces(ipa))

    if not normalized:
        return None

    tokens: list[IpaToken] = []
    pos = 0
    length = len(normalized)

    while pos < length:
        matched = False
        for entry_key, entry_token in table:
            entry_len = len(entry_key)
            if pos + entry_len <= length and normalized[pos:pos + entry_len] == entry_key:
                tokens.append(entry_token)
                pos += entry_len
                matched = True
                break
        if not matched:
            return None

    return tokens


# ── Garde contre faux positifs ────────────────────────────────────────

# Categories considerees comme operandes (gauche/droite d'un operateur)
_OPERAND_CATS = frozenset({
    "nombre", "lettre", "grec", "bracket", "connecteur", "fonction", "unite",
})


def _is_valid_math_sequence(tokens: list[IpaToken]) -> bool:
    """Verifie qu'une sequence de tokens forme une formule math valide.

    Regles :
    1. Au moins 2 tokens (pour prefix comme √9) ou 3 pour le cas general
    2. Au moins un token symbole/fonction/grec (sinon c'est nombre/sigle)
    3. Operateurs binaires : operande a gauche ET a droite
       (sauf - ou + en position 0 = unaire)
    4. Operateurs postfix : operande a gauche
    5. Operateurs prefix : operande a droite
    """
    if len(tokens) < 2:
        return False

    # Regle 2 : au moins un token specifiquement math
    has_math = False
    for tok in tokens:
        if tok.category in ("symbole", "fonction", "grec", "bracket", "connecteur", "unite"):
            has_math = True
            break
    if not has_math:
        return False

    # Si seulement 2 tokens, au moins un doit etre un operateur prefix/postfix,
    # une fonction, ou un connecteur (pour eviter "2 x" → faux positif)
    if len(tokens) == 2:
        values = {tok.value for tok in tokens if tok.category == "symbole"}
        cats = {tok.category for tok in tokens}
        has_prefix_or_postfix = bool(values & (_PREFIX_OPS | _POSTFIX_OPS))
        has_function = "fonction" in cats
        has_connecteur = "connecteur" in cats
        if not (has_prefix_or_postfix or has_function or has_connecteur):
            return False

    # Verifier la structure operateurs/operandes
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok.category != "symbole":
            continue
        sym = tok.value

        # Ignorer les tokens speciaux non-operateur
        if sym in ("indice", "puissance", "enne", "prime",
                    "racine_carree", "factorielle"):
            continue

        if sym in _BINARY_OPS:
            # Signe unaire en debut de formule
            if i == 0 and sym in ("+", "-", "−"):
                if i + 1 >= n:
                    return False
                continue
            # Operande a gauche
            if i == 0:
                return False
            left = tokens[i - 1]
            if left.category not in _OPERAND_CATS and left.value not in _POSTFIX_OPS and left.category != "symbole":
                # Accepter aussi si le token precedent est un symbole postfix
                if left.value not in ("²", "³", "!", "°", "%", "‰",
                                      "enne", "prime", "factorielle"):
                    return False
            # Operande a droite
            if i + 1 >= n:
                return False

        elif sym in _POSTFIX_OPS:
            if i == 0:
                return False

        elif sym in _PREFIX_OPS:
            if i + 1 >= n:
                return False

    return True


# ── Reconstruction de la formule ──────────────────────────────────────

def _reconstruct_maths(tokens: list[IpaToken]) -> str | None:
    """Reconstruit une formule mathematique depuis les tokens IPA.

    Gere : nombres, lettres (variables), grec, symboles, fonctions,
    brackets, connecteurs (smart parens), multiplication implicite.
    """
    parts: list[str] = []
    n = len(tokens)
    i = 0

    while i < n:
        tok = tokens[i]
        cat = tok.category
        val = tok.value

        if cat == "nombre":
            # Reconstruire le nombre : accumuler les tokens nombre consecutifs
            num_tokens: list[IpaToken] = []
            while i < n and tokens[i].category == "nombre":
                num_tokens.append(tokens[i])
                i += 1
            # Verifier si "virgule" suit pour decimal
            while i < n and tokens[i].category == "special" and tokens[i].value in ("virgule", "et"):
                if tokens[i].value == "virgule" and i + 1 < n and tokens[i + 1].category == "nombre":
                    num_tokens.append(tokens[i])
                    i += 1
                    while i < n and tokens[i].category == "nombre":
                        num_tokens.append(tokens[i])
                        i += 1
                else:
                    break
            num_str = _reconstruct_nombre(num_tokens)
            if num_str is None:
                return None
            parts.append(num_str)
            continue

        elif cat == "lettre":
            # Variable dans une formule → minuscule
            parts.append(val.lower() if isinstance(val, str) and len(val) == 1 else str(val).lower())
            i += 1

        elif cat == "grec":
            parts.append(str(val))
            i += 1

        elif cat == "symbole":
            sym = str(val)
            if sym == "√":
                # √ — son IPA inclut deja le "de" (ʁasin kaʁe də)
                parts.append("√")
                i += 1
            elif sym == "racine_carree":
                # Forme sans "de" (si rencontree)
                parts.append("√")
                if i + 1 < n and tokens[i + 1].category == "connecteur" and tokens[i + 1].value == "de":
                    i += 1
                i += 1
            elif sym == "indice":
                # Subscript : _contenu
                parts.append("_")
                i += 1
                # Consommer le contenu du subscript (chiffres/lettres)
                sub_parts: list[str] = []
                while i < n:
                    if tokens[i].category == "nombre":
                        sub_num: list[IpaToken] = []
                        while i < n and tokens[i].category == "nombre":
                            sub_num.append(tokens[i])
                            i += 1
                        sub_val = _reconstruct_number(sub_num)
                        sub_parts.append(str(sub_val))
                    elif tokens[i].category == "lettre":
                        sub_parts.append(str(tokens[i].value).lower())
                        i += 1
                    else:
                        break
                parts.append("".join(sub_parts))
                continue
            elif sym == "puissance":
                # Exposant general : ^contenu → superscript
                i += 1
                # Consommer signe eventuel
                exp_sign = ""
                if i < n and tokens[i].category == "special" and tokens[i].value == "moins":
                    exp_sign = "⁻"
                    i += 1
                # Consommer "enne" → n
                if i < n and tokens[i].category == "symbole" and tokens[i].value == "enne":
                    parts.append(exp_sign + "ⁿ")
                    i += 1
                elif i < n and tokens[i].category == "nombre":
                    exp_num: list[IpaToken] = []
                    while i < n and tokens[i].category == "nombre":
                        exp_num.append(tokens[i])
                        i += 1
                    exp_val = _reconstruct_number(exp_num)
                    # Convertir en superscript unicode
                    sup_map = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                               "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
                    exp_str = "".join(sup_map.get(c, c) for c in str(exp_val))
                    parts.append(exp_sign + exp_str)
                else:
                    # Pas de contenu apres puissance → juste ^
                    parts.append("^")
                continue
            elif sym == "enne":
                # "enne" seul (variable n en contexte exposant)
                parts.append("n")
                i += 1
            elif sym == "prime":
                parts.append("′")
                i += 1
            elif sym == "factorielle":
                parts.append("!")
                i += 1
            elif sym == "∞":
                parts.append("∞")
                i += 1
            else:
                # Symbole normal (operateur, etc.)
                parts.append(sym)
                i += 1

        elif cat == "fonction":
            # Nom de fonction
            func_name = str(val)
            parts.append(func_name)
            i += 1

        elif cat == "bracket":
            parts.append(str(val))
            i += 1

        elif cat == "unite":
            parts.append(str(val))
            i += 1

        elif cat == "connecteur":
            conn = str(val)
            if conn == "de":
                # Smart paren inverse : lettre/fonction + "de" + operande → f(x)
                # Ouvrir une parenthese
                parts.append("(")
                i += 1
                # Collecter l'operande + decorateurs
                inner_parts: list[str] = []
                depth = 1
                while i < n and depth > 0:
                    t = tokens[i]
                    # Un autre "de" ouvre un niveau de profondeur
                    if t.category == "connecteur" and t.value == "de":
                        depth += 1
                        inner_parts.append("(")
                        i += 1
                        continue
                    # Fin de l'operande : operateur binaire ou autre connecteur
                    if t.category == "symbole" and t.value in _BINARY_OPS:
                        break
                    if t.category == "connecteur" and t.value == "par":
                        break
                    # Token qui fait partie de l'operande
                    if t.category == "nombre":
                        num_tokens2: list[IpaToken] = []
                        while i < n and tokens[i].category == "nombre":
                            num_tokens2.append(tokens[i])
                            i += 1
                        # Gerer virgule decimale
                        while i < n and tokens[i].category == "special" and tokens[i].value == "virgule":
                            if i + 1 < n and tokens[i + 1].category == "nombre":
                                num_tokens2.append(tokens[i])
                                i += 1
                                while i < n and tokens[i].category == "nombre":
                                    num_tokens2.append(tokens[i])
                                    i += 1
                            else:
                                break
                        ns = _reconstruct_nombre(num_tokens2)
                        if ns:
                            inner_parts.append(ns)
                        continue
                    elif t.category == "lettre":
                        inner_parts.append(str(t.value).lower())
                        i += 1
                    elif t.category == "grec":
                        inner_parts.append(str(t.value))
                        i += 1
                    elif t.category == "symbole":
                        sv = str(t.value)
                        if sv in ("²", "³"):
                            inner_parts.append(sv)
                            i += 1
                        elif sv == "puissance":
                            # Exposant dans l'operande
                            i += 1
                            esign = ""
                            if i < n and tokens[i].category == "special" and tokens[i].value == "moins":
                                esign = "⁻"
                                i += 1
                            if i < n and tokens[i].category == "symbole" and tokens[i].value == "enne":
                                inner_parts.append(esign + "ⁿ")
                                i += 1
                            elif i < n and tokens[i].category == "nombre":
                                en: list[IpaToken] = []
                                while i < n and tokens[i].category == "nombre":
                                    en.append(tokens[i])
                                    i += 1
                                ev = _reconstruct_number(en)
                                smap = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                                        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
                                inner_parts.append(esign + "".join(smap.get(c, c) for c in str(ev)))
                            else:
                                inner_parts.append("^")
                            continue
                        elif sv == "prime":
                            inner_parts.append("′")
                            i += 1
                        elif sv in _POSTFIX_OPS:
                            inner_parts.append(sv)
                            i += 1
                        else:
                            # Fin de l'operande (operateur non-decorateur)
                            break
                    elif t.category == "fonction":
                        inner_parts.append(str(t.value))
                        i += 1
                    elif t.category == "bracket":
                        inner_parts.append(str(t.value))
                        i += 1
                    else:
                        break
                # Fermer les parens
                for _ in range(depth):
                    inner_parts.append(")")
                parts.extend(inner_parts)
                continue
            elif conn == "par":
                # "par" entre unites → /
                parts.append("/")
                i += 1
            elif conn == "facteur_de":
                # facteur de → ouvrir parenthese (meme logique que "de")
                parts.append("(")
                i += 1
                fd_parts: list[str] = []
                while i < n:
                    t = tokens[i]
                    if t.category == "symbole" and t.value in _BINARY_OPS:
                        break
                    if t.category == "connecteur":
                        break
                    if t.category == "nombre":
                        nt: list[IpaToken] = []
                        while i < n and tokens[i].category == "nombre":
                            nt.append(tokens[i])
                            i += 1
                        ns2 = _reconstruct_nombre(nt)
                        if ns2:
                            fd_parts.append(ns2)
                        continue
                    elif t.category == "lettre":
                        fd_parts.append(str(t.value).lower())
                        i += 1
                    elif t.category == "grec":
                        fd_parts.append(str(t.value))
                        i += 1
                    elif t.category == "symbole" and t.value in ("²", "³", "prime"):
                        fd_parts.append("²" if t.value == "²" else ("³" if t.value == "³" else "′"))
                        i += 1
                    else:
                        break
                fd_parts.append(")")
                parts.extend(fd_parts)
                continue
            elif conn == "abs_de":
                # Valeur absolue → |...|
                parts.append("|")
                i += 1
                abs_parts: list[str] = []
                while i < n:
                    t = tokens[i]
                    if t.category == "symbole" and t.value in _BINARY_OPS:
                        break
                    if t.category == "connecteur" and t.value not in ("de",):
                        break
                    if t.category == "nombre":
                        nt2: list[IpaToken] = []
                        while i < n and tokens[i].category == "nombre":
                            nt2.append(tokens[i])
                            i += 1
                        ns3 = _reconstruct_nombre(nt2)
                        if ns3:
                            abs_parts.append(ns3)
                        continue
                    elif t.category == "lettre":
                        abs_parts.append(str(t.value).lower())
                        i += 1
                    elif t.category == "grec":
                        abs_parts.append(str(t.value))
                        i += 1
                    elif t.category == "symbole":
                        sv = str(t.value)
                        if sv in _BINARY_OPS:
                            abs_parts.append(sv)
                            i += 1
                        elif sv in _POSTFIX_OPS:
                            abs_parts.append(sv)
                            i += 1
                        else:
                            break
                    else:
                        break
                abs_parts.append("|")
                parts.extend(abs_parts)
                continue
            else:
                i += 1

        elif cat == "special":
            sp = str(val)
            if sp == "moins":
                parts.append("-")
                i += 1
            elif sp == "et":
                # Absorbe (ne produit rien dans une formule math)
                i += 1
            else:
                i += 1

        else:
            # Token inconnu — echec
            return None

    formula = "".join(parts)
    return formula if formula else None


# ── Verification aller-retour (API publique) ──────────────────────────

def reconnaitre_maths_ipa(ipa: str) -> LectureFormuleResult | None:
    """Reconnait une formule mathematique a partir de sa transcription IPA.

    Etapes :
    1. Tokenisation avec la table math etendue
    2. Validation de la sequence (garde contre faux positifs)
    3. Reconstruction de la formule
    4. Forward pass (lire_maths) pour verification aller-retour
    5. Comparaison IPA forward vs IPA input

    Returns:
        LectureFormuleResult si reconnu et verifie, None sinon.
    """
    tokens = _tokenize_ipa_math(ipa)
    if not tokens:
        return None

    if not _is_valid_math_sequence(tokens):
        return None

    formula_str = _reconstruct_maths(tokens)
    if not formula_str:
        return None

    try:
        result = lire_maths(formula_str)
    except Exception:
        return None

    # Verification aller-retour
    forward_ipa = _normalize_ipa_for_compare(result.phone)
    input_ipa = _normalize_ipa_for_compare(ipa)

    if forward_ipa == input_ipa:
        return result

    return None


# ── Detection de spans math dans les phrases ──────────────────────────

def _is_math_token(ipa_word: str) -> bool:
    """Verifie si un mot IPA individuel peut etre un token math.

    Retourne True si le mot se tokenise entierement en tokens
    de categories math (nombre, lettre, symbole, fonction, grec,
    bracket, connecteur, special, unite).
    """
    tokens = _tokenize_ipa_math(ipa_word)
    if not tokens:
        return False
    for tok in tokens:
        if tok.category in ("nombre", "lettre", "symbole", "fonction",
                            "grec", "bracket", "connecteur", "special",
                            "unite", "fragment"):
            continue
        return False
    return True


def detect_formula_spans(
    ipa_words: list[str],
    *,
    min_span: int = 3,
    max_span: int = 20,
) -> list[tuple[int, int, LectureFormuleResult]]:
    """Detecte les spans de mots IPA formant des formules mathematiques.

    Meme patron que detect_number_spans (greedy longest-first).
    Pre-filtre avec _is_math_token, puis valide avec reconnaitre_maths_ipa.

    Args:
        ipa_words: Liste de mots IPA.
        min_span: Taille minimum de span (defaut: 3).
        max_span: Taille maximum de span (defaut: 20).

    Returns:
        Liste de (start, end, LectureFormuleResult) triee par position.
        Les spans ne se chevauchent pas (greedy plus long d'abord).
    """
    n = len(ipa_words)
    if n < min_span:
        return []

    # Phase 1 : pre-filtrer les mots qui sont des tokens math
    is_math = [_is_math_token(w) for w in ipa_words]

    # Phase 2 : greedy longest-first
    results: list[tuple[int, int, LectureFormuleResult]] = []
    used: set[int] = set()

    for span_len in range(min(max_span, n), min_span - 1, -1):
        for i in range(n - span_len + 1):
            end = i + span_len
            if any(j in used for j in range(i, end)):
                continue
            if not all(is_math[j] for j in range(i, end)):
                continue

            merged_ipa = " ".join(ipa_words[i:end])
            result = reconnaitre_maths_ipa(merged_ipa)
            if result is not None:
                results.append((i, end, result))
                used.update(range(i, end))
                break  # passer au span suivant (greedy)

    results.sort(key=lambda x: x[0])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Tolerance STT pour les formules math
# ══════════════════════════════════════════════════════════════════════════════

_math_stt_lookup_table: list[tuple[str, IpaToken]] | None = None


def _build_math_stt_lookup_table() -> list[tuple[str, IpaToken]]:
    """Table de lookup math etendue avec normalisation STT.

    Part de la table math standard et :
    - Ajoute les variantes CTC (wit→8, diz→10, kat→4)
    - Ajoute les versions normalisees STT (ɑ→a, ɔ→o, ɛ→e, schwa removal)
    """
    global _math_stt_lookup_table
    if _math_stt_lookup_table is not None:
        return _math_stt_lookup_table

    # Partir de la table math standard
    base = dict(_build_math_lookup_table())

    # Ajouter les variantes CTC (seulement si pas deja presentes)
    variants = _build_stt_variants()
    for phone, (val, key) in variants.items():
        nphone = _nfc(phone)
        if nphone not in base:
            base[nphone] = IpaToken("nombre", val, key, nphone)

    # Variantes CTC auto-generees pour toutes les categories
    variant_entries: dict[str, IpaToken] = {}
    for ipa_key, token in list(base.items()):
        for variant in _generate_variants(ipa_key):
            if variant not in base and variant not in variant_entries:
                variant_entries[variant] = token
    base.update(variant_entries)

    # Re-normaliser les entrees existantes (strip schwas, ɑ→a, ɔ→o, ɛ→e)
    extended: dict[str, IpaToken] = {}
    for ipa_key, token in base.items():
        if ipa_key:
            extended[ipa_key] = token
        norm_key = _normalize_ipa_for_stt(ipa_key)
        if norm_key and norm_key != ipa_key and norm_key not in extended:
            extended[norm_key] = token

    _math_stt_lookup_table = sorted(
        extended.items(), key=lambda x: len(x[0]), reverse=True,
    )
    return _math_stt_lookup_table


def _tokenize_ipa_math_stt(ipa: str) -> list[IpaToken] | None:
    """Tokenise une chaine IPA math avec tolerance STT.

    Pre-normalise l'IPA (schwas, ɑ→a, ɔ→o, ɛ→e) et utilise
    la table math etendue avec variantes CTC.
    """
    table = _build_math_stt_lookup_table()
    normalized = _normalize_ipa_for_stt(ipa)

    if not normalized:
        return None

    tokens: list[IpaToken] = []
    pos = 0
    length = len(normalized)

    while pos < length:
        matched = False
        for entry_key, entry_token in table:
            entry_len = len(entry_key)
            if pos + entry_len <= length and normalized[pos:pos + entry_len] == entry_key:
                tokens.append(entry_token)
                pos += entry_len
                matched = True
                break
        if not matched:
            return None

    return tokens


def _is_math_token_stt(ipa_word: str) -> bool:
    """Verifie si un mot IPA peut etre un token math (version STT tolerante)."""
    tokens = _tokenize_ipa_math_stt(ipa_word)
    if not tokens:
        return False
    for tok in tokens:
        if tok.category in ("nombre", "lettre", "symbole", "fonction",
                            "grec", "bracket", "connecteur", "special",
                            "unite", "fragment"):
            continue
        return False
    return True


def reconnaitre_maths_ipa_stt(ipa: str) -> LectureFormuleResult | None:
    """Reconnait une formule math a partir d'IPA produit par un pipeline STT.

    Version tolerante aux variantes CTC :
    - Schwas (ə) inseres entre consonnes
    - ɑ standalone normalise en a, ɔ en o, ɛ en e
    - Variantes de phone (wit pour huit, diz pour dix, etc.)
    - Verification aller-retour relaxee (compare apres normalisation STT)
    - Tolerance Levenshtein pour les formules longues

    Returns:
        LectureFormuleResult si reconnu, None sinon.
    """
    tokens = _tokenize_ipa_math_stt(ipa)
    if not tokens:
        return None

    # Filtrer les tokens "lettre" parasites causes par le schwa → E
    has_nombre = any(t.category == "nombre" for t in tokens)
    has_lettre_e = any(
        t.category == "lettre" and t.value == "E" for t in tokens
    )
    if has_nombre and has_lettre_e:
        # Ne filtrer que les lettres E (schwas residuels), garder les autres
        filtered = [
            t for t in tokens
            if not (t.category == "lettre" and t.value == "E")
        ]
        if filtered:
            tokens = filtered

    if not _is_valid_math_sequence(tokens):
        return None

    formula_str = _reconstruct_maths(tokens)
    if not formula_str:
        return None

    try:
        result = lire_maths(formula_str)
    except Exception:
        return None

    # Verification relaxee : compare apres normalisation STT
    forward_norm = _normalize_ipa_for_stt(result.phone)
    input_norm = _normalize_ipa_for_stt(ipa)

    if forward_norm == input_norm:
        return result

    # Tolerance Levenshtein pour les formules longues
    max_dist = _stt_max_distance(len(input_norm))
    if max_dist > 0 and _levenshtein(forward_norm, input_norm) <= max_dist:
        return result

    return None


def detect_formula_spans_stt(
    ipa_words: list[str],
    *,
    min_span: int = 3,
    max_span: int = 20,
) -> list[tuple[int, int, LectureFormuleResult]]:
    """Detecte les spans de formules math avec tolerance STT.

    Version tolerante de detect_formula_spans pour l'IPA produit
    par un pipeline STT (schwas, variantes CTC, normalisation vocalique).

    Args:
        ipa_words: Liste de mots IPA.
        min_span: Taille minimum de span (defaut: 3).
        max_span: Taille maximum de span (defaut: 20).

    Returns:
        Liste de (start, end, LectureFormuleResult) triee par position.
    """
    n = len(ipa_words)
    if n < min_span:
        return []

    is_math = [_is_math_token_stt(w) for w in ipa_words]

    results: list[tuple[int, int, LectureFormuleResult]] = []
    used: set[int] = set()

    for span_len in range(min(max_span, n), min_span - 1, -1):
        for i in range(n - span_len + 1):
            end = i + span_len
            if any(j in used for j in range(i, end)):
                continue
            if not all(is_math[j] for j in range(i, end)):
                continue

            merged_ipa = " ".join(ipa_words[i:end])
            result = reconnaitre_maths_ipa_stt(merged_ipa)
            if result is not None:
                results.append((i, end, result))
                used.update(range(i, end))
                break

    results.sort(key=lambda x: x[0])
    return results
