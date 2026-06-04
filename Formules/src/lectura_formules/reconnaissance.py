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


def _normalize_ipa_for_stt(ipa: str) -> str:
    """Normalise une chaine IPA pour la reconnaissance STT.

    - NFC
    - Strip espaces
    - Supprime les schwas (ə) inseres par le CTC/lexique
    - Normalise ɑ (U+0251) standalone en a (U+0061)
    - Normalise g (U+0067) en ɡ (U+0261)
    """
    s = _nfc(_strip_spaces(ipa))
    s = _RE_SCHWA.sub("", s)
    s = _RE_ALPHA_STANDALONE.sub("a", s)
    s = s.replace("g", "ɡ")  # ASCII g → IPA script g
    return s


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

    # Verification relaxee : compare apres normalisation STT (strip schwas, ɑ→a)
    forward_norm = _normalize_ipa_for_stt(result.phone)
    input_norm = _normalize_ipa_for_stt(ipa)

    if forward_norm == input_norm:
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


def detect_number_spans(
    ipa_words: list[str],
    *,
    min_span: int = 2,
    max_span: int = 8,
) -> list[tuple[int, int, LectureFormuleResult]]:
    """Detecte les spans de mots IPA formant des nombres/formules.

    Scanne la liste de mots IPA pour trouver des sequences contigues
    qui forment des nombres reconnaissables. Utilise _is_number_token
    comme pre-filtre puis reconnaitre_ipa_stt pour la validation.

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
