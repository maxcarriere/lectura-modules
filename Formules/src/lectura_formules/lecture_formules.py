"""Lecture algorithmique des formules — module autonome transversal.

Fournit 3 facettes :
  - Tokeniseur : identification du type + display_fr
  - G2P : transcription phonétique IPA
  - Aligneur : events décomposés avec groupement par composant

Zéro dépendance externe. Embarque les tables numReader comme dicts Python
pour la lecture de nombres, sigles, dates, téléphones, ordinaux, fractions,
notations scientifiques, formules mathématiques, numéros, heures, monnaies,
pourcentages, intervalles, coordonnées GPS et pages/chapitres.

Licence : CC-BY-SA-4.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from lectura_formules._chargeur import (
    unites as _load_unites,
    lettres as _load_lettres,
    symboles as _load_symboles,
    grec as _load_grec,
    ordinaux as _load_ordinaux,
    mois as _load_mois,
    virgule as _load_virgule,
    fois as _load_fois,
    dix as _load_dix,
    exposant as _load_exposant,
    echelles as _load_echelles,
    heure_words as _load_heure_words,
    devises as _load_devises,
    pourcent_words as _load_pourcent_words,
    gps_directions as _load_gps_directions,
    gps_units as _load_gps_units,
    intervalle_bounds as _load_intervalle_bounds,
)

# ══════════════════════════════════════════════════════════════════════════════
# Types de sortie
# ══════════════════════════════════════════════════════════════════════════════

Span = tuple[int, int]


@dataclass
class EventFormuleLecture:
    """Événement de lecture aligné sur le texte source."""
    ortho: str
    phone: str
    span_source: Span = (0, 0)
    composant: int = 0
    sound_id: str = ""
    span_fr: Span = (0, 0)
    span_num: Span = (0, 0)
    span_num_extra: Span = (0, 0)
    span_rom: Span = (0, 0)


@dataclass
class LectureFormuleResult:
    """Résultat de lecture d'une formule."""
    display_fr: str
    phone: str
    events: list[EventFormuleLecture] = field(default_factory=list)
    display_num: str = ""
    display_rom: str = ""
    valeur: int | float | str = ""

    def composants(self) -> list[list[EventFormuleLecture]]:
        """Regroupe les events par composant (pour mode block)."""
        if not self.events:
            return []
        groups: dict[int, list[EventFormuleLecture]] = {}
        for evt in self.events:
            groups.setdefault(evt.composant, []).append(evt)
        return [groups[k] for k in sorted(groups)]


@dataclass
class OptionsLecture:
    """Options pour la lecture des formules."""
    fraction_mode: str = "hybride"   # "hybride", "ordinal", "standard"
    decimal_method: str = "m2"       # "m1" (reste comme entier), "m2" (groupes de 3)
    heure_mot_minutes: bool = False  # dire "minutes" quand format h/colon
    monnaie_dire_centimes: bool = True  # inclure les centimes
    romain_actif: bool = True        # calculer display_rom pour les nombres
    auto_convert_sci: bool = False   # convertir automatiquement décimal ↔ scientifique
    heure_minuit_midi: bool = False  # 0h → "minuit", 12h → "midi"


# ══════════════════════════════════════════════════════════════════════════════
# Donnees metier chargees depuis JSON (via _chargeur)
# ══════════════════════════════════════════════════════════════════════════════
# Si le fichier JSON est absent (mode Niveau 1 / API), on initialise un
# flag _MODE_API = True et les fonctions publiques deleguent au serveur.

_MODE_API = False
try:
    _UNITES = _load_unites()
    _LETTRES = _load_lettres()
    _SYMBOLES = _load_symboles()
    _GREC = _load_grec()
    _ORDINAUX = _load_ordinaux()
    _MOIS = _load_mois()
    _VIRGULE = _load_virgule()
    _FOIS = _load_fois()
    _DIX = _load_dix()
    _EXPOSANT = _load_exposant()
except FileNotFoundError:
    _MODE_API = True
    # Placeholders — jamais accedes en mode API (les fonctions publiques
    # deleguent au serveur avant d'atteindre le code local)
    _UNITES = {}
    _LETTRES = {}
    _SYMBOLES = {}
    _GREC = {}
    _ORDINAUX = {}
    _MOIS = {}
    _VIRGULE = ("", "")
    _FOIS = ("", "")
    _DIX = ("", "")
    _EXPOSANT = ("", "")

# Constantes maths importées du tokeniseur (source unique)
from lectura_tokeniseur.maths import (
    MathToken as _MathToken,
    tokenize_maths as _tokenize_maths,
    FUNCTION_LIKE_VARS as _FUNCTION_LIKE_VARS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithme cœur : nombre → français + IPA
# ══════════════════════════════════════════════════════════════════════════════

def _u(key: str) -> tuple[str, str]:
    """Raccourci : retourne (texte, phone) depuis _UNITES."""
    t, p, _v = _UNITES[key]
    return (t, p)


def _bloc_0_999(n: int, feminin: bool = False) -> list[tuple[str, str]]:
    """Convertit un entier 0–999 en liste de (texte, phone).

    Gère les règles françaises : vingt-et-un, soixante-dix,
    quatre-vingts, quatre-vingt-dix, etc.
    """
    if n == 0:
        return []
    if n < 0 or n > 999:
        raise ValueError(f"_bloc_0_999 : n={n} hors limites")

    parts: list[tuple[str, str]] = []
    centaines = n // 100
    reste = n % 100

    # -- Centaines --
    if centaines > 0:
        if centaines == 1:
            parts.append(_u("100"))
        else:
            parts.append(_u(str(centaines)))
            if reste == 0:
                parts.append(_u("100s"))  # deux-cents (pluriel)
            else:
                parts.append(_u("100"))   # deux-cent-... (pas de s)

    # -- Dizaines + unités --
    if reste == 0:
        pass
    elif reste <= 16:
        # 1–16 : formes directes
        if reste == 1 and feminin:
            parts.append(_u("1_fem"))
        else:
            parts.append(_u(str(reste)))
    elif reste <= 19:
        # 17–19 : dix-sept, dix-huit, dix-neuf
        parts.append(_u("10"))
        parts.append(_u(str(reste - 10)))
    elif reste <= 69:
        dizaine = (reste // 10) * 10
        unite = reste % 10
        if unite == 0:
            parts.append(_u(str(dizaine)))
        elif unite == 1:
            # vingt-et-un, trente-et-un, etc.
            parts.append(_u("20t" if dizaine == 20 else str(dizaine)))
            if feminin:
                parts.append(_u("et_1_fem"))
            else:
                parts.append(_u("et_1"))
        elif unite == 11 and dizaine == 60:
            # soixante-et-onze
            parts.append(_u("60"))
            parts.append(_u("et_11"))
        else:
            parts.append(_u("20t" if dizaine == 20 else str(dizaine)))
            if unite == 1 and feminin:
                parts.append(_u("1_fem"))
            else:
                parts.append(_u(str(unite)))
    elif reste <= 79:
        # 70–79 : soixante-dix, soixante-et-onze, ...
        unite79 = reste - 60
        parts.append(_u("60"))
        if unite79 == 10:
            parts.append(_u("10"))
        elif unite79 == 11:
            parts.append(_u("et_11"))
        elif unite79 <= 16:
            parts.append(_u(str(unite79)))
        else:
            # 77–79 : soixante-dix-sept, etc.
            parts.append(_u("10"))
            parts.append(_u(str(unite79 - 10)))
    elif reste == 80:
        parts.append(_u("4"))
        parts.append(_u("20s"))  # quatre-vingts
    elif reste <= 99:
        # 81–99 : quatre-vingt-un, quatre-vingt-dix, etc.
        parts.append(_u("4"))
        parts.append(_u("20"))  # vingt sans liaison dans quatre-vingt-X
        unite99 = reste - 80
        if unite99 <= 16:
            if unite99 == 1 and feminin:
                parts.append(_u("1_fem"))
            else:
                parts.append(_u(str(unite99)))
        else:
            # 97–99 : quatre-vingt-dix-sept, etc.
            parts.append(_u("10"))
            parts.append(_u(str(unite99 - 10)))

    return parts


def _decomposer_blocs(n: int) -> list[tuple[int, int]]:
    """Décompose un entier en blocs de 3 chiffres.

    Retourne [(position, valeur), ...] du bloc le plus significatif
    au moins significatif. Position 0=unités, 1=milliers, 2=millions, etc.
    """
    if n == 0:
        return [(0, 0)]

    blocs: list[tuple[int, int]] = []
    position = 0
    remaining = abs(n)
    while remaining > 0:
        bloc_val = remaining % 1000
        blocs.append((position, bloc_val))
        remaining //= 1000
        position += 1

    blocs.reverse()  # du plus significatif au moins
    return blocs


_ECHELLES = {} if _MODE_API else _load_echelles()


def _assembler_blocs(
    blocs: list[tuple[int, int]],
    feminin: bool = False,
) -> list[tuple[str, str]]:
    """Assemble des blocs avec les mots d'échelle (mille, million, etc.)."""
    parts: list[tuple[str, str]] = []

    for position, valeur in blocs:
        if valeur == 0:
            continue

        if position == 0:
            # Bloc des unités — pas de mot d'échelle
            parts.extend(_bloc_0_999(valeur, feminin=feminin))
        elif position in _ECHELLES:
            sing, plur, prefixer_un = _ECHELLES[position]
            if valeur == 1:
                if prefixer_un:
                    parts.append(_u("1"))
                parts.append(_u(sing))
            else:
                # Feminin=False pour les blocs d'échelle (deux mille, pas deux milles)
                parts.extend(_bloc_0_999(valeur, feminin=False))
                parts.append(_u(plur if valeur > 1 and position >= 2 else sing))
        else:
            # Position > 4 : pas gérée, fallback digits
            parts.extend(_bloc_0_999(valeur, feminin=False))

    return parts


def _nombre_vers_francais(
    n: int,
    feminin: bool = False,
) -> list[tuple[str, str]]:
    """Convertit un entier en liste de (texte_fr, phone_ipa).

    Gère : 0–999'999'999'999 (billions), échelles, toutes les règles
    françaises (quatre-vingts, deux-cents, vingt-et-un, pas de "un mille").
    """
    if n < 0:
        result: list[tuple[str, str]] = [_SYMBOLES["-"]]
        result.extend(_nombre_vers_francais(abs(n), feminin))
        return result
    if n == 0:
        return [_u("0")]

    blocs = _decomposer_blocs(n)
    return _assembler_blocs(blocs, feminin=feminin)


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires de construction d'events
# ══════════════════════════════════════════════════════════════════════════════


def _compute_span_rom(
    events: list[EventFormuleLecture],
    n: int,
    display_rom: str,
) -> None:
    """Calcule span_rom pour chaque event en alignant français ↔ romain.

    Décompose le nombre en groupes positionnels (milliers, centaines,
    dizaines+unités), calcule la portion romaine de chaque groupe,
    puis assigne les spans correspondants.  Pour les dizaines+unités
    simples (1-69), une décomposition plus fine est appliquée.
    """
    if not events or not display_rom or n <= 0:
        return

    try:
        from lectura_formules.romains import int_to_roman
    except ImportError:
        return

    # ── 1. Découpage positionnel du romain ──────────────────────────────
    thousands = n // 1000
    hundreds = (n % 1000) // 100
    tens_units = n % 100

    rom_spans: dict[str, Span] = {}
    pos = 0

    for group_val, group_name in [
        (thousands * 1000, "th"),
        (hundreds * 100, "h"),
        (tens_units, "tu"),
    ]:
        if group_val > 0:
            rom_str = int_to_roman(group_val)
            rom_spans[group_name] = (pos, pos + len(rom_str))
            pos += len(rom_str)

    # ── 2. Répartir les events dans les groupes ─────────────────────────
    # Chercher les mots-clés structurels pour identifier les frontières
    mille_idx = -1
    cent_idx = -1

    for i, evt in enumerate(events):
        if evt.ortho == "mille":
            mille_idx = i
            break

    start = mille_idx + 1 if mille_idx >= 0 else 0
    for i in range(start, len(events)):
        if events[i].ortho in ("cent", "cents"):
            cent_idx = i
            break

    for i, evt in enumerate(events):
        if mille_idx >= 0 and i <= mille_idx:
            group = "th"
        elif cent_idx >= 0 and i <= cent_idx:
            group = "h"
        else:
            group = "tu"
        if group in rom_spans:
            evt.span_rom = rom_spans[group]

    # ── 3. Affinage pour les dizaines+unités simples (1-69) ─────────────
    if tens_units == 0 or "tu" not in rom_spans:
        return

    tu_span = rom_spans["tu"]
    tu_events = [evt for evt in events if evt.span_rom == tu_span]

    if len(tu_events) <= 1:
        return

    # Table inverse : ortho → valeur faciale
    _val_by_ortho: dict[str, int] = {}
    for _k, (t, _p, v) in _UNITES.items():
        _val_by_ortho[t] = v

    tu_base = tu_span[0]

    if tens_units <= 69:
        # Pour 1-69, la valeur cumulée produit un préfixe romain cohérent.
        cumul = 0
        prev_len = 0
        for evt in tu_events:
            face_val = _val_by_ortho.get(evt.ortho, 0)
            cumul += face_val
            if cumul <= 0:
                continue
            try:
                rom_cumul = int_to_roman(cumul)
                cur_len = len(rom_cumul)
            except ValueError:
                cur_len = prev_len
            evt.span_rom = (tu_base + prev_len, tu_base + cur_len)
            prev_len = cur_len
        return

    # ── 70-99 : décomposition préfixe/suffixe ────────────────────────
    # 70/80/90 atomiques (tous les events ont déjà le même tu_span)
    if tens_units % 10 == 0:
        return

    # Dernier event = groupe unités, le reste = groupe dizaines
    last_evt = tu_events[-1]
    face_val = _val_by_ortho.get(last_evt.ortho, 0)
    tu_rom = display_rom[tu_span[0]:tu_span[1]]
    split_len = -1

    # Essai 1 : n − valeur_faciale_dernier_event (fonctionne pour 70-89)
    tens_try = tens_units - face_val
    if 0 < tens_try < tens_units:
        try:
            tens_rom = int_to_roman(tens_try)
            if tu_rom.startswith(tens_rom):
                split_len = len(tens_rom)
        except ValueError:
            pass

    # Essai 2 : arrondi à la dizaine inférieure (fallback pour 91-99)
    if split_len < 0:
        tens_try2 = (tens_units // 10) * 10
        if tens_try2 > 0:
            try:
                tens_rom = int_to_roman(tens_try2)
                if tu_rom.startswith(tens_rom):
                    split_len = len(tens_rom)
            except ValueError:
                pass

    if split_len < 0:
        return  # pas de découpage trouvé, laisser atomique

    tens_span_rom = (tu_span[0], tu_span[0] + split_len)
    units_span_rom = (tu_span[0] + split_len, tu_span[1])
    for evt in tu_events[:-1]:
        evt.span_rom = tens_span_rom
    tu_events[-1].span_rom = units_span_rom


def _format_display_num(n_str: str) -> str:
    """Formate un nombre avec des apostrophes (séparateur de milliers) si >= 5 chiffres."""
    sign = ""
    digits = n_str
    if digits and digits[0] in "-−±":
        sign = digits[0]
        digits = digits[1:]
    if len(digits) < 5 or not digits.isdigit():
        return n_str
    parts: list[str] = []
    while digits:
        if len(digits) > 3:
            parts.insert(0, digits[-3:])
            digits = digits[:-3]
        else:
            parts.insert(0, digits)
            digits = ""
    return sign + "'".join(parts)


def _format_display_decimal(text_clean: str) -> str:
    """Formate un nombre décimal avec apostrophes dans la partie entière seulement.

    La partie entière est formatée normalement (séparateur de milliers).
    La partie décimale est laissée telle quelle (pas de séparateurs).
    Ex: "12345.00250025" → "12'345.00250025"
    """
    sign = ""
    rest = text_clean
    if rest and rest[0] in "-−+±":
        sign = rest[0]
        rest = rest[1:]
    sep = "," if "," in rest else "."
    if sep not in rest:
        return text_clean
    int_part, dec_part = rest.split(sep, 1)
    # Formater partie entière uniquement
    if len(int_part) >= 5 and int_part.isdigit():
        formatted_int = _format_display_num(int_part)
    else:
        formatted_int = int_part
    return sign + formatted_int + sep + dec_part


def _adjust_spans_for_apostrophes(
    events: list[EventFormuleLecture],
    plain: str,
    formatted: str,
) -> None:
    """Ajuste span_num après formatage avec apostrophes.

    Les mots de magnitude (mille, million, milliard) reçoivent le span
    de l'apostrophe. Les autres events voient leur span converti
    des positions chiffres vers les positions caractères formatées.
    """
    # Mapping chiffre → position caractère dans la chaîne formatée
    d2c: list[int] = []
    apos_chars: list[int] = []
    for i, c in enumerate(formatted):
        if c == "'":
            apos_chars.append(i)
        else:
            d2c.append(i)

    if not apos_chars:
        return

    # Mapping mot magnitude → apostrophe (depuis la droite)
    mag_to_apos: dict[str, int] = {}
    if len(apos_chars) >= 1:
        mag_to_apos["mille"] = apos_chars[-1]
    if len(apos_chars) >= 2:
        mag_to_apos["million"] = mag_to_apos["millions"] = apos_chars[-2]
    if len(apos_chars) >= 3:
        mag_to_apos["milliard"] = mag_to_apos["milliards"] = apos_chars[-3]

    # Trouver la position du séparateur décimal dans 'formatted'
    dot_pos = -1
    for sep_char in (",", "."):
        if sep_char in formatted:
            dot_pos = formatted.index(sep_char)
            break

    for evt in events:
        s, e = evt.span_num
        if s >= e:
            continue

        ortho_lower = evt.ortho.lower()
        if ortho_lower in mag_to_apos:
            apos = mag_to_apos[ortho_lower]
            # Ne remapper que si le span est dans la partie entière
            if dot_pos < 0 or s < dot_pos:
                evt.span_num = (apos, apos + 1)
                continue
        # Conversion position chiffre → position formatée
        if 0 <= s < len(d2c) and 0 < e <= len(d2c):
            evt.span_num = (d2c[s], d2c[e - 1] + 1)


def _compute_span_num(
    events: list[EventFormuleLecture],
    display_num: str,
    method: str = "m2",
) -> None:
    """Calcule span_num pour chaque event d'un nombre.

    Décompose le nombre en groupes de magnitude (milliards, millions,
    milliers, unités), puis en sous-groupes (centaines, dizaines+unités).
    Gère aussi les décimaux (virgule) et le signe négatif.
    """
    if not events or not display_num:
        return

    # ── Décimaux : séparer à "virgule" ──────────────────────────────────
    dot_pos = -1
    for c in (".", ","):
        if c in display_num:
            dot_pos = display_num.index(c)
            break

    virgule_idx = -1
    for i, evt in enumerate(events):
        if evt.ortho == "virgule":
            virgule_idx = i
            break

    if virgule_idx >= 0 and dot_pos >= 0:
        # Partie entière
        int_events = events[:virgule_idx]
        int_str = display_num[:dot_pos]
        _assign_span_num_integer(int_events, int_str, offset=0)
        # Virgule
        events[virgule_idx].span_num = (dot_pos, dot_pos + 1)
        # Partie décimale
        dec_events = events[virgule_idx + 1:]
        dec_str = display_num[dot_pos + 1:]
        if dec_events:
            _assign_span_num_decimal(dec_events, dec_str, offset=dot_pos + 1, method=method)
        return

    # ── Entier pur ──────────────────────────────────────────────────────
    _assign_span_num_integer(events, display_num, offset=0)


def _assign_span_num_zeros(
    events: list[EventFormuleLecture],
    ei: int,
    count: int,
    offset: int,
) -> int:
    """Assigne les spans pour un run de zéros consécutifs. Retourne le nouvel ei."""
    z_start = offset
    z_end = offset + count
    if count < 3:
        for k in range(count):
            if ei < len(events):
                events[ei].span_num = (offset + k, offset + k + 1)
                ei += 1
    else:
        # "N fois zéro" → nombre_parts + "fois" + "zéro"
        n_parts = _nombre_vers_francais(count)
        total_events = len(n_parts) + 2
        for _ in range(total_events):
            if ei < len(events):
                events[ei].span_num = (z_start, z_end)
                ei += 1
    return ei


def _assign_span_num_decimal(
    events: list[EventFormuleLecture],
    dec_str: str,
    offset: int,
    method: str = "m2",
) -> None:
    """Assigne span_num pour la partie décimale (après la virgule).

    M2 : segments (runs de ≥3 zéros fusionnés, portions groupées par 3).
    M1 : zéros initiaux + reste entier surligné d'un bloc.
    M3 : chiffre par chiffre (1 event = 1 chiffre).
    """
    if not events:
        return
    if len(events) == 1:
        events[0].span_num = (offset, offset + len(dec_str))
        return

    ei = 0

    if method == "m3":
        # M3 : 1 chiffre = 1 event
        for k, ch in enumerate(dec_str):
            if ei < len(events):
                events[ei].span_num = (offset + k, offset + k + 1)
                ei += 1
    elif method == "m1":
        # M1 : zéros initiaux + reste comme un seul bloc
        lz = _count_leading_zeros(dec_str)
        rest = dec_str[lz:]
        if lz > 0:
            ei = _assign_span_num_zeros(events, ei, lz, offset)
        if rest:
            rest_start = offset + lz
            rest_end = offset + len(dec_str)
            while ei < len(events):
                events[ei].span_num = (rest_start, rest_end)
                ei += 1
    else:
        # M2 : segmentation par runs de zéros ≥3 + groupes de 3
        pos = 0
        for seg_str, seg_type in _segment_decimal_zeros(dec_str):
            seg_offset = offset + pos
            if seg_type == "zeros":
                ei = _assign_span_num_zeros(events, ei, len(seg_str), seg_offset)
            else:
                # Digit segment : zéros initiaux (1-2) + groupes de 3
                lz = _count_leading_zeros(seg_str)
                rest = seg_str[lz:]
                if lz > 0:
                    ei = _assign_span_num_zeros(events, ei, lz, seg_offset)
                if rest:
                    for grp in _group_by_3_left(rest):
                        n = int(grp)
                        grp_offset = offset + pos + lz
                        grp_end = grp_offset + len(grp)
                        if n > 0:
                            grp_parts = _nombre_vers_francais(n)
                            for _ in range(len(grp_parts)):
                                if ei < len(events):
                                    events[ei].span_num = (grp_offset, grp_end)
                                    ei += 1
                        lz += len(grp)  # avancer la position interne
            pos += len(seg_str)


def _assign_span_num_integer(
    events: list[EventFormuleLecture],
    num_str: str,
    offset: int,
) -> None:
    """Assigne span_num pour un nombre entier (sans virgule)."""
    if not events or not num_str:
        return

    # Ignorer le signe "moins"
    start_evt = 0
    if events and events[0].ortho == "moins":
        start_evt = 1
    evts = events[start_evt:]

    if not evts:
        return

    # Un seul event ou un seul chiffre → tout surligner
    if len(evts) == 1 or len(num_str) <= 1:
        for e in evts:
            e.span_num = (offset, offset + len(num_str))
        return

    # Valeur absolue
    try:
        abs_n = abs(int(num_str))
    except ValueError:
        for e in evts:
            e.span_num = (offset, offset + len(num_str))
        return

    # ── Groupes de magnitude (depuis la droite) ────────────────────────
    digits = str(abs_n)
    total = len(digits)
    mag_groups = []
    r = total
    # units (0-999)
    u_len = min(3, r)
    mag_groups.insert(0, ("units", r - u_len, r))
    r -= u_len
    # thousands
    if r > 0:
        t_len = min(3, r)
        mag_groups.insert(0, ("th", r - t_len, r))
        r -= t_len
    # millions
    if r > 0:
        m_len = min(3, r)
        mag_groups.insert(0, ("mil", r - m_len, r))
        r -= m_len
    # milliards
    if r > 0:
        mag_groups.insert(0, ("mrd", 0, r))

    # Convertir en positions absolues dans display_num
    for i in range(len(mag_groups)):
        name, s, e = mag_groups[i]
        mag_groups[i] = (name, offset + s, offset + e)

    mag_spans = {name: (s, e) for name, s, e in mag_groups}

    # Étendre les groupes quand les groupes inférieurs sont tous à zéro
    # (ex: 2000 → th=(0,1) étendu à (0,4) car units=0)
    extended_groups: set[str] = set()
    mag_order = [n for n, _, _ in mag_groups]
    full_end = offset + len(num_str)
    for i, gn in enumerate(mag_order):
        # Valeur des groupes suivants (plus bas)
        remainder = abs_n
        if gn == "mrd":
            remainder = abs_n % 1_000_000_000
        elif gn == "mil":
            remainder = abs_n % 1_000_000
        elif gn == "th":
            remainder = abs_n % 1_000
        else:
            continue  # units = dernier groupe, rien à étendre
        if remainder == 0 and total < 5:
            mag_spans[gn] = (mag_spans[gn][0], full_end)
            extended_groups.add(gn)

    # ── Répartir les events dans les groupes de magnitude ───────────────
    # Chercher les mots-clés frontières : milliard(s), million(s), mille
    boundary_indices: dict[str, int] = {}
    for i, evt in enumerate(evts):
        o = evt.ortho.lower().rstrip("s")
        if o == "milliard" and "mrd" not in boundary_indices:
            boundary_indices["mrd"] = i
        elif o == "million" and "mil" not in boundary_indices:
            boundary_indices["mil"] = i
        elif o == "mille" and "th" not in boundary_indices:
            boundary_indices["th"] = i

    # Assigner chaque event à un groupe de magnitude
    evt_groups: dict[str, list[int]] = {}
    for i, evt in enumerate(evts):
        if "mrd" in boundary_indices and i <= boundary_indices["mrd"]:
            g = "mrd"
        elif "mil" in boundary_indices and i <= boundary_indices["mil"]:
            g = "mil"
        elif "th" in boundary_indices and i <= boundary_indices["th"]:
            g = "th"
        else:
            g = "units"
        evt_groups.setdefault(g, []).append(i)

    # ── Dans chaque groupe, décomposer centaines / dizaines+unités ──────
    _MAGNITUDE_WORDS = {"mille", "million", "millions", "milliard", "milliards"}

    for gname, indices in evt_groups.items():
        if gname not in mag_spans:
            continue
        gspan = mag_spans[gname]
        g_len = gspan[1] - gspan[0]

        # Séparer les mots-clés de magnitude (mille/million/milliard)
        # qui gardent toujours le span du groupe entier
        mag_indices = [idx for idx in indices if evts[idx].ortho.lower() in _MAGNITUDE_WORDS]
        val_indices = [idx for idx in indices if idx not in mag_indices]

        for idx in mag_indices:
            evts[idx].span_num = gspan

        # Valeur du groupe (0-999)
        if gname == "mrd":
            gval = abs_n // 1_000_000_000
        elif gname == "mil":
            gval = (abs_n % 1_000_000_000) // 1_000_000
        elif gname == "th":
            gval = (abs_n % 1_000_000) // 1_000
        else:
            gval = abs_n % 1_000

        # Span effectif : sauter les zéros internes non prononcés
        # (ex: "025" → "25" pour 1025, "005" → "5" pour 1005)
        # Ne pas toucher les groupes étendus (ex: "2000" pour 2000)
        val_span = gspan
        is_extended = gname in extended_groups
        if not is_extended and gval > 0:
            sig = len(str(gval))
            if sig < g_len:
                val_span = (gspan[1] - sig, gspan[1])
        val_len = val_span[1] - val_span[0]

        if val_len <= 1 or len(val_indices) <= 1:
            # Groupe d'un seul chiffre ou un seul event → atomique
            for idx in val_indices:
                evts[idx].span_num = val_span
            continue

        # Chercher "cent(s)" dans ce groupe
        cent_idx_local = -1
        for idx in val_indices:
            if evts[idx].ortho.lower().rstrip("s") == "cent":
                cent_idx_local = idx
                break

        if cent_idx_local < 0:
            # Pas de "cent" → tout est dizaines+unités
            _assign_tens_units(evts, val_indices, val_span, abs_n, gname)
            continue

        # Séparer centaines / dizaines+unités
        h_indices = [idx for idx in val_indices if idx <= cent_idx_local]
        tu_indices = [idx for idx in val_indices if idx > cent_idx_local]

        if not tu_indices:
            # Rien après cent (ex: 200, 100) → atomique
            for idx in h_indices:
                evts[idx].span_num = val_span
        else:
            # Centaines = premier chiffre, dizaines+unités = reste
            h_span = (val_span[0], val_span[0] + 1)
            tu_span_local = (val_span[0] + 1, val_span[1])
            # Sauter le zéro de dizaine dans tu_span (ex: "05" → "5")
            tu_val = gval % 100
            if tu_val > 0:
                tu_sig = len(str(tu_val))
                tu_len = tu_span_local[1] - tu_span_local[0]
                if tu_sig < tu_len:
                    tu_span_local = (tu_span_local[1] - tu_sig, tu_span_local[1])
            for idx in h_indices:
                evts[idx].span_num = h_span
            _assign_tens_units(evts, tu_indices, tu_span_local, abs_n, gname)


def _assign_tens_units(
    evts: list[EventFormuleLecture],
    indices: list[int],
    span: Span,
    abs_n: int,
    gname: str,
) -> None:
    """Assigne span_num pour un sous-groupe dizaines+unités (0-99).

    Règles :
    - Chiffre d'unité = 0 (10, 20, …, 90) → atomique
    - Sinon : dernier event = chiffre d'unité, le reste = chiffre de dizaine
    """
    s_len = span[1] - span[0]
    if s_len <= 1 or len(indices) <= 1:
        for idx in indices:
            evts[idx].span_num = span
        return

    # Déterminer la valeur du sous-groupe
    if gname == "mrd":
        tu_val = abs_n // 1_000_000_000
    elif gname == "mil":
        tu_val = (abs_n % 1_000_000_000) // 1_000_000
    elif gname == "th":
        tu_val = (abs_n % 1_000_000) // 1_000
    else:
        tu_val = abs_n % 1_000
    tu_val = tu_val % 100  # juste les dizaines+unités

    # Chiffre d'unité = 0 → atomique (70, 80, 90, etc.)
    if tu_val % 10 == 0:
        for idx in indices:
            evts[idx].span_num = span
        return

    # 2 chiffres : dernier event = unité, le reste = dizaine
    if s_len == 2:
        tens_span = (span[0], span[0] + 1)
        units_span = (span[0] + 1, span[1])
        for idx in indices[:-1]:
            evts[idx].span_num = tens_span
        evts[indices[-1]].span_num = units_span
    else:
        for idx in indices:
            evts[idx].span_num = span


def _make_result(
    events: list[EventFormuleLecture],
    display_num: str = "",
    display_rom: str = "",
    valeur: int | float | str = "",
) -> LectureFormuleResult:
    """Construit un LectureFormuleResult à partir d'events."""
    display = "-".join(e.ortho for e in events)
    phone = " ".join(e.phone for e in events)
    # Calculer span_fr pour chaque event
    offset = 0
    for evt in events:
        evt.span_fr = (offset, offset + len(evt.ortho))
        offset += len(evt.ortho) + 1  # +1 pour le "-"
    # Calculer span_num à partir de span_source (si pas déjà calculé)
    # span_source contient les positions dans le texte source ;
    # display_num ≈ texte source → on soustrait l'offset de base.
    # _compute_span_num() peut avoir déjà assigné les spans → ne pas écraser.
    if display_num and events:
        already_computed = any(
            e.span_num and e.span_num != (0, 0) for e in events
        )
        if not already_computed:
            valid = [e.span_source for e in events
                     if e.span_source and e.span_source[0] < e.span_source[1]]
            if valid:
                base = min(s[0] for s in valid)
                num_len = len(display_num)
                for evt in events:
                    ss = evt.span_source
                    if ss and ss[0] < ss[1]:
                        evt.span_num = (max(0, ss[0] - base),
                                        min(ss[1] - base, num_len))
    return LectureFormuleResult(
        display_fr=display, phone=phone, events=events,
        display_num=display_num, display_rom=display_rom, valeur=valeur,
    )


def _events_from_parts(
    parts: list[tuple[str, str]],
    span: Span,
    text: str,
    composant: int = 0,
) -> list[EventFormuleLecture]:
    """Construit des events avec spans répartis sur le texte source.

    Les spans sont distribués proportionnellement sur le texte source.
    """
    if not parts:
        return []

    src_start, src_end = span
    src_len = src_end - src_start
    n_parts = len(parts)

    events: list[EventFormuleLecture] = []
    for i, (ortho, phone) in enumerate(parts):
        # Distribution proportionnelle des spans
        seg_start = src_start + (i * src_len) // n_parts
        seg_end = src_start + ((i + 1) * src_len) // n_parts
        if seg_end == seg_start and src_len > 0:
            seg_end = seg_start + 1
        events.append(EventFormuleLecture(
            ortho=ortho,
            phone=phone,
            span_source=(seg_start, seg_end),
            composant=composant,
        ))
    return events


def _digits_span_events(
    parts: list[tuple[str, str]],
    digit_text: str,
    offset: int,
    composant: int = 0,
) -> list[EventFormuleLecture]:
    """Construit des events pour un nombre avec alignement chiffre-par-chiffre."""
    return _events_from_parts(parts, (offset, offset + len(digit_text)), digit_text,
                              composant=composant)


# ══════════════════════════════════════════════════════════════════════════════
# Lecteurs par type de formule
# ══════════════════════════════════════════════════════════════════════════════

# -- NOMBRE --------------------------------------------------------------------

def lire_nombre(
    text: str,
    span: Span = (0, 0),
    feminin: bool = False,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit un nombre entier ou décimal.

    Composants : 1 seul (composant=0 pour tous les events).
    """
    if options is None:
        options = OptionsLecture()

    text_clean = text.replace(" ", "").replace("'", "").replace("\u202f", "")

    # Retirer les suffixes non numériques (ex: "42.0254€" → "42.0254")
    _strip = text_clean.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ€$£¥%°")
    if _strip and _strip != text_clean:
        text_clean = _strip

    # Nombre décimal ?
    if "," in text_clean or "." in text_clean:
        return _lire_decimal(text_clean, span, options)

    # Signe ?
    negatif = text_clean.startswith("-") or text_clean.startswith("−")
    plus_ou_moins = not negatif and text_clean.startswith("±")
    positif = not negatif and not plus_ou_moins and text_clean.startswith("+")
    if negatif:
        text_clean = text_clean.lstrip("-−")
    elif plus_ou_moins:
        text_clean = text_clean.lstrip("±")
    elif positif:
        text_clean = text_clean.lstrip("+")

    # Normaliser les zéros initiaux (002568 → 2568)
    text_clean = text_clean.lstrip("0") or "0"

    try:
        n = int(text_clean)
    except ValueError:
        return _epeler_texte(text, span)

    val = -n if negatif else n

    # Auto-conversion en scientifique si > 15 chiffres
    if options.auto_convert_sci and len(text_clean) > 15:
        sign = "-" if negatif else "+" if positif else ""
        sci_str = _decimal_to_sci(sign + text_clean)
        if sci_str is not None:
            return lire_scientifique(sci_str, span, options=OptionsLecture(auto_convert_sci=False))

    # Lecture en français (valeur absolue, signe ajouté manuellement)
    parts: list[tuple[str, str]] = []
    if negatif:
        parts.append(_SYMBOLES["-"])
    elif plus_ou_moins:
        parts.append(_SYMBOLES["±"])
    elif positif:
        parts.append(_SYMBOLES["+"])
    parts.extend(_nombre_vers_francais(n, feminin=feminin))
    events = _events_from_parts(parts, span, text, composant=0)

    # Chiffres romains : uniquement pour les entiers positifs sans signe
    display_rom = ""
    if options.romain_actif and not negatif and not positif and not plus_ou_moins and 1 <= n <= 39999:
        try:
            from lectura_formules.romains import int_to_roman
            display_rom = int_to_roman(n)
            _compute_span_rom(events, n, display_rom)
        except (ImportError, ValueError):
            pass

    _compute_span_num(events, text_clean)
    formatted = _format_display_num(text_clean)
    if formatted != text_clean:
        _adjust_spans_for_apostrophes(events, text_clean, formatted)

    # Ajouter le signe au display_num et ajuster les spans
    if negatif or plus_ou_moins or positif:
        sign_char = "±" if plus_ou_moins else "-" if negatif else "+"
        sign_ortho = "plus ou moins" if plus_ou_moins else "moins" if negatif else "plus"
        formatted = sign_char + formatted
        sign_len = len(sign_char)
        for evt in events:
            if evt.ortho == sign_ortho:
                evt.span_num = (0, sign_len)
                break
        for evt in events:
            if evt.ortho != sign_ortho and evt.span_num and evt.span_num != (0, 0):
                s, e = evt.span_num
                evt.span_num = (s + sign_len, e + sign_len)

    return _make_result(events, display_num=formatted, display_rom=display_rom, valeur=val)


def _lire_decimal(
    text_clean: str, span: Span,
    options: OptionsLecture | None = None,
) -> LectureFormuleResult:
    """Lit un nombre décimal avec méthode M1 ou M2."""
    if options is None:
        options = OptionsLecture()

    # Gestion du signe
    negatif = text_clean.startswith("-") or text_clean.startswith("−")
    plus_ou_moins = not negatif and text_clean.startswith("±")
    positif = not negatif and not plus_ou_moins and text_clean.startswith("+")
    sans_signe = text_clean
    if negatif:
        sans_signe = text_clean.lstrip("-−")
    elif plus_ou_moins:
        sans_signe = text_clean.lstrip("±")
    elif positif:
        sans_signe = text_clean.lstrip("+")

    sep = "," if "," in sans_signe else "."
    partie_ent, partie_dec = sans_signe.split(sep, 1)

    # Normaliser les zéros initiaux de la partie entière (002.568 → 2.568)
    partie_ent = partie_ent.lstrip("0") or "0"
    sans_signe = partie_ent + sep + partie_dec
    sign_char = "±" if plus_ou_moins else "-" if negatif else "+" if positif else ""
    text_clean = sign_char + sans_signe

    # Auto-conversion en scientifique si total > 15 chiffres
    total_digits = len(partie_ent.lstrip("0")) + len(partie_dec.rstrip("0"))
    if options.auto_convert_sci and total_digits > 15:
        sci_str = _decimal_to_sci(text_clean)
        if sci_str is not None:
            return lire_scientifique(sci_str, span, options=OptionsLecture(auto_convert_sci=False))

    parts: list[tuple[str, str]] = []

    # Signe
    if negatif:
        parts.append(_SYMBOLES["-"])
    elif plus_ou_moins:
        parts.append(_SYMBOLES["±"])
    elif positif:
        parts.append(_SYMBOLES["+"])

    # Partie entière
    n_ent = int(partie_ent) if partie_ent else 0
    parts.extend(_nombre_vers_francais(n_ent))

    # Virgule
    parts.append(_VIRGULE)

    # Partie décimale selon la méthode
    method = options.decimal_method if options else "m2"
    if method == "m1":
        parts.extend(_decimal_m1(partie_dec))
    elif method == "m3":
        parts.extend(_decimal_m3(partie_dec))
    else:
        parts.extend(_decimal_m2(partie_dec))

    # Valeur numérique
    try:
        valeur = float(text_clean.replace(",", ".").replace("−", "-"))
    except ValueError:
        valeur = text_clean

    # Calculer span_num sur la partie sans signe, puis décaler si signe
    events = _events_from_parts(parts, span, text_clean, composant=0)

    if negatif or plus_ou_moins or positif:
        # Calculer spans sur sans_signe (sans le caractère de signe)
        sign_events = events[:1]   # l'event "moins"/"plus"/"plus ou moins"
        rest_events = events[1:]
        _compute_span_num(rest_events, sans_signe, method=method)
        # Décaler tous les spans pour le caractère de signe
        sign_len = len(sign_char)
        for evt in rest_events:
            if evt.span_num and evt.span_num != (0, 0):
                s, e = evt.span_num
                evt.span_num = (s + sign_len, e + sign_len)
        sign_events[0].span_num = (0, sign_len)
    else:
        _compute_span_num(events, sans_signe, method=method)

    formatted = _format_display_decimal(text_clean)
    # Ajuster les spans si la partie entière a des apostrophes
    if formatted != text_clean:
        _adjust_spans_for_apostrophes(events, text_clean, formatted)
    return _make_result(events, display_num=formatted, valeur=valeur)


def _count_leading_zeros(s: str) -> int:
    """Compte les zéros initiaux dans une chaîne de chiffres."""
    count = 0
    for ch in s:
        if ch == "0":
            count += 1
        else:
            break
    return count


def _leading_zeros_parts(lz: int) -> list[tuple[str, str]]:
    """Construit les parts pour des zéros consécutifs.

    1-2 zéros : les lire individuellement.
    ≥3 zéros : "N fois zéro".
    """
    if lz <= 0:
        return []
    if lz < 3:
        return [_u("0") for _ in range(lz)]
    # "N fois zéro"
    result: list[tuple[str, str]] = []
    result.extend(_nombre_vers_francais(lz))
    result.append(_FOIS)
    result.append(_u("0"))
    return result


def _group_by_3_left(s: str) -> list[str]:
    """Groupe les chiffres par 3 depuis la gauche.

    Ex: "25124" → ["251", "24"]
    Ex: "5" → ["5"]
    """
    groups = []
    i = 0
    while i < len(s):
        end = min(i + 3, len(s))
        groups.append(s[i:end])
        i = end
    return groups


def _segment_decimal_zeros(s: str) -> list[tuple[str, str]]:
    """Segmente une chaîne décimale en runs de zéros (≥3) et portions de chiffres.

    Retourne [(segment_str, "zeros"|"digits"), ...].
    Ex: "0000002540000024" → [("000000","zeros"),("254","digits"),("00000","zeros"),("24","digits")]
    Ex: "002500045800001457" → [("0025","digits"),("000","zeros"),("458","digits"),("0000","zeros"),("1457","digits")]
    """
    segments: list[tuple[str, str]] = []
    i = 0
    n = len(s)
    while i < n:
        # Vérifier si on a un run de zéros ≥3
        if s[i] == "0":
            j = i
            while j < n and s[j] == "0":
                j += 1
            if j - i >= 3:
                segments.append((s[i:j], "zeros"))
                i = j
                continue
        # Pas un run de zéros ≥3 : avancer jusqu'au prochain run ≥3
        j = i
        while j < n:
            if s[j] == "0":
                k = j
                while k < n and s[k] == "0":
                    k += 1
                if k - j >= 3:
                    break  # trouvé, arrêter le segment digits ici
                j = k
            else:
                j += 1
        segments.append((s[i:j], "digits"))
        i = j
    return segments


def _decimal_m2(partie_dec: str) -> list[tuple[str, str]]:
    """Méthode M2 : groupes de 3, runs de ≥3 zéros fusionnés en "N fois zéro".

    Ex: "0000002540000024"
      → six fois zéro | deux-cent-cinquante-quatre | cinq fois zéro | vingt-quatre
    Ex: "002500045800001457"
      → zéro zéro | vingt-cinq | trois fois zéro | quatre-cent-cinquante-huit
        | quatre fois zéro | cent-quarante-cinq | sept
    """
    parts: list[tuple[str, str]] = []

    for seg_str, seg_type in _segment_decimal_zeros(partie_dec):
        if seg_type == "zeros":
            parts.extend(_leading_zeros_parts(len(seg_str)))
        else:
            # Segment de chiffres : zéros initiaux (1-2) + groupes de 3
            lz = _count_leading_zeros(seg_str)
            rest = seg_str[lz:]
            parts.extend(_leading_zeros_parts(lz))
            if rest:
                for grp in _group_by_3_left(rest):
                    n = int(grp)
                    if n > 0:
                        parts.extend(_nombre_vers_francais(n))

    return parts


def _decimal_m1(partie_dec: str) -> list[tuple[str, str]]:
    """Méthode M1 : zéros initiaux + reste comme entier complet."""
    parts: list[tuple[str, str]] = []

    lz = _count_leading_zeros(partie_dec)
    rest = partie_dec[lz:]

    # Zéros initiaux
    parts.extend(_leading_zeros_parts(lz))

    # Reste comme entier complet
    if rest:
        n = int(rest)
        if n > 0:
            parts.extend(_nombre_vers_francais(n))

    return parts


def _decimal_m3(partie_dec: str) -> list[tuple[str, str]]:
    """Méthode M3 : chiffre par chiffre.

    Ex: "14"  → ("un", …), ("quatre", …)
    Ex: "003" → ("zéro", …), ("zéro", …), ("trois", …)
    """
    parts: list[tuple[str, str]] = []
    for ch in partie_dec:
        parts.append(_u(ch))
    return parts


# -- SIGLE ---------------------------------------------------------------------

def lire_sigle(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit un sigle lettre par lettre, chiffres comme nombres.

    Composants : 1 composant par lettre/chiffre-groupe.
    Ex: "SNCF" → 4 composants ; "B2B" → 3 composants.
    """
    events: list[EventFormuleLecture] = []
    src_start = span[0]
    comp_idx = 0

    i = 0
    while i < len(text):
        ch = text[i]
        pos = src_start + i

        if ch.isdigit():
            # Accumuler le groupe de chiffres
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            group = text[i:j]
            n = int(group)
            parts = _nombre_vers_francais(n)
            for p_ortho, p_phone in parts:
                events.append(EventFormuleLecture(
                    ortho=p_ortho, phone=p_phone,
                    span_source=(pos, src_start + j),
                    composant=comp_idx,
                ))
            comp_idx += 1
            i = j
        elif ch.isalpha() and ch.upper() in _LETTRES:
            ortho, phone = _LETTRES[ch.upper()]
            events.append(EventFormuleLecture(
                ortho=ortho, phone=phone,
                span_source=(pos, pos + 1),
                composant=comp_idx,
            ))
            comp_idx += 1
            i += 1
        else:
            # Ignorer les caractères non alpha-numériques (points dans W.W.F.)
            i += 1

    return _make_result(events, display_num=text)


# -- DATE ----------------------------------------------------------------------

_DATE_RE = re.compile(
    r"(\d{1,2})\s*[/\-\.]\s*(\d{1,2})\s*[/\-\.]\s*(\d{2,4})"
)


def lire_date(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit une date JJ/MM/AAAA.

    Composants : jour (0), mois (1), année (2).
    """
    m = _DATE_RE.match(text.strip())
    if not m:
        return _epeler_texte(text, span)

    jour_str, mois_str, annee_str = m.group(1), m.group(2), m.group(3)
    jour = int(jour_str)
    mois = int(mois_str)
    annee = int(annee_str)
    # Année à 2 chiffres
    if annee < 100:
        annee += 2000 if annee < 50 else 1900

    events: list[EventFormuleLecture] = []
    src = span[0]

    # -- Jour (composant 0) --
    jour_start = src + m.start(1)
    jour_end = src + m.end(1)
    if jour == 1:
        events.append(EventFormuleLecture(
            ortho="premier", phone="pʁømje",
            span_source=(jour_start, jour_end),
            composant=0,
        ))
    else:
        parts = _nombre_vers_francais(jour)
        events.extend(_events_from_parts(parts, (jour_start, jour_end), jour_str,
                                         composant=0))

    # -- Mois (composant 1) --
    mois_start = src + m.start(2)
    mois_end = src + m.end(2)
    if mois in _MOIS:
        t_mois, p_mois = _MOIS[mois]
        events.append(EventFormuleLecture(
            ortho=t_mois, phone=p_mois,
            span_source=(mois_start, mois_end),
            composant=1,
        ))
    else:
        parts = _nombre_vers_francais(mois)
        events.extend(_events_from_parts(parts, (mois_start, mois_end), mois_str,
                                         composant=1))

    # -- Année (composant 2) --
    annee_start = src + m.start(3)
    annee_end = src + m.end(3)
    parts_annee = _nombre_vers_francais(annee)
    events.extend(_events_from_parts(parts_annee, (annee_start, annee_end), annee_str,
                                     composant=2))

    # -- span_num par composant (positions dans display_num) --
    display = text.strip()
    j_s, j_e = m.start(1), m.end(1)
    mo_s, mo_e = m.start(2), m.end(2)
    a_s, a_e = m.start(3), m.end(3)
    for e in events:
        if e.composant == 0:
            e.span_num = (j_s, j_e)
        elif e.composant == 1:
            e.span_num = (mo_s, mo_e)
        elif e.composant == 2:
            e.span_num = (a_s, a_e)

    return _make_result(events, display_num=display)


# -- TELEPHONE -----------------------------------------------------------------

_TEL_CLEAN_RE = re.compile(r"[\s.\-]")


def _normalize_phone(text: str) -> str:
    """Normalise un numéro de téléphone français sans séparateur.

    10 chiffres commençant par 0 → XX.XX.XX.XX.XX
    """
    digits = text.replace(" ", "").replace(".", "").replace("-", "")
    if len(digits) == 10 and digits[0] == "0" and digits.isdigit():
        return ".".join(digits[i:i + 2] for i in range(0, 10, 2))
    return text


def lire_telephone(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit un numéro de téléphone par groupes de chiffres.

    Respecte le groupement original (espaces, points, tirets).
    Composants : 1 composant par groupe de chiffres.
    """
    text = _normalize_phone(text)
    events: list[EventFormuleLecture] = []
    src_start = span[0]
    comp_idx = 0
    i = 0

    # Préfixe "+"
    stripped = text.lstrip()
    if stripped.startswith("+"):
        plus_pos = text.index("+")
        events.append(EventFormuleLecture(
            ortho="plus", phone="plys",
            span_source=(src_start + plus_pos, src_start + plus_pos + 1),
            composant=comp_idx,
        ))
        events[-1].span_num = (plus_pos, plus_pos + 1)
        i = plus_pos + 1

    # Parcourir le texte par groupes de chiffres contigus
    while i < len(text):
        ch = text[i]
        if ch.isdigit():
            # Extraire le groupe de chiffres contigus
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            group = text[i:j]
            grp_start = i
            grp_end = j

            # Lire les zéros initiaux + reste comme nombre
            # Tout le bloc a le même span_num (surlignage groupé)
            k = 0
            while k < len(group) - 1 and group[k] == "0":
                t0, p0 = _u("0")
                events.append(EventFormuleLecture(
                    ortho=t0, phone=p0,
                    span_source=(src_start + grp_start, src_start + grp_end),
                    composant=comp_idx,
                ))
                events[-1].span_num = (grp_start, grp_end)
                k += 1

            # Lire le reste comme un nombre
            remainder = group[k:]
            if remainder:
                n = int(remainder)
                parts = _nombre_vers_francais(n)
                for p_ortho, p_phone in parts:
                    events.append(EventFormuleLecture(
                        ortho=p_ortho, phone=p_phone,
                        span_source=(src_start + grp_start, src_start + grp_end),
                        composant=comp_idx,
                    ))
                    events[-1].span_num = (grp_start, grp_end)

            comp_idx += 1
            i = j
        else:
            i += 1

    return _make_result(events, display_num=text)


# -- ORDINAL ------------------------------------------------------------------

_ORDINAL_RE = re.compile(
    r"(\d+)\s*(er|re|ère|e|ème|ème|ième|eme|ier|ière|nd|nde)\b",
    re.IGNORECASE,
)
_ORDINAL_ROMAN_RE = re.compile(
    r"([IVXLCDM]+)\s*(er|re|ère|e|ème|ème|ième|eme|ier|ière|nd|nde)\b",
    re.IGNORECASE,
)


def lire_ordinal(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit un ordinal : 1er → premier, 2e → deuxième, etc.

    Composants : nombre (0), suffixe ordinal (1).
    Pour 1er/1ère/2nd/2nde : un seul composant (0) car fusionné.
    """
    m = _ORDINAL_RE.match(text.strip())
    if not m:
        # Tenter avec chiffres romains
        m = _ORDINAL_ROMAN_RE.match(text.strip())
        if not m:
            return _epeler_texte(text, span)
        from lectura_formules.romains import roman_to_int
        try:
            n = roman_to_int(m.group(1))
        except (ValueError, KeyError):
            return _epeler_texte(text, span)
    else:
        n = int(m.group(1))
    suffix = m.group(2).lower()
    src = span[0]
    num_start = src + m.start(1)
    num_end = src + m.end(1)
    suf_start = src + m.start(2)
    suf_end = src + m.end(2)

    events: list[EventFormuleLecture] = []

    # Chiffres romains pour les ordinaux
    display_rom = ""
    if 1 <= n <= 39999:
        try:
            from lectura_formules.romains import int_to_roman
            rom_base = int_to_roman(n)
            # Ajouter le suffixe ordinal au display_rom
            if n == 1:
                if suffix in ("re", "ère", "ière"):
                    display_rom = rom_base + "re"
                else:
                    display_rom = rom_base + "er"
            elif n == 2 and suffix in ("nd", "nde"):
                display_rom = rom_base + suffix
            else:
                display_rom = rom_base + "e"
        except (ImportError, ValueError):
            pass
    rom_full = (0, int(len(display_rom))) if display_rom else (0, 0)

    if n == 1:
        # 1er/1re/1ère → premier/première (composant 0 unique)
        if suffix in ("re", "ère", "ière"):
            evt = EventFormuleLecture(
                ortho="première", phone="pʁømjɛʁ",
                span_source=(num_start, suf_end),
                composant=0,
            )
            evt.span_rom = rom_full
            events.append(evt)
        else:
            evt = EventFormuleLecture(
                ortho="premier", phone="pʁømje",
                span_source=(num_start, suf_end),
                composant=0,
            )
            evt.span_rom = rom_full
            events.append(evt)
    elif n == 2 and suffix in ("nd", "nde"):
        if suffix == "nde":
            evt = EventFormuleLecture(
                ortho="seconde", phone="səɡɔ̃d",
                span_source=(num_start, suf_end),
                composant=0,
            )
            evt.span_rom = rom_full
            events.append(evt)
        else:
            evt = EventFormuleLecture(
                ortho="second", phone="səɡɔ̃",
                span_source=(num_start, suf_end),
                composant=0,
            )
            evt.span_rom = rom_full
            events.append(evt)
    else:
        # Nombre cardinal (composant 0) + suffixe ordinal (composant 1)
        parts = _nombre_vers_francais(n)
        num_text = m.group(1)
        for ortho, phone in parts:
            events.append(EventFormuleLecture(
                ortho=ortho, phone=phone,
                span_source=(num_start, num_end),
                composant=0,
            ))
        # Calculer span_num structurel (centaines/dizaines/unités)
        _assign_span_num_integer(events, num_text, offset=0)

        # Calculer span_rom sur les events cardinaux AVANT la substitution ordinale
        if display_rom:
            _compute_span_rom(events, n, display_rom)

        # Suffixe ordinal : appliquer au dernier mot cardinal
        last_cardinal = parts[-1][0] if parts else ""
        if last_cardinal in _ORDINAUX:
            ord_t, ord_p = _ORDINAUX[last_cardinal]
            if events:
                last_evt = events[-1]
                last_span = last_evt.span_num or (0, len(num_text))
                last_rom = last_evt.span_rom
                events[-1] = EventFormuleLecture(
                    ortho=ord_t, phone=ord_p,
                    span_source=(num_start + last_span[0], suf_end),
                    composant=1,
                )
                # Étendre span_num pour inclure le suffixe
                events[-1].span_num = (last_span[0], len(num_text) + len(suffix))
                # Propager span_rom du cardinal et l'étendre au suffixe ordinal
                if last_rom and last_rom != (0, 0):
                    events[-1].span_rom = (last_rom[0], len(display_rom))
        else:
            # Fallback : ajouter "ième" (composant 1)
            events.append(EventFormuleLecture(
                ortho="ième", phone="jɛm",
                span_source=(suf_start, suf_end),
                composant=1,
            ))
            events[-1].span_num = (len(num_text), len(num_text) + len(suffix))
            # Étendre span_rom du dernier event cardinal pour inclure le suffixe
            if display_rom and len(events) >= 2:
                prev = events[-2]
                if prev.span_rom and prev.span_rom != (0, 0):
                    prev.span_rom = (prev.span_rom[0], len(display_rom))

    return _make_result(events, display_num=text.strip(), display_rom=display_rom)


# -- FRACTION ------------------------------------------------------------------

_FRACTION_RE = re.compile(r"([-+]?\d+)\s*/\s*(\d+)")


def lire_fraction(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit une fraction selon le mode choisi.

    Composants : numérateur (0), "sur/de" (1), dénominateur (2).
    """
    if options is None:
        options = OptionsLecture()

    m = _FRACTION_RE.match(text.strip())
    if not m:
        return _epeler_texte(text, span)

    num = int(m.group(1))
    den = int(m.group(2))
    src = span[0]
    num_start = src + m.start(1)
    num_end = src + m.end(1)
    den_start = src + m.start(2)
    den_end = src + m.end(2)

    events: list[EventFormuleLecture] = []

    if options.fraction_mode == "standard":
        # Numérateur (composant 0) + "sur" (composant 1) + dénominateur (composant 2)
        parts_num = _nombre_vers_francais(num)
        events.extend(_events_from_parts(parts_num, (num_start, num_end), m.group(1),
                                         composant=0))
        events.append(EventFormuleLecture(
            ortho="sur", phone="syʁ",
            span_source=(num_end, den_start),
            composant=1,
        ))
        parts_den = _nombre_vers_francais(den)
        events.extend(_events_from_parts(parts_den, (den_start, den_end), m.group(2),
                                         composant=2))
    elif options.fraction_mode == "hybride":
        events = _fraction_hybride(num, den, num_start, num_end, den_start, den_end,
                                   m.group(1), m.group(2))
    else:
        # ordinal
        events = _fraction_ordinal(num, den, num_start, num_end, den_start, den_end,
                                   m.group(1), m.group(2))

    return _make_result(events, display_num=text.strip())


def _fraction_hybride(
    num: int, den: int,
    num_start: int, num_end: int,
    den_start: int, den_end: int,
    num_text: str, den_text: str,
) -> list[EventFormuleLecture]:
    """Mode hybride : cas spéciaux (demi, tiers, quart) sinon ordinal.

    Composants : numérateur (0), dénominateur (2).
    """
    events: list[EventFormuleLecture] = []
    pluriel = abs(num) > 1

    # Cas spéciaux
    if den == 2:
        parts_num = _nombre_vers_francais(num)
        events.extend(_events_from_parts(parts_num, (num_start, num_end), num_text,
                                         composant=0))
        ortho = "demis" if pluriel else "demi"
        events.append(EventFormuleLecture(
            ortho=ortho, phone="dəmi",
            span_source=(den_start, den_end),
            composant=2,
        ))
        return events

    if den == 3:
        parts_num = _nombre_vers_francais(num)
        events.extend(_events_from_parts(parts_num, (num_start, num_end), num_text,
                                         composant=0))
        events.append(EventFormuleLecture(
            ortho="tiers", phone="tjɛʁ",
            span_source=(den_start, den_end),
            composant=2,
        ))
        return events

    if den == 4:
        parts_num = _nombre_vers_francais(num)
        events.extend(_events_from_parts(parts_num, (num_start, num_end), num_text,
                                         composant=0))
        ortho = "quarts" if pluriel else "quart"
        events.append(EventFormuleLecture(
            ortho=ortho, phone="kaʁ",
            span_source=(den_start, den_end),
            composant=2,
        ))
        return events

    # Dénominateurs simples (5-16, 100, 1000) : ordinal
    if 5 <= den <= 16 or den in (100, 1000):
        return _fraction_ordinal(num, den, num_start, num_end, den_start, den_end,
                                 num_text, den_text)

    # Autres dénominateurs : "sur" (standard)
    events: list[EventFormuleLecture] = []
    parts_num = _nombre_vers_francais(num)
    events.extend(_events_from_parts(parts_num, (num_start, num_end), num_text,
                                     composant=0))
    events.append(EventFormuleLecture(
        ortho="sur", phone="syʁ",
        span_source=(num_end, den_start),
        composant=1,
    ))
    parts_den = _nombre_vers_francais(den)
    events.extend(_events_from_parts(parts_den, (den_start, den_end), den_text,
                                     composant=2))
    return events


def _fraction_ordinal(
    num: int, den: int,
    num_start: int, num_end: int,
    den_start: int, den_end: int,
    num_text: str, den_text: str,
) -> list[EventFormuleLecture]:
    """Mode ordinal : numérateur (composant 0) + dénominateur ordinal (composant 2)."""
    events: list[EventFormuleLecture] = []

    # Numérateur (composant 0)
    parts_num = _nombre_vers_francais(num)
    events.extend(_events_from_parts(parts_num, (num_start, num_end), num_text,
                                     composant=0))

    # Dénominateur ordinal (composant 2)
    parts_den = _nombre_vers_francais(den)
    if parts_den:
        last_cardinal = parts_den[-1][0]
        if last_cardinal in _ORDINAUX:
            ord_t, ord_p = _ORDINAUX[last_cardinal]
            # Ajouter "s" si pluriel
            if abs(num) > 1:
                ord_t += "s"
            # Tous les events sauf le dernier (partie cardinale)
            for t, p in parts_den[:-1]:
                events.append(EventFormuleLecture(
                    ortho=t, phone=p,
                    span_source=(den_start, den_end),
                    composant=2,
                ))
            events.append(EventFormuleLecture(
                ortho=ord_t, phone=ord_p,
                span_source=(den_start, den_end),
                composant=2,
            ))
        else:
            # Fallback
            for t, p in parts_den:
                events.append(EventFormuleLecture(
                    ortho=t, phone=p,
                    span_source=(den_start, den_end),
                    composant=2,
                ))
            events.append(EventFormuleLecture(
                ortho="ième", phone="jɛm",
                span_source=(den_start, den_end),
                composant=2,
            ))
    return events


# -- SCIENTIFIQUE --------------------------------------------------------------

_SCI_RE = re.compile(
    r"([+-]?\d+(?:[.,]\d+)?)\s*[eE×x]\s*([+-]?\d+(?:[.,]\d+)?)"
)


def _sci_to_decimal(text: str) -> str | None:
    """Convertit une notation scientifique en nombre décimal si ≤ 15 chiffres significatifs."""
    m = _SCI_RE.match(text.strip())
    if not m:
        return None
    try:
        mant_str = m.group(1).replace(",", ".")
        exp_str = m.group(2).replace(",", ".")
        val = float(mant_str) * (10 ** float(exp_str))
        # Vérifier si le résultat est un entier raisonnable
        if val == int(val) and abs(val) < 10**15:
            return str(int(val))
        # Ou un décimal raisonnable
        s = f"{val:.15g}"
        # Compter les chiffres significatifs
        digits = s.replace("-", "").replace(".", "").lstrip("0")
        if len(digits) <= 15 and "e" not in s.lower():
            return s
    except (ValueError, OverflowError):
        pass
    return None


def _decimal_to_sci(text: str) -> str | None:
    """Convertit un nombre décimal très grand/petit en notation scientifique."""
    try:
        cleaned = text.replace(",", ".").replace("'", "").replace(" ", "").replace("\u202f", "")
        val = float(cleaned)
        if val == 0:
            return None
        s = f"{val:.6e}"  # Ex: "1.234568e+17"
        return s
    except (ValueError, OverflowError):
        return None


def lire_scientifique(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit une notation scientifique : 1.23e-5 → un virgule vingt-trois
    fois dix exposant moins cinq.

    Composants : mantisse (0), "fois dix exposant" (1), exposant (2).
    """
    if options is None:
        options = OptionsLecture()

    # Auto-conversion : si le résultat est raisonnable, lire comme nombre
    if options.auto_convert_sci:
        decimal_str = _sci_to_decimal(text)
        if decimal_str is not None:
            return lire_nombre(decimal_str, span, options=options)

    m = _SCI_RE.match(text.strip())
    if not m:
        return _epeler_texte(text, span)

    mantisse_str = m.group(1)
    exposant_str = m.group(2)
    src = span[0]
    display_num = text.strip()

    # Positions dans display_num
    mant_dn_start = m.start(1)
    mant_dn_end = m.end(1)
    sep_dn_start = m.end(1)
    sep_dn_end = m.start(2)
    exp_dn_start = m.start(2)
    exp_dn_end = m.end(2)

    events: list[EventFormuleLecture] = []

    # Mantisse (composant 0)
    mant_start = src + m.start(1)
    mant_end = src + m.end(1)
    mant_result = lire_nombre(mantisse_str, span=(mant_start, mant_end))
    mant_events: list[EventFormuleLecture] = []
    for evt in mant_result.events:
        new_evt = EventFormuleLecture(
            ortho=evt.ortho, phone=evt.phone,
            span_source=evt.span_source,
            composant=0,
        )
        mant_events.append(new_evt)
    # Recalculer span_num pour la mantisse (as standalone number)
    _compute_span_num(mant_events, mantisse_str)
    # Décaler les spans vers la position dans display_num
    for evt in mant_events:
        if evt.span_num and evt.span_num != (0, 0):
            s0, e0 = evt.span_num
            evt.span_num = (mant_dn_start + s0, mant_dn_start + e0)
        else:
            evt.span_num = (mant_dn_start, mant_dn_end)
    events.extend(mant_events)

    # "fois dix exposant" (composant 1) → surligne le séparateur "e"
    sep_start = src + m.end(1)
    sep_end = src + m.start(2)
    for word, phone in [("fois", "fwa"), ("dix", "dis"), ("exposant", "ɛkspozɑ̃")]:
        evt = EventFormuleLecture(
            ortho=word, phone=phone,
            span_source=(sep_start, sep_end),
            composant=1,
        )
        evt.span_num = (sep_dn_start, sep_dn_end)
        events.append(evt)

    # Exposant (composant 2)
    exp_start = src + m.start(2)
    exp_end = src + m.end(2)
    exp_result = lire_nombre(exposant_str, span=(exp_start, exp_end))
    exp_events: list[EventFormuleLecture] = []
    for evt in exp_result.events:
        new_evt = EventFormuleLecture(
            ortho=evt.ortho, phone=evt.phone,
            span_source=evt.span_source,
            composant=2,
        )
        new_evt.span_num = (exp_dn_start, exp_dn_end)
        exp_events.append(new_evt)
    events.extend(exp_events)

    return _make_result(events, display_num=display_num)


# -- MATHS ---------------------------------------------------------------------
# _tokenize_maths, _requalify_unit_vars et les constantes maths sont désormais
# dans lectura_tokeniseur.maths (importés en haut du fichier).
# Seul _GREEK_CHARS reste ici car il dépend de _GREC (table G2P locale).
_GREEK_CHARS = set(_GREC.keys())


def _smart_parens(tokens: list[_MathToken]) -> list[_MathToken]:
    """Transforme les parenthèses selon le contexte.

    - function / var-function-like + (  →  "de", supprime )
    - number / var-non-function + (   →  "facteur de", supprime )
    - (…)/(…)  →  supprime les 4 parenthèses
    - |…|  →  abs_open / abs_close
    """
    n = len(tokens)

    # 1. Associer les ( et ) correspondantes
    paren_pairs: dict[int, int] = {}
    stack: list[int] = []
    for i, tok in enumerate(tokens):
        if tok[0] == "(" and tok[1] == "bracket":
            stack.append(i)
        elif tok[0] == ")" and tok[1] == "bracket":
            if stack:
                paren_pairs[stack.pop()] = i

    # 2. Associer les |…| pour valeur absolue
    pipe_pairs: dict[int, int] = {}
    pipe_open: int | None = None
    for i, tok in enumerate(tokens):
        if tok[0] == "|" and tok[1] == "bracket":
            if pipe_open is None:
                pipe_open = i
            else:
                pipe_pairs[pipe_open] = i
                pipe_open = None
    pipe_closes = set(pipe_pairs.values())

    # 3. Détecter (…)/(…) → suppression des 4 parenthèses
    frac_opens: set[int] = set()
    frac_closes: set[int] = set()
    for o1, c1 in list(paren_pairs.items()):
        if c1 + 2 < n:
            if tokens[c1 + 1][0] == "/" and tokens[c1 + 1][1] == "operator":
                if tokens[c1 + 2][0] == "(" and tokens[c1 + 2][1] == "bracket":
                    o2 = c1 + 2
                    if o2 in paren_pairs:
                        c2 = paren_pairs[o2]
                        frac_opens.update({o1, o2})
                        frac_closes.update({c1, c2})

    # 4. Déterminer le connecteur pour chaque ( selon le contexte précédent
    def _effective_prev(result: list[_MathToken]) -> _MathToken | None:
        idx = len(result) - 1
        while idx >= 0 and result[idx][1] in ("subscript", "superscript", "prime"):
            idx -= 1
        return result[idx] if idx >= 0 else None

    close_type: dict[int, str] = {}
    result: list[_MathToken] = []

    for i, tok in enumerate(tokens):
        txt, ttype, extra = tok

        # |
        if txt == "|" and ttype == "bracket":
            if i in pipe_pairs:
                result.append(("|", "abs_open", ""))
            elif i in pipe_closes:
                result.append(("|", "abs_close", ""))
            else:
                result.append(tok)
            continue

        # (
        if txt == "(" and ttype == "bracket":
            if i in frac_opens:
                continue
            if i in paren_pairs and result:
                ci = paren_pairs[i]
                prev = _effective_prev(result)
                if prev is not None:
                    pt = prev[1]
                    ptx = prev[0]
                    if pt == "function" or (pt == "greek"):
                        close_type[ci] = "de"
                        result.append(("(", "paren_de", ""))
                        continue
                    if pt == "variable":
                        if ptx in _FUNCTION_LIKE_VARS:
                            close_type[ci] = "de"
                            result.append(("(", "paren_de", ""))
                        else:
                            close_type[ci] = "facteur"
                            result.append(("(", "paren_facteur", ""))
                        continue
                    if pt == "number":
                        close_type[ci] = "facteur"
                        result.append(("(", "paren_facteur", ""))
                        continue
                    if pt in ("paren_de_close", "paren_facteur_close"):
                        close_type[ci] = "facteur"
                        result.append(("(", "paren_facteur", ""))
                        continue
            result.append(tok)
            continue

        # )
        if txt == ")" and ttype == "bracket":
            if i in frac_closes:
                continue
            if i in close_type:
                ct = close_type[i]
                result.append((")", f"paren_{ct}_close", ""))
                continue
            result.append(tok)
            continue

        result.append(tok)

    return result


def lire_maths(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit une formule mathématique en combinant nombres, lettres,
    symboles et fonctions.

    Composants : 1 composant par token de formule (nombre, opérateur,
    variable, fonction, parenthèse, etc.).
    """
    if options is None:
        options = OptionsLecture()

    # Phase 1 : Tokenisation
    tokens = _tokenize_maths(text)

    # Phase 2 : Transformation des parenthèses
    tokens = _smart_parens(tokens)

    # Phase 3 : Génération des événements
    events: list[EventFormuleLecture] = []
    src = span[0]
    comp_idx = 0
    n_tok = len(tokens)

    # Position courante dans le texte source (pour les spans)
    char_pos = 0
    tok_positions: list[int] = []
    tmp_i = 0
    for tok in tokens:
        # Avancer dans le texte pour trouver la position de ce token
        # (les tokens supprimés par smart_parens ne sont pas dans la liste)
        tok_positions.append(tmp_i)
        tmp_i += len(tok[0])
    # Recalculer les positions en parcourant le texte original
    tok_positions = _compute_token_positions(text, tokens)

    func_paren_stack: list[int] = []

    ti = 0
    while ti < n_tok:
        tok_text, tok_type, tok_extra = tokens[ti]
        tpos = tok_positions[ti]
        pos = src + tpos
        tlen = len(tok_text)
        tok_span = (tpos, tpos + tlen)
        evt_start = len(events)

        if tok_type == "number":
            # √ ne s'applique qu'au nombre entier/décimal suivant :
            # √3/2 → "racine carrée de trois sur deux" (pas "trois demis")
            prev_is_sqrt = (ti > 0 and tokens[ti - 1][0] == "√"
                            and tokens[ti - 1][1] == "operator")
            # Vérifier si c'est une fraction : [sign] number / number
            if not prev_is_sqrt and ti + 2 < n_tok and tokens[ti + 1][0] == "/" and tokens[ti + 1][1] == "operator" and tokens[ti + 2][1] == "number":
                num_str = tok_text
                den_str = tokens[ti + 2][0]
                # Vérifier si un signe précède le numérateur
                sign_prefix = ""
                has_sign = False
                if (ti > 0 and tokens[ti - 1][1] == "operator"
                        and tokens[ti - 1][0] in ("-", "+", "−")
                        and len(events) > 0
                        and events[-1].ortho in ("moins", "plus")):
                    sign_prefix = "-" if tokens[ti - 1][0] in ("-", "−") else "+"
                    has_sign = True
                try:
                    num_v = int(num_str)
                    den_v = int(den_str)
                    if den_v in (2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000):
                        frac_text = f"{num_str}/{den_str}"
                        frac_start = pos
                        if has_sign:
                            # Retirer l'event de signe déjà émis
                            events.pop()
                            comp_idx -= 1
                            frac_text = f"{sign_prefix}{num_str}/{den_str}"
                            frac_start = src + tok_positions[ti - 1]
                        end_pos = src + tok_positions[ti + 2] + len(den_str)
                        frac_result = lire_fraction(
                            frac_text, span=(frac_start, end_pos), options=options,
                        )
                        # Copier les events en décalant span_num
                        frac_offset = tok_positions[ti - 1] if has_sign else tpos
                        for evt in frac_result.events:
                            new_evt = EventFormuleLecture(
                                ortho=evt.ortho, phone=evt.phone,
                                span_source=evt.span_source,
                                composant=comp_idx,
                            )
                            if evt.span_num and evt.span_num != (0, 0):
                                s, e = evt.span_num
                                new_evt.span_num = (frac_offset + s, frac_offset + e)
                            events.append(new_evt)
                        comp_idx += 1
                        ti += 3
                        continue
                except ValueError:
                    pass

            # Nombre suivi d'unité(s) → décomposer le nombre normalement
            if ti + 1 < n_tok and tokens[ti + 1][1] == "unit":
                grp_result = lire_nombre(tok_text, span=(pos, pos + tlen), options=options)
                for evt in grp_result.events:
                    events.append(EventFormuleLecture(
                        ortho=evt.ortho, phone=evt.phone,
                        span_source=evt.span_source,
                        composant=comp_idx,
                    ))
                # Calculer span_num décomposé puis décaler vers la position dans display_num
                _compute_span_num(events[evt_start:], tok_text)
                for evt in events[evt_start:]:
                    if evt.span_num and evt.span_num != (0, 0):
                        s, e = evt.span_num
                        evt.span_num = (tpos + s, tpos + e)
                comp_idx += 1
                ti += 1
                continue

            # Nombre normal — surligné d'un seul bloc dans les formules
            grp_result = lire_nombre(tok_text, span=(pos, pos + tlen), options=options)
            num_span = (tpos, tpos + tlen)
            for evt in grp_result.events:
                e = EventFormuleLecture(
                    ortho=evt.ortho, phone=evt.phone,
                    span_source=evt.span_source,
                    composant=comp_idx,
                )
                e.span_num = num_span
                events.append(e)
            comp_idx += 1
            ti += 1
            continue

        elif tok_type == "superscript":
            converted = tok_extra
            if converted == "2":
                t_sym, p_sym = _SYMBOLES["²"]
                events.append(EventFormuleLecture(
                    ortho=t_sym, phone=p_sym,
                    span_source=(pos, pos + tlen),
                    composant=comp_idx,
                ))
            elif converted == "3":
                t_sym, p_sym = _SYMBOLES["³"]
                events.append(EventFormuleLecture(
                    ortho=t_sym, phone=p_sym,
                    span_source=(pos, pos + tlen),
                    composant=comp_idx,
                ))
            else:
                # Général : "puissance" + nombre (gérer signe négatif)
                events.append(EventFormuleLecture(
                    ortho="puissance", phone="pɥisɑ̃s",
                    span_source=(pos, pos + tlen),
                    composant=comp_idx,
                ))
                sign_part = ""
                num_part = converted
                if num_part.startswith("-"):
                    events.append(EventFormuleLecture(
                        ortho="moins", phone="mwɛ̃",
                        span_source=(pos, pos + tlen),
                        composant=comp_idx,
                    ))
                    num_part = num_part[1:]
                elif num_part.startswith("+"):
                    num_part = num_part[1:]
                if num_part == "n":
                    events.append(EventFormuleLecture(
                        ortho="enne", phone="ɛn",
                        span_source=(pos, pos + tlen),
                        composant=comp_idx,
                    ))
                elif num_part.isdigit():
                    parts = _nombre_vers_francais(int(num_part))
                    for t, p in parts:
                        events.append(EventFormuleLecture(
                            ortho=t, phone=p,
                            span_source=(pos, pos + tlen),
                            composant=comp_idx,
                        ))
            comp_idx += 1

        elif tok_type == "subscript":
            content = tok_extra
            events.append(EventFormuleLecture(
                ortho="indice", phone="ɛ̃dis",
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            # Contenu : chiffres comme nombres, lettres comme variables
            ci = 0
            while ci < len(content):
                c = content[ci]
                if c.isdigit():
                    j2 = ci
                    while j2 < len(content) and content[j2].isdigit():
                        j2 += 1
                    parts = _nombre_vers_francais(int(content[ci:j2]))
                    for t, p in parts:
                        events.append(EventFormuleLecture(
                            ortho=t, phone=p,
                            span_source=(pos, pos + tlen),
                            composant=comp_idx,
                        ))
                    ci = j2
                elif c.isalpha():
                    t_l, p_l = _LETTRES.get(c, (c, c))
                    events.append(EventFormuleLecture(
                        ortho=t_l, phone=p_l,
                        span_source=(pos, pos + tlen),
                        composant=comp_idx,
                    ))
                    ci += 1
                else:
                    ci += 1
            comp_idx += 1

        elif tok_type == "variable":
            t_l, p_l = _LETTRES.get(tok_text, (tok_text, tok_text))
            events.append(EventFormuleLecture(
                ortho=t_l, phone=p_l,
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            comp_idx += 1

        elif tok_type == "function":
            wl = tok_text.lower()
            if wl in _SYMBOLES:
                t_fn, p_fn = _SYMBOLES[wl]
            else:
                t_fn, p_fn = tok_text, tok_text
            events.append(EventFormuleLecture(
                ortho=t_fn, phone=p_fn,
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            comp_idx += 1

        elif tok_type == "greek":
            t_gr, p_gr = _GREC.get(tok_text, (tok_text, tok_text))
            events.append(EventFormuleLecture(
                ortho=t_gr, phone=p_gr,
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            comp_idx += 1

        elif tok_type == "unit":
            if tok_text in _SYMBOLES:
                t_u, p_u = _SYMBOLES[tok_text]
            else:
                t_u, p_u = tok_text, tok_text
            events.append(EventFormuleLecture(
                ortho=t_u, phone=p_u,
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            comp_idx += 1

        elif tok_type == "operator":
            ch = tok_text
            if ch == "√":
                events.append(EventFormuleLecture(
                    ortho="racine carrée", phone="ʁasin kaʁe",
                    span_source=(pos, pos + tlen),
                    composant=comp_idx,
                ))
                events.append(EventFormuleLecture(
                    ortho="de", phone="də",
                    span_source=(pos, pos + tlen),
                    composant=comp_idx,
                ))
            elif ch == "/":
                # "/" entre unités → "par", sinon "sur"
                prev_unit = ti > 0 and tokens[ti - 1][1] == "unit"
                next_unit = ti + 1 < n_tok and tokens[ti + 1][1] == "unit"
                if prev_unit and next_unit:
                    events.append(EventFormuleLecture(
                        ortho="par", phone="paʁ",
                        span_source=(pos, pos + tlen),
                        composant=comp_idx,
                    ))
                elif ch in _SYMBOLES:
                    t_sym, p_sym = _SYMBOLES[ch]
                    events.append(EventFormuleLecture(
                        ortho=t_sym, phone=p_sym,
                        span_source=(pos, pos + tlen),
                        composant=comp_idx,
                    ))
            elif ch in _SYMBOLES:
                t_sym, p_sym = _SYMBOLES[ch]
                events.append(EventFormuleLecture(
                    ortho=t_sym, phone=p_sym,
                    span_source=(pos, pos + tlen),
                    composant=comp_idx,
                ))
            comp_idx += 1

        elif tok_type == "prime":
            events.append(EventFormuleLecture(
                ortho="prime", phone="pʁim",
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            comp_idx += 1

        elif tok_type == "factorial":
            events.append(EventFormuleLecture(
                ortho="factorielle", phone="faktɔʁjɛl",
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            comp_idx += 1

        elif tok_type == "second":
            events.append(EventFormuleLecture(
                ortho="seconde", phone="səɡɔ̃d",
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            comp_idx += 1

        # Smart parens
        elif tok_type == "paren_de":
            events.append(EventFormuleLecture(
                ortho="de", phone="də",
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            func_paren_stack.append(len(events) - 1)
            comp_idx += 1

        elif tok_type == "paren_facteur":
            events.append(EventFormuleLecture(
                ortho="facteur de", phone="faktœʁ də",
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            func_paren_stack.append(len(events) - 1)
            comp_idx += 1

        elif tok_type in ("paren_de_close", "paren_facteur_close"):
            if func_paren_stack:
                de_idx = func_paren_stack.pop()
                events[de_idx].span_num_extra = (tpos, tpos + tlen)

        elif tok_type == "abs_open":
            events.append(EventFormuleLecture(
                ortho="valeur absolue de", phone="valœʁ apsoly də",
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            func_paren_stack.append(len(events) - 1)
            comp_idx += 1

        elif tok_type == "abs_close":
            if func_paren_stack:
                de_idx = func_paren_stack.pop()
                events[de_idx].span_num_extra = (tpos, tpos + tlen)

        elif tok_type == "bracket":
            ch = tok_text
            if ch in _SYMBOLES:
                t_sym, p_sym = _SYMBOLES[ch]
                events.append(EventFormuleLecture(
                    ortho=t_sym, phone=p_sym,
                    span_source=(pos, pos + tlen),
                    composant=comp_idx,
                ))
                comp_idx += 1

        elif tok_type == "separator":
            if tok_text in _SYMBOLES:
                t_sep, p_sep = _SYMBOLES[tok_text]
            else:
                t_sep, p_sep = tok_text, tok_text
            events.append(EventFormuleLecture(
                ortho=t_sep, phone=p_sep,
                span_source=(pos, pos + tlen),
                composant=comp_idx,
            ))
            # Pas d'incrémentation comp_idx — le séparateur n'est pas un composant séparé

        # Catch-all: assign span_num to all new events that don't have one
        for ei in range(evt_start, len(events)):
            if events[ei].span_num == (0, 0):
                events[ei].span_num = tok_span
        ti += 1

    return _make_result(events, display_num=text)


def _compute_token_positions(text: str, tokens: list[_MathToken]) -> list[int]:
    """Calcule la position de chaque token dans le texte original.

    Parcourt le texte original et associe chaque token à sa position.
    Gère les tokens supprimés par _smart_parens (parenthèses de fraction).
    """
    positions: list[int] = []
    ti = 0
    scan = 0
    n_text = len(text)

    for tok in tokens:
        tok_text = tok[0]
        # Chercher tok_text dans le texte à partir de scan
        while scan < n_text:
            if text[scan] in (" ", "\t", "\u202f"):
                scan += 1
                continue
            if text[scan:scan + len(tok_text)] == tok_text:
                positions.append(scan)
                scan += len(tok_text)
                break
            # Token supprimé (paren de fraction) → avancer d'un char
            scan += 1
        else:
            # Pas trouvé → utiliser la dernière position connue
            positions.append(positions[-1] if positions else 0)

    return positions


# -- NUMERO --------------------------------------------------------------------

_NUMERO_SPLIT_RE = re.compile(r"[\s.]+")


def lire_numero(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit un numéro mixte alphanumérique.

    Composants : 1 composant par groupe (lettres/chiffres).
    Ex: "AB 123 CD" → 3 composants.
    """
    events: list[EventFormuleLecture] = []
    src = span[0]
    comp_idx = 0
    i = 0

    # Préfixe "N°" ou "n°"
    if len(text) >= 2 and text[0] in ("N", "n") and text[1] == "°":
        events.append(EventFormuleLecture(
            ortho="numéro", phone="nymeʁo",
            span_source=(src, src + 2),
            composant=comp_idx,
        ))
        events[-1].span_num = (0, 2)
        comp_idx += 1
        i = 2

    # On parcourt le texte en segments alpha/digit/séparateur
    while i < len(text):
        ch = text[i]
        pos = src + i

        if ch in (" ", ".", "\t"):
            i += 1
            continue

        if ch == "-":
            events.append(EventFormuleLecture(
                ortho="tiret", phone="tiʁɛ",
                span_source=(pos, pos + 1),
                composant=comp_idx,
            ))
            events[-1].span_num = (i, i + 1)
            comp_idx += 1
            i += 1
            continue

        if ch.isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            group = text[i:j]
            # Zéros initiaux → lire individuellement, puis reste comme nombre
            # Tout le bloc (zéros + reste) surligné ensemble
            if len(group) > 1 and group[0] == "0":
                lz = _count_leading_zeros(group)
                rest_g = group[lz:]
                for k in range(lz):
                    t_d, p_d = _u("0")
                    events.append(EventFormuleLecture(
                        ortho=t_d, phone=p_d,
                        span_source=(src + i + k, src + i + k + 1),
                        composant=comp_idx,
                    ))
                    events[-1].span_num = (i, j)
                if rest_g:
                    n_g = int(rest_g)
                    parts_g = _nombre_vers_francais(n_g)
                    evts_g = _events_from_parts(
                        parts_g, (src + i + lz, src + j), rest_g,
                        composant=comp_idx,
                    )
                    for evt in evts_g:
                        evt.span_num = (i, j)
                    events.extend(evts_g)
            else:
                n = int(group)
                parts = _nombre_vers_francais(n)
                evts = _events_from_parts(parts, (pos, src + j), group,
                                          composant=comp_idx)
                for evt in evts:
                    evt.span_num = (i, j)
                events.extend(evts)
            comp_idx += 1
            i = j
        elif ch.isalpha():
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
            # Épeler chaque lettre (même composant pour le groupe)
            for k in range(i, j):
                c = text[k]
                t_l, p_l = _LETTRES.get(c.upper(), (c, c))
                events.append(EventFormuleLecture(
                    ortho=t_l, phone=p_l,
                    span_source=(src + k, src + k + 1),
                    composant=comp_idx,
                ))
                events[-1].span_num = (k, k + 1)
            comp_idx += 1
            i = j
        else:
            i += 1

    return _make_result(events, display_num=text)


# -- FALLBACK ------------------------------------------------------------------

def _epeler_texte(text: str, span: Span) -> LectureFormuleResult:
    """Fallback : épelle caractère par caractère."""
    events: list[EventFormuleLecture] = []
    src = span[0]
    for i, ch in enumerate(text):
        pos = src + i
        if ch in _LETTRES:
            t, p = _LETTRES[ch]
        elif ch in _GREC:
            t, p = _GREC[ch]
        elif ch in _SYMBOLES:
            t, p = _SYMBOLES[ch]
        elif ch.isdigit():
            t, p = _u(ch)
        elif ch == " ":
            continue
        else:
            t, p = ch, ch
        events.append(EventFormuleLecture(
            ortho=t, phone=p,
            span_source=(pos, pos + 1),
        ))
    return _make_result(events, display_num=text)


# -- HEURE ---------------------------------------------------------------------

_HEURE_RE_H = re.compile(
    r"^(\d{1,2})[hH](\d{1,2})?(?:min(\d{1,2}))?(?:s)?$"
)
_HEURE_RE_HMS = re.compile(r"^(\d{1,2})[hH](\d{1,2})min(\d{1,2})s?$")
_HEURE_RE_HMIN = re.compile(r"^(\d{1,2})[hH](\d{1,2})min$")
_HEURE_RE_COLON = re.compile(r"^(\d{1,2}):(\d{2})$")
_HEURE_RE_MIN = re.compile(r"^(\d{1,2})min$")
_HEURE_RE_SEC = re.compile(r"^(\d{1,2})s$")
# Durée : 3'35" ou 3'35
_HEURE_RE_MINSEC = re.compile(r"""^(\d{1,2})['\u2032](\d{1,2})["\u2033]?$""")

_HEURE_WORDS = {} if _MODE_API else _load_heure_words()


def _parse_heure(text: str) -> dict | None:
    """Parse un format heure/durée. Retourne dict ou None."""
    s = text.strip()

    m = _HEURE_RE_HMS.match(s)
    if m:
        h, mi, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if h <= 23 and mi <= 59 and sec <= 59:
            return {"hours": h, "minutes": mi, "seconds": sec, "format": "hms"}

    m = _HEURE_RE_HMIN.match(s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if h <= 23 and mi <= 59:
            return {"hours": h, "minutes": mi, "seconds": None, "format": "hmin"}

    m = _HEURE_RE_H.match(s)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2)) if m.group(2) else None
        sec = None
        if h <= 23 and (mi is None or mi <= 59):
            return {"hours": h, "minutes": mi, "seconds": sec, "format": "h"}

    m = _HEURE_RE_COLON.match(s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if h <= 23 and mi <= 59:
            return {"hours": h, "minutes": mi, "seconds": None, "format": "colon"}

    m = _HEURE_RE_MINSEC.match(s)
    if m:
        mi, sec = int(m.group(1)), int(m.group(2))
        if mi <= 59 and sec <= 59:
            return {"hours": None, "minutes": mi, "seconds": sec, "format": "minsec"}

    m = _HEURE_RE_MIN.match(s)
    if m:
        mi = int(m.group(1))
        if mi <= 59:
            return {"hours": None, "minutes": mi, "seconds": None, "format": "min_only"}

    m = _HEURE_RE_SEC.match(s)
    if m:
        sec = int(m.group(1))
        if sec <= 59:
            return {"hours": None, "minutes": None, "seconds": sec, "format": "s_only"}

    return None


def lire_heure(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit une heure/durée.

    Composants : heures (0), minutes (1), secondes (2).
    """
    if options is None:
        options = OptionsLecture()

    data = _parse_heure(text)
    if data is None:
        return _epeler_texte(text, span)

    hours = data.get("hours")
    minutes = data.get("minutes")
    seconds = data.get("seconds")
    fmt = data["format"]
    # Pour h/colon : pas de "minutes" par défaut, sauf si option activée
    if fmt in ("h", "colon"):
        add_min_word = options.heure_mot_minutes
    else:
        add_min_word = True

    events: list[EventFormuleLecture] = []
    src = span[0]
    comp = 0

    # Heures — avec option minuit/midi
    use_minuit_midi = (options.heure_minuit_midi
                       and hours is not None
                       and hours in (0, 12)
                       and fmt in ("h", "colon"))
    if use_minuit_midi:
        word_mm = "minuit" if hours == 0 else "midi"
        phone_mm = "minɥi" if hours == 0 else "midi"
        events.append(EventFormuleLecture(
            ortho=word_mm, phone=phone_mm,
            span_source=span, composant=comp,
            sound_id=f"dir_{word_mm}",
        ))
        comp += 1
    elif hours is not None:
        parts = _nombre_vers_francais(hours, feminin=True)
        evts = _events_from_parts(parts, span, text, composant=comp)
        events.extend(evts)
        word = "heure" if hours == 1 else "heures"
        t_w, p_w = _HEURE_WORDS[word]
        events.append(EventFormuleLecture(
            ortho=t_w, phone=p_w, span_source=span, composant=comp,
        ))
        comp += 1

    # Minutes
    if minutes is not None:
        parts = _nombre_vers_francais(minutes, feminin=True)
        evts = _events_from_parts(parts, span, text, composant=comp)
        events.extend(evts)
        if add_min_word or fmt in ("hms", "hmin", "min_only"):
            word = "minute" if minutes == 1 else "minutes"
            t_w, p_w = _HEURE_WORDS[word]
            events.append(EventFormuleLecture(
                ortho=t_w, phone=p_w, span_source=span, composant=comp,
            ))
        comp += 1

    # Secondes
    if seconds is not None:
        parts = _nombre_vers_francais(seconds, feminin=True)
        evts = _events_from_parts(parts, span, text, composant=comp)
        events.extend(evts)
        word = "seconde" if seconds == 1 else "secondes"
        t_w, p_w = _HEURE_WORDS[word]
        events.append(EventFormuleLecture(
            ortho=t_w, phone=p_w, span_source=span, composant=comp,
        ))

    display_num = text.strip()

    # -- span_num par composant (positions dans display_num) --
    _H_LABELS = {"heure", "heures", "minute", "minutes", "seconde", "secondes"}
    _MM_LABELS = {"minuit", "midi"}
    digit_groups = list(re.finditer(r'\d+', display_num))
    for e in events:
        comp = e.composant
        if comp < len(digit_groups):
            g = digit_groups[comp]
            if e.ortho in _MM_LABELS:
                # Minuit/midi → surligne chiffres + séparateur (ex: "0h", "12h")
                sep_end = g.end()
                if comp + 1 < len(digit_groups):
                    sep_end = digit_groups[comp + 1].start()
                else:
                    # Pas de groupe suivant : inclure jusqu'à la fin ou le prochain chiffre
                    rest = display_num[g.end():]
                    extra = 0
                    for ch in rest:
                        if ch.isdigit():
                            break
                        extra += 1
                    sep_end = g.end() + extra
                e.span_num = (g.start(), sep_end)
            elif e.ortho in _H_LABELS:
                # Mot temporel → surligne le séparateur (h, min, s, :)
                sep_start = g.end()
                if comp + 1 < len(digit_groups):
                    sep_end = digit_groups[comp + 1].start()
                else:
                    sep_end = len(display_num)
                if sep_start < sep_end:
                    e.span_num = (sep_start, sep_end)
                else:
                    e.span_num = (g.start(), g.end())
            else:
                # Mot numérique → surligne les chiffres
                e.span_num = (g.start(), g.end())

    return _make_result(events, display_num=display_num, valeur=display_num)


# -- MONNAIE -------------------------------------------------------------------

_DEVISES = {} if _MODE_API else _load_devises()

_SYM_TO_ISO: dict[str, str] = {v["symbole"]: k for k, v in _DEVISES.items()
                                 if v.get("symbole")}

_MONNAIE_RE_POST = re.compile(
    r"^([0-9][0-9 ']*[0-9]*[.,]?\d*)\s*([€$£¥])$"
)
# 42€50 : montant€centimes
_MONNAIE_RE_INFIX = re.compile(
    r"^([0-9][0-9 ']*[0-9]*)([€$£¥])(\d{1,2})$"
)
_MONNAIE_RE_PRE = re.compile(
    r"^([€$£¥])\s*([0-9][0-9 ']*[0-9]*[.,]?\d*)$"
)
_MONNAIE_RE_ISO_POST = re.compile(
    r"^([0-9][0-9 ']*[0-9]*[.,]?\d*)\s*(EUR|USD|GBP|CHF|JPY)$", re.IGNORECASE
)
_MONNAIE_RE_ISO_PRE = re.compile(
    r"^(EUR|USD|GBP|CHF|JPY)\s*([0-9][0-9 ']*[0-9]*[.,]?\d*)$", re.IGNORECASE
)
# Note: signs (-/+/±) are stripped in _parse_monnaie before matching these regex


def _parse_monnaie(text: str) -> dict | None:
    """Parse un montant avec devise. Retourne dict ou None."""
    s = text.strip()
    negative = False
    if s.startswith("-") or s.startswith("−"):
        negative = True
        s = s.lstrip("-−").strip()
    amount_str = None
    currency = None

    m = _MONNAIE_RE_INFIX.match(s)
    if m:
        major_str = re.sub(r"['\s]", "", m.group(1))
        currency = _SYM_TO_ISO.get(m.group(2))
        minor_str = m.group(3)
        if currency and currency in _DEVISES:
            major = int(major_str) if major_str else 0
            minor = int(minor_str)
            if len(minor_str) == 1:
                minor *= 10
            return {"currency": currency, "major": major, "minor": minor, "negative": negative}
    m = _MONNAIE_RE_POST.match(s)
    if m:
        amount_str, currency = m.group(1), _SYM_TO_ISO.get(m.group(2))
    if not currency:
        m = _MONNAIE_RE_PRE.match(s)
        if m:
            amount_str, currency = m.group(2), _SYM_TO_ISO.get(m.group(1))
    if not currency:
        m = _MONNAIE_RE_ISO_POST.match(s)
        if m:
            amount_str, currency = m.group(1), m.group(2).upper()
    if not currency:
        m = _MONNAIE_RE_ISO_PRE.match(s)
        if m:
            currency, amount_str = m.group(1).upper(), m.group(2)

    if not currency or not amount_str or currency not in _DEVISES:
        return None

    cleaned = re.sub(r"['\s]", "", amount_str).replace(",", ".")
    extra_decimals = ""
    if "." in cleaned:
        int_part, dec_part = cleaned.split(".", 1)
        major = int(int_part) if int_part else 0
        if len(dec_part) <= 2:
            # 1-2 décimales → centimes normaux
            if len(dec_part) == 1:
                minor = int(dec_part) * 10
            else:
                minor = int(dec_part)
        else:
            # >2 décimales → centimes + décimales supplémentaires
            minor = int(dec_part[:2])
            extra_decimals = dec_part[2:]
    else:
        major = int(cleaned)
        minor = 0

    return {"currency": currency, "major": major, "minor": minor,
            "extra_decimals": extra_decimals, "negative": negative}


def lire_monnaie(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit un montant avec monnaie.

    Composants : montant majeur (0), "et" (1), montant mineur (2).
    """
    if options is None:
        options = OptionsLecture()

    data = _parse_monnaie(text)
    if data is None:
        return _epeler_texte(text, span)

    currency = data["currency"]
    major = data["major"]
    minor = data["minor"]
    extra_decimals = data.get("extra_decimals", "")
    negative = data.get("negative", False)
    cur = _DEVISES[currency]

    events: list[EventFormuleLecture] = []

    # Signe négatif
    if negative:
        t_m, p_m = _SYMBOLES["-"]
        events.append(EventFormuleLecture(
            ortho=t_m, phone=p_m, span_source=span, composant=0,
        ))

    # Partie majeure (composant 0) — toujours inclure, même si major=0
    if True:
        parts = _nombre_vers_francais(major)
        evts = _events_from_parts(parts, span, text, composant=0)
        events.extend(evts)
        # Mot devise
        idx = 0 if major == 1 else 1
        t_dev = cur["majeur"][idx]
        p_dev = cur["phone_maj"][idx]
        events.append(EventFormuleLecture(
            ortho=t_dev, phone=p_dev, span_source=span, composant=0,
        ))
        # Suffixe (CHF → "suisse")
        if "suffixe" in cur:
            t_suf, p_suf = cur["suffixe"]
            events.append(EventFormuleLecture(
                ortho=t_suf, phone=p_suf, span_source=span, composant=0,
            ))

    # Connecteur "et" (composant 1) + Partie mineure (composant 2)
    if extra_decimals:
        # >2 décimales : lire "virgule" + toutes les décimales chiffre par chiffre
        t_v, p_v = _VIRGULE
        events.append(EventFormuleLecture(
            ortho=t_v, phone=p_v, span_source=span, composant=1,
        ))
        all_dec = f"{minor:02d}{extra_decimals}"
        for ch in all_dec:
            t_ch, p_ch = _u(ch)
            events.append(EventFormuleLecture(
                ortho=t_ch, phone=p_ch, span_source=span, composant=2,
            ))
    elif minor > 0 and cur.get("mineur") and options.monnaie_dire_centimes:
        events.append(EventFormuleLecture(
            ortho="et", phone="e", span_source=span, composant=1,
        ))
        parts = _nombre_vers_francais(minor)
        evts = _events_from_parts(parts, span, text, composant=2)
        events.extend(evts)
        idx = 0 if minor == 1 else 1
        t_min = cur["mineur"][idx]
        p_min = cur["phone_min"][idx]
        events.append(EventFormuleLecture(
            ortho=t_min, phone=p_min, span_source=span, composant=2,
        ))

    # display_num + span_num
    sign_prefix = "-" if negative else ""
    sign_len = len(sign_prefix)
    major_str = str(major)
    major_start = sign_len
    major_end = sign_len + len(major_str)
    sym = cur.get("symbole")
    if sym:
        if minor > 0:
            display_num = f"{sign_prefix}{major},{minor:02d}{sym}"
            minor_start = major_end + 1  # après la virgule
            minor_end = minor_start + 2
            sym_start = minor_end
            sym_end = sym_start + len(sym)
        else:
            display_num = f"{sign_prefix}{major}{sym}"
            minor_start = minor_end = 0
            sym_start = major_end
            sym_end = sym_start + len(sym)
    else:
        if minor > 0:
            display_num = f"{sign_prefix}{major},{minor:02d} {currency}"
            minor_start = major_end + 1
            minor_end = minor_start + 2
            sym_start = minor_end + 1  # espace
            sym_end = sym_start + len(currency)
        else:
            display_num = f"{sign_prefix}{major} {currency}"
            minor_start = minor_end = 0
            sym_start = major_end + 1
            sym_end = sym_start + len(currency)

    # Assigner span_num à chaque événement
    cur_maj_label = cur["majeur"][0 if major == 1 else 1]
    cur_min_label = cur["mineur"][0 if minor == 1 else 1] if cur.get("mineur") else ""
    cur_suf_label = cur["suffixe"][0] if "suffixe" in cur else ""
    for evt in events:
        if evt.ortho == "moins":
            evt.span_num = (0, sign_len)
        elif evt.composant == 0:
            if evt.ortho in (cur_maj_label, cur_suf_label):
                evt.span_num = (sym_start, sym_end)
            else:
                evt.span_num = (major_start, major_end)
        elif evt.composant == 1:
            # "et" → virgule
            evt.span_num = (major_end, major_end + 1)
        elif evt.composant == 2:
            if evt.ortho == cur_min_label:
                evt.span_num = (sym_start, sym_end)
            else:
                evt.span_num = (minor_start, minor_end)

    valeur = major + minor / 100 if minor else major
    return _make_result(events, display_num=display_num, valeur=valeur)


# -- POURCENTAGE ---------------------------------------------------------------

_POURCENT_RE = re.compile(r"^([-+±]?[0-9][0-9 ']*\.?[0-9]*)([%‰])$")

_POURCENT_WORDS = {} if _MODE_API else _load_pourcent_words()


def lire_pourcentage(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit un pourcentage ou pour-mille.

    Composants : nombre (0), "pour cent/mille" (1).
    """
    if options is None:
        options = OptionsLecture()

    s = text.strip()
    m = _POURCENT_RE.match(s)
    if not m:
        return _epeler_texte(text, span)

    number_str = re.sub(r"['\s]", "", m.group(1))
    symbol = m.group(2)

    # Lire le nombre (possiblement décimal)
    num_result = lire_nombre(number_str, span=span, options=options)
    # Re-assigner composant=0
    for evt in num_result.events:
        evt.composant = 0

    events = list(num_result.events)

    # Ajouter "pour cent" / "pour mille"
    t_pct, p_pct = _POURCENT_WORDS[symbol]
    sym_pos = s.index(symbol)
    sym_span = (span[0] + sym_pos, span[0] + sym_pos + len(symbol))
    pct_evt = EventFormuleLecture(
        ortho=t_pct, phone=p_pct, span_source=sym_span, composant=1,
    )
    # span_num direct : le symbole est à la fin du display_num
    pct_evt.span_num = (len(number_str), len(number_str) + len(symbol))
    events.append(pct_evt)

    display_num = number_str + symbol
    try:
        valeur = float(number_str)
    except ValueError:
        valeur = number_str
    return _make_result(events, display_num=display_num, valeur=valeur)


# -- INTERVALLE ----------------------------------------------------------------

_INTERVALLE_RE = re.compile(r"^([\[\]])([^;,]+)[;,]([^;,]+)([\[\]])$")

_INTERVALLE_BOUNDS = set() if _MODE_API else _load_intervalle_bounds()
_INTERVALLE_NUM_RE = re.compile(r"^-?\d+\.?\d*$")


def _is_valid_bound(val: str) -> bool:
    """Vérifie qu'une borne est valide."""
    if val in _INTERVALLE_BOUNDS:
        return True
    return bool(_INTERVALLE_NUM_RE.match(val.replace("'", "").replace(" ", "")))


def _read_bound(val: str, span: Span, composant: int,
                options: OptionsLecture) -> list[EventFormuleLecture]:
    """Lit une borne d'intervalle."""
    events: list[EventFormuleLecture] = []
    if val in ("+∞", "+inf"):
        events.append(EventFormuleLecture(
            ortho="plus", phone="plys", span_source=span, composant=composant))
        events.append(EventFormuleLecture(
            ortho="infini", phone="ɛ̃fini", span_source=span, composant=composant))
    elif val in ("-∞", "-inf"):
        events.append(EventFormuleLecture(
            ortho="moins", phone="mwɛ̃", span_source=span, composant=composant))
        events.append(EventFormuleLecture(
            ortho="infini", phone="ɛ̃fini", span_source=span, composant=composant))
    elif val in ("∞", "inf"):
        events.append(EventFormuleLecture(
            ortho="infini", phone="ɛ̃fini", span_source=span, composant=composant))
    else:
        result = lire_nombre(val, span=span, options=options)
        for evt in result.events:
            evt.composant = composant
        events.extend(result.events)
    return events


def lire_intervalle(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit un intervalle mathématique.

    Composants : borne gauche (0), connecteur (1), borne droite (2).
    """
    if options is None:
        options = OptionsLecture()

    s = text.strip()
    m = _INTERVALLE_RE.match(s)
    if not m:
        return _epeler_texte(text, span)

    left_val = m.group(2).strip()
    right_val = m.group(3).strip()

    if not _is_valid_bound(left_val) or not _is_valid_bound(right_val):
        return _epeler_texte(text, span)

    events: list[EventFormuleLecture] = []

    left_bracket = m.group(1)
    right_bracket = m.group(4)
    display_num = f"{left_bracket}{left_val};{right_val}{right_bracket}"

    # Positions dans display_num
    lb_start = 0
    lb_end = len(left_bracket)
    lv_start = lb_end
    lv_end = lv_start + len(left_val)
    sep_start = lv_end
    sep_end = sep_start + 1  # ";"
    rv_start = sep_end
    rv_end = rv_start + len(right_val)
    rb_start = rv_end
    rb_end = rb_start + len(right_bracket)

    # "de" (connecteur ouvert, composant 1) → surligne les deux crochets
    evt_de = EventFormuleLecture(
        ortho="de", phone="də", span_source=span, composant=1,
    )
    evt_de.span_num = (lb_start, lb_end)
    evt_de.span_num_extra = (rb_start, rb_end)
    events.append(evt_de)

    # Borne gauche (composant 0)
    left_events = _read_bound(left_val, span, composant=0, options=options)
    for evt in left_events:
        evt.span_num = (lv_start, lv_end)
    events.extend(left_events)

    # "a" (connecteur, composant 1) → surligne le ";"
    evt_a = EventFormuleLecture(
        ortho="a", phone="a", span_source=span, composant=1,
    )
    evt_a.span_num = (sep_start, sep_end)
    events.append(evt_a)

    # Borne droite (composant 2)
    right_events = _read_bound(right_val, span, composant=2, options=options)
    for evt in right_events:
        evt.span_num = (rv_start, rv_end)
    events.extend(right_events)

    # Crochet fermant (composant 1) — implicite via display
    # Ajouter un event silencieux pour le crochet fermant si nécessaire
    # (le surlignage du crochet fermant se fait naturellement à la fin)

    return _make_result(events, display_num=display_num, valeur=display_num)


# -- GPS -----------------------------------------------------------------------

_GPS_DMS_RE = re.compile(
    r"(\d{1,3})°(\d{1,2})'(?:(\d{1,2})\"?)?\s*([NSEOW])", re.IGNORECASE
)
_GPS_DD_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)°\s*([NSEOW])", re.IGNORECASE
)

_GPS_DIRECTIONS = {} if _MODE_API else _load_gps_directions()
_GPS_UNITS = {} if _MODE_API else _load_gps_units()


def lire_gps(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit des coordonnées GPS (DMS ou DD).

    Composants : 1 par coordonnée (lat=0, lon=1).
    """
    if options is None:
        options = OptionsLecture()

    s = text.strip()
    events: list[EventFormuleLecture] = []
    display_parts: list[str] = []
    comp = 0

    # Essayer DMS
    dms_matches = list(_GPS_DMS_RE.finditer(s))
    if dms_matches:
        for m in dms_matches:
            deg = int(m.group(1))
            mi = int(m.group(2))
            sec = int(m.group(3)) if m.group(3) else None
            direction = m.group(4).upper()
            if direction == "W":
                direction = "O"

            # Construire display_num pour ce segment et calculer positions
            dp = f"{deg}°{mi}'"
            if sec is not None:
                dp += f'{sec}"'
            dp += direction
            dp_offset = sum(len(p) + 1 for p in display_parts)  # +1 pour espace

            # Positions dans dp
            deg_str = str(deg)
            pos_deg = (dp_offset, dp_offset + len(deg_str))
            pos_deg_sym = (pos_deg[1], pos_deg[1] + 1)  # °
            mi_str = str(mi)
            pos_mi = (pos_deg_sym[1], pos_deg_sym[1] + len(mi_str))
            pos_mi_sym = (pos_mi[1], pos_mi[1] + 1)  # '
            if sec is not None:
                sec_str = str(sec)
                pos_sec = (pos_mi_sym[1], pos_mi_sym[1] + len(sec_str))
                pos_sec_sym = (pos_sec[1], pos_sec[1] + 1)  # "
                pos_dir = (pos_sec_sym[1], pos_sec_sym[1] + len(direction))
            else:
                pos_dir = (pos_mi_sym[1], pos_mi_sym[1] + len(direction))

            # Degrés
            parts = _nombre_vers_francais(deg)
            evts = _events_from_parts(parts, span, text, composant=comp)
            for evt in evts:
                evt.span_num = pos_deg
            events.extend(evts)
            word = "degré" if deg == 1 else "degrés"
            t_w, p_w = _GPS_UNITS[word]
            evt_deg_sym = EventFormuleLecture(
                ortho=t_w, phone=p_w, span_source=span, composant=comp,
            )
            evt_deg_sym.span_num = pos_deg_sym
            events.append(evt_deg_sym)

            # Minutes
            parts = _nombre_vers_francais(mi, feminin=True)
            evts = _events_from_parts(parts, span, text, composant=comp)
            for evt in evts:
                evt.span_num = pos_mi
            events.extend(evts)
            word = "minute" if mi == 1 else "minutes"
            t_w, p_w = _GPS_UNITS[word]
            evt_mi_sym = EventFormuleLecture(
                ortho=t_w, phone=p_w, span_source=span, composant=comp,
            )
            evt_mi_sym.span_num = pos_mi_sym
            events.append(evt_mi_sym)

            # Secondes
            if sec is not None:
                parts = _nombre_vers_francais(sec, feminin=True)
                evts = _events_from_parts(parts, span, text, composant=comp)
                for evt in evts:
                    evt.span_num = pos_sec
                events.extend(evts)
                word = "seconde" if sec == 1 else "secondes"
                t_w, p_w = _GPS_UNITS[word]
                evt_sec_sym = EventFormuleLecture(
                    ortho=t_w, phone=p_w, span_source=span, composant=comp,
                )
                evt_sec_sym.span_num = pos_sec_sym
                events.append(evt_sec_sym)

            # Direction
            t_dir, p_dir, sid_dir = _GPS_DIRECTIONS[direction]
            evt_dir = EventFormuleLecture(
                ortho=t_dir, phone=p_dir, span_source=span, composant=comp,
                sound_id=sid_dir,
            )
            evt_dir.span_num = pos_dir
            events.append(evt_dir)

            display_parts.append(dp)
            comp += 1

        display_num = " ".join(display_parts)
        return _make_result(events, display_num=display_num, valeur=display_num)

    # Essayer DD
    dd_matches = list(_GPS_DD_RE.finditer(s))
    if dd_matches:
        for m in dd_matches:
            deg_str = m.group(1)
            direction = m.group(2).upper()
            if direction == "W":
                direction = "O"

            dp = f"{deg_str}°{direction}"
            dp_offset = sum(len(p) + 1 for p in display_parts)  # +1 pour espace

            # Positions dans dp
            pos_num = (dp_offset, dp_offset + len(deg_str))
            pos_deg_dir = (dp_offset + len(deg_str), dp_offset + len(deg_str) + 1 + len(direction))  # °N

            # Lire le nombre (possiblement décimal)
            num_result = lire_nombre(deg_str, span=span, options=options)
            for evt in num_result.events:
                evt.composant = comp
                evt.span_num = pos_num
            events.extend(num_result.events)

            # "degrés" → surligne °
            t_w, p_w = _GPS_UNITS["degrés"]
            evt_deg = EventFormuleLecture(
                ortho=t_w, phone=p_w, span_source=span, composant=comp,
            )
            evt_deg.span_num = pos_deg_dir
            events.append(evt_deg)

            # Direction → surligne aussi °N ensemble
            t_dir, p_dir, sid_dir = _GPS_DIRECTIONS[direction]
            evt_dir = EventFormuleLecture(
                ortho=t_dir, phone=p_dir, span_source=span, composant=comp,
                sound_id=sid_dir,
            )
            evt_dir.span_num = pos_deg_dir
            events.append(evt_dir)

            display_parts.append(dp)
            comp += 1

        display_num = " ".join(display_parts)
        return _make_result(events, display_num=display_num, valeur=display_num)

    return _epeler_texte(text, span)


# -- PAGE / CHAPITRE -----------------------------------------------------------

_PAGE_RE = re.compile(r"^(p|P|page)\.?\s*(\d+)$")
_CHAP_RE = re.compile(r"^(chap|ch|Ch)\.?\s*(\d+)$")


def lire_page_chapitre(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit une référence page ou chapitre.

    Composants : préfixe (0), nombre (1).
    """
    if options is None:
        options = OptionsLecture()

    s = text.strip()
    events: list[EventFormuleLecture] = []

    # Page
    m = _PAGE_RE.match(s)
    if m:
        prefix_raw = s[:s.index(m.group(2))]
        number_str = m.group(2)
        evt_prefix = EventFormuleLecture(
            ortho="page", phone="paʒ", span_source=span, composant=0,
        )
        evt_prefix.span_num = (0, len(prefix_raw))
        events.append(evt_prefix)
        n = int(number_str)
        parts = _nombre_vers_francais(n)
        evts = _events_from_parts(parts, span, text, composant=1)
        for evt in evts:
            evt.span_num = (len(prefix_raw), len(prefix_raw) + len(number_str))
        events.extend(evts)
        display_num = prefix_raw + number_str
        return _make_result(events, display_num=display_num, valeur=n)

    # Chapitre
    m = _CHAP_RE.match(s)
    if m:
        prefix_raw = s[:s.index(m.group(2))]
        number_str = m.group(2)
        evt_prefix = EventFormuleLecture(
            ortho="chapitre", phone="ʃapitʁ", span_source=span, composant=0,
        )
        evt_prefix.span_num = (0, len(prefix_raw))
        events.append(evt_prefix)
        n = int(number_str)
        parts = _nombre_vers_francais(n)
        evts = _events_from_parts(parts, span, text, composant=1)
        for evt in evts:
            evt.span_num = (len(prefix_raw), len(prefix_raw) + len(number_str))
        events.extend(evts)
        display_num = prefix_raw + number_str
        return _make_result(events, display_num=display_num, valeur=n)

    return _epeler_texte(text, span)


# ══════════════════════════════════════════════════════════════════════════════
# API publique
# ══════════════════════════════════════════════════════════════════════════════

# Dispatch table : formule_type → lecteur
_LECTEURS: dict[str, object] = {
    # 9 existants
    "nombre":       lire_nombre,
    "sigle":        lire_sigle,
    "date":         lire_date,
    "telephone":    lire_telephone,
    "ordinal":      lire_ordinal,
    "fraction":     lire_fraction,
    "scientifique": lire_scientifique,
    "maths":        lire_maths,
    "numero":       lire_numero,
    # 6 nouveaux
    "heure":           lire_heure,
    "monnaie":         lire_monnaie,
    "pourcentage":     lire_pourcentage,
    "intervalle":      lire_intervalle,
    "gps":             lire_gps,
    "page_chapitre":   lire_page_chapitre,
}


def lire_formule(
    formule_type: str,
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    options: OptionsLecture | None = None,
    feminin: bool = False,
) -> LectureFormuleResult:
    """Point d'entree unique pour la lecture algorithmique des formules.

    Parameters
    ----------
    formule_type : str
        Type de formule (nombre, sigle, date, telephone, ordinal,
        fraction, scientifique, maths, numero).
    text : str
        Texte source de la formule.
    span : tuple[int, int]
        Position (start, end) dans le texte original.
    children : list | None
        Sous-tokens du Tokeniseur (pour les formules composites).
    options : OptionsLecture | None
        Options de lecture (mode fraction, methode decimale).
    feminin : bool
        Si True, utilise les formes feminines (une au lieu de un).

    Returns
    -------
    LectureFormuleResult
        Texte lu, IPA et evenements alignes.
    """
    # Mode API : deleguer au serveur quand les donnees locales sont absentes
    if _MODE_API:
        from lectura_formules._api_client import lire_formule_api
        result = lire_formule_api(formule_type, text, span=list(span), feminin=feminin)
        return LectureFormuleResult(
            display_fr=result.get("display_fr", ""),
            phone=result.get("phone", ""),
            events=[
                EventFormuleLecture(
                    ortho=e.get("ortho", ""),
                    phone=e.get("phone", ""),
                    span_source=tuple(e.get("span_source", (0, 0))),
                    composant=e.get("composant", 0),
                )
                for e in result.get("events", [])
            ],
            display_num=result.get("display_num", ""),
            display_rom=result.get("display_rom", ""),
            valeur=result.get("valeur", ""),
        )

    ftype = formule_type.lower()
    logger.debug("lire_formule() type=%s text=%r", ftype, text)

    lecteur = _LECTEURS.get(ftype)
    if lecteur is None:
        logger.warning("Unrecognized formule type %r, falling back to spelling", ftype)
        return _epeler_texte(text, span)

    # Passer les kwargs pertinents
    kwargs: dict[str, object] = {
        "text": text,
        "span": span,
        "children": children,
    }
    if ftype == "nombre":
        kwargs["feminin"] = feminin
        kwargs["options"] = options
    if ftype in ("fraction", "scientifique", "heure", "monnaie", "pourcentage",
                 "intervalle", "gps", "page_chapitre"):
        kwargs["options"] = options

    return lecteur(**kwargs)


def enrichir_formules(
    tokens: list,
    options: OptionsLecture | None = None,
) -> list:
    """Enrichit les tokens Formule avec display_fr et lecture.

    Pour chaque token dont type.value == 'formule', calcule la lecture
    et assigne display_fr sur le token (duck-typing via setattr).
    Retourne la liste des tokens inchangés (modification in-place).
    """
    if options is None:
        options = OptionsLecture()

    count = 0
    for tok in tokens:
        ttype = getattr(tok, "type", None)
        if ttype is None:
            continue
        tname = str(getattr(ttype, "value", str(ttype))).lower()
        if tname != "formule":
            continue

        text = getattr(tok, "text", "")
        tok_span = getattr(tok, "span", (0, len(text)))
        children = getattr(tok, "children", None)
        ft = getattr(tok, "formule_type", None)
        ftype = getattr(ft, "value", str(ft)).lower() if ft else "nombre"

        lecture = lire_formule(
            formule_type=ftype,
            text=text,
            span=tok_span,
            children=children,
            options=options,
        )
        setattr(tok, "display_fr", lecture.display_fr)
        setattr(tok, "lecture", lecture)
        count += 1

    logger.info("enrichir_formules() enriched %s formule tokens", count)
    return tokens
