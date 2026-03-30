"""Lecture algorithmique des formules — module autonome transversal.

Fournit 3 facettes :
  - Tokeniseur : identification du type + display_fr
  - G2P : transcription phonétique IPA
  - Syllabeur/Aligneur : events décomposés avec groupement par composant

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
    decimal_method: str = "m2"       # "m1" (groupes de 3), "m2" (nombre entier)
    heure_mot_minutes: bool = True   # dire "minutes" quand format h/colon
    monnaie_dire_centimes: bool = True  # inclure les centimes
    romain_actif: bool = True        # calculer display_rom pour les nombres


# ══════════════════════════════════════════════════════════════════════════════
# Tables embarquées (extraites des CSV numReader)
# ══════════════════════════════════════════════════════════════════════════════

# -- Unités de base : label → (texte, phone, valeur) --------------------------
_UNITES: dict[str, tuple[str, str, int]] = {
    "0":    ("zéro",             "zeʁo",       0),
    "1":    ("un",               "ɛ̃",          1),
    "2":    ("deux",             "dø",          2),
    "3":    ("trois",            "tʁwa",        3),
    "4":    ("quatre",           "katʁ",        4),
    "5":    ("cinq",             "sɛ̃k",        5),
    "6":    ("six",              "sis",          6),
    "7":    ("sept",             "sɛt",          7),
    "8":    ("huit",             "ɥit",          8),
    "9":    ("neuf",             "nœf",          9),
    "10":   ("dix",              "dis",         10),
    "11":   ("onze",             "ɔ̃z",         11),
    "12":   ("douze",            "duz",         12),
    "13":   ("treize",           "tʁɛz",       13),
    "14":   ("quatorze",         "katɔʁz",     14),
    "15":   ("quinze",           "kɛ̃z",        15),
    "16":   ("seize",            "sɛz",         16),
    "20":   ("vingt",            "vɛ̃",         20),
    "30":   ("trente",           "tʁɑ̃t",       30),
    "40":   ("quarante",         "kaʁɑ̃t",      40),
    "50":   ("cinquante",        "sɛ̃kɑ̃t",     50),
    "60":   ("soixante",         "swasɑ̃t",     60),
    "100":  ("cent",             "sɑ̃",        100),
    "1000": ("mille",            "mil",       1000),
    # Variantes spéciales
    "et":       ("et",       "e",    0),
    "1_fem":    ("une",      "yn",   1),
    "et_1":     ("et un",    "e ɛ̃",  1),
    "et_1_fem": ("et une",   "e yn", 1),
    "et_11":    ("et onze",  "e ɔ̃z", 11),
    "20t":      ("vingt",    "vɛ̃t",  20),  # vingt devant unité (liaison t)
    "20s":      ("vingts",   "vɛ̃",   20),  # quatre-vingts (pluriel)
    "100s":     ("cents",    "sɑ̃",  100),  # deux-cents (pluriel)
    # Échelles
    "million":    ("million",   "miljɔ̃",   1_000_000),
    "millions":   ("millions",  "miljɔ̃",   1_000_000),
    "milliard":   ("milliard",  "miljaʁ",  1_000_000_000),
    "milliards":  ("milliards", "miljaʁ",  1_000_000_000),
    "billion":    ("billion",   "bijɔ̃",    1_000_000_000_000),
    "billions":   ("billions",  "bijɔ̃",    1_000_000_000_000),
}

# -- Lettres : caractère → (texte, phone) -------------------------------------
_LETTRES: dict[str, tuple[str, str]] = {
    "A": ("a", "a"),       "a": ("a", "a"),
    "B": ("bé", "be"),     "b": ("bé", "be"),
    "C": ("cé", "se"),     "c": ("cé", "se"),
    "D": ("dé", "de"),     "d": ("dé", "de"),
    "E": ("e", "ə"),       "e": ("e", "ə"),
    "F": ("effe", "ɛf"),   "f": ("effe", "ɛf"),
    "G": ("gé", "ʒe"),     "g": ("gé", "ʒe"),
    "H": ("ache", "aʃ"),   "h": ("ache", "aʃ"),
    "I": ("i", "i"),       "i": ("i", "i"),
    "J": ("ji", "ʒi"),     "j": ("ji", "ʒi"),
    "K": ("ka", "ka"),     "k": ("ka", "ka"),
    "L": ("elle", "ɛl"),   "l": ("elle", "ɛl"),
    "M": ("emme", "ɛm"),   "m": ("emme", "ɛm"),
    "N": ("enne", "ɛn"),   "n": ("enne", "ɛn"),
    "O": ("o", "o"),       "o": ("o", "o"),
    "P": ("pé", "pe"),     "p": ("pé", "pe"),
    "Q": ("ku", "ky"),     "q": ("ku", "ky"),
    "R": ("erre", "ɛʁ"),   "r": ("erre", "ɛʁ"),
    "S": ("esse", "ɛs"),   "s": ("esse", "ɛs"),
    "T": ("té", "te"),     "t": ("té", "te"),
    "U": ("u", "y"),       "u": ("u", "y"),
    "V": ("vé", "ve"),     "v": ("vé", "ve"),
    "W": ("double-vé", "dublə ve"), "w": ("double-vé", "dublə ve"),
    "X": ("ix", "iks"),    "x": ("ix", "iks"),
    "Y": ("i-grec", "i ɡʁɛk"), "y": ("i-grec", "i ɡʁɛk"),
    "Z": ("zède", "zɛd"),  "z": ("zède", "zɛd"),
}

# -- Symboles mathématiques : symbole → (texte, phone) ------------------------
_SYMBOLES: dict[str, tuple[str, str]] = {
    "+":  ("plus",                 "plys"),
    "-":  ("moins",                "mwɛ̃"),
    "−":  ("moins",                "mwɛ̃"),
    "±":  ("plus ou moins",        "plyz u mwɛ̃"),
    "=":  ("égal",                 "eɡal"),
    "≠":  ("différent de",         "difeʁɑ̃ də"),
    "<":  ("inférieur à",          "ɛ̃feʁjœʁ a"),
    ">":  ("supérieur à",          "sypeʁjœʁ a"),
    "≤":  ("inférieur ou égal à",  "ɛ̃feʁjœʁ u eɡal a"),
    "≥":  ("supérieur ou égal à",  "sypeʁjœʁ u eɡal a"),
    "×":  ("fois",                 "fwa"),
    "*":  ("fois",                 "fwa"),
    "÷":  ("divisé par",           "divize paʁ"),
    "/":  ("sur",                  "syʁ"),
    "^":  ("puissance",            "pɥisɑ̃s"),
    "√":  ("racine carrée de",     "ʁasin kaʁe də"),
    "∞":  ("infini",               "ɛ̃fini"),
    "%":  ("pour cent",            "puʁ sɑ̃"),
    "‰":  ("pour mille",           "puʁ mil"),
    "²":  ("au carré",             "o kaʁe"),
    "³":  ("au cube",              "o kyb"),
    "∑":  ("somme",                "sɔm"),
    "∏":  ("produit",              "pʁodɥi"),
    "∫":  ("intégrale",            "ɛ̃teɡʁal"),
    "∂":  ("dérivée partielle",    "deʁive paʁsjɛl"),
    "∇":  ("nabla",                "nabla"),
    "∈":  ("appartient à",         "apaʁtjɛ̃ a"),
    "∉":  ("n'appartient pas à",   "napaʁtjɛ̃ pa a"),
    "⊂":  ("inclus dans",          "ɛ̃kly dɑ̃"),
    "∪":  ("union",                "ynjɔ̃"),
    "∩":  ("intersection",         "ɛ̃tɛʁseksjɔ̃"),
    "→":  ("donne",                "dɔn"),
    "←":  ("reçoit",               "ʁəswa"),
    "↔":  ("équivalent à",         "ekivalɑ̃ a"),
    "⇒":  ("implique",             "ɛ̃plik"),
    "⇔":  ("équivalent à",         "ekivalɑ̃ a"),
    "(":  ("ouvrez la parenthèse", "uvʁe la paʁɑ̃tɛz"),
    ")":  ("fermez la parenthèse", "fɛʁme la paʁɑ̃tɛz"),
    "[":  ("crochet ouvrant",      "kʁoʃɛ uvʁɑ̃"),
    "]":  ("crochet fermant",      "kʁoʃɛ fɛʁmɑ̃"),
    # Fonctions math
    "sin":  ("sinus",                    "sinys"),
    "cos":  ("cosinus",                  "kosinys"),
    "tan":  ("tangente",                 "tɑ̃ʒɑ̃t"),
    "exp":  ("exponentielle",            "ɛksponɑ̃sjɛl"),
    "ln":   ("logarithme népérien",      "loɡaʁitm nepeʁjɛ̃"),
    "log":  ("logarithme",               "loɡaʁitm"),
    "sqrt": ("racine carrée",            "ʁasin kaʁe"),
    "abs":  ("valeur absolue",           "valœʁ apsoly"),
}

# -- Lettres grecques : caractère → (texte, phone) ----------------------------
_GREC: dict[str, tuple[str, str]] = {
    "Α": ("alpha", "alfa"),     "α": ("alpha", "alfa"),
    "Β": ("bêta", "bɛta"),     "β": ("bêta", "bɛta"),
    "Γ": ("gamma", "ɡama"),    "γ": ("gamma", "ɡama"),
    "Δ": ("delta", "dɛlta"),   "δ": ("delta", "dɛlta"),
    "Ε": ("epsilon", "ɛpsilɔ̃"), "ε": ("epsilon", "ɛpsilɔ̃"),
    "Ζ": ("zêta", "zɛta"),     "ζ": ("zêta", "zɛta"),
    "Η": ("êta", "ɛta"),       "η": ("êta", "ɛta"),
    "Θ": ("thêta", "tɛta"),    "θ": ("thêta", "tɛta"),
    "Ι": ("iota", "jota"),     "ι": ("iota", "jota"),
    "Κ": ("kappa", "kapa"),    "κ": ("kappa", "kapa"),
    "Λ": ("lambda", "lɑ̃bda"),  "λ": ("lambda", "lɑ̃bda"),
    "Μ": ("mu", "my"),         "μ": ("mu", "my"),
    "Ν": ("nu", "ny"),         "ν": ("nu", "ny"),
    "Ξ": ("ksi", "ksi"),       "ξ": ("ksi", "ksi"),
    "Ο": ("omicron", "omikʁɔ̃"), "ο": ("omicron", "omikʁɔ̃"),
    "Π": ("pi", "pi"),         "π": ("pi", "pi"),
    "Ρ": ("rhô", "ʁo"),       "ρ": ("rhô", "ʁo"),
    "Σ": ("sigma", "siɡma"),   "σ": ("sigma", "siɡma"),
    "ς": ("sigma", "siɡma"),
    "Τ": ("tau", "to"),        "τ": ("tau", "to"),
    "Υ": ("upsilon", "ypsilɔn"), "υ": ("upsilon", "ypsilɔn"),
    "Φ": ("phi", "fi"),        "φ": ("phi", "fi"),
    "Χ": ("khi", "ki"),        "χ": ("khi", "ki"),
    "Ψ": ("psi", "psi"),       "ψ": ("psi", "psi"),
    "Ω": ("oméga", "omeɡa"),   "ω": ("oméga", "omeɡa"),
    "'": ("prime", "pʁim"),     "′": ("prime", "pʁim"),
}

# -- Ordinaux : cardinal → (texte_ordinal, phone_ordinal) ---------------------
_ORDINAUX: dict[str, tuple[str, str]] = {
    "un":         ("unième",        "ynjɛm"),
    "deux":       ("deuxième",      "døzjɛm"),
    "trois":      ("troisième",     "tʁwazjɛm"),
    "quatre":     ("quatrième",     "katʁjɛm"),
    "cinq":       ("cinquième",     "sɛ̃kjɛm"),
    "six":        ("sixième",       "sizjɛm"),
    "sept":       ("septième",      "sɛtjɛm"),
    "huit":       ("huitième",      "ɥitjɛm"),
    "neuf":       ("neuvième",      "nœvjɛm"),
    "dix":        ("dixième",       "dizjɛm"),
    "onze":       ("onzième",       "ɔ̃zjɛm"),
    "douze":      ("douzième",      "duzjɛm"),
    "treize":     ("treizième",     "tʁɛzjɛm"),
    "quatorze":   ("quatorzième",   "katɔʁzjɛm"),
    "quinze":     ("quinzième",     "kɛ̃zjɛm"),
    "seize":      ("seizième",      "sɛzjɛm"),
    "vingt":      ("vingtième",     "vɛ̃tjɛm"),
    "trente":     ("trentième",     "tʁɑ̃tjɛm"),
    "quarante":   ("quarantième",   "kaʁɑ̃tjɛm"),
    "cinquante":  ("cinquantième",  "sɛ̃kɑ̃tjɛm"),
    "soixante":   ("soixantième",   "swasɑ̃tjɛm"),
    "cent":       ("centième",      "sɑ̃tjɛm"),
    "mille":      ("millième",      "miljɛm"),
    "million":    ("millionième",   "miljɔnjɛm"),
    "milliard":   ("milliardième",  "miljaʁdjɛm"),
    "billion":    ("billionième",   "biljɔnjɛm"),
    # Formes spéciales
    "_premier_m":  ("premier",  "pʁømje"),
    "_premier_f":  ("première", "pʁømjɛʁ"),
    "_second_m":   ("second",   "səɡɔ̃"),
    "_second_f":   ("seconde",  "səɡɔ̃d"),
    "_demi":       ("demi",     "dəmi"),
    "_tiers":      ("tiers",    "tjɛʁ"),
    "_quart":      ("quart",    "kaʁ"),
    "_quarts":     ("quarts",   "kaʁ"),
}

# -- Mois : numéro 1-12 → (texte, phone) --------------------------------------
_MOIS: dict[int, tuple[str, str]] = {
    1:  ("janvier",   "ʒɑ̃vje"),
    2:  ("février",   "fevʁije"),
    3:  ("mars",      "maʁs"),
    4:  ("avril",     "avʁil"),
    5:  ("mai",       "mɛ"),
    6:  ("juin",      "ʒɥɛ̃"),
    7:  ("juillet",   "ʒɥijɛ"),
    8:  ("août",      "ut"),
    9:  ("septembre", "sɛptɑ̃bʁ"),
    10: ("octobre",   "ɔktɔbʁ"),
    11: ("novembre",  "novɑ̃bʁ"),
    12: ("décembre",  "desɑ̃bʁ"),
}

# -- Mots-outils ---------------------------------------------------------------
_VIRGULE = ("virgule", "viʁɡyl")
_FOIS    = ("fois",    "fwa")
_DIX     = ("dix",     "dis")
_EXPOSANT = ("exposant", "ɛkspozɑ̃")

# Exposants unicode → valeur
_SUPERSCRIPTS: dict[str, str] = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "ⁿ": "n",
}


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
            parts.append(_u(str(dizaine)))
            if feminin:
                parts.append(_u("et_1_fem"))
            else:
                parts.append(_u("et_1"))
        elif unite == 11 and dizaine == 60:
            # soixante-et-onze
            parts.append(_u("60"))
            parts.append(_u("et_11"))
        else:
            parts.append(_u(str(dizaine)))
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
        parts.append(_u("20"))  # pas de s car suivi d'unité
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


# Échelles : position → (singulier, pluriel, prefixer_un)
_ECHELLES: dict[int, tuple[str, str, bool]] = {
    1: ("1000", "1000", False),          # mille (pas "un mille")
    2: ("million", "millions", True),     # un million, deux millions
    3: ("milliard", "milliards", True),
    4: ("billion", "billions", True),
}


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

def _make_result(
    events: list[EventFormuleLecture],
    display_num: str = "",
    display_rom: str = "",
    valeur: int | float | str = "",
) -> LectureFormuleResult:
    """Construit un LectureFormuleResult à partir d'events."""
    display = "-".join(e.ortho for e in events)
    phone = "".join(e.phone.replace(" ", "") for e in events)
    # Calculer span_fr pour chaque event
    offset = 0
    for evt in events:
        evt.span_fr = (offset, offset + len(evt.ortho))
        offset += len(evt.ortho) + 1  # +1 pour le "-"
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

    # Nombre décimal ?
    if "," in text_clean or "." in text_clean:
        return _lire_decimal(text_clean, span, options)

    # Nombre négatif ?
    negatif = text_clean.startswith("-") or text_clean.startswith("−")
    if negatif:
        text_clean = text_clean.lstrip("-−")

    try:
        n = int(text_clean)
    except ValueError:
        # Fallback : épeler les caractères
        return _epeler_texte(text, span)

    if negatif:
        n = -n

    parts = _nombre_vers_francais(n, feminin=feminin)
    events = _events_from_parts(parts, span, text, composant=0)

    # Chiffres romains
    display_rom = ""
    if options.romain_actif and isinstance(n, int) and 1 <= abs(n) <= 39999:
        try:
            from lectura_formules.romains import int_to_roman
            display_rom = int_to_roman(abs(n))
        except (ImportError, ValueError):
            pass

    return _make_result(events, display_num=text_clean, display_rom=display_rom, valeur=n)


def _lire_decimal(
    text_clean: str, span: Span,
    options: OptionsLecture | None = None,
) -> LectureFormuleResult:
    """Lit un nombre décimal avec méthode M1 ou M2."""
    if options is None:
        options = OptionsLecture()

    sep = "," if "," in text_clean else "."
    partie_ent, partie_dec = text_clean.split(sep, 1)

    parts: list[tuple[str, str]] = []

    # Partie entière
    n_ent = int(partie_ent) if partie_ent else 0
    parts.extend(_nombre_vers_francais(n_ent))

    # Virgule
    parts.append(_VIRGULE)

    # Partie décimale selon la méthode
    method = options.decimal_method if options else "m2"
    if method == "m1":
        parts.extend(_decimal_m1(partie_dec))
    else:
        parts.extend(_decimal_m2(partie_dec))

    # Valeur numérique
    try:
        valeur = float(text_clean.replace(",", "."))
    except ValueError:
        valeur = text_clean

    events = _events_from_parts(parts, span, text_clean, composant=0)
    return _make_result(events, display_num=text_clean, valeur=valeur)


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
    """Construit les parts pour les zéros initiaux.

    ≤3 zéros : les lire individuellement.
    >3 zéros : "N fois zéro".
    """
    if lz <= 0:
        return []
    if lz <= 3:
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


def _decimal_m2(partie_dec: str) -> list[tuple[str, str]]:
    """Méthode M2 : zéros initiaux + groupes de 3 depuis la gauche."""
    parts: list[tuple[str, str]] = []

    lz = _count_leading_zeros(partie_dec)
    rest = partie_dec[lz:]

    # Zéros initiaux
    parts.extend(_leading_zeros_parts(lz))

    # Grouper le reste par 3
    if rest:
        groups = _group_by_3_left(rest)
        for grp in groups:
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

    return _make_result(events)


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

    return _make_result(events)


# -- TELEPHONE -----------------------------------------------------------------

_TEL_CLEAN_RE = re.compile(r"[\s.\-]")


def lire_telephone(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit un numéro de téléphone par paires de chiffres.

    Composants : 1 composant par paire de chiffres.
    Ex: "06 12 34 56 78" → 5 composants.
    """
    events: list[EventFormuleLecture] = []
    src_start = span[0]

    # Extraire les chiffres avec leurs positions dans le texte source
    digits: list[tuple[str, int]] = []
    for i, ch in enumerate(text):
        if ch.isdigit():
            digits.append((ch, src_start + i))

    # Lire par paires
    i = 0
    comp_idx = 0
    while i < len(digits):
        if i + 1 < len(digits):
            d0, pos0 = digits[i]
            d1, pos1 = digits[i + 1]
            pair_start = pos0
            pair_end = pos1 + 1

            if d0 == "0":
                # Paire avec zéro initial : lire chiffre par chiffre
                t0, p0 = _u(d0)
                events.append(EventFormuleLecture(
                    ortho=t0, phone=p0,
                    span_source=(pos0, pos0 + 1),
                    composant=comp_idx,
                ))
                t1, p1 = _u(d1)
                events.append(EventFormuleLecture(
                    ortho=t1, phone=p1,
                    span_source=(pos1, pos1 + 1),
                    composant=comp_idx,
                ))
            else:
                # Paire normale : lire comme nombre
                n = int(d0 + d1)
                parts = _nombre_vers_francais(n)
                for p_ortho, p_phone in parts:
                    events.append(EventFormuleLecture(
                        ortho=p_ortho, phone=p_phone,
                        span_source=(pair_start, pair_end),
                        composant=comp_idx,
                    ))
            comp_idx += 1
            i += 2
        else:
            # Chiffre isolé en fin
            d = digits[i][0]
            d_pos = digits[i][1]
            t, p = _u(d)
            events.append(EventFormuleLecture(
                ortho=t, phone=p,
                span_source=(d_pos, d_pos + 1),
                composant=comp_idx,
            ))
            comp_idx += 1
            i += 1

    return _make_result(events)


# -- ORDINAL ------------------------------------------------------------------

_ORDINAL_RE = re.compile(
    r"(\d+)\s*(er|re|ère|e|ème|ème|ième|eme|ier|ière)\b",
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
        return _epeler_texte(text, span)

    n = int(m.group(1))
    suffix = m.group(2).lower()
    src = span[0]
    num_start = src + m.start(1)
    num_end = src + m.end(1)
    suf_start = src + m.start(2)
    suf_end = src + m.end(2)

    events: list[EventFormuleLecture] = []

    if n == 1:
        # 1er/1re/1ère → premier/première (composant 0 unique)
        if suffix in ("re", "ère", "ière"):
            events.append(EventFormuleLecture(
                ortho="première", phone="pʁømjɛʁ",
                span_source=(num_start, suf_end),
                composant=0,
            ))
        else:
            events.append(EventFormuleLecture(
                ortho="premier", phone="pʁømje",
                span_source=(num_start, suf_end),
                composant=0,
            ))
    elif n == 2 and suffix in ("nd", "nde"):
        if suffix == "nde":
            events.append(EventFormuleLecture(
                ortho="seconde", phone="səɡɔ̃d",
                span_source=(num_start, suf_end),
                composant=0,
            ))
        else:
            events.append(EventFormuleLecture(
                ortho="second", phone="səɡɔ̃",
                span_source=(num_start, suf_end),
                composant=0,
            ))
    else:
        # Nombre cardinal (composant 0) + suffixe ordinal (composant 1)
        parts = _nombre_vers_francais(n)
        events.extend(_events_from_parts(parts, (num_start, num_end), m.group(1),
                                         composant=0))
        # Suffixe ordinal : appliquer au dernier mot cardinal
        last_cardinal = parts[-1][0] if parts else ""
        if last_cardinal in _ORDINAUX:
            ord_t, ord_p = _ORDINAUX[last_cardinal]
            # Remplacer le dernier event par sa forme ordinale (composant 1)
            if events:
                last_evt = events[-1]
                events[-1] = EventFormuleLecture(
                    ortho=ord_t, phone=ord_p,
                    span_source=(last_evt.span_source[0], suf_end),
                    composant=1,
                )
        else:
            # Fallback : ajouter "ième" (composant 1)
            events.append(EventFormuleLecture(
                ortho="ième", phone="jɛm",
                span_source=(suf_start, suf_end),
                composant=1,
            ))

    return _make_result(events)


# -- FRACTION ------------------------------------------------------------------

_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


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

    return _make_result(events)


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
    pluriel = num > 1

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

    # Sinon : ordinal
    return _fraction_ordinal(num, den, num_start, num_end, den_start, den_end,
                             num_text, den_text)


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
            if num > 1:
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


def lire_scientifique(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit une notation scientifique : 1.23e-5 → un virgule vingt-trois
    fois dix exposant moins cinq.

    Composants : mantisse (0), "fois dix exposant" (1), exposant (2).
    """
    m = _SCI_RE.match(text.strip())
    if not m:
        return _epeler_texte(text, span)

    mantisse_str = m.group(1)
    exposant_str = m.group(2)
    src = span[0]

    events: list[EventFormuleLecture] = []

    # Mantisse (composant 0)
    mant_start = src + m.start(1)
    mant_end = src + m.end(1)
    mant_result = lire_nombre(mantisse_str, span=(mant_start, mant_end))
    for evt in mant_result.events:
        events.append(EventFormuleLecture(
            ortho=evt.ortho, phone=evt.phone,
            span_source=evt.span_source,
            composant=0,
        ))

    # "fois dix exposant" (composant 1)
    sep_start = src + m.end(1)
    sep_end = src + m.start(2)
    events.append(EventFormuleLecture(
        ortho="fois", phone="fwa",
        span_source=(sep_start, sep_end),
        composant=1,
    ))
    events.append(EventFormuleLecture(
        ortho="dix", phone="dis",
        span_source=(sep_start, sep_end),
        composant=1,
    ))
    events.append(EventFormuleLecture(
        ortho="exposant", phone="ɛkspozɑ̃",
        span_source=(sep_start, sep_end),
        composant=1,
    ))

    # Exposant (composant 2)
    exp_start = src + m.start(2)
    exp_end = src + m.end(2)
    exp_result = lire_nombre(exposant_str, span=(exp_start, exp_end))
    for evt in exp_result.events:
        events.append(EventFormuleLecture(
            ortho=evt.ortho, phone=evt.phone,
            span_source=evt.span_source,
            composant=2,
        ))

    return _make_result(events)


# -- MATHS ---------------------------------------------------------------------

# Caractères reconnus comme opérateurs maths
_MATHS_OPS = set("+-−±=≠<>≤≥×*÷/^√∞²³∑∏∫∂∇∈∉⊂∪∩→←↔⇒⇔")
_MATHS_BRACKETS = set("()[]{}⟨⟩")
_GREEK_CHARS = set(_GREC.keys())
_FUNC_NAMES = {"sin", "cos", "tan", "exp", "ln", "log", "sqrt", "abs"}


def lire_maths(
    text: str,
    span: Span = (0, 0),
    children: list[object] | None = None,
    **_kw: object,
) -> LectureFormuleResult:
    """Lit une formule mathématique en combinant nombres, lettres,
    symboles et fonctions.

    Composants : 1 composant par token de formule (nombre, opérateur,
    variable, fonction, parenthèse, etc.).
    """
    events: list[EventFormuleLecture] = []
    src = span[0]
    i = 0
    comp_idx = 0

    while i < len(text):
        ch = text[i]
        pos = src + i

        # Espaces
        if ch in (" ", "\t", "\u202f"):
            i += 1
            continue

        # ² et ³ (symboles directs — avant les exposants génériques)
        if ch in ("²", "³"):
            t_sym, p_sym = _SYMBOLES[ch]
            events.append(EventFormuleLecture(
                ortho=t_sym, phone=p_sym,
                span_source=(pos, pos + 1),
                composant=comp_idx,
            ))
            comp_idx += 1
            i += 1
            continue

        # Exposants unicode (⁰¹⁴⁵⁶⁷⁸⁹ⁿ — mais pas ²³ traités ci-dessus)
        if ch in _SUPERSCRIPTS:
            sup_val = _SUPERSCRIPTS[ch]
            if sup_val.isdigit():
                # Accumuler les exposants consécutifs
                j = i
                sup_digits = ""
                while j < len(text) and text[j] in _SUPERSCRIPTS:
                    sv = _SUPERSCRIPTS[text[j]]
                    if sv.isdigit():
                        sup_digits += sv
                        j += 1
                    else:
                        break
                if sup_digits:
                    n_sup = int(sup_digits)
                    events.append(EventFormuleLecture(
                        ortho="exposant", phone="ɛkspozɑ̃",
                        span_source=(pos, src + j),
                        composant=comp_idx,
                    ))
                    parts = _nombre_vers_francais(n_sup)
                    for t, p in parts:
                        events.append(EventFormuleLecture(
                            ortho=t, phone=p,
                            span_source=(pos, src + j),
                            composant=comp_idx,
                        ))
                    comp_idx += 1
                    i = j
                    continue
            elif sup_val == "n":
                events.append(EventFormuleLecture(
                    ortho="exposant", phone="ɛkspozɑ̃",
                    span_source=(pos, pos + 1),
                    composant=comp_idx,
                ))
                events.append(EventFormuleLecture(
                    ortho="enne", phone="ɛn",
                    span_source=(pos, pos + 1),
                    composant=comp_idx,
                ))
                comp_idx += 1
                i += 1
                continue

        # Chiffres
        if ch.isdigit():
            j = i
            while j < len(text) and (text[j].isdigit() or text[j] in ".,"):
                j += 1
            group = text[i:j]
            grp_result = lire_nombre(group, span=(pos, src + j))
            for evt in grp_result.events:
                events.append(EventFormuleLecture(
                    ortho=evt.ortho, phone=evt.phone,
                    span_source=evt.span_source,
                    composant=comp_idx,
                ))
            comp_idx += 1
            i = j
            continue

        # Lettres grecques
        if ch in _GREEK_CHARS:
            t_gr, p_gr = _GREC[ch]
            events.append(EventFormuleLecture(
                ortho=t_gr, phone=p_gr,
                span_source=(pos, pos + 1),
                composant=comp_idx,
            ))
            comp_idx += 1
            i += 1
            continue

        # Opérateurs maths
        if ch in _MATHS_OPS:
            if ch in _SYMBOLES:
                t_sym, p_sym = _SYMBOLES[ch]
                events.append(EventFormuleLecture(
                    ortho=t_sym, phone=p_sym,
                    span_source=(pos, pos + 1),
                    composant=comp_idx,
                ))
                comp_idx += 1
            i += 1
            continue

        # Parenthèses/crochets
        if ch in _MATHS_BRACKETS:
            if ch in _SYMBOLES:
                t_sym, p_sym = _SYMBOLES[ch]
                events.append(EventFormuleLecture(
                    ortho=t_sym, phone=p_sym,
                    span_source=(pos, pos + 1),
                    composant=comp_idx,
                ))
                comp_idx += 1
            i += 1
            continue

        # Lettres latines (fonctions connues ou variables)
        if ch.isalpha():
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
            word = text[i:j]
            word_lower = word.lower()

            if word_lower in _FUNC_NAMES:
                # Fonction connue
                t_fn, p_fn = _SYMBOLES[word_lower]
                events.append(EventFormuleLecture(
                    ortho=t_fn, phone=p_fn,
                    span_source=(pos, src + j),
                    composant=comp_idx,
                ))
                comp_idx += 1
            elif len(word) == 1:
                # Variable d'une lettre → épeler
                t_l, p_l = _LETTRES.get(word, (word, word))
                events.append(EventFormuleLecture(
                    ortho=t_l, phone=p_l,
                    span_source=(pos, pos + 1),
                    composant=comp_idx,
                ))
                comp_idx += 1
            else:
                # Mot multi-lettres inconnu → épeler (1 composant pour tout le mot)
                for k, c in enumerate(word):
                    t_l, p_l = _LETTRES.get(c, (c, c))
                    events.append(EventFormuleLecture(
                        ortho=t_l, phone=p_l,
                        span_source=(src + i + k, src + i + k + 1),
                        composant=comp_idx,
                    ))
                comp_idx += 1
            i = j
            continue

        # Caractère inconnu → ignorer
        i += 1

    return _make_result(events)


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

    # On parcourt le texte en segments alpha/digit/séparateur
    i = 0
    while i < len(text):
        ch = text[i]
        pos = src + i

        if ch in (" ", ".", "-", "\t"):
            i += 1
            continue

        if ch.isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            group = text[i:j]
            n = int(group)
            parts = _nombre_vers_francais(n)
            events.extend(_events_from_parts(parts, (pos, src + j), group,
                                             composant=comp_idx))
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
            comp_idx += 1
            i = j
        else:
            i += 1

    return _make_result(events)


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
    return _make_result(events)


# -- HEURE ---------------------------------------------------------------------

_HEURE_RE_H = re.compile(
    r"^(\d{1,2})[hH](\d{1,2})?(?:min(\d{1,2}))?(?:s)?$"
)
_HEURE_RE_HMS = re.compile(r"^(\d{1,2})[hH](\d{1,2})min(\d{1,2})s?$")
_HEURE_RE_HMIN = re.compile(r"^(\d{1,2})[hH](\d{1,2})min$")
_HEURE_RE_COLON = re.compile(r"^(\d{1,2}):(\d{2})$")
_HEURE_RE_MIN = re.compile(r"^(\d{1,2})min$")
_HEURE_RE_SEC = re.compile(r"^(\d{1,2})s$")

_HEURE_WORDS: dict[str, tuple[str, str]] = {
    "heure":    ("heure",    "œʁ"),
    "heures":   ("heures",   "œʁ"),
    "minute":   ("minute",   "minyt"),
    "minutes":  ("minutes",  "minyt"),
    "seconde":  ("seconde",  "səɡɔ̃d"),
    "secondes": ("secondes", "səɡɔ̃d"),
}


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

    # Heures
    if hours is not None:
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
    return _make_result(events, display_num=display_num, valeur=display_num)


# -- MONNAIE -------------------------------------------------------------------

_DEVISES: dict[str, dict] = {
    "EUR": {"symbole": "€",  "majeur": ("euro", "euros"),
            "mineur": ("centime", "centimes"), "phone_maj": ("øʁo", "øʁo"),
            "phone_min": ("sɑ̃tim", "sɑ̃tim")},
    "USD": {"symbole": "$",  "majeur": ("dollar", "dollars"),
            "mineur": ("centime", "centimes"), "phone_maj": ("dolaʁ", "dolaʁ"),
            "phone_min": ("sɑ̃tim", "sɑ̃tim")},
    "GBP": {"symbole": "£",  "majeur": ("livre", "livres"),
            "mineur": ("centime", "centimes"), "phone_maj": ("livʁ", "livʁ"),
            "phone_min": ("sɑ̃tim", "sɑ̃tim")},
    "CHF": {"symbole": None, "majeur": ("franc", "francs"),
            "mineur": ("centime", "centimes"), "phone_maj": ("fʁɑ̃", "fʁɑ̃"),
            "phone_min": ("sɑ̃tim", "sɑ̃tim"),
            "suffixe": ("suisse", "sɥis")},
    "JPY": {"symbole": "¥",  "majeur": ("yen", "yens"),
            "mineur": None, "phone_maj": ("jɛn", "jɛn")},
}

_SYM_TO_ISO: dict[str, str] = {v["symbole"]: k for k, v in _DEVISES.items()
                                 if v.get("symbole")}

_MONNAIE_RE_POST = re.compile(
    r"^([0-9][0-9 ']*[0-9]*[.,]?\d*)\s*([€$£¥])$"
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


def _parse_monnaie(text: str) -> dict | None:
    """Parse un montant avec devise. Retourne dict ou None."""
    s = text.strip()
    amount_str = None
    currency = None

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
    if "." in cleaned:
        int_part, dec_part = cleaned.split(".", 1)
        major = int(int_part) if int_part else 0
        if len(dec_part) == 1:
            minor = int(dec_part) * 10
        elif len(dec_part) >= 2:
            minor = int(dec_part[:2])
        else:
            minor = 0
    else:
        major = int(cleaned)
        minor = 0

    return {"currency": currency, "major": major, "minor": minor}


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
    cur = _DEVISES[currency]

    events: list[EventFormuleLecture] = []

    # Partie majeure (composant 0)
    if major > 0 or minor == 0:
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
    if minor > 0 and cur.get("mineur") and options.monnaie_dire_centimes:
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

    # display_num
    sym = cur.get("symbole")
    if sym:
        if minor > 0:
            display_num = f"{major},{minor:02d}{sym}"
        else:
            display_num = f"{major}{sym}"
    else:
        if minor > 0:
            display_num = f"{major},{minor:02d} {currency}"
        else:
            display_num = f"{major} {currency}"

    valeur = major + minor / 100 if minor else major
    return _make_result(events, display_num=display_num, valeur=valeur)


# -- POURCENTAGE ---------------------------------------------------------------

_POURCENT_RE = re.compile(r"^([0-9][0-9 ']*\.?[0-9]*)([%‰])$")

_POURCENT_WORDS: dict[str, tuple[str, str]] = {
    "%":  ("pour cent",  "puʁ sɑ̃"),
    "‰": ("pour mille", "puʁ mil"),
}


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
    events.append(EventFormuleLecture(
        ortho=t_pct, phone=p_pct, span_source=span, composant=1,
    ))

    display_num = number_str + symbol
    try:
        valeur = float(number_str)
    except ValueError:
        valeur = number_str
    return _make_result(events, display_num=display_num, valeur=valeur)


# -- INTERVALLE ----------------------------------------------------------------

_INTERVALLE_RE = re.compile(r"^([\[\]])([^;,]+)[;,]([^;,]+)([\[\]])$")

_INTERVALLE_BOUNDS = {"+∞", "-∞", "∞", "+inf", "-inf", "inf"}
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

    # "de" (connecteur ouvert, composant 1)
    events.append(EventFormuleLecture(
        ortho="de", phone="də", span_source=span, composant=1,
    ))

    # Borne gauche (composant 0)
    events.extend(_read_bound(left_val, span, composant=0, options=options))

    # "à" (connecteur, composant 1)
    events.append(EventFormuleLecture(
        ortho="à", phone="a", span_source=span, composant=1,
    ))

    # Borne droite (composant 2)
    events.extend(_read_bound(right_val, span, composant=2, options=options))

    left_bracket = m.group(1)
    right_bracket = m.group(4)
    display_num = f"{left_bracket}{left_val};{right_val}{right_bracket}"
    return _make_result(events, display_num=display_num, valeur=display_num)


# -- GPS -----------------------------------------------------------------------

_GPS_DMS_RE = re.compile(
    r"(\d{1,3})°(\d{1,2})'(?:(\d{1,2})\"?)?\s*([NSEOW])", re.IGNORECASE
)
_GPS_DD_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)°\s*([NSEOW])", re.IGNORECASE
)

_GPS_DIRECTIONS: dict[str, tuple[str, str]] = {
    "N": ("nord",  "nɔʁ"),
    "S": ("sud",   "syd"),
    "E": ("est",   "ɛst"),
    "O": ("ouest", "wɛst"),
    "W": ("ouest", "wɛst"),
}

_GPS_UNITS: dict[str, tuple[str, str]] = {
    "degré":    ("degré",    "dəɡʁe"),
    "degrés":   ("degrés",   "dəɡʁe"),
    "minute":   ("minute",   "minyt"),
    "minutes":  ("minutes",  "minyt"),
    "seconde":  ("seconde",  "səɡɔ̃d"),
    "secondes": ("secondes", "səɡɔ̃d"),
}


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

            # Degrés
            parts = _nombre_vers_francais(deg)
            evts = _events_from_parts(parts, span, text, composant=comp)
            events.extend(evts)
            word = "degré" if deg == 1 else "degrés"
            t_w, p_w = _GPS_UNITS[word]
            events.append(EventFormuleLecture(
                ortho=t_w, phone=p_w, span_source=span, composant=comp,
            ))

            # Minutes
            parts = _nombre_vers_francais(mi, feminin=True)
            evts = _events_from_parts(parts, span, text, composant=comp)
            events.extend(evts)
            word = "minute" if mi == 1 else "minutes"
            t_w, p_w = _GPS_UNITS[word]
            events.append(EventFormuleLecture(
                ortho=t_w, phone=p_w, span_source=span, composant=comp,
            ))

            # Secondes
            if sec is not None:
                parts = _nombre_vers_francais(sec, feminin=True)
                evts = _events_from_parts(parts, span, text, composant=comp)
                events.extend(evts)
                word = "seconde" if sec == 1 else "secondes"
                t_w, p_w = _GPS_UNITS[word]
                events.append(EventFormuleLecture(
                    ortho=t_w, phone=p_w, span_source=span, composant=comp,
                ))

            # Direction
            t_dir, p_dir = _GPS_DIRECTIONS[direction]
            events.append(EventFormuleLecture(
                ortho=t_dir, phone=p_dir, span_source=span, composant=comp,
            ))

            dp = f"{deg}°{mi}'"
            if sec is not None:
                dp += f'{sec}"'
            dp += direction
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

            # Lire le nombre (possiblement décimal)
            num_result = lire_nombre(deg_str, span=span, options=options)
            for evt in num_result.events:
                evt.composant = comp
            events.extend(num_result.events)

            # "degrés"
            t_w, p_w = _GPS_UNITS["degrés"]
            events.append(EventFormuleLecture(
                ortho=t_w, phone=p_w, span_source=span, composant=comp,
            ))

            # Direction
            t_dir, p_dir = _GPS_DIRECTIONS[direction]
            events.append(EventFormuleLecture(
                ortho=t_dir, phone=p_dir, span_source=span, composant=comp,
            ))

            display_parts.append(f"{deg_str}°{direction}")
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
        events.append(EventFormuleLecture(
            ortho="page", phone="paʒ", span_source=span, composant=0,
        ))
        n = int(number_str)
        parts = _nombre_vers_francais(n)
        evts = _events_from_parts(parts, span, text, composant=1)
        events.extend(evts)
        display_num = prefix_raw + number_str
        return _make_result(events, display_num=display_num, valeur=n)

    # Chapitre
    m = _CHAP_RE.match(s)
    if m:
        prefix_raw = s[:s.index(m.group(2))]
        number_str = m.group(2)
        events.append(EventFormuleLecture(
            ortho="chapitre", phone="ʃapitʁ", span_source=span, composant=0,
        ))
        n = int(number_str)
        parts = _nombre_vers_francais(n)
        evts = _events_from_parts(parts, span, text, composant=1)
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
    """Point d'entrée unique pour la lecture algorithmique des formules.

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
        Options de lecture (mode fraction, méthode décimale).
    feminin : bool
        Si True, utilise les formes féminines (une au lieu de un).

    Returns
    -------
    LectureFormuleResult
        Texte lu, IPA et événements alignés.
    """
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
    if ftype in ("fraction", "heure", "monnaie", "pourcentage",
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
