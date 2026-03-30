"""Lectura Syllabeur Complet — Analyseur syllabique du français avec groupes de lecture.

Fichier unique, autonome, zéro dépendance Python.
Phonémiseur pluggable avec backend eSpeak-NG par défaut.

Architecture en 2 étapes :
    E1 : Groupes de lecture (élisions, liaisons, enchaînements)
    E2 : Syllabation sur les groupes avec alignement formules

Usage rapide :
    from lectura_syllabeur import LecturaSyllabeur

    syl = LecturaSyllabeur()                        # eSpeak par défaut
    result = syl.analyze("chocolat")
    for s in result.syllabes:
        print(f"{s.ortho} -> /{s.phone}/")

Usage complet (avec groupes de lecture) :
    from lectura_syllabeur import LecturaSyllabeur, MotAnalyse, OptionsGroupes

    mots = [
        MotAnalyse(token=..., phone="lez", liaison="Lz"),
        MotAnalyse(token=..., phone="ɑ̃fɑ̃", liaison="none"),
    ]
    r = syl.analyser_complet(mots)
    print(f"{r.nb_groupes} groupes, {r.nb_syllabes} syllabes")

IPA direct (sans phonémiseur) :
    sylls = syl.syllabify_ipa("ʃɔkɔla")            # -> ["ʃɔ", "kɔ", "la"]

Pré-requis système (mode eSpeak) :
    sudo apt install espeak-ng        # Linux
    brew install espeak               # macOS
    choco install espeak-ng           # Windows

Copyright (c) 2025 Lectura — Licence CC BY-SA 4.0.
Voir LICENCE.txt et ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__version__ = "2.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# Données embarquées
# ══════════════════════════════════════════════════════════════════════════════

# Alphabet IPA du français — classification par type et sous-type
_ALPHABET_IPA: dict[str, dict] = {
    # Voyelles orales
    "i": {"type": "voyelle", "sous_type": "orale"},
    "y": {"type": "voyelle", "sous_type": "orale"},
    "u": {"type": "voyelle", "sous_type": "orale"},
    "e": {"type": "voyelle", "sous_type": "orale"},
    "ø": {"type": "voyelle", "sous_type": "orale"},
    "o": {"type": "voyelle", "sous_type": "orale"},
    "ɛ": {"type": "voyelle", "sous_type": "orale"},
    "œ": {"type": "voyelle", "sous_type": "orale"},
    "ɔ": {"type": "voyelle", "sous_type": "orale"},
    "a": {"type": "voyelle", "sous_type": "orale"},
    "ə": {"type": "voyelle", "sous_type": "orale"},
    # Voyelles nasales
    "ɑ̃": {"type": "voyelle", "sous_type": "nasale"},
    "ɛ̃": {"type": "voyelle", "sous_type": "nasale"},
    "ɔ̃": {"type": "voyelle", "sous_type": "nasale"},
    "œ̃": {"type": "voyelle", "sous_type": "nasale"},
    # Consonnes occlusives
    "p": {"type": "consonne", "sous_type": "occlusive", "voisee": False},
    "b": {"type": "consonne", "sous_type": "occlusive", "voisee": True},
    "t": {"type": "consonne", "sous_type": "occlusive", "voisee": False},
    "d": {"type": "consonne", "sous_type": "occlusive", "voisee": True},
    "k": {"type": "consonne", "sous_type": "occlusive", "voisee": False},
    "ɡ": {"type": "consonne", "sous_type": "occlusive", "voisee": True},
    # Consonnes fricatives
    "f": {"type": "consonne", "sous_type": "fricative", "voisee": False},
    "v": {"type": "consonne", "sous_type": "fricative", "voisee": True},
    "s": {"type": "consonne", "sous_type": "fricative", "voisee": False},
    "z": {"type": "consonne", "sous_type": "fricative", "voisee": True},
    "ʃ": {"type": "consonne", "sous_type": "fricative", "voisee": False},
    "ʒ": {"type": "consonne", "sous_type": "fricative", "voisee": True},
    # Consonnes nasales
    "m": {"type": "consonne", "sous_type": "nasale", "voisee": True},
    "n": {"type": "consonne", "sous_type": "nasale", "voisee": True},
    "ɲ": {"type": "consonne", "sous_type": "nasale", "voisee": True},
    "ŋ": {"type": "consonne", "sous_type": "nasale", "voisee": True},
    # Consonnes liquides
    "l": {"type": "consonne", "sous_type": "liquide", "voisee": True},
    "ʁ": {"type": "consonne", "sous_type": "liquide", "voisee": True},
    # Semi-voyelles
    "j": {"type": "semi-voyelle"},
    "w": {"type": "semi-voyelle"},
    "ɥ": {"type": "semi-voyelle"},
}

# Table phonème → graphèmes possibles (pour l'aligneur)
_PHONE_TO_GRAPHEMES: dict[str, list[str]] = {
    "v": ["v", "w", "f", "f_"],
    "z": ["z", "s", "x", "zz", "s_", "x_"],
    "ʒ": ["j", "g", "ge", "j'"],
    "ʃ": ["ch", "sh", "sch", "sc", "s", "x"],
    "f": ["f", "ff", "ph"],
    "s": ["s", "ss", "c", "ç", "t", "sc", "x", "z", "s'", "sth"],
    "ʁ": ["r", "rr", "rh", "j", "h"],
    "l": ["l", "ll", "l'"],
    "m": ["m", "mm", "m'"],
    "n": ["n", "nn", "n'", "n_"],
    "ɲ": ["gn", "ñ", "nn", "ni", "ny"],
    "ŋ": ["ng", "n", "g"],
    "b": ["b", "bb"],
    "d": ["d", "dd", "z", "j", "g", "t", "d'"],
    "ɡ": ["g", "gu", "gh", "c", "gg", "ggu"],
    "k": ["c", "qu", "k", "x", "cc", "ck", "ch", "q", "kh", "cqu", "cq", "qu'", "g"],
    "p": ["p", "pp", "b", "p_"],
    "t": ["t", "tt", "th", "pt", "gt", "t'", "t_", "d_", "d"],
    "ɥ": ["u", "ü"],
    "j": ["y", "ill", "i", "ï", "î", "y'", "ll", "il", "í"],
    "w": ["o", "ou", "w", "u"],
    "ɛ̃": ["in", "im", "ain", "ein", "en", "ym", "yn", "ïn", "în", "aim", "eim", "un", "um"],
    "ɑ̃": ["an", "am", "en", "em", "aon", "ân"],
    "œ̃": ["un", "um", "eu", "en", "in"],
    "ɔ̃": ["on", "om", "aon", "un", "ôn", "um"],
    "ə": ["e", "on", "ai", "œ", "æ", "u", "o"],
    "y": ["u", "û", "eu", "eû", "ü"],
    "i": ["i", "ï", "y", "î", "ee", "e", "ea", "u"],
    "u": ["ou", "où", "oo", "u", "oû", "ow", "e", "ü"],
    "ø": ["eu", "œu", "e", "ai", "oe", "œ"],
    "e": ["é", "er", "ez", "e", "ed", "et", "ai", "ê", "ay", "aî", "aï", "ë", "oe", "es", "æ", "a", "œ"],
    "o": ["o", "ô", "au", "eau", "aw", "a", "e"],
    "œ": ["eu", "œu", "ue", "u", "e", "i", "oe", "œ", "u"],
    "ɛ": ["è", "ê", "ai", "ei", "ay", "e", "et", "ey", "aî", "aï", "é", "ea", "ë", "a", "es", "est", "êt", "æ"],
    "ɔ": ["o", "au", "u", "a", "ü", "oa", "ô"],
    "a": ["a", "à", "â", "e", "i", "î"],
    # Blocs multi-phonèmes
    "ɡz": ["x²"],
    "ks": ["x²", "xc"],
    "wa": ["oi", "oy", "oê"],
    "wɛ̃": ["oin"],
    "ɥi": ["ui", "uy"],
    "jɛ̃": ["ien"],
    "dʒ": ["j²", "g²"],
    "waj": ["oy²"],
    "ɛj": ["ay²", "ey²", "a²"],
    "ɥij": ["uy²"],
    "tʃ": ["ch", "cc", "tj"],
    "nj": ["gn", "ñ²", "nn"],
    "ɛ̃n": ["in²"],
    "ɑ̃n": ["an²", "en²"],
    "œ̃n": ["un²"],
    "ɔ̃n": ["on²"],
    "jɛ̃n": ["ien²"],
    "wɛ̃n": ["oin²"],
    "ij": ["i²", "y²"],
    "dz": ["z²"],
    "aj": ["y²", "i²"],
    "ej": ["ay²", "ey²", "a²"],
    "ei": ["ay²", "ey²"],
    "œl": ["le"],
    "ju": ["ue"],
    "nju": ["new"],
    "əl": ["le"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Mapping eSpeak → IPA
# ══════════════════════════════════════════════════════════════════════════════

_ESPEAK_TO_IPA: dict[str, str] = {
    "i": "i", "e": "e", "E": "ɛ", "a": "a", "O": "ɔ",
    "o": "o", "u": "u", "y": "y", "Y": "ø", "W": "œ",
    "@": "ə",
    "A~": "ɑ̃", "E~": "ɛ̃", "O~": "ɔ̃", "W~": "œ̃",
    "j": "j", "w": "w", "H": "ɥ",
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
    "f": "f", "v": "v", "s": "s", "z": "z", "S": "ʃ", "Z": "ʒ",
    "m": "m", "n": "n", "n^": "ɲ", "N": "ŋ",
    "l": "l", "R": "ʁ",
}


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


def iter_phonemes(ipa: str) -> list[str]:
    """Découpe une chaîne IPA en phonèmes, regroupant les combining marks Unicode.

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
    """Vrai si le phonème est une voyelle IPA (orale ou nasale)."""
    if not phoneme:
        return False
    if phoneme in _VOYELLES:
        return True
    return bool(phoneme[0] in _VOYELLES)


def est_consonne(phoneme: str) -> bool:
    """Vrai si le phonème est une consonne IPA."""
    return bool(phoneme and phoneme in _CONSONNES)


def est_semi_voyelle(phoneme: str) -> bool:
    """Vrai si le phonème est une semi-voyelle IPA."""
    return bool(phoneme and phoneme in _SEMI_VOYELLES)


# ══════════════════════════════════════════════════════════════════════════════
# Modèles de données — Syllabation de base
# ══════════════════════════════════════════════════════════════════════════════

Span = tuple[int, int]


@dataclass
class Phoneme:
    """Phonème individuel avec correspondance graphème."""

    ipa: str
    grapheme: str = ""


@dataclass
class GroupePhonologique:
    """Groupe de phonèmes (attaque, noyau ou coda d'une syllabe)."""

    phonemes: list[Phoneme] = field(default_factory=list)

    @property
    def phone(self) -> str:
        return "".join(p.ipa for p in self.phonemes)

    @property
    def grapheme(self) -> str:
        return "".join(p.grapheme for p in self.phonemes)


@dataclass
class Syllabe:
    """Syllabe décomposée en attaque/noyau/coda avec correspondance orthographique."""

    phone: str
    ortho: str
    span: Span
    attaque: GroupePhonologique = field(default_factory=GroupePhonologique)
    noyau: GroupePhonologique = field(default_factory=GroupePhonologique)
    coda: GroupePhonologique = field(default_factory=GroupePhonologique)


@dataclass
class ResultatAnalyse:
    """Résultat complet de l'analyse syllabique d'un mot."""

    mot: str
    phone: str
    syllabes: list[Syllabe] = field(default_factory=list)

    @property
    def nb_syllabes(self) -> int:
        return len(self.syllabes)

    def format_simple(self) -> str:
        """Retourne "mot -> /phone/ (n syllabes)"."""
        parts = [s.phone for s in self.syllabes]
        return f"{self.mot} -> /{'.'.join(parts)}/ ({self.nb_syllabes} syll.)"

    def format_detail(self) -> str:
        """Retourne un affichage détaillé avec attaque/noyau/coda."""
        lines = [f"{self.mot} -> /{self.phone}/"]
        for i, s in enumerate(self.syllabes, 1):
            att = s.attaque.phone or "-"
            noy = s.noyau.phone or "-"
            cod = s.coda.phone or "-"
            lines.append(
                f"  σ{i}: /{s.phone}/ <<{s.ortho}>> "
                f"[{s.span[0]}:{s.span[1]}] "
                f"att={att} noy={noy} cod={cod}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Modèles de données — Groupes de lecture (E1) et formules
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MotAnalyse:
    """Mot avec ses annotations G2P.

    Attributs :
        token : Token du tokeniseur (ou None si non disponible)
        phone : IPA du mot (ex: "le", "ɑ̃fɑ̃")
        liaison : Label liaison (none, Lz, Lt, Ln, Lr, Lp)
        pos : POS tag optionnel
    """
    token: object | None = None
    phone: str = ""
    liaison: str = "none"
    pos: str = ""

    @property
    def text(self) -> str:
        """Texte du mot (depuis le token ou chaîne vide)."""
        if self.token is not None and hasattr(self.token, "text"):
            return self.token.text
        return ""

    @property
    def span(self) -> Span:
        """Span du mot (depuis le token ou (0,0))."""
        if self.token is not None and hasattr(self.token, "span"):
            return self.token.span
        return (0, 0)


@dataclass
class EventFormule:
    """Un cran de lecture pour une formule."""
    ortho: str
    phone: str
    span_source: Span = (0, 0)
    span_lecture: Span = (0, 0)


@dataclass
class LectureFormule:
    """Lecture pré-calculée d'un token FORMULE (fournie par numReader)."""
    display_fr: str
    events: list[EventFormule] = field(default_factory=list)


@dataclass
class OptionsGroupes:
    """Options pour la construction des groupes de lecture."""
    gerer_elisions: bool = True
    gerer_liaisons: bool = True
    gerer_enchainement: bool = True
    ajouter_schwas_finaux: bool = False


@dataclass
class GroupeLecture:
    """Groupe de lecture : mots liés par élision, liaison ou enchaînement."""
    mots: list[MotAnalyse] = field(default_factory=list)
    phone_groupe: str = ""
    span: Span = (0, 0)
    jonctions: list[str] = field(default_factory=list)
    est_formule: bool = False
    lecture: LectureFormule | None = None

    @property
    def text(self) -> str:
        """Texte du groupe (concaténation des mots)."""
        return " ".join(m.text for m in self.mots)


@dataclass
class ResultatGroupe:
    """Résultat de syllabation d'un groupe de lecture."""
    groupe: GroupeLecture
    syllabes: list[Syllabe] = field(default_factory=list)


@dataclass
class ResultatSyllabation:
    """Résultat complet de la syllabation avec groupes de lecture."""
    texte_original: str
    groupes: list[ResultatGroupe] = field(default_factory=list)
    options: OptionsGroupes = field(default_factory=OptionsGroupes)

    @property
    def nb_syllabes(self) -> int:
        return sum(len(rg.syllabes) for rg in self.groupes)

    @property
    def nb_groupes(self) -> int:
        return len(self.groupes)

    def format_ligne1(self) -> str:
        """Groupes de lecture (non syllabé)."""
        parts: list[str] = []
        for rg in self.groupes:
            if rg.groupe.est_formule and rg.groupe.lecture:
                parts.append(f"[{rg.groupe.lecture.display_fr}]")
            else:
                parts.append(rg.groupe.text)
        return " | ".join(parts)

    def format_ligne2(self) -> str:
        """Syllabes."""
        parts: list[str] = []
        for rg in self.groupes:
            syl_parts = [s.phone for s in rg.syllabes]
            parts.append(".".join(syl_parts))
        return " | ".join(parts)

    def format_detail(self) -> str:
        """Affichage détaillé des groupes et syllabes."""
        lines: list[str] = []
        lines.append(f"Texte : {self.texte_original}")
        lines.append(f"Groupes : {self.nb_groupes}  Syllabes : {self.nb_syllabes}")
        lines.append(f"Ligne 1 : {self.format_ligne1()}")
        lines.append(f"Ligne 2 : {self.format_ligne2()}")
        lines.append("")
        for gi, rg in enumerate(self.groupes, 1):
            g = rg.groupe
            jonc = ", ".join(g.jonctions) if g.jonctions else "-"
            formule_mark = " [FORMULE]" if g.est_formule else ""
            lines.append(f"  G{gi}: <<{g.text}>>{formule_mark}  /{g.phone_groupe}/  jonctions={jonc}")
            for si, s in enumerate(rg.syllabes, 1):
                att = s.attaque.phone or "-"
                noy = s.noyau.phone or "-"
                cod = s.coda.phone or "-"
                lines.append(
                    f"    σ{si}: /{s.phone}/ <<{s.ortho}>> "
                    f"[{s.span[0]}:{s.span[1]}] "
                    f"att={att} noy={noy} cod={cod}"
                )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Protocol Phonemizer
# ══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class Phonemizer(Protocol):
    """Interface pour brancher n'importe quel phonémiseur."""

    def phonemize(self, word: str) -> str: ...


# ══════════════════════════════════════════════════════════════════════════════
# Backend eSpeak-NG
# ══════════════════════════════════════════════════════════════════════════════


class EspeakPhonemizer:
    """Phonémiseur basé sur eSpeak-NG."""

    def __init__(self, lang: str = "fr") -> None:
        self._lang = lang
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                subprocess.run(
                    ["espeak-ng", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                self._available = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._available = False
                logger.warning("eSpeak-NG not available, phonemization will fail")
        return self._available

    def phonemize(self, word: str) -> str:
        if not self.is_available():
            raise RuntimeError(
                "eSpeak-NG n'est pas installe. "
                "Installez-le (sudo apt install espeak-ng) ou "
                "fournissez un phonemiseur custom."
            )

        proc = subprocess.run(
            ["espeak-ng", "-v", self._lang, "--ipa", "-q", word],
            capture_output=True,
            timeout=5,
        )
        raw = proc.stdout.decode("utf-8", errors="replace").strip()

        cleaned = raw.replace("ˈ", "").replace("ˌ", "").replace("-", "")
        cleaned = cleaned.replace(" ", "")

        return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# Adaptateur pour Lectura G2P (duck typing)
# ══════════════════════════════════════════════════════════════════════════════


class _G2PAdapter:
    """Adapte un objet LecturaG2P (méthode .predict) au Protocol Phonemizer."""

    def __init__(self, g2p_obj: object) -> None:
        if not hasattr(g2p_obj, "predict"):
            raise TypeError(
                "L'objet G2P doit avoir une méthode .predict(word) -> str"
            )
        self._g2p = g2p_obj

    def phonemize(self, word: str) -> str:
        return self._g2p.predict(word)  # type: ignore[union-attr]


# ══════════════════════════════════════════════════════════════════════════════
# Syllabeur (algorithme par sonorité)
# ══════════════════════════════════════════════════════════════════════════════

COMPOUND_PHONEMES: set[str] = {"tʃ", "dʒ", "ts", "ɡz"}


@dataclass(frozen=True)
class _SonorityClasses:
    O: set[str]  # Obstruantes
    N: set[str]  # Nasales
    L: set[str]  # Liquides
    Y: set[str]  # Semi-voyelles
    V: set[str]  # Voyelles


def _build_sonority_classes() -> _SonorityClasses:
    """Construit les 5 classes de sonorité depuis l'alphabet IPA embarqué."""
    O: set[str] = set()
    N: set[str] = set()
    L: set[str] = set()
    Y: set[str] = set()
    V: set[str] = set()

    for ph, meta in _ALPHABET_IPA.items():
        t = meta.get("type")
        if t == "voyelle":
            V.add(ph)
        elif t == "semi-voyelle":
            Y.add(ph)
        elif t == "consonne":
            st = meta.get("sous_type")
            if st == "nasale":
                N.add(ph)
            elif st == "liquide":
                L.add(ph)
            else:
                O.add(ph)

    O.add("s")
    L.add("s")
    O |= COMPOUND_PHONEMES

    return _SonorityClasses(O=O, N=N, L=L, Y=Y, V=V)


# Singleton — les classes ne changent pas
_SONORITY = _build_sonority_classes()


def _class_of(token: str, classes: _SonorityClasses) -> str:
    if token in classes.V or est_voyelle(token):
        return "V"
    if token in classes.Y or est_semi_voyelle(token):
        return "Y"
    if token in classes.L:
        return "L"
    if token in classes.N:
        return "N"
    return "O"


def _merge_compounds_ipa(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            pair = tokens[i] + tokens[i + 1]
            if pair in COMPOUND_PHONEMES:
                out.append(pair)
                i += 2
                continue
        out.append(tokens[i])
        i += 1
    return out


def _split_central_cluster(
    cluster: list[str],
    classes: _SonorityClasses,
    *,
    s_liquid_in_attack: bool = True,
) -> int:
    if len(cluster) <= 1:
        return 0

    left = 0
    right = len(cluster)

    def is_liquid_attack(c: str) -> bool:
        return c in classes.L and (s_liquid_in_attack or c != "s")

    layers = ["Y", "L", "N", "O"]

    for layer in layers:
        if right > left:
            c = cluster[right - 1]
            if layer == "Y" and c in classes.Y:
                right -= 1
            elif layer == "L" and is_liquid_attack(c):
                right -= 1
            elif layer == "N" and c in classes.N:
                right -= 1
            elif layer == "O" and c in classes.O:
                right -= 1

        if right > left:
            c = cluster[left]
            if layer == "Y" and c in classes.Y:
                left += 1
            elif layer == "L" and c in classes.L:
                left += 1
            elif layer == "N" and c in classes.N:
                left += 1
            elif layer == "O" and c in classes.O:
                left += 1

    if right > left:
        i = left
        while (
            i < right
            and _class_of(cluster[i], classes) in {"L", "Y"}
            and cluster[i] != "s"
        ):
            i += 1
        left = i

    return left


def _syllabify_ipa(phone: str) -> list[str]:
    """Découpe une chaîne IPA en syllabes par le modèle de sonorité."""
    tokens = _merge_compounds_ipa(iter_phonemes(phone))

    if not tokens:
        return [phone] if phone else []

    vowel_idx = [i for i, t in enumerate(tokens) if _class_of(t, _SONORITY) == "V"]
    if len(vowel_idx) <= 1:
        return ["".join(tokens)]

    boundaries: list[int] = []
    for vi, vj in zip(vowel_idx, vowel_idx[1:]):
        cluster = tokens[vi + 1 : vj]
        k = _split_central_cluster(cluster, _SONORITY)
        boundaries.append((vi + 1) + k)

    out: list[str] = []
    start = 0
    for b in boundaries:
        out.append("".join(tokens[start:b]))
        start = b
    out.append("".join(tokens[start:]))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Aligneur graphème-phonème
# ══════════════════════════════════════════════════════════════════════════════

_LETTRES_MUETTES_POSSIBLES = {
    "e", "s", "t", "d", "p", "h", "x", "g", "c", "l",
    "m", "n", "b", "f", "r", "v", "z", "q",
}

_IGNORED_CHARS = {"-", " ", "'", "\u2019", "_"}


def _alignement_v2(
    ortho: str,
    phone: str,
    phone_to_graphs: dict[str, list[str]],
) -> tuple[list[str], list[str], bool]:
    ortho_orig = ortho
    ortho_norm = ortho.lower()
    phone_tokens = iter_phonemes(phone.lower())

    bloc_defs = [
        (b, iter_phonemes(b.lower()))
        for b in phone_to_graphs
    ]
    bloc_defs.sort(key=lambda x: len(x[1]), reverse=True)

    def generate_segmentations(max_segmentations: int = 32) -> list[list[str]]:
        segs: list[list[str]] = []

        def dfs_seg(pos: int, current: list[str]) -> None:
            if len(segs) >= max_segmentations:
                return
            if pos == len(phone_tokens):
                segs.append(current[:])
                return
            for bloc, toks in bloc_defs:
                ll = len(toks)
                if phone_tokens[pos : pos + ll] == toks:
                    current.append(bloc)
                    dfs_seg(pos + ll, current)
                    current.pop()
            current.append(phone_tokens[pos])
            dfs_seg(pos + 1, current)
            current.pop()

        dfs_seg(0, [])
        return segs

    segmentations = generate_segmentations()

    all_results: list[tuple[list[str], list[str], int]] = []
    M = len(ortho_orig)

    for elements0 in segmentations:

        def dfs(
            elements: list[str],
            i_el: int,
            pos_ortho: int,
            align_ph: list[str],
            align_gr: list[str],
            pending: str,
            muettes: int,
            allow_split: bool = True,
        ) -> None:
            N = len(elements)

            while pos_ortho < M and ortho_orig[pos_ortho] in _IGNORED_CHARS:
                if align_gr:
                    align_gr[-1] += ortho_orig[pos_ortho]
                pos_ortho += 1

            if i_el == N:
                if pending and align_gr:
                    align_gr[-1] += pending

                if pos_ortho < M and align_gr:
                    for k in range(pos_ortho, M):
                        ch = ortho_orig[k]
                        if ch.lower() in _LETTRES_MUETTES_POSSIBLES:
                            align_gr[-1] += ch + "°"
                            muettes += 1
                        else:
                            return

                all_results.append((align_ph[:], align_gr[:], muettes))
                return

            phon = elements[i_el]

            possible_here = False
            for g in phone_to_graphs.get(phon, []):
                g_norm = g.replace("²", "")
                if ortho_norm.startswith(g_norm, pos_ortho):
                    possible_here = True
                    break

            before = len(all_results)

            for g in phone_to_graphs.get(phon, []):
                g_norm = g.replace("²", "")
                ll = len(g_norm)

                if ortho_norm.startswith(g_norm, pos_ortho):
                    align_ph.append(phon)
                    align_gr.append(pending + g)
                    dfs(
                        elements, i_el + 1, pos_ortho + ll,
                        align_ph, align_gr, "", muettes,
                        allow_split=True,
                    )
                    align_ph.pop()
                    align_gr.pop()

            after = len(all_results)
            produced_solution = after > before

            if allow_split and not produced_solution and not possible_here:
                sub_phons = iter_phonemes(phon.lower())
                if len(sub_phons) > 1:
                    new_elements = (
                        elements[:i_el] + sub_phons + elements[i_el + 1 :]
                    )
                    dfs(
                        new_elements, i_el, pos_ortho,
                        align_ph, align_gr, pending, muettes,
                        allow_split=False,
                    )

            if pos_ortho < M and not possible_here:
                ch = ortho_orig[pos_ortho]
                if ch.lower() in _LETTRES_MUETTES_POSSIBLES:
                    mu = ch + "°"
                    if ch.lower() == "h":
                        dfs(
                            elements, i_el, pos_ortho + 1,
                            align_ph, align_gr, pending + mu, muettes + 1,
                            allow_split=True,
                        )
                    else:
                        if align_gr:
                            align_gr[-1] += mu
                            dfs(
                                elements, i_el, pos_ortho + 1,
                                align_ph, align_gr, pending, muettes + 1,
                                allow_split=True,
                            )
                            align_gr[-1] = align_gr[-1][: -len(mu)]
                        else:
                            dfs(
                                elements, i_el, pos_ortho + 1,
                                align_ph, align_gr, pending + mu, muettes + 1,
                                allow_split=True,
                            )

        dfs(elements0, 0, 0, [], [], "", 0, allow_split=True)

    if not all_results:
        return [], [], False

    def _muette_penalty(gr: list[str]) -> int:
        penalty = 0
        pos = 0
        for g in gr:
            for ch in g:
                if ch == "°":
                    penalty += pos
                else:
                    pos += 1
        return penalty

    def _score(
        sol: tuple[list[str], list[str], int],
    ) -> tuple[int, int, int, int]:
        _, gr, muettes = sol
        pass2_count = sum(1 for g in gr if g not in phone_to_graphs)
        muette_pos_penalty = _muette_penalty(gr)
        return (muettes, muette_pos_penalty, pass2_count, len(gr))

    best = min(all_results, key=_score)
    align_ph, align_gr, _ = best

    # Réinjection de la casse originale
    new_gr: list[str] = []
    idx = 0
    lw = len(ortho_orig)

    for g in align_gr:
        rebuilt = ""
        for ch in g:
            if ch in {"°", "²"}:
                rebuilt += ch
            else:
                if idx < lw:
                    rebuilt += ortho_orig[idx]
                    idx += 1
                else:
                    rebuilt += ch
        new_gr.append(rebuilt)

    return align_ph, new_gr, True


def _build_spans(ortho: str, align_gr: list[str]) -> list[Span]:
    spans: list[Span] = []
    pos = 0
    for gr in align_gr:
        g = gr.replace("²", "").replace("°", "")
        if not g:
            spans.append((pos, pos))
            continue
        end = pos + len(g)
        spans.append((pos, end))
        pos = end
    return spans


def _phonemise_alignment(
    align_ph: list[str],
    align_gr: list[str],
    spans: list[Span],
) -> tuple[list[str], list[str], list[Span]]:
    new_ph: list[str] = []
    new_gr: list[str] = []
    new_spans: list[Span] = []

    for ph, gr, span in zip(align_ph, align_gr, spans):
        tokens = iter_phonemes(ph)
        if len(tokens) == 1:
            new_ph.append(ph)
            new_gr.append(gr)
            new_spans.append(span)
            continue

        first = tokens[0]
        new_ph.append(first)
        new_gr.append(gr[0] if gr else "")
        g0 = gr[0] if gr else ""
        if g0:
            new_spans.append((span[0], span[0] + len(g0)))
        else:
            new_spans.append((span[0], span[0]))

        rest = "".join(tokens[1:])
        if rest:
            new_ph.append(rest)
            new_gr.append(gr[1:] if len(gr) > 1 else "")
            g1 = gr[1:] if len(gr) > 1 else ""
            if g1:
                g1_clean = g1.replace("°", "").replace("²", "")
                new_spans.append((span[0] + len(g0), span[0] + len(g0) + len(g1_clean)))
            else:
                new_spans.append((span[1], span[1]))

    return new_ph, new_gr, new_spans


def _aligner(
    ortho: str,
    phone: str,
) -> tuple[list[str], list[str], list[Span], bool]:
    """Aligne graphèmes et phonèmes."""
    align_ph, align_gr, ok = _alignement_v2(ortho, phone, _PHONE_TO_GRAPHEMES)
    if not ok:
        return [], [], [], False
    spans = _build_spans(ortho, align_gr)
    dec_ph, dec_gr, dec_spans = _phonemise_alignment(align_ph, align_gr, spans)
    return dec_ph, dec_gr, dec_spans, True


# ══════════════════════════════════════════════════════════════════════════════
# Construction des syllabes riches
# ══════════════════════════════════════════════════════════════════════════════


def _decouper_syllabe(syll_phone: str) -> tuple[str, str, str] | None:
    """Décompose une syllabe en (attaque, noyau, coda)."""
    phonemes = iter_phonemes(syll_phone)
    vowel_indices = [i for i, p in enumerate(phonemes) if est_voyelle(p)]
    if not vowel_indices:
        return None
    v = vowel_indices[0]
    return "".join(phonemes[:v]), phonemes[v], "".join(phonemes[v + 1 :])


def _build_syllabes(
    syll_phones: list[str],
    dec_ph: list[str],
    dec_gr: list[str],
    dec_spans: list[Span],
    word_offset: int,
    alignment_ok: bool,
) -> list[Syllabe]:
    """Construit les objets Syllabe depuis la syllabation et l'alignement."""
    syllabes: list[Syllabe] = []
    cursor = 0

    for syll_phone in syll_phones:
        syll_phonemes = iter_phonemes(syll_phone)
        decomp = _decouper_syllabe(syll_phone)

        if not alignment_ok or not dec_ph:
            # Mode dégradé : pas d'alignement
            if decomp is not None:
                att_str, noy_str, cod_str = decomp
                attaque = GroupePhonologique(
                    phonemes=[Phoneme(ipa=p) for p in iter_phonemes(att_str)]
                    if att_str else []
                )
                noyau = GroupePhonologique(phonemes=[Phoneme(ipa=noy_str)])
                coda = GroupePhonologique(
                    phonemes=[Phoneme(ipa=p) for p in iter_phonemes(cod_str)]
                    if cod_str else []
                )
            else:
                attaque = GroupePhonologique(
                    phonemes=[Phoneme(ipa=p) for p in syll_phonemes]
                )
                noyau = GroupePhonologique()
                coda = GroupePhonologique()

            syllabes.append(Syllabe(
                phone=syll_phone, ortho="", span=(0, 0),
                attaque=attaque, noyau=noyau, coda=coda,
            ))
            continue

        # Mode normal : on a l'alignement
        mapped: list[tuple[str, str, Span]] = []
        pos = cursor
        for sph in syll_phonemes:
            if pos < len(dec_ph):
                mapped.append((sph, dec_gr[pos], dec_spans[pos]))
                pos += 1
            else:
                mapped.append((sph, "", (0, 0)))
        cursor = pos

        # Calculer ortho et span de la syllabe
        syll_ortho_parts: list[str] = []
        rel_start: float = float("inf")
        rel_end = 0

        for _, gr, sp in mapped:
            clean_gr = gr.replace("°", "").replace("²", "")
            syll_ortho_parts.append(clean_gr)
            if sp[0] < sp[1]:
                rel_start = min(rel_start, sp[0])
                rel_end = max(rel_end, sp[1])

        if rel_start == float("inf") or rel_start >= rel_end:
            rel_start = 0
            rel_end = 0

        abs_start = word_offset + rel_start
        abs_end = word_offset + rel_end
        syll_ortho = "".join(syll_ortho_parts)

        if decomp is not None:
            att_str, noy_str, cod_str = decomp
            att_phonemes_list = iter_phonemes(att_str) if att_str else []
            cod_phonemes_list = iter_phonemes(cod_str) if cod_str else []

            idx = 0
            att_ph: list[Phoneme] = []
            for p in att_phonemes_list:
                gr = mapped[idx][1].replace("°", "").replace("²", "") if idx < len(mapped) else ""
                att_ph.append(Phoneme(ipa=p, grapheme=gr))
                idx += 1

            noy_ph: list[Phoneme] = []
            gr = mapped[idx][1].replace("°", "").replace("²", "") if idx < len(mapped) else ""
            noy_ph.append(Phoneme(ipa=noy_str, grapheme=gr))
            idx += 1

            cod_ph: list[Phoneme] = []
            for p in cod_phonemes_list:
                gr = mapped[idx][1].replace("°", "").replace("²", "") if idx < len(mapped) else ""
                cod_ph.append(Phoneme(ipa=p, grapheme=gr))
                idx += 1

            attaque = GroupePhonologique(phonemes=att_ph)
            noyau = GroupePhonologique(phonemes=noy_ph)
            coda = GroupePhonologique(phonemes=cod_ph)
        else:
            attaque = GroupePhonologique(
                phonemes=[
                    Phoneme(ipa=m[0], grapheme=m[1].replace("°", "").replace("²", ""))
                    for m in mapped
                ]
            )
            noyau = GroupePhonologique()
            coda = GroupePhonologique()

        syllabes.append(Syllabe(
            phone=syll_phone,
            ortho=syll_ortho,
            span=(int(abs_start), int(abs_end)),
            attaque=attaque,
            noyau=noyau,
            coda=coda,
        ))

    return syllabes


# ══════════════════════════════════════════════════════════════════════════════
# E1 — Groupes de lecture
# ══════════════════════════════════════════════════════════════════════════════

# Consonnes de liaison (IPA)
_LIAISON_CONSONNES: dict[str, str] = {
    "Lz": "z",
    "Lt": "t",
    "Ln": "n",
    "Lr": "ʁ",
    "Lp": "p",
}


def _phone_starts_with_vowel(phone: str) -> bool:
    """Vrai si la chaîne IPA commence par une voyelle."""
    if not phone:
        return False
    phonemes = iter_phonemes(phone)
    if not phonemes:
        return False
    return est_voyelle(phonemes[0])


def _phone_ends_with_consonne(phone: str) -> bool:
    """Vrai si la chaîne IPA finit par une consonne."""
    if not phone:
        return False
    phonemes = iter_phonemes(phone)
    if not phonemes:
        return False
    return est_consonne(phonemes[-1])


def _phone_ends_with_schwa(phone: str) -> bool:
    """Vrai si la chaîne IPA finit par un schwa (ə)."""
    if not phone:
        return False
    phonemes = iter_phonemes(phone)
    return phonemes[-1] == "ə" if phonemes else False


def construire_groupes(
    mots: list[MotAnalyse],
    options: OptionsGroupes | None = None,
) -> list[GroupeLecture]:
    """E1 — Construit les groupes de lecture depuis une liste de mots analysés.

    Parcourt les mots séquentiellement et les regroupe selon :
    - Élisions (l'enfant → 1 groupe)
    - Liaisons (les‿enfants → 1 groupe)
    - Enchaînements (avec‿elle → 1 groupe)
    """
    if options is None:
        options = OptionsGroupes()

    if not mots:
        return []

    groupes: list[GroupeLecture] = []
    current_mots: list[MotAnalyse] = [mots[0]]
    current_phones: list[str] = [mots[0].phone]
    current_jonctions: list[str] = []

    for i in range(1, len(mots)):
        mot_courant = mots[i]
        mot_precedent = mots[i - 1]
        jonction = ""

        # Vérifier si on doit fusionner

        # Élision : le mot précédent finit par apostrophe (on détecte par le texte)
        if options.gerer_elisions:
            prev_text = mot_precedent.text
            if prev_text.endswith("'") or (
                i >= 2 and hasattr(mots[i - 1], "token") and
                mot_precedent.token is not None and
                hasattr(mot_precedent.token, "text")
            ):
                # Vérifier si le token entre les deux est une apostrophe
                # On se base sur la liaison entre les phones
                pass

        # Liaison : mot précédent a un label de liaison et mot courant commence par voyelle
        if options.gerer_liaisons and mot_precedent.liaison != "none":
            if _phone_starts_with_vowel(mot_courant.phone):
                liaison_consonne = _LIAISON_CONSONNES.get(mot_precedent.liaison, "")
                if liaison_consonne:
                    jonction = f"liaison_{liaison_consonne}"
                    current_mots.append(mot_courant)
                    # Le phone de liaison est déjà dans le phone du mot précédent
                    # (appliqué par le G2P via appliquer_liaison())
                    current_phones.append(mot_courant.phone)
                    current_jonctions.append(jonction)
                    continue

        # Enchaînement : consonne finale de mot1 + voyelle initiale de mot2
        if options.gerer_enchainement:
            if (_phone_ends_with_consonne(mot_precedent.phone) and
                    _phone_starts_with_vowel(mot_courant.phone)):
                jonction = "enchainement"
                current_mots.append(mot_courant)
                current_phones.append(mot_courant.phone)
                current_jonctions.append(jonction)
                continue

        # Pas de fusion → fermer le groupe courant et en ouvrir un nouveau
        phone_groupe = "".join(current_phones)
        span_start = current_mots[0].span[0] if current_mots else 0
        span_end = current_mots[-1].span[1] if current_mots else 0
        groupes.append(GroupeLecture(
            mots=current_mots,
            phone_groupe=phone_groupe,
            span=(span_start, span_end),
            jonctions=current_jonctions,
        ))
        current_mots = [mot_courant]
        current_phones = [mot_courant.phone]
        current_jonctions = []

    # Fermer le dernier groupe
    if current_mots:
        phone_groupe = "".join(current_phones)
        span_start = current_mots[0].span[0] if current_mots else 0
        span_end = current_mots[-1].span[1] if current_mots else 0

        # Schwa final
        if options.ajouter_schwas_finaux and _phone_ends_with_schwa(phone_groupe):
            pass  # Le schwa est déjà dans le phone, il sera prononcé

        groupes.append(GroupeLecture(
            mots=current_mots,
            phone_groupe=phone_groupe,
            span=(span_start, span_end),
            jonctions=current_jonctions,
        ))

    return groupes


# ══════════════════════════════════════════════════════════════════════════════
# Conversion G2P → Syllabeur
# ══════════════════════════════════════════════════════════════════════════════


def lecture_depuis_g2p(result: object) -> LectureFormule:
    """Convertit un LectureFormuleResult (G2P) en LectureFormule (Syllabeur).

    Permet de rester indépendant du module G2P : accepte tout objet
    avec attributs display_fr et events (chaque event ayant ortho,
    phone, span_source).
    """
    events_syl: list[EventFormule] = []
    for evt in getattr(result, "events", []):
        events_syl.append(EventFormule(
            ortho=getattr(evt, "ortho", ""),
            phone=getattr(evt, "phone", ""),
            span_source=getattr(evt, "span_source", (0, 0)),
            span_lecture=(0, 0),
        ))
    return LectureFormule(
        display_fr=getattr(result, "display_fr", ""),
        events=events_syl,
    )


def _valider_spans_formule(lecture: LectureFormule) -> None:
    """Valide la cohérence des spans dans une LectureFormule.

    Vérifie que les span_source ne se chevauchent pas de façon incohérente
    et que start <= end pour chaque event.
    """
    for i, evt in enumerate(lecture.events):
        s, e = evt.span_source
        if s > e:
            raise ValueError(
                f"EventFormule #{i} ({evt.ortho!r}) : span_source "
                f"incohérent ({s}, {e}) — start > end"
            )


# ══════════════════════════════════════════════════════════════════════════════
# E2 — Syllabation sur les groupes
# ══════════════════════════════════════════════════════════════════════════════


def _syllabes_depuis_lecture(
    lecture: LectureFormule,
) -> list[Syllabe]:
    """Construit les syllabes à partir d'une LectureFormule.

    Chaque EventFormule = 1 syllabe (mode progressif).
    Pour un mode block, l'appelant pré-fusionne les events par composant
    avant de les passer au Syllabeur.
    """
    syllabes: list[Syllabe] = []
    for evt in lecture.events:
        syllabes.append(Syllabe(
            phone=evt.phone,
            ortho=evt.ortho,
            span=evt.span_source,
            attaque=GroupePhonologique(),
            noyau=GroupePhonologique(phonemes=[Phoneme(ipa=evt.phone)]),
            coda=GroupePhonologique(),
        ))
    return syllabes


def syllabifier_groupes(
    groupes: list[GroupeLecture],
    lectures_formules: dict[int, LectureFormule] | None = None,
) -> list[ResultatGroupe]:
    """E2 — Syllabifie chaque groupe de lecture.

    Pour les groupes de mots : syllabation IPA + alignement.
    Pour les groupes FORMULE avec lecture : événements pré-calculés.
    Chaque EventFormule = 1 syllabe (mode progressif).
    """
    if lectures_formules is None:
        lectures_formules = {}

    resultats: list[ResultatGroupe] = []

    for gi, groupe in enumerate(groupes):
        if groupe.est_formule and groupe.lecture is not None:
            # FORMULE avec lecture pré-calculée
            _valider_spans_formule(groupe.lecture)
            syllabes = _syllabes_depuis_lecture(groupe.lecture)
            resultats.append(ResultatGroupe(groupe=groupe, syllabes=syllabes))
            continue

        if groupe.est_formule and gi in lectures_formules:
            # FORMULE avec lecture fournie par dict
            lecture = lectures_formules[gi]
            _valider_spans_formule(lecture)
            groupe.lecture = lecture
            syllabes = _syllabes_depuis_lecture(lecture)
            resultats.append(ResultatGroupe(groupe=groupe, syllabes=syllabes))
            continue

        # Mot(s) normal(aux) → syllabation standard
        phone = groupe.phone_groupe
        if not phone:
            resultats.append(ResultatGroupe(groupe=groupe, syllabes=[]))
            continue

        # Syllabation IPA
        syll_phones = _syllabify_ipa(phone)

        # Alignement : on utilise le texte combiné du groupe
        ortho = " ".join(m.text for m in groupe.mots)
        # Pour l'alignement, retirer les espaces si le groupe est uni
        if len(groupe.mots) == 1:
            ortho = groupe.mots[0].text
            word_offset = groupe.mots[0].span[0]
        else:
            # Multi-mots : concaténer sans espaces pour l'alignement IPA
            ortho = "".join(m.text for m in groupe.mots)
            word_offset = groupe.mots[0].span[0] if groupe.mots else 0

        dec_ph, dec_gr, dec_spans, ok = _aligner(ortho, phone)
        syllabes = _build_syllabes(syll_phones, dec_ph, dec_gr, dec_spans, word_offset, ok)
        resultats.append(ResultatGroupe(groupe=groupe, syllabes=syllabes))

    return resultats


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale
# ══════════════════════════════════════════════════════════════════════════════


class LecturaSyllabeur:
    """Analyseur syllabique complet du français.

    Combine phonémisation, syllabation par sonorité, alignement
    graphème-phonème, et groupes de lecture (E1+E2).

    Parameters
    ----------
    phonemizer : Phonemizer | None
        Objet avec méthode ``phonemize(word) -> str``.
        Si None, utilise eSpeak-NG par défaut.
        Si l'objet a une méthode ``predict`` (comme LecturaG2P),
        il sera automatiquement adapté.
    """

    def __init__(self, phonemizer: Phonemizer | object | None = None) -> None:
        if phonemizer is None:
            self._phonemizer: Phonemizer = EspeakPhonemizer()
        elif isinstance(phonemizer, Phonemizer):
            self._phonemizer = phonemizer
        elif hasattr(phonemizer, "predict"):
            self._phonemizer = _G2PAdapter(phonemizer)
        elif hasattr(phonemizer, "phonemize"):
            self._phonemizer = phonemizer  # type: ignore[assignment]
        else:
            raise TypeError(
                "Le phonemiseur doit avoir une methode .phonemize(word) ou .predict(word)"
            )

    @classmethod
    def with_espeak(cls, lang: str = "fr") -> LecturaSyllabeur:
        """Crée un syllabeur avec le backend eSpeak-NG."""
        return cls(phonemizer=EspeakPhonemizer(lang=lang))

    # ── API rétrocompatible (identique au syllabeur simple) ──

    def analyze(self, word: str, phone: str | None = None) -> ResultatAnalyse:
        """Analyse syllabique complète d'un mot.

        Parameters
        ----------
        word : str
            Mot français à analyser.
        phone : str | None
            Transcription IPA manuelle.

        Returns
        -------
        ResultatAnalyse
        """
        logger.debug("analyze() word=%r phone=%r", word, phone)
        if phone is None:
            phone = self._phonemizer.phonemize(word)

        syll_phones = _syllabify_ipa(phone)
        dec_ph, dec_gr, dec_spans, ok = _aligner(word, phone)
        syllabes = _build_syllabes(syll_phones, dec_ph, dec_gr, dec_spans, 0, ok)

        return ResultatAnalyse(mot=word, phone=phone, syllabes=syllabes)

    def analyze_text(self, text: str) -> list[ResultatAnalyse]:
        """Analyse syllabique de chaque mot d'un texte."""
        words = re.findall(r"[a-zA-ZÀ-ÿ\u0100-\u024F]+(?:['-][a-zA-ZÀ-ÿ]+)*", text)
        return [self.analyze(w) for w in words]

    def syllabify_ipa(self, phone: str) -> list[str]:
        """Découpage syllabique bas-niveau sur de l'IPA brut.

        >>> syl.syllabify_ipa("ʃɔkɔla")
        ['ʃɔ', 'kɔ', 'la']
        """
        return _syllabify_ipa(phone)

    # ── API complète avec groupes de lecture ──

    def analyser_complet(
        self,
        mots: list[MotAnalyse],
        lectures_formules: dict[int, LectureFormule] | None = None,
        options: OptionsGroupes | None = None,
    ) -> ResultatSyllabation:
        """Analyse complète E1 + E2 : groupes de lecture puis syllabation.

        Parameters
        ----------
        mots : list[MotAnalyse]
            Liste de mots avec annotations G2P (phone, liaison, etc.)
        lectures_formules : dict[int, LectureFormule] | None
            Lectures pré-calculées pour les formules (index groupe -> lecture)
        options : OptionsGroupes | None
            Options de regroupement (élisions, liaisons, enchaînements, schwas)

        Returns
        -------
        ResultatSyllabation
        """
        if options is None:
            options = OptionsGroupes()

        logger.debug("analyser_complet() called with %s mots", len(mots))

        # E1 : construire les groupes de lecture
        groupes = construire_groupes(mots, options)

        # E2 : syllabifier chaque groupe
        resultats_groupes = syllabifier_groupes(
            groupes, lectures_formules,
        )

        # Reconstituer le texte original
        texte = " ".join(m.text for m in mots)

        result = ResultatSyllabation(
            texte_original=texte,
            groupes=resultats_groupes,
            options=options,
        )
        logger.info("analyser_complet() produced %s groupes, %s syllabes",
                     result.nb_groupes, result.nb_syllabes)
        return result

    def construire_groupes(
        self,
        mots: list[MotAnalyse],
        options: OptionsGroupes | None = None,
    ) -> list[GroupeLecture]:
        """E1 seul : construit les groupes de lecture.

        Parameters
        ----------
        mots : list[MotAnalyse]
            Liste de mots avec annotations G2P.
        options : OptionsGroupes | None
            Options de regroupement.

        Returns
        -------
        list[GroupeLecture]
        """
        return construire_groupes(mots, options)

    def syllabifier_groupes(
        self,
        groupes: list[GroupeLecture],
        lectures_formules: dict[int, LectureFormule] | None = None,
    ) -> list[ResultatGroupe]:
        """E2 seul : syllabifie des groupes de lecture.

        Parameters
        ----------
        groupes : list[GroupeLecture]
            Groupes de lecture (sortie de E1).
        lectures_formules : dict[int, LectureFormule] | None
            Lectures pré-calculées pour les formules.

        Returns
        -------
        list[ResultatGroupe]
        """
        return syllabifier_groupes(groupes, lectures_formules)


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    syl = LecturaSyllabeur()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        results = syl.analyze_text(text)
        for r in results:
            print(r.format_detail())
            print()
    else:
        print("Lectura Syllabeur Complet — Mode interactif (Ctrl+C pour quitter)")
        print()
        try:
            while True:
                word = input("Mot > ").strip()
                if not word:
                    continue
                if word.startswith("/"):
                    # Mode IPA direct
                    phone = word[1:].strip()
                    sylls = syl.syllabify_ipa(phone)
                    print(f"  /{phone}/ -> {'.'.join(sylls)} ({len(sylls)} syll.)")
                else:
                    result = syl.analyze(word)
                    print(result.format_detail())
                print()
        except (KeyboardInterrupt, EOFError):
            print()
