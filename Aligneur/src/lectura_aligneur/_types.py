"""Types de donnees de l'Aligneur-Syllabeur.

Dataclasses et type aliases utilises dans l'API publique.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Span = tuple[int, int]


# ══════════════════════════════════════════════════════════════════════════════
# Syllabation de base
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Phoneme:
    """Phoneme individuel avec correspondance grapheme."""

    ipa: str
    grapheme: str = ""


@dataclass
class GroupePhonologique:
    """Groupe de phonemes (attaque, noyau ou coda d'une syllabe)."""

    phonemes: list[Phoneme] = field(default_factory=list)

    @property
    def phone(self) -> str:
        return "".join(p.ipa for p in self.phonemes)

    @property
    def grapheme(self) -> str:
        return "".join(p.grapheme for p in self.phonemes)


@dataclass
class Syllabe:
    """Syllabe decomposee en attaque/noyau/coda avec correspondance orthographique."""

    phone: str
    ortho: str
    span: Span
    attaque: GroupePhonologique = field(default_factory=GroupePhonologique)
    noyau: GroupePhonologique = field(default_factory=GroupePhonologique)
    coda: GroupePhonologique = field(default_factory=GroupePhonologique)


@dataclass
class ResultatAnalyse:
    """Resultat complet de l'analyse syllabique d'un mot."""

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
        """Retourne un affichage detaille avec attaque/noyau/coda."""
        lines = [f"{self.mot} -> /{self.phone}/"]
        for i, s in enumerate(self.syllabes, 1):
            att = s.attaque.phone or "-"
            noy = s.noyau.phone or "-"
            cod = s.coda.phone or "-"
            lines.append(
                f"  \u03c3{i}: /{s.phone}/ <<{s.ortho}>> "
                f"[{s.span[0]}:{s.span[1]}] "
                f"att={att} noy={noy} cod={cod}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Groupes de lecture (E1) et formules
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MotAnalyse:
    """Mot avec ses annotations G2P.

    Attributs :
        token : Token du tokeniseur (ou None si non disponible)
        phone : IPA du mot (ex: "le", "\u0251\u0303f\u0251\u0303")
        liaison : Label liaison (none, Lz, Lt, Ln, Lr, Lp)
        pos : POS tag optionnel
    """
    token: object | None = None
    phone: str = ""
    liaison: str = "none"
    pos: str = ""
    ponctuation_avant: bool = False
    elision_avant: bool = False
    est_formule: bool = False

    @property
    def text(self) -> str:
        """Texte du mot (depuis le token ou chaine vide)."""
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
    """Lecture pre-calculee d'un token FORMULE (fournie par numReader)."""
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
    """Groupe de lecture : mots lies par elision, liaison ou enchainement."""
    mots: list[MotAnalyse] = field(default_factory=list)
    phone_groupe: str = ""
    span: Span = (0, 0)
    jonctions: list[str] = field(default_factory=list)
    est_formule: bool = False
    lecture: LectureFormule | None = None

    @property
    def text(self) -> str:
        """Texte du groupe (concatenation des mots)."""
        return " ".join(m.text for m in self.mots)


@dataclass
class ResultatGroupe:
    """Resultat de syllabation d'un groupe de lecture."""
    groupe: GroupeLecture
    syllabes: list[Syllabe] = field(default_factory=list)


@dataclass
class ResultatSyllabation:
    """Resultat complet de la syllabation avec groupes de lecture."""
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
        """Groupes de lecture (non syllabe)."""
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
        """Affichage detaille des groupes et syllabes."""
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
                    f"    \u03c3{si}: /{s.phone}/ <<{s.ortho}>> "
                    f"[{s.span[0]}:{s.span[1]}] "
                    f"att={att} noy={noy} cod={cod}"
                )
        return "\n".join(lines)
