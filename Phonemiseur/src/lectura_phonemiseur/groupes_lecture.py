"""Groupes de lecture — regroupement phonetique (elisions, liaisons, enchainements).

Opere sur la sortie du pipeline G2P (ResultatPhraseG2P) pour construire
les groupes de lecture avant la syllabation.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lectura_phonemiseur._chargeur import liaison_consonnes as _get_liaison_consonnes
from lectura_phonemiseur.utils.ipa import est_consonne, est_voyelle, iter_phonemes

# Import conditionnel pour ne pas casser si pipeline_formules n'est pas dispo
try:
    from lectura_phonemiseur.pipeline_formules import MotAnalyseG2P, ResultatPhraseG2P
except ImportError:  # pragma: no cover
    MotAnalyseG2P = None  # type: ignore[assignment,misc]
    ResultatPhraseG2P = None  # type: ignore[assignment,misc]


# ══════════════════════════════════════════════════════════════════════════════
# Types
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OptionsGroupes:
    """Options pour la construction des groupes de lecture."""

    gerer_elisions: bool = True
    gerer_liaisons: bool = True
    gerer_enchainement: bool = True
    ajouter_schwas_finaux: bool = False


@dataclass
class GroupeLecture:
    """Groupe de lecture : mots lies par elision, liaison ou enchainement.

    Attributs :
        mots : liste de MotAnalyseG2P composant le groupe
        phone_groupe : IPA concatene du groupe
        span : (debut, fin) en caracteres dans le texte original
        jonctions : type de jonction entre les mots ('elision', 'liaison_z', 'enchainement')
        est_formule : True si le groupe contient une formule
        lecture : lecture pre-calculee d'une formule (si applicable)
    """

    mots: list = field(default_factory=list)
    phone_groupe: str = ""
    span: tuple[int, int] = (0, 0)
    jonctions: list[str] = field(default_factory=list)
    est_formule: bool = False
    lecture: object | None = None

    @property
    def text(self) -> str:
        """Texte du groupe (concatenation des mots)."""
        return " ".join(m.text for m in self.mots)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers phonetiques
# ══════════════════════════════════════════════════════════════════════════════


def _phone_starts_with_vowel(phone: str) -> bool:
    """Vrai si la chaine IPA commence par une voyelle."""
    if not phone:
        return False
    phonemes = iter_phonemes(phone)
    if not phonemes:
        return False
    return est_voyelle(phonemes[0])


def _phone_ends_with_consonne(phone: str) -> bool:
    """Vrai si la chaine IPA finit par une consonne."""
    if not phone:
        return False
    phonemes = iter_phonemes(phone)
    if not phonemes:
        return False
    return est_consonne(phonemes[-1])


def _phone_ends_with_schwa(phone: str) -> bool:
    """Vrai si la chaine IPA finit par un schwa (ə)."""
    if not phone:
        return False
    phonemes = iter_phonemes(phone)
    return phonemes[-1] == "ə" if phonemes else False


# ── Schwas pedagogiques ──────────────────────────────────────────────────────

# Ortho finit par e/es (sauf é/è/ê/ë)
_RE_E_MUET = re.compile(r"(?<![éèêë])es?$", re.IGNORECASE)
# Verbe finit par -ent (3e pers. pluriel)
_RE_VERB_ENT = re.compile(r"ent$", re.IGNORECASE)


def ajouter_schwa_final(ortho: str, pos: str, phone: str) -> str:
    """Ajoute un ə pedagogique final si le mot a un e-muet non prononce.

    Conditions :
    - ortho finit par e/es (sauf é/è/ê/ë) OU ortho finit par -ent avec POS=VER
    - ET l'IPA ne finit pas deja par une voyelle
    """
    if not phone or not ortho:
        return phone

    has_e_muet = bool(_RE_E_MUET.search(ortho))
    has_verb_ent = (
        bool(_RE_VERB_ENT.search(ortho))
        and isinstance(pos, str)
        and pos.upper().startswith("VER")
    )

    if not (has_e_muet or has_verb_ent):
        return phone

    phonemes = iter_phonemes(phone)
    if phonemes and est_voyelle(phonemes[-1]):
        return phone

    return phone + "ə"


# ══════════════════════════════════════════════════════════════════════════════
# Construction des groupes de lecture
# ══════════════════════════════════════════════════════════════════════════════

_LIAISON_CONSONNES = _get_liaison_consonnes


def construire_groupes_lecture(
    result: object,
    options: OptionsGroupes | None = None,
) -> list[GroupeLecture]:
    """Construit les groupes de lecture depuis un ResultatPhraseG2P.

    Parcourt les mots sequentiellement et les regroupe selon :
    - Elisions (l'enfant → 1 groupe)
    - Liaisons (les‿enfants → 1 groupe)
    - Enchainements (avec‿elle → 1 groupe)

    Parameters
    ----------
    result : ResultatPhraseG2P
        Sortie de ``analyser_phrase_complete()``.
    options : OptionsGroupes | None
        Options de regroupement.

    Returns
    -------
    list[GroupeLecture]
    """
    if options is None:
        options = OptionsGroupes()

    mots = getattr(result, "mots", [])
    if not mots:
        return []

    groupes: list[GroupeLecture] = []
    current_mots = [mots[0]]
    current_phones: list[str] = [mots[0].phone]
    current_jonctions: list[str] = []

    for i in range(1, len(mots)):
        mot_courant = mots[i]
        mot_precedent = mots[i - 1]

        # Deduire ponctuation_avant : soit le mot precedent est une ponctuation,
        # soit le mot courant a un attribut explicite ponctuation_avant
        ponctuation_avant = (
            getattr(mot_precedent, "est_ponctuation", False)
            or getattr(mot_courant, "ponctuation_avant", False)
        )

        # Deduire elision_avant : le mot precedent finit par une apostrophe,
        # ou le mot courant a un attribut explicite elision_avant
        elision_avant = (
            mot_precedent.text.endswith("'")
            or getattr(mot_courant, "elision_avant", False)
        )

        # La ponctuation ou une formule interdit toute fusion entre les mots
        est_ponctuation = getattr(mot_courant, "est_ponctuation", False)
        est_formule_courant = getattr(mot_courant, "est_formule", False)
        est_formule_precedent = getattr(mot_precedent, "est_formule", False)

        if ponctuation_avant or est_ponctuation or est_formule_courant or est_formule_precedent:
            pass  # tombe dans le « pas de fusion » ci-dessous
        else:
            # Elision : apostrophe entre les deux mots (m'appelle, l'enfant)
            if options.gerer_elisions and elision_avant:
                current_mots.append(mot_courant)
                current_phones.append(mot_courant.phone)
                current_jonctions.append("elision")
                continue

            # Liaison : mot precedent a un label de liaison et mot courant commence par voyelle
            liaison_precedent = getattr(mot_precedent, "liaison", "")
            if options.gerer_liaisons and liaison_precedent and liaison_precedent != "none":
                if _phone_starts_with_vowel(mot_courant.phone):
                    liaison_consonne = _LIAISON_CONSONNES().get(liaison_precedent, "")
                    if liaison_consonne:
                        current_mots.append(mot_courant)
                        current_phones.append(mot_courant.phone)
                        current_jonctions.append(f"liaison_{liaison_consonne}")
                        continue

            # Enchainement : consonne finale de mot1 + voyelle initiale de mot2
            if options.gerer_enchainement:
                if (_phone_ends_with_consonne(mot_precedent.phone)
                        and _phone_starts_with_vowel(mot_courant.phone)):
                    current_mots.append(mot_courant)
                    current_phones.append(mot_courant.phone)
                    current_jonctions.append("enchainement")
                    continue

        # Pas de fusion → fermer le groupe courant et en ouvrir un nouveau
        _fermer_groupe(groupes, current_mots, current_phones, current_jonctions)
        current_mots = [mot_courant]
        current_phones = [mot_courant.phone]
        current_jonctions = []

    # Fermer le dernier groupe
    if current_mots:
        _fermer_groupe(groupes, current_mots, current_phones, current_jonctions)

    return groupes


def _fermer_groupe(
    groupes: list[GroupeLecture],
    mots: list,
    phones: list[str],
    jonctions: list[str],
) -> None:
    """Ferme un groupe de lecture et l'ajoute a la liste."""
    phone_groupe = "".join(phones)
    is_formule = any(getattr(m, "est_formule", False) for m in mots)

    # Recuperer la lecture de formule si present
    lecture = None
    for m in mots:
        lec = getattr(m, "lecture", None)
        if lec is not None:
            lecture = lec
            break

    # Calculer le span du groupe depuis les spans des mots
    first_span = getattr(mots[0], "span", (0, 0))
    last_span = getattr(mots[-1], "span", (0, 0))
    group_span = (first_span[0], last_span[1])

    groupes.append(GroupeLecture(
        mots=list(mots),
        phone_groupe=phone_groupe,
        span=group_span,
        jonctions=list(jonctions),
        est_formule=is_formule,
        lecture=lecture,
    ))
