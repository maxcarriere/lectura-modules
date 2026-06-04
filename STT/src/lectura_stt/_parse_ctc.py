"""Parsing de la sortie CTC en structure mot-par-mot.

Convertit une chaine IPA brute du decodeur CTC (ex: "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d .")
en une liste de mots IPA avec ponctuation et liaisons.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Marqueurs CTC v2
LIAISON_MARKERS = {"[z]", "[t]", "[n]", "[ʁ]", "[p]"}
ELISION_MARKER = "[']"
COMPOUND_MARKER = "[-]"
PUNCT_TOKENS = {",", ".", "?", "!", "…"}


@dataclass
class ParseResult:
    """Resultat du parsing de la sortie CTC.

    Attributes
    ----------
    mots_ipa : list[str]
        Phones concatenes par mot : ["bɔ̃ʒuʁ", "lə", "mɔ̃d"]
    ponctuation_finale : str
        Ponctuation finale detectee (".", "?", "!", "") ou "".
    liaisons : list[str]
        Liaison avant chaque mot ("z", "t", "n", ...) ou "" si aucune.
        Meme longueur que mots_ipa.
    """

    mots_ipa: list[str] = field(default_factory=list)
    ponctuation_finale: str = ""
    liaisons: list[str] = field(default_factory=list)


def parse_ctc_output(ipa_str: str) -> ParseResult:
    """Parse la sortie CTC en mots IPA + ponctuation + liaisons.

    La sortie CTC utilise les conventions suivantes :
    - Phones separes par des espaces : "b ɔ̃ ʒ u ʁ"
    - ``|`` : separateur de mot (groupe de lecture)
    - ``,``, ``.``, ``?``, ``!``, ``…`` : ponctuation
    - ``[z]``, ``[t]``, ``[n]``, ``[ʁ]``, ``[p]`` : liaisons
    - ``[']`` : elision (clitique attache au mot suivant)
    - ``[-]`` : frontiere de mot compose

    Parameters
    ----------
    ipa_str : str
        Chaine IPA brute du decodeur CTC.

    Returns
    -------
    ParseResult
        Structure contenant les mots IPA, la ponctuation et les liaisons.
    """
    if not ipa_str or not ipa_str.strip():
        return ParseResult()

    tokens = ipa_str.split()
    mots_ipa: list[str] = []
    liaisons: list[str] = []
    ponctuation_finale = ""

    current_phones: list[str] = []
    pending_liaison = ""

    def _flush_word() -> None:
        """Flush le mot courant dans la liste."""
        if current_phones:
            mots_ipa.append("".join(current_phones))
            liaisons.append(pending_liaison)
            current_phones.clear()

    for tok in tokens:
        if tok == "|":
            _flush_word()
            pending_liaison = ""

        elif tok in LIAISON_MARKERS:
            _flush_word()
            pending_liaison = tok[1:-1]  # "[z]" → "z"

        elif tok == ELISION_MARKER:
            # Elision : le clitique precedent est rattache au mot suivant
            # On ajoute une apostrophe au mot courant pour marquer l'elision
            if current_phones:
                # Rattacher le clitique avec apostrophe au mot suivant
                clitic_ipa = "".join(current_phones)
                current_phones.clear()
                # On stocke le clitique avec apostrophe pour le prochain mot
                current_phones.append(clitic_ipa + "'")
            pending_liaison = ""

        elif tok == COMPOUND_MARKER:
            # Frontiere de mot compose : on separe en mots distincts
            _flush_word()
            pending_liaison = ""

        elif tok in PUNCT_TOKENS:
            _flush_word()
            pending_liaison = ""
            # Normaliser … → .
            ponctuation_finale = "." if tok == "…" else tok

        else:
            current_phones.append(tok)

    # Flush le dernier mot
    _flush_word()

    return ParseResult(
        mots_ipa=mots_ipa,
        ponctuation_finale=ponctuation_finale,
        liaisons=liaisons,
    )
