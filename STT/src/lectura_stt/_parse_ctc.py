"""Parsing de la sortie CTC en structure mot-par-mot.

Convertit une chaine IPA brute du decodeur CTC (ex: "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d .")
en une liste de mots IPA avec ponctuation et liaisons.

Deux niveaux de parsing :
  - ``parse_ctc_output`` : retourne un ParseResult (mots_ipa + liaisons + ponctuation)
  - ``parse_ctc_v2``     : retourne une liste de segments enrichis (type, liaison,
    elision, compound) pour le pipeline avance

Copyright (C) 2025-2026 Max Carriere
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
            # Elision : le clitique precedent est rattache au mot suivant.
            # Un clitique valide est toujours un seul phone (l, d, ʒ, k, m,
            # n, s, t).  Si le CTC a fusionne le mot precedent avec le
            # clitique (ex: "ɛ m [']" au lieu de "ʒ ə | m [']"), on separe :
            # seul le dernier phone est le clitique, les phones precedents
            # forment un mot distinct.
            if current_phones:
                if len(current_phones) > 1:
                    clitic_phone = current_phones[-1]
                    mots_ipa.append("".join(current_phones[:-1]))
                    liaisons.append(pending_liaison)
                    current_phones.clear()
                    current_phones.append(clitic_phone + "'")
                else:
                    clitic_ipa = current_phones[0]
                    current_phones.clear()
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


def parse_ctc_v2(ipa_str: str) -> list[dict]:
    """Parse une sequence CTC v2 avec marqueurs explicites.

    Retourne une liste de segments enrichis ::

        {"type": "word", "ipa": "il"}
        {"type": "word", "ipa": "ɑ̃fɑ̃", "liaison_before": "z"}
        {"type": "word", "ipa": "akademi", "elision_before": True}
        {"type": "word", "ipa": "ɡʁɑ̃", "compound_after": True}
        {"type": "punct", "value": "."}

    Conventions CTC v2 :
        - ``|``                : separateur de mot (groupe de lecture)
        - ``[z]``, ``[t]``... : liaisons
        - ``[']``             : elision (clitique → mot suivant)
        - ``[-]``             : frontiere intra-compose
        - ``,``, ``.``, ``?``, ``!``, ``…`` : ponctuation
    """
    if not ipa_str or not ipa_str.strip():
        return []

    tokens = ipa_str.split()
    segments: list[dict] = []
    current_phones: list[str] = []
    pending_liaison: str | None = None
    pending_elision = False

    def _flush(extra: dict | None = None) -> None:
        if current_phones:
            seg: dict = {"type": "word", "ipa": "".join(current_phones)}
            if pending_liaison:
                seg["liaison_before"] = pending_liaison
            if pending_elision:
                seg["elision_before"] = True
            if extra:
                seg.update(extra)
            segments.append(seg)
            current_phones.clear()

    for tok in tokens:
        if tok == "|":
            _flush()
            pending_liaison = None
            pending_elision = False

        elif tok in LIAISON_MARKERS:
            _flush()
            pending_liaison = tok[1:-1]
            pending_elision = False

        elif tok == ELISION_MARKER:
            if current_phones:
                if len(current_phones) > 1:
                    # CTC a fusionne le mot precedent avec le clitique
                    # (ex: "ɛ m [']" → separe en mot "ɛ" + clitique "m")
                    clitic_phone = current_phones.pop()
                    _flush()
                    seg_clitic: dict = {"type": "word", "ipa": clitic_phone,
                                        "is_clitic": True}
                    segments.append(seg_clitic)
                else:
                    seg_clitic2: dict = {"type": "word", "ipa": current_phones[0],
                                         "is_clitic": True}
                    segments.append(seg_clitic2)
                    current_phones.clear()
            pending_elision = True
            pending_liaison = None

        elif tok == COMPOUND_MARKER:
            _flush({"compound_after": True})
            pending_liaison = None
            pending_elision = False

        elif tok in PUNCT_TOKENS:
            _flush()
            pending_liaison = None
            pending_elision = False
            val = "." if tok == "…" else tok
            segments.append({"type": "punct", "value": val})

        else:
            current_phones.append(tok)

    # Flush le dernier mot
    if current_phones:
        seg_final: dict = {"type": "word", "ipa": "".join(current_phones)}
        if pending_liaison:
            seg_final["liaison_before"] = pending_liaison
        if pending_elision:
            seg_final["elision_before"] = True
        segments.append(seg_final)

    return segments
