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

# --- Post-traitement : confusion CTC pronom + ɛm --------------------
# Phones IPA vocaliques (premier caractere Unicode du token).
_VOWELS_IPA = frozenset("aeɛəioɔuyøœɑ")

# Pronoms sujets en IPA (formes exactes pour le cas 2 : mot separe).
_PRONOUN_IPA = frozenset(("ʒə", "ty", "il", "ɛl", "ɔ̃", "nu", "vu"))

# Prefixes pronominaux pour le cas 1 (mot fusionne).
# Du plus long au plus court pour eviter les correspondances partielles.
_PRONOUN_PREFIXES = ("ʒə", "ty", "il", "ɛl", "nu", "vu", "ɔ̃")


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

    # Post-traitement : corriger les confusions CTC connues
    mots_ipa, liaisons = _postprocess_em_confusion(mots_ipa, liaisons)

    return ParseResult(
        mots_ipa=mots_ipa,
        ponctuation_finale=ponctuation_finale,
        liaisons=liaisons,
    )


# ── Post-traitement ɛm ──────────────────────────────────────────────


def _is_vowel_start(s: str) -> bool:
    """Teste si une chaine IPA commence par un phone vocalique."""
    return bool(s) and s[0] in _VOWELS_IPA


def _postprocess_em_confusion(
    mots_ipa: list[str],
    liaisons: list[str],
) -> tuple[list[str], list[str]]:
    """Corrige la confusion CTC ou le decodeur produit ɛm au lieu de m'.

    Le CTC confond regulierement le clitique m' (me) avec le nom de la
    lettre M (/ɛm/) dans les constructions pronom + m' + verbe, et ne
    produit ni separateur ``|`` ni marqueur d'elision ``[']``.

    Deux cas sont traites :

    1. **Mot fusionne** — le pronom et le clitique sont dans le meme mot.
       ``"tyɛmapɛl"`` → ``["ty", "m'apɛl"]``

    2. **Mot separe** — le pronom est un mot a part, mais le mot suivant
       commence par ``ɛm`` + voyelle.
       ``["ty", "ɛmapɛl"]`` → ``["ty", "m'apɛl"]``
    """
    new_mots: list[str] = []
    new_liaisons: list[str] = []

    for i, mot in enumerate(mots_ipa):
        liaison = liaisons[i] if i < len(liaisons) else ""
        handled = False

        # Cas 1 : mot fusionne — pronom + ɛm + voyelle dans un seul mot
        for prefix in _PRONOUN_PREFIXES:
            marker = prefix + "ɛm"
            if mot.startswith(marker):
                rest = mot[len(marker):]
                if _is_vowel_start(rest):
                    new_mots.append(prefix)
                    new_liaisons.append(liaison)
                    new_mots.append("m'" + rest)
                    new_liaisons.append("")
                    handled = True
                    break

        # Cas 2 : mot separe — le mot precedent est un pronom sujet,
        # le mot courant commence par ɛm + voyelle
        if not handled and mot.startswith("ɛm") and len(mot) > 2:
            rest = mot[2:]
            if _is_vowel_start(rest) and new_mots and new_mots[-1] in _PRONOUN_IPA:
                new_mots.append("m'" + rest)
                new_liaisons.append(liaison)
                handled = True

        if not handled:
            new_mots.append(mot)
            new_liaisons.append(liaison)

    return new_mots, new_liaisons
