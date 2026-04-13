"""Post-traitement pour le modèle unifié.

- Corrections G2P POS-aware : homographes (mot+POS), table plate, règles
- Dénasalisation pour les liaisons
- Reconstruction IPA finale avec liaison
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from lectura_nlp.utils.ipa import iter_phonemes, est_voyelle, est_consonne
from lectura_nlp._chargeur import (
    denasalisation as _load_denas,
    liaison_consonnes as _load_liaison_consonnes,
)

_DENAS_MAP = _load_denas()
_LIAISON_CONSONNES = _load_liaison_consonnes()


def appliquer_denasalisation(ipa: str, denas: str) -> str:
    """Applique la dénasalisation si spécifiée.

    denas format : "ɔ̃>ɔ" (voyelle_nasale > voyelle_orale)
    """
    if not denas or ">" not in denas:
        return ipa
    src, dst = denas.split(">", 1)
    return ipa.replace(src, dst)


def appliquer_liaison(
    tokens: list[str],
    phones: list[str],
    liaisons: list[str],
    denas_list: list[str] | None = None,
) -> list[str]:
    """Applique les liaisons entre tokens consécutifs.

    Pour chaque token avec une liaison non-"none", si le token suivant
    commence par une voyelle, on ajoute la consonne de liaison au phone
    du token courant.

    Args:
        tokens: Liste des formes
        phones: Liste des IPA par token
        liaisons: Liste des labels liaison par token
        denas_list: Liste optionnelle des dénasalisations par token

    Returns:
        Liste des IPA modifiés avec liaisons appliquées
    """
    if not tokens:
        return phones

    result = list(phones)
    n = len(tokens)

    for i in range(n - 1):
        liaison = liaisons[i]
        if liaison == "none" or liaison not in _LIAISON_CONSONNES:
            continue

        # Le token suivant commence-t-il par une voyelle ?
        next_phone = result[i + 1]
        if not next_phone:
            continue
        next_phonemes = iter_phonemes(next_phone)
        if not next_phonemes or not est_voyelle(next_phonemes[0]):
            continue

        # Appliquer dénasalisation si nécessaire
        if denas_list and denas_list[i]:
            result[i] = appliquer_denasalisation(result[i], denas_list[i])

        # Ajouter la consonne de liaison
        consonne = _LIAISON_CONSONNES[liaison]
        result[i] = result[i] + consonne

    return result


# ── Voyelles ortho pour détection ex+voyelle ────────────────────────────

_VOYELLES_ORTHO = set("aeiouyàâäéèêëïîôùûüæœ")

# ── Règles G2P ──────────────────────────────────────────────────────────


def _fix_ex_consonne(word: str, ipa: str) -> str:
    """ex + consonne : insérer 's' si pred = ɛk + consonne.

    Exemples : expression ɛkpʁɛsjɔ̃ → ɛkspʁɛsjɔ̃, extrême ɛktʁɛm → ɛkstʁɛm
    """
    w = word.lower()
    if len(w) < 3 or w[:2] != "ex":
        return ipa
    # ex doit être suivi d'une consonne (pas une voyelle)
    if w[2] in _VOYELLES_ORTHO:
        return ipa
    phonemes = iter_phonemes(ipa)
    if len(phonemes) < 3:
        return ipa
    # pred doit commencer par ɛk + consonne (pas déjà ɛks)
    if phonemes[0] != "ɛ" or phonemes[1] != "k":
        return ipa
    if phonemes[2] == "s":
        return ipa  # déjà correct
    if not est_consonne(phonemes[2]):
        return ipa
    # Insérer s après k
    phonemes.insert(2, "s")
    return "".join(phonemes)


def _fix_ex_voyelle(word: str, ipa: str) -> str:
    """ex + voyelle : remplacer ɛk par ɛɡz ou ez par ɛɡz.

    Exemples : exemple ɛkɑ̃pl → ɛɡzɑ̃pl, existence ezistɑ̃s → ɛɡzistɑ̃s
    """
    w = word.lower()
    if len(w) < 3 or w[:2] != "ex":
        return ipa
    # ex doit être suivi d'une voyelle
    if w[2] not in _VOYELLES_ORTHO:
        return ipa
    phonemes = iter_phonemes(ipa)
    if len(phonemes) < 2:
        return ipa
    # Pattern 1 : pred commence par ɛk → remplacer par ɛɡz
    if phonemes[0] == "ɛ" and phonemes[1] == "k":
        return "ɛɡz" + "".join(phonemes[2:])
    # Pattern 2 : pred commence par ez → remplacer par ɛɡz
    if phonemes[0] == "e" and phonemes[1] == "z":
        return "ɛɡz" + "".join(phonemes[2:])
    # Pattern 3 : pred commence par ɛz → remplacer par ɛɡz
    if phonemes[0] == "ɛ" and phonemes[1] == "z":
        return "ɛɡz" + "".join(phonemes[2:])
    return ipa


# Regex ortho : obstruent + liquide + i + voyelle
_YOD_ORTHO_RE = re.compile(r"[bcdfgkpstvz][rl]i[aeéèêëiîoôuûüyà]")
_LIQUIDES_IPA = {"l", "ʁ"}
_VOYELLES_IPA = {
    "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə",
}


def _fix_yod(word: str, ipa: str) -> str:
    """Insérer i manquant : obstruent+liquide+j+voyelle → +i avant j.

    Exemples : oublier ublje → ublije, crier kʁje → kʁije
    """
    if not _YOD_ORTHO_RE.search(word.lower()):
        return ipa
    phonemes = iter_phonemes(ipa)
    if len(phonemes) < 3:
        return ipa
    result = []
    modified = False
    i = 0
    while i < len(phonemes):
        ph = phonemes[i]
        # Chercher séquence liquide + j + voyelle (sans i avant j)
        if (
            ph in _LIQUIDES_IPA
            and i + 1 < len(phonemes)
            and phonemes[i + 1] == "j"
        ):
            # Vérifier que ce n'est pas déjà précédé de i
            if not result or result[-1] != "i":
                # Vérifier qu'une voyelle suit j (ou j est en fin)
                if i + 2 < len(phonemes):
                    base = phonemes[i + 2][0] if phonemes[i + 2] else ""
                    if base in _VOYELLES_IPA:
                        result.append(ph)
                        result.append("i")  # insérer i avant j
                        modified = True
                        i += 1
                        continue
        result.append(ph)
        i += 1
    return "".join(result) if modified else ipa


def appliquer_regles_g2p(word: str, ipa: str) -> str:
    """Applique toutes les règles G2P en séquence."""
    ipa = _fix_ex_consonne(word, ipa)
    ipa = _fix_ex_voyelle(word, ipa)
    ipa = _fix_yod(word, ipa)
    return ipa


# ── Table de corrections ────────────────────────────────────────────────

_CORRECTIONS_TABLE: dict[str, str] | None = None


def charger_corrections(path: str | Path) -> None:
    """Charge la table JSON mot→IPA."""
    global _CORRECTIONS_TABLE
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            _CORRECTIONS_TABLE = json.load(f)
    else:
        _CORRECTIONS_TABLE = None


# ── Table d'homographes (POS-aware) ───────────────────────────────────

_HOMOGRAPHES_TABLE: dict[str, dict[str, str]] | None = None


def charger_homographes(path: str | Path) -> None:
    """Charge la table JSON mot→{POS→IPA} pour les homographes."""
    global _HOMOGRAPHES_TABLE, _CORRECTIONS_TABLE
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            _HOMOGRAPHES_TABLE = json.load(f)
        # Retirer de la table corrections plate les mots présents
        # dans la table homographes (éviter conflit)
        if _CORRECTIONS_TABLE and _HOMOGRAPHES_TABLE:
            for mot in _HOMOGRAPHES_TABLE:
                _CORRECTIONS_TABLE.pop(mot, None)
    else:
        _HOMOGRAPHES_TABLE = None


def corriger_g2p(word: str, ipa: str, pos: str | None = None) -> str:
    """Pipeline complet : 1) homographes  2) table plate  3) règles."""
    w = word.lower()
    # 1) Homographes (prioritaire si POS fourni)
    if _HOMOGRAPHES_TABLE and w in _HOMOGRAPHES_TABLE:
        entry = _HOMOGRAPHES_TABLE[w]
        if pos and pos in entry:
            return entry[pos]
    # 2) Table plate
    if _CORRECTIONS_TABLE and w in _CORRECTIONS_TABLE:
        return _CORRECTIONS_TABLE[w]
    # 3) Règles
    ipa = appliquer_regles_g2p(word, ipa)
    return ipa
