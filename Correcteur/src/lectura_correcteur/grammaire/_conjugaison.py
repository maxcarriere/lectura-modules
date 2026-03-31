"""Regles de conjugaison : pronom sujet + verbe.

Simplifie par rapport au POC : pas de lookup par phone (pas de IPA).
Correction par suffixe direct.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import PRONOM_PERSONNE, SUJETS_3PL

# Terminaisons attendues par personne (indicatif present, 1er groupe)
_SUFFIXES_ATTENDUS: dict[str, list[str]] = {
    "1": ["e", "s"],      # je mange, je finis
    "2": ["es", "s"],     # tu manges, tu finis
    "3": ["e", "t", "d"],  # il mange, il finit, il prend
    "1p": ["ons"],        # nous mangeons
    "2p": ["ez"],         # vous mangez
    "3p": ["ent"],        # ils mangent
}


def verifier_conjugaisons(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Applique les regles de conjugaison sur la phrase.

    Regles :
    3. ils/elles + VER en -e -> -ent
    5. Pronom sujet + VER -> forcer conjugaison correcte (par suffixe)
    """
    if not mots:
        return mots, []

    origs = originaux if originaux else mots
    result = list(mots)
    corrections: list[Correction] = []
    n = len(result)

    for i in range(n):
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]

        # Regle 3 : ils/elles + VER en -e -> -ent
        if i > 0 and pos in ("VER", "AUX"):
            prev_is_3pl = (
                result[i - 1].lower() in SUJETS_3PL
                or (i - 1 < len(origs) and origs[i - 1].lower() in SUJETS_3PL)
            )
            if (
                prev_is_3pl
                and curr.endswith("e")
                and not curr.endswith(("ent", "nt"))
            ):
                candidate = curr + "nt"
                if lexique is None or lexique.existe(candidate):
                    ancien = result[i]
                    result[i] = candidate
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=candidate,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="ils/elles + verbe -> -ent",
                    ))
                    continue

        # Regle 5 simplifie : Pronom sujet + VER -> correction par suffixe
        if i > 0 and pos in ("VER", "AUX"):
            pronom_info = _trouver_pronom_sujet(result, origs, i)
            if pronom_info is not None:
                personne, nombre = pronom_info
                correction = _corriger_par_suffixe(
                    curr, personne, nombre, lexique,
                )
                if correction and correction.lower() != curr.lower():
                    ancien = result[i]
                    result[i] = transferer_casse(curr, correction)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication=f"Conjugaison P{personne}",
                    ))

    return result, corrections


def _trouver_pronom_sujet(
    mots: list[str], origs: list[str], idx_verbe: int,
) -> tuple[str, str] | None:
    """Cherche le pronom sujet le plus proche avant le verbe."""
    for j in range(idx_verbe - 1, max(-1, idx_verbe - 4), -1):
        mot = mots[j].lower()
        orig = origs[j].lower() if j < len(origs) else ""
        for candidate in (mot, orig):
            if candidate in PRONOM_PERSONNE:
                return PRONOM_PERSONNE[candidate]
    return None


def _corriger_par_suffixe(
    mot: str, personne: str, nombre: str, lexique,
) -> str | None:
    """Corrige la conjugaison par ajustement de suffixe.

    Pour la 2e personne : "mange" -> "manges"
    Pour la 3e pluriel : "mange" -> "mangent"
    """
    low = mot.lower()
    key = personne + nombre  # ex: "3p", "2", "1"

    # Tu + verbe en -e sans -s final -> ajouter -s
    if key == "2" and low.endswith("e") and not low.endswith("es"):
        candidate = mot + "s"
        if lexique is None or lexique.existe(candidate):
            return candidate

    # 3e pluriel : -e -> -ent (deja gere par regle 3, mais cas ou prev != ils/elles)
    if key == "3p" and low.endswith("e") and not low.endswith(("ent", "nt")):
        candidate = mot + "nt"
        if lexique is None or lexique.existe(candidate):
            return candidate

    return None
