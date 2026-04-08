"""Sous-module grammaire : regles d'accord, conjugaison, homophones, PP, negation."""

from lectura_correcteur.grammaire._accord import verifier_accords
from lectura_correcteur.grammaire._conjugaison import verifier_conjugaisons
from lectura_correcteur.grammaire._homophones import verifier_homophones
from lectura_correcteur.grammaire._negation import verifier_negation
from lectura_correcteur.grammaire._participe import (
    verifier_participes_passes,
    verifier_pp_accord_etre,
)

__all__ = [
    "appliquer_grammaire",
    "verifier_accords",
    "verifier_conjugaisons",
    "verifier_homophones",
    "verifier_negation",
    "verifier_participes_passes",
    "verifier_pp_accord_etre",
]


def appliquer_grammaire(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list]:
    """Applique toutes les regles grammaticales sur la phrase.

    Returns:
        Tuple (mots_corriges, liste_de_Correction).
    """
    from lectura_correcteur._types import Correction

    result = list(mots)
    origs = originaux if originaux else mots
    corrections: list[Correction] = []

    # 1. Accords (det+nom, det+adj+nom, det+nom+ver, genre)
    result_acc, corr_acc = verifier_accords(
        result, pos_tags, morpho, lexique, origs,
    )
    result = result_acc
    corrections.extend(corr_acc)

    # 2. Conjugaisons (pronom+verbe)
    result_conj, corr_conj = verifier_conjugaisons(
        result, pos_tags, morpho, lexique, origs,
    )
    result = result_conj
    corrections.extend(corr_conj)

    # 3. Participes passes (auxiliaire + infinitif -> PP)
    result_pp, corr_pp = verifier_participes_passes(
        result, pos_tags, morpho, lexique, origs,
    )
    result = result_pp
    corrections.extend(corr_pp)

    # 3b. Accord PP avec sujet quand auxiliaire = etre
    result_ppetre, corr_ppetre = verifier_pp_accord_etre(
        result, pos_tags, morpho, lexique, origs,
    )
    result = result_ppetre
    corrections.extend(corr_ppetre)

    # 4. Homophones contextuels (AVANT negation pour eviter desalignement POS)
    result_homo, corr_homo = verifier_homophones(
        result, pos_tags, morpho, lexique, origs,
    )
    result = result_homo
    corrections.extend(corr_homo)

    # 5. Negation (inserer ne)
    result_neg, corr_neg = verifier_negation(
        result, pos_tags, morpho, lexique, origs,
    )
    result = result_neg
    corrections.extend(corr_neg)

    return result, corrections
