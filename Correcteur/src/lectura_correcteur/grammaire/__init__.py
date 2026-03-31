"""Sous-module grammaire : regles d'accord, conjugaison et homophones."""

from lectura_correcteur.grammaire._accord import verifier_accords
from lectura_correcteur.grammaire._conjugaison import verifier_conjugaisons
from lectura_correcteur.grammaire._homophones import verifier_homophones

__all__ = [
    "appliquer_grammaire",
    "verifier_accords",
    "verifier_conjugaisons",
    "verifier_homophones",
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

    # Accords (det+nom, det+adj+nom, det+nom+ver)
    result_acc, corr_acc = verifier_accords(
        result, pos_tags, morpho, lexique, origs,
    )
    result = result_acc
    corrections.extend(corr_acc)

    # Conjugaisons (pronom+verbe)
    result_conj, corr_conj = verifier_conjugaisons(
        result, pos_tags, morpho, lexique, origs,
    )
    result = result_conj
    corrections.extend(corr_conj)

    # Homophones contextuels
    result_homo, corr_homo = verifier_homophones(
        result, pos_tags, morpho, lexique, origs,
    )
    result = result_homo
    corrections.extend(corr_homo)

    return result, corrections
