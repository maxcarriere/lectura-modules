"""Tokenizer : events lectura-formules → sequence de token IDs.

Convertit les events (EventFormuleLecture) d'un LectureFormuleResult
en une sequence de token IDs semantiques pour l'entrainement du modele
CTC formules.
"""

from __future__ import annotations

from lectura_formules.lecture_formules import LectureFormuleResult

from lectura_stt_formules._vocab import ORTHO_TO_TOKENS, SPACE


def events_to_token_sequence(result: LectureFormuleResult) -> list[int]:
    """Convertit les events d'un resultat de lecture en sequence de tokens.

    Insere un token SPACE entre les composants (groupes de telephone,
    jour/mois/annee, etc.) identifies par le champ ``composant`` des events.

    Args:
        result: Resultat d'une fonction lire_* de lectura-formules.

    Returns:
        Liste de token IDs (sans BLANK — ceux-ci sont ajoutes par le CTC).

    Raises:
        ValueError: Si un ortho d'event n'est pas dans le vocabulaire.
    """
    tokens: list[int] = []
    prev_composant = -1

    for evt in result.events:
        # Inserer un separateur entre composants differents
        if evt.composant != prev_composant and prev_composant >= 0:
            tokens.append(SPACE)
        prev_composant = evt.composant

        ortho = evt.ortho.lower().strip()
        if ortho in ORTHO_TO_TOKENS:
            tokens.extend(ORTHO_TO_TOKENS[ortho])
        else:
            raise ValueError(f"ortho inconnu: {ortho!r}")

    return tokens
