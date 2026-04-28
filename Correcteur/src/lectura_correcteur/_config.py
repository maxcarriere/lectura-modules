"""Configuration du correcteur."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorrecteurConfig:
    """Configuration globale du correcteur."""

    activer_orthographe: bool = True
    activer_grammaire: bool = True
    activer_syntaxe: bool = True
    activer_resegmentation: bool = True
    activer_coherence: bool = False
    activer_scoring: bool = False
    seuil_remplacement: float = 0.15
    activer_azerty: bool = False
    activer_sms: bool = False
    activer_negation: bool = False
    activer_viterbi: bool = False
    seuil_confiance_pos: float = 0.7
    max_suggestions: int = 5
    distance_suggestions: int = 2
    seuil_freq_suspicion: float = 0.0  # freq en dessous de laquelle un mot
    # present dans le lexique est quand meme soumis aux candidats d=1/d=2.
    # 0.0 = desactive (comportement par defaut : seuls les OOV sont corriges).
    # Ex: 1.0 = mots archaiques/rares aussi candidats a la correction.
