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
    max_suggestions: int = 5
    distance_suggestions: int = 2
