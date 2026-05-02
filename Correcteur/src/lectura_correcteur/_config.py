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
    activer_azerty: bool = True
    activer_sms: bool = False
    activer_negation: bool = False
    activer_viterbi: bool = False
    seuil_confiance_pos: float = 0.7
    max_suggestions: int = 5
    distance_suggestions: int = 2
    activer_double_tagging: bool = False
    seuil_freq_suspicion: float = 5.0  # freq en dessous de laquelle un mot
    # present dans le lexique est quand meme soumis aux candidats d=1/d=2.
    # 0.0 = desactive (seuls les OOV sont corriges).
    # 5.0 = mots rares aussi candidats a la correction (aligne verificateur).
    activer_editeur_homophones: bool = True  # BiLSTM edit tagger pour homophones
    seuil_editeur: float = 0.95  # seuil de confiance minimum pour accepter
    # une prediction du BiLSTM editeur. 0.95 = conservative (haute precision).
    activer_lm: bool = True  # Modele de langue n-gram pour homophones phonetiques
    chemin_lm: str = ""  # Chemin vers ngram.db (vide = auto-detect dans data/)
