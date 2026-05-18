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
    activer_negation: bool = True
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
    activer_lm: bool = False  # Modele de langue n-gram generique (desactive, remplace par lm_homophones)
    chemin_lm: str = ""  # Chemin vers ngram.db (vide = auto-detect dans data/)
    activer_lm_homophones: bool = True  # LM trigramme specialise homophones
    chemin_lm_homophones: str = ""  # Chemin vers homophones_trigrams.db
    activer_pos_ngram: bool = True  # N-gram POS pour validation des corrections
    chemin_pos_ngram: str = ""  # Chemin vers pos_ngram.db
    activer_analyse_viterbi: bool = False  # Viterbi trigramme POS+forme (OFF par defaut)
    viterbi_bonus_original: float = 2.0  # Biais conservateur forme originale
    viterbi_bonus_lm: float = 1.0  # Poids bonus LM homophones
    viterbi_w_emission: float = 1.0  # Poids emissions
    viterbi_w_transition: float = 1.0  # Poids transitions POS n-gram
    activer_viterbi_morpho: bool = False  # Viterbi POS+Morpho aval grammaire (OFF par defaut)
    viterbi_morpho_bonus_current: float = 2.0  # Bonus PM tag concordant avec POS actuel
    viterbi_morpho_w_emission: float = 1.0  # Poids emissions PM
    viterbi_morpho_w_transition: float = 1.0  # Poids transitions PM n-gram
    viterbi_morpho_use_variants: bool = False  # Inclure variantes flexionnelles
    activer_tagger_hybride: bool = False  # Tagger hybride G2P+overrides (OFF par defaut)
    seuil_freq_voisin: float = 50.0  # Freq au-dela de laquelle un voisin d=1 rend le mot "ambigu"
