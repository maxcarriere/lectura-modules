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
    lm_homophones_ratio: float = 3.0  # Ratio minimum score_best/score_current pour homophones ambigus
    activer_accord_pm: bool = False       # Accord guide par PM n-gram (OFF par defaut)
    accord_pm_seuil_violation: float = -10.0  # PM bigram logp en dessous = violation
    accord_pm_seuil_delta: float = 2.0    # Delta minimum pour accepter la correction


@dataclass
class CorrecteurV2Config:
    """Configuration du pipeline V2 a 3 passes."""

    # Passe 2 — POS Viterbi
    seuil_ancrage_pos: float = 0.90
    seuil_correction_pos: float = 0.70
    w_g2p_emission: float = 3.0
    w_freq_emission: float = 0.5
    w_conserv_emission: float = 5.0

    # Passe 3 — Morpho Viterbi
    seuil_delta_nombre: float = 3.0
    seuil_delta_genre: float = 5.0
    seuil_delta_personne: float = 2.0
    seuil_delta_inf_pp: float = 2.5
    morpho_use_variants: bool = False


@dataclass
class CorrecteurV3Config:
    """Configuration du pipeline V3 (Ortho + G2P/P2G).

    Le V3 remplace les passes 2+3 du V2 par un roundtrip G2P→P2G.
    La passe 1 (orthographe) est identique au V2.
    """

    # Passe 2 — Ancrage (memes semantiques que V2)
    seuil_ancrage_pos: float = 0.90  # confiance G2P pour ancrer un mot

    # Passe 2 — P2G roundtrip
    seuil_confiance_p2g: float = 0.70   # confiance P2G minimum pour corriger
    bonus_forme_originale: float = 0.15  # bonus de confiance ajoute a la forme originale
    activer_guard_pos_ngram: bool = True  # cross-check POS n-gram
    activer_guard_lm_homo: bool = True    # cross-check LM homophones

    # Fallback V2 si P2G indisponible
    fallback_v2: bool = True


@dataclass
class CorrecteurV4Config:
    """Configuration du pipeline V4 (Ortho + P2G sans ortho_words).

    Le V4 remplace les passes 2+3 du V2 par un P2G sans ortho_words,
    qui fournit un meilleur tagging POS/Morpho sur texte fautif.
    La passe 1 (orthographe) est identique au V2/V3.
    """

    seuil_confiance_p2g: float = 0.60     # plus bas que V3 (0.70) car P2G pur est plus fiable
    bonus_forme_originale: float = 0.10   # bonus conservateur pour la forme d'entree
    fallback_v2: bool = True              # fallback si P2G indisponible


@dataclass
class CorrecteurV5Config(CorrecteurConfig):
    """Configuration V5 : V1 + P2G comme etiqueteur POS/MORPHO.

    Herite de CorrecteurConfig (toutes les options V1 restent valables).
    V5 desactive par defaut le LM trigramme et l'editeur BiLSTM qui
    causent des FP (est→ait, Mes→Mais). Les homophones sont geres
    par la detection P2G.
    """

    activer_p2g_tagger: bool = True   # utiliser P2G pour POS/Morpho
    fallback_lexique: bool = True     # fallback LexiqueTagger si P2G indisponible
    activer_lm_homophones: bool = False  # remplace par detection P2G
    activer_editeur_homophones: bool = False  # remplace par detection P2G
    activer_homophones_p2g: bool = True  # detection homophones via P2G


@dataclass
class CorrecteurV6Config:
    """Configuration V6 : Pipeline dual G2P/P2G zero-FP.

    Architecture en 3 etapes :
      1. Preprocessing orthographique conservateur (OOV → candidat proche)
      2. Analyse duale G2P + P2G (enrichissement sans correction)
      3. Corrections ciblees (homophones, accords, participe passe)

    Contrainte primaire : zero faux positif sur texte propre.
    """

    # Etape 1 — Orthographe conservatrice
    ortho_distance_max: int = 1
    ortho_frequence_min: float = 0.5

    # Etape 3a — Homophones via divergence P2G
    homophone_confiance_min: float = 0.75
    homophone_top_k: int = 5

    # Etape 3b — Accord morphologique
    accord_fenetre: int = 2

    # Etape 3c — Participe passe
    participe_confiance_min: float = 0.80

    # Preprocessing
    bypass_markdown: bool = True  # Bypass correction sur phrases markdown/LaTeX

    # Etape 3 — Activation des regles (True = actif)
    activer_p2g_global: bool = True        # P2G source de verite
    activer_homophones_p2g: bool = True    # Homophones via divergence P2G
    activer_accords: bool = True           # Accord morpho (ADJ/PART nombre/genre)
    activer_accord_det_nom: bool = True    # Accord DET-NOM (les chien -> chiens)
    activer_accord_attribut: bool = True   # ADJ accord via verbe d'etat
    activer_pp_etre: bool = True           # PP accord avec sujet (auxiliaire etre)
    activer_pp_avoir_genre: bool = True    # PP+avoir invariable (genre du PP)
    activer_accord_nom_adj: bool = True    # Accord NOM+ADJ en nombre
    activer_verbe_p2g: bool = True         # Correction verbe via P2G
    activer_accent_p2g: bool = True        # Accent via P2G
    activer_accent_lexique: bool = True    # Accent fallback lexique
    activer_negation: bool = True          # Insertion de ne/n' devant verbe si absent
    activer_homophones_struct: bool = True # Homophones structurels (sans P2G)
    activer_accord_sujet_verbe: bool = True  # Accord sujet-verbe (conjugaison)

    # Phase 2 — Suggestions (recall elargi, confiance reduite)
    activer_phase2: bool = True

    # P2G lex_select (v7)
    p2g_lex_select: bool = True       # Activer lex_select du P2G
    p2g_lex_threshold: float = 0.92   # Seuil softmax minimum pour lex_select

    # Debug
    mode_analyse: bool = False  # True = analyser() retourne les MotV6 sans corriger
