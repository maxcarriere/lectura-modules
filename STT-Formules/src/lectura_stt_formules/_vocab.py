"""Vocabulaire de sortie pour le modele CTC formules.

87 tokens semantiques (0-86) correspondant aux mots prononces dans la
lecture d'une formule francaise. La decomposition suit exactement les
events produits par les fonctions lire_* de lectura-formules.
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════
# Tokens de controle (2)
# ══════════════════════════════════════════════════════════════════════════

BLANK = 0       # CTC blank
SPACE = 1       # Separateur de composants

# ══════════════════════════════════════════════════════════════════════════
# Nombres atomiques (18) — 0-16 + une
# ══════════════════════════════════════════════════════════════════════════

ZERO = 2
UN = 3
DEUX = 4
TROIS = 5
QUATRE = 6
CINQ = 7
SIX = 8
SEPT = 9
HUIT = 10
NEUF = 11
DIX = 12
ONZE = 13
DOUZE = 14
TREIZE = 15
QUATORZE = 16
QUINZE = 17
SEIZE = 18
UNE = 19

# ══════════════════════════════════════════════════════════════════════════
# Dizaines (5)
# ══════════════════════════════════════════════════════════════════════════

VINGT = 20
TRENTE = 21
QUARANTE = 22
CINQUANTE = 23
SOIXANTE = 24

# ══════════════════════════════════════════════════════════════════════════
# Echelles (4)
# ══════════════════════════════════════════════════════════════════════════

CENT = 25
MILLE = 26
MILLION = 27
MILLIARD = 28

# ══════════════════════════════════════════════════════════════════════════
# Connecteurs (4)
# ══════════════════════════════════════════════════════════════════════════

ET = 29
VIRGULE = 30
MOINS = 31
PLUS = 32

# ══════════════════════════════════════════════════════════════════════════
# Mois (12)
# ══════════════════════════════════════════════════════════════════════════

JANVIER = 33
FEVRIER = 34
MARS = 35
AVRIL = 36
MAI = 37
JUIN = 38
JUILLET = 39
AOUT = 40
SEPTEMBRE = 41
OCTOBRE = 42
NOVEMBRE = 43
DECEMBRE = 44

# ══════════════════════════════════════════════════════════════════════════
# Heure (4)
# ══════════════════════════════════════════════════════════════════════════

HEURE = 45
MINUTE = 46
SECONDE_T = 47
MIDI = 48

# ══════════════════════════════════════════════════════════════════════════
# Devises (4)
# ══════════════════════════════════════════════════════════════════════════

EURO = 49
DOLLAR = 50
CENTIME = 51
LIVRE = 52

# ══════════════════════════════════════════════════════════════════════════
# Pourcentage (2)
# ══════════════════════════════════════════════════════════════════════════

POURCENT = 53
POURMILLE = 54

# ══════════════════════════════════════════════════════════════════════════
# Ordinaux / fractions (6)
# ══════════════════════════════════════════════════════════════════════════

PREMIER = 55
IEME = 56
SUR = 57
DEMI = 58
TIERS = 59
QUART = 60

# ══════════════════════════════════════════════════════════════════════════
# Lettres A-Z (26) — pour epeler les sigles
# ══════════════════════════════════════════════════════════════════════════

A_LETTER = 61
B_LETTER = 62
C_LETTER = 63
D_LETTER = 64
E_LETTER = 65
F_LETTER = 66
G_LETTER = 67
H_LETTER = 68
I_LETTER = 69
J_LETTER = 70
K_LETTER = 71
L_LETTER = 72
M_LETTER = 73
N_LETTER = 74
O_LETTER = 75
P_LETTER = 76
Q_LETTER = 77
R_LETTER = 78
S_LETTER = 79
T_LETTER = 80
U_LETTER = 81
V_LETTER = 82
W_LETTER = 83
X_LETTER = 84
Y_LETTER = 85
Z_LETTER = 86

# ══════════════════════════════════════════════════════════════════════════
# Total : 87 tokens (0-86)
# ══════════════════════════════════════════════════════════════════════════

NUM_TOKENS = 87

# ══════════════════════════════════════════════════════════════════════════
# Table ID → nom lisible
# ══════════════════════════════════════════════════════════════════════════

VOCAB: dict[int, str] = {
    BLANK: "<blank>",
    SPACE: "<space>",
    # Nombres atomiques
    ZERO: "ZERO", UN: "UN", DEUX: "DEUX", TROIS: "TROIS", QUATRE: "QUATRE",
    CINQ: "CINQ", SIX: "SIX", SEPT: "SEPT", HUIT: "HUIT", NEUF: "NEUF",
    DIX: "DIX", ONZE: "ONZE", DOUZE: "DOUZE", TREIZE: "TREIZE",
    QUATORZE: "QUATORZE", QUINZE: "QUINZE", SEIZE: "SEIZE", UNE: "UNE",
    # Dizaines
    VINGT: "VINGT", TRENTE: "TRENTE", QUARANTE: "QUARANTE",
    CINQUANTE: "CINQUANTE", SOIXANTE: "SOIXANTE",
    # Echelles
    CENT: "CENT", MILLE: "MILLE", MILLION: "MILLION", MILLIARD: "MILLIARD",
    # Connecteurs
    ET: "ET", VIRGULE: "VIRGULE", MOINS: "MOINS", PLUS: "PLUS",
    # Mois
    JANVIER: "JANVIER", FEVRIER: "FEVRIER", MARS: "MARS", AVRIL: "AVRIL",
    MAI: "MAI", JUIN: "JUIN", JUILLET: "JUILLET", AOUT: "AOUT",
    SEPTEMBRE: "SEPTEMBRE", OCTOBRE: "OCTOBRE", NOVEMBRE: "NOVEMBRE",
    DECEMBRE: "DECEMBRE",
    # Heure
    HEURE: "HEURE", MINUTE: "MINUTE", SECONDE_T: "SECONDE_T", MIDI: "MIDI",
    # Devises
    EURO: "EURO", DOLLAR: "DOLLAR", CENTIME: "CENTIME", LIVRE: "LIVRE",
    # Pourcentage
    POURCENT: "POURCENT", POURMILLE: "POURMILLE",
    # Ordinaux / fractions
    PREMIER: "PREMIER", IEME: "IEME", SUR: "SUR",
    DEMI: "DEMI", TIERS: "TIERS", QUART: "QUART",
    # Lettres
    A_LETTER: "A", B_LETTER: "B", C_LETTER: "C", D_LETTER: "D",
    E_LETTER: "E", F_LETTER: "F", G_LETTER: "G", H_LETTER: "H",
    I_LETTER: "I", J_LETTER: "J", K_LETTER: "K", L_LETTER: "L",
    M_LETTER: "M", N_LETTER: "N", O_LETTER: "O", P_LETTER: "P",
    Q_LETTER: "Q", R_LETTER: "R", S_LETTER: "S", T_LETTER: "T",
    U_LETTER: "U", V_LETTER: "V", W_LETTER: "W", X_LETTER: "X",
    Y_LETTER: "Y", Z_LETTER: "Z",
}

# ══════════════════════════════════════════════════════════════════════════
# Mapping ortho (depuis events) → liste de token IDs
#
# Les cles sont les valeurs .ortho.lower() des EventFormuleLecture
# produits par les fonctions lire_* de lectura-formules.
# ══════════════════════════════════════════════════════════════════════════

ORTHO_TO_TOKENS: dict[str, list[int]] = {
    # --- Nombres atomiques ---
    "zéro": [ZERO],
    "un": [UN],
    "deux": [DEUX],
    "trois": [TROIS],
    "quatre": [QUATRE],
    "cinq": [CINQ],
    "six": [SIX],
    "sept": [SEPT],
    "huit": [HUIT],
    "neuf": [NEUF],
    "dix": [DIX],
    "onze": [ONZE],
    "douze": [DOUZE],
    "treize": [TREIZE],
    "quatorze": [QUATORZE],
    "quinze": [QUINZE],
    "seize": [SEIZE],
    "une": [UNE],

    # --- Dizaines ---
    "vingt": [VINGT],
    "vingts": [VINGT],       # 80 → quatre-vingts
    "trente": [TRENTE],
    "quarante": [QUARANTE],
    "cinquante": [CINQUANTE],
    "soixante": [SOIXANTE],

    # --- Echelles ---
    "cent": [CENT],
    "cents": [CENT],         # 200 → deux cents
    "mille": [MILLE],
    "million": [MILLION],
    "millions": [MILLION],
    "milliard": [MILLIARD],
    "milliards": [MILLIARD],

    # --- Connecteurs ---
    "et": [ET],
    "et un": [ET, UN],       # 21 → vingt et un (event mono-ortho)
    "et une": [ET, UNE],     # 21 feminin
    "et onze": [ET, ONZE],   # 71 → soixante et onze
    "virgule": [VIRGULE],
    "moins": [MOINS],
    "plus": [PLUS],

    # --- Mois ---
    "janvier": [JANVIER],
    "février": [FEVRIER],
    "mars": [MARS],
    "avril": [AVRIL],
    "mai": [MAI],
    "juin": [JUIN],
    "juillet": [JUILLET],
    "août": [AOUT],
    "septembre": [SEPTEMBRE],
    "octobre": [OCTOBRE],
    "novembre": [NOVEMBRE],
    "décembre": [DECEMBRE],

    # --- Heure ---
    "heure": [HEURE],
    "heures": [HEURE],
    "minute": [MINUTE],
    "minutes": [MINUTE],
    "seconde": [SECONDE_T],
    "secondes": [SECONDE_T],
    "midi": [MIDI],

    # --- Devises ---
    "euro": [EURO],
    "euros": [EURO],
    "dollar": [DOLLAR],
    "dollars": [DOLLAR],
    "centime": [CENTIME],
    "centimes": [CENTIME],
    "livre": [LIVRE],
    "livres": [LIVRE],

    # --- Pourcentage ---
    "pour cent": [POURCENT],
    "pour mille": [POURMILLE],

    # --- Ordinaux ---
    "premier": [PREMIER],
    "première": [PREMIER],

    # Ordinaux composes : base + IEME (singulier et pluriel)
    "unième": [UN, IEME],
    "deuxième": [DEUX, IEME],
    "troisième": [TROIS, IEME],
    "quatrième": [QUATRE, IEME],
    "cinquième": [CINQ, IEME],
    "sixième": [SIX, IEME],
    "septième": [SEPT, IEME],
    "huitième": [HUIT, IEME],
    "neuvième": [NEUF, IEME],
    "dixième": [DIX, IEME],
    "onzième": [ONZE, IEME],
    "douzième": [DOUZE, IEME],
    "treizième": [TREIZE, IEME],
    "quatorzième": [QUATORZE, IEME],
    "quinzième": [QUINZE, IEME],
    "seizième": [SEIZE, IEME],
    "vingtième": [VINGT, IEME],
    "trentième": [TRENTE, IEME],
    "quarantième": [QUARANTE, IEME],
    "cinquantième": [CINQUANTE, IEME],
    "soixantième": [SOIXANTE, IEME],
    "centième": [CENT, IEME],
    "millième": [MILLE, IEME],
    "millionième": [MILLION, IEME],
    "milliardième": [MILLIARD, IEME],
    # Pluriels (fractions : 3/5 → "trois cinquièmes")
    "unièmes": [UN, IEME],
    "deuxièmes": [DEUX, IEME],
    "troisièmes": [TROIS, IEME],
    "quatrièmes": [QUATRE, IEME],
    "cinquièmes": [CINQ, IEME],
    "sixièmes": [SIX, IEME],
    "septièmes": [SEPT, IEME],
    "huitièmes": [HUIT, IEME],
    "neuvièmes": [NEUF, IEME],
    "dixièmes": [DIX, IEME],
    "onzièmes": [ONZE, IEME],
    "douzièmes": [DOUZE, IEME],
    "treizièmes": [TREIZE, IEME],
    "quatorzièmes": [QUATORZE, IEME],
    "quinzièmes": [QUINZE, IEME],
    "seizièmes": [SEIZE, IEME],
    "vingtièmes": [VINGT, IEME],
    "trentièmes": [TRENTE, IEME],
    "quarantièmes": [QUARANTE, IEME],
    "cinquantièmes": [CINQUANTE, IEME],
    "soixantièmes": [SOIXANTE, IEME],
    "centièmes": [CENT, IEME],
    "millièmes": [MILLE, IEME],
    "millionièmes": [MILLION, IEME],
    "milliardièmes": [MILLIARD, IEME],

    # Ordinaux composes avec "et"
    "et unième": [ET, UN, IEME],
    "et onzième": [ET, ONZE, IEME],
    "et unièmes": [ET, UN, IEME],
    "et onzièmes": [ET, ONZE, IEME],

    # --- Fractions ---
    "sur": [SUR],
    "demi": [DEMI],
    "tiers": [TIERS],
    "quart": [QUART],
    "quarts": [QUART],

    # --- Lettres (sigles) ---
    # Les ortho des lettres epelees suivent les noms du fichier donnees_formules.json
    "a": [A_LETTER],
    "bé": [B_LETTER],
    "cé": [C_LETTER],
    "dé": [D_LETTER],
    "e": [E_LETTER],
    "effe": [F_LETTER],
    "gé": [G_LETTER],
    "ache": [H_LETTER],
    "i": [I_LETTER],
    "ji": [J_LETTER],
    "ka": [K_LETTER],
    "elle": [L_LETTER],
    "emme": [M_LETTER],
    "enne": [N_LETTER],
    "o": [O_LETTER],
    "pé": [P_LETTER],
    "ku": [Q_LETTER],
    "erre": [R_LETTER],
    "esse": [S_LETTER],
    "té": [T_LETTER],
    "u": [U_LETTER],
    "vé": [V_LETTER],
    "double-vé": [W_LETTER],
    "ix": [X_LETTER],
    "i-grec": [Y_LETTER],
    "zède": [Z_LETTER],
}


def token_ids_to_names(token_ids: list[int]) -> list[str]:
    """Convertit une liste de token IDs en noms lisibles."""
    return [VOCAB.get(tid, f"?{tid}") for tid in token_ids]


def vocab_to_json() -> dict[str, int]:
    """Retourne le vocabulaire sous forme {nom: id} pour export JSON."""
    return {name: tid for tid, name in sorted(VOCAB.items())}
