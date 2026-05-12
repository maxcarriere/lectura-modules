"""Perturbateurs de phrases francaises pour generation de corpus d'entrainement.

Chaque perturbateur prend un contexte de phrase enrichie (tokens + infos lexique)
et retourne une version fautive d'un mot, avec metadata sur l'erreur.

Types d'erreurs generes :
  PHON  — Confusions phonetiques (homophones du lexique)
  ACC   — Erreurs d'accord (genre, nombre)
  CONJ  — Erreurs de conjugaison (personne, temps)
  HOMO  — Homophones grammaticaux (a/a, et/est, son/sont...)
  PP    — Confusion participe passe / infinitif (-e/-er)
  ACCENT — Suppression ou modification d'accents
  TYPO  — Fautes de frappe (AZERTY, transposition)
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class TokenEnrichi:
    """Token d'une phrase avec infos lexique."""
    forme: str
    index: int
    lemme: str = ""
    pos: str = ""          # NOM, VER, ADJ, ADV, PRE, CON, ART, AUX, PRO...
    genre: str = ""        # m, f
    nombre: str = ""       # s, p
    phone: str = ""
    freq: float = 0.0
    dans_lexique: bool = False
    mode: str = ""
    temps: str = ""
    personne: str = ""


@dataclass
class Erreur:
    """Description d'une erreur injectee."""
    position: int
    original: str
    perturbe: str
    type_erreur: str       # PHON, ACC, CONJ, HOMO, PP, ACCENT, TYPO


# ---------------------------------------------------------------------------
# Homophones grammaticaux — table de confusion
# ---------------------------------------------------------------------------

# Chaque groupe = des formes interchangeables en contexte.
# On remplace un mot correct par un autre du meme groupe.
HOMOPHONES_GRAM: list[list[str]] = [
    ["a", "à"],
    ["et", "est"],
    ["ou", "où"],
    ["on", "ont"],
    ["son", "sont"],
    ["ce", "se"],
    ["ces", "ses"],
    ["la", "là"],
    ["ma", "m'a"],
    ["ta", "t'a"],
    ["peu", "peut", "peux"],
    ["mais", "mes"],
    ["ni", "n'y"],
    ["si", "s'y"],
    ["dans", "d'en"],
    ["sans", "s'en"],
    ["leur", "leurs"],
    ["ça", "sa"],
    ["quand", "quant"],
    ["près", "prêt"],
]

# Index inverse : mot -> groupe (pour lookup rapide)
_HOMO_INDEX: dict[str, list[str]] = {}
for _group in HOMOPHONES_GRAM:
    for _mot in _group:
        _HOMO_INDEX[_mot] = _group


# ---------------------------------------------------------------------------
# AZERTY — touches voisines
# ---------------------------------------------------------------------------

VOISINS_AZERTY: dict[str, str] = {
    "a": "zqs", "z": "aeqs", "e": "zrds", "r": "etfd",
    "t": "rygf", "y": "tuhg", "u": "yijh", "i": "uokj",
    "o": "iplk", "p": "olm",
    "q": "azws", "s": "qzedwx", "d": "serfxc",
    "f": "drtgcv", "g": "ftyhvb", "h": "gyujbn",
    "j": "huikn", "k": "jiol", "l": "kopm",
    "m": "lp",
    "w": "qsxa", "x": "wsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhj",
}


# ---------------------------------------------------------------------------
# Accents — table de correspondances
# ---------------------------------------------------------------------------

# Correspondances accent specifiques pour le francais
CONFUSIONS_ACCENT: dict[str, list[str]] = {
    "é": ["e", "è", "ê"],
    "è": ["e", "é", "ê"],
    "ê": ["e", "é", "è"],
    "à": ["a"],
    "â": ["a"],
    "ù": ["u"],
    "û": ["u"],
    "ï": ["i"],
    "î": ["i"],
    "ô": ["o"],
    "ç": ["c"],
    "œ": ["oe"],
    "æ": ["ae"],
}


# ---------------------------------------------------------------------------
# Perturbateur : HOMOPHONES GRAMMATICAUX
# ---------------------------------------------------------------------------

def perturber_homophone(
    tokens: list[TokenEnrichi],
    idx: int,
) -> Erreur | None:
    """Remplace un homophone grammatical par un autre du meme groupe."""
    mot = tokens[idx].forme.lower()
    groupe = _HOMO_INDEX.get(mot)
    if not groupe:
        return None
    # Choisir un autre mot du groupe
    alternatives = [m for m in groupe if m != mot]
    if not alternatives:
        return None
    remplacement = random.choice(alternatives)
    return Erreur(
        position=idx,
        original=tokens[idx].forme,
        perturbe=remplacement,
        type_erreur="HOMO",
    )


# ---------------------------------------------------------------------------
# Perturbateur : ACCORD NOMBRE
# ---------------------------------------------------------------------------

def perturber_accord_nombre(
    tokens: list[TokenEnrichi],
    idx: int,
    lexique: Any,
) -> Erreur | None:
    """Change le nombre d'un nom ou adjectif (singulier <-> pluriel)."""
    tok = tokens[idx]
    if tok.pos not in ("NOM", "ADJ") or not tok.lemme:
        return None

    formes = lexique.formes_de(tok.lemme, tok.pos)
    if not formes:
        return None

    if tok.nombre == "p":
        # Pluriel -> chercher le singulier
        cibles = [
            f for f in formes
            if f.get("nombre") == "s"
            and f.get("genre", tok.genre) == tok.genre
        ]
    elif tok.nombre == "s":
        # Singulier -> chercher le pluriel
        cibles = [
            f for f in formes
            if f.get("nombre") == "p"
            and f.get("genre", tok.genre) == tok.genre
        ]
    else:
        return None

    if not cibles:
        return None

    cible = cibles[0]
    ortho = cible.get("ortho", "")
    if not ortho or ortho.lower() == tok.forme.lower():
        return None

    return Erreur(
        position=idx,
        original=tok.forme,
        perturbe=ortho,
        type_erreur="ACC",
    )


# ---------------------------------------------------------------------------
# Perturbateur : ACCORD GENRE
# ---------------------------------------------------------------------------

def perturber_accord_genre(
    tokens: list[TokenEnrichi],
    idx: int,
    lexique: Any,
) -> Erreur | None:
    """Change le genre d'un adjectif (masculin <-> feminin)."""
    tok = tokens[idx]
    if tok.pos != "ADJ" or not tok.lemme or not tok.genre:
        return None

    formes = lexique.formes_de(tok.lemme, "ADJ")
    if not formes:
        return None

    genre_cible = "f" if tok.genre == "m" else "m"
    cibles = [
        f for f in formes
        if f.get("genre") == genre_cible
        and f.get("nombre", tok.nombre) == tok.nombre
    ]
    if not cibles:
        return None

    ortho = cibles[0].get("ortho", "")
    if not ortho or ortho.lower() == tok.forme.lower():
        return None

    return Erreur(
        position=idx,
        original=tok.forme,
        perturbe=ortho,
        type_erreur="ACC",
    )


# ---------------------------------------------------------------------------
# Perturbateur : CONJUGAISON
# ---------------------------------------------------------------------------

def perturber_conjugaison(
    tokens: list[TokenEnrichi],
    idx: int,
    lexique: Any,
) -> Erreur | None:
    """Remplace un verbe conjugue par une mauvaise forme (personne/nombre).

    Strategie permissive : si le mot est un verbe dans le lexique,
    on le remplace par une forme conjuguee differente du meme lemme,
    au meme temps si possible, sinon au present de l'indicatif.
    """
    tok = tokens[idx]
    if tok.pos not in ("VER", "AUX") or not tok.lemme:
        return None
    # Exclure les infinitifs et participes (pas de "conjugaison" a corriger)
    if tok.forme.lower().endswith(("er", "ir", "re", "ant", "é", "ée", "és", "ées")):
        return None

    formes = lexique.formes_de(tok.lemme, tok.pos)
    if not formes:
        return None

    # Privilegier les formes au meme temps/mode si connus
    candidats = []
    if tok.temps and tok.mode:
        candidats = [
            f for f in formes
            if f.get("temps") == tok.temps
            and f.get("mode") == tok.mode
            and f.get("ortho", "").lower() != tok.forme.lower()
        ]

    # Fallback : formes du present de l'indicatif
    if not candidats:
        candidats = [
            f for f in formes
            if f.get("mode") == "indicatif"
            and f.get("temps") in ("présent", "imparfait")
            and f.get("ortho", "").lower() != tok.forme.lower()
        ]

    if not candidats:
        return None

    cible = random.choice(candidats)
    ortho = cible.get("ortho", "")
    if not ortho or ortho.lower() == tok.forme.lower():
        return None

    return Erreur(
        position=idx,
        original=tok.forme,
        perturbe=ortho,
        type_erreur="CONJ",
    )


# ---------------------------------------------------------------------------
# Perturbateur : PARTICIPE PASSE / INFINITIF
# ---------------------------------------------------------------------------

_AUXILIAIRES = frozenset(
    "ai as a avons avez ont avais avait avions aviez avaient "
    "eus eut eûmes eûtes eurent aurai auras aura aurons aurez auront "
    "suis es est sommes êtes sont étais était étions étiez étaient "
    "fus fut fûmes fûtes furent serai seras sera serons serez seront".split()
)
_MODAUX = frozenset(
    "va vas vais vont allons allez allait allaient "
    "veux veut voulons voulez veulent voulait voulaient "
    "dois doit devons devez doivent devait devaient "
    "peux peut pouvons pouvez peuvent pouvait pouvaient "
    "faut fallait".split()
)


def perturber_pp_infinitif(
    tokens: list[TokenEnrichi],
    idx: int,
    lexique: Any,
) -> Erreur | None:
    """Confond participe passe et infinitif pour les verbes du 1er groupe.

    Plus permissif que la V1 : ne requiert pas un POS tag precis.
    """
    mot = tokens[idx].forme.lower()

    # Contexte gauche
    mot_gauche = tokens[idx - 1].forme.lower() if idx > 0 else ""

    # Cas 1 : mot en -é (PP) -> -er (infinitif)
    # Ex: "a mangé" -> "a manger"
    if mot.endswith("é") and len(mot) > 2:
        remplacement = mot[:-1] + "er"
        if lexique.existe(remplacement):
            return Erreur(
                position=idx,
                original=tokens[idx].forme,
                perturbe=remplacement,
                type_erreur="PP",
            )

    # Cas 2 : mot en -er (infinitif) -> -é (PP)
    # Ex: "va manger" -> "va mangé", "pour manger" -> "pour mangé"
    if mot.endswith("er") and len(mot) > 3:
        remplacement = mot[:-2] + "é"
        if lexique.existe(remplacement):
            return Erreur(
                position=idx,
                original=tokens[idx].forme,
                perturbe=remplacement,
                type_erreur="PP",
            )

    # Cas 3 : PP feminin -ée -> -er
    if mot.endswith("ée") and len(mot) > 3:
        remplacement = mot[:-2] + "er"
        if lexique.existe(remplacement):
            return Erreur(
                position=idx,
                original=tokens[idx].forme,
                perturbe=remplacement,
                type_erreur="PP",
            )

    # Cas 4 : PP pluriel -és -> -er
    if mot.endswith("és") and len(mot) > 3:
        remplacement = mot[:-2] + "er"
        if lexique.existe(remplacement):
            return Erreur(
                position=idx,
                original=tokens[idx].forme,
                perturbe=remplacement,
                type_erreur="PP",
            )

    return None


# ---------------------------------------------------------------------------
# Perturbateur : PHONETIQUE (homophones du lexique)
# ---------------------------------------------------------------------------

def _est_variante_triviale(original: str, candidat: str) -> bool:
    """Detecte les variantes triviales (pluriel en -s, feminin en -e).

    Ces cas relevent de ACC, pas de PHON.
    """
    a, b = original.lower(), candidat.lower()
    # Difference d'un seul -s/-x final
    if a + "s" == b or b + "s" == a:
        return True
    if a + "x" == b or b + "x" == a:
        return True
    # Difference d'un seul -e final
    if a + "e" == b or b + "e" == a:
        return True
    # Difference -es final
    if a + "es" == b or b + "es" == a:
        return True
    return False


# Mots trop courts ou fonctionnels a exclure de PHON
# (ils sont couverts par HOMO ou trop ambigus)
_PHON_EXCLUS = frozenset(
    "le la les un une de des du en et est a au aux il elle on je tu nous vous "
    "ils elles ce qui que ne pas dans par pour avec son sa ses sur se si me te "
    "lui y ni ou où".split()
)


def perturber_phonetique(
    tokens: list[TokenEnrichi],
    idx: int,
    lexique: Any,
) -> Erreur | None:
    """Remplace un mot par un homophone (meme prononciation, graphie differente)."""
    tok = tokens[idx]
    mot = tok.forme.lower()

    # Exclure les mots fonctionnels courts (couverts par HOMO)
    if mot in _PHON_EXCLUS or len(mot) < 3:
        return None

    phone = tok.phone
    if not phone:
        phone = lexique.phone_de(tok.forme)
    if not phone:
        return None

    homos = lexique.homophones(phone)
    alternatives = []
    for h in homos:
        ortho = h.get("ortho", "")
        if not ortho or ortho.lower() == mot:
            continue
        freq = h.get("freq") or h.get("freq_opensubs") or 0
        if not freq or float(freq) <= 0:
            continue
        # Exclure les variantes triviales (singulier/pluriel)
        if _est_variante_triviale(mot, ortho):
            continue
        # Exclure les graphies trop differentes en longueur
        if abs(len(ortho) - len(mot)) > 3:
            continue
        # Exclure les mots avec des tirets/caracteres speciaux
        if "-" in ortho or "'" in ortho:
            continue
        alternatives.append((ortho, float(freq)))

    if not alternatives:
        return None

    # Ponderer par la frequence (les mots frequents = erreurs plus realistes)
    total_freq = sum(f for _, f in alternatives)
    if total_freq > 0:
        r = random.random() * total_freq
        cumul = 0.0
        remplacement = alternatives[0][0]
        for ortho, freq in alternatives:
            cumul += freq
            if r <= cumul:
                remplacement = ortho
                break
    else:
        remplacement = random.choice(alternatives)[0]

    return Erreur(
        position=idx,
        original=tok.forme,
        perturbe=remplacement,
        type_erreur="PHON",
    )


# ---------------------------------------------------------------------------
# Perturbateur : ACCENTS
# ---------------------------------------------------------------------------

def perturber_accent(
    tokens: list[TokenEnrichi],
    idx: int,
) -> Erreur | None:
    """Supprime ou modifie un accent dans le mot."""
    mot = tokens[idx].forme
    # Trouver les positions accentuees
    positions_accent = []
    for i, c in enumerate(mot):
        if c in CONFUSIONS_ACCENT:
            positions_accent.append(i)

    if not positions_accent:
        return None

    # Choisir une position aleatoire
    pos = random.choice(positions_accent)
    char_original = mot[pos]
    remplacement_char = random.choice(CONFUSIONS_ACCENT[char_original])

    # Reconstruire le mot
    nouveau = mot[:pos] + remplacement_char + mot[pos + 1:]
    if nouveau == mot:
        return None

    return Erreur(
        position=idx,
        original=tokens[idx].forme,
        perturbe=nouveau,
        type_erreur="ACCENT",
    )


# ---------------------------------------------------------------------------
# Perturbateur : TYPO (AZERTY)
# ---------------------------------------------------------------------------

def perturber_typo(
    tokens: list[TokenEnrichi],
    idx: int,
) -> Erreur | None:
    """Simule une faute de frappe realiste (clavier AZERTY)."""
    mot = tokens[idx].forme
    if len(mot) < 3:
        return None

    operation = random.choice(["substitution", "transposition", "suppression", "doublement"])

    if operation == "substitution":
        # Remplacer un caractere par un voisin AZERTY
        positions = [i for i, c in enumerate(mot) if c.lower() in VOISINS_AZERTY]
        if not positions:
            return None
        pos = random.choice(positions)
        c = mot[pos].lower()
        voisins = VOISINS_AZERTY[c]
        remplacement = random.choice(list(voisins))
        if mot[pos].isupper():
            remplacement = remplacement.upper()
        nouveau = mot[:pos] + remplacement + mot[pos + 1:]

    elif operation == "transposition":
        # Echanger deux caracteres adjacents
        pos = random.randint(0, len(mot) - 2)
        chars = list(mot)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        nouveau = "".join(chars)

    elif operation == "suppression":
        # Supprimer un caractere (pas le premier ni le dernier)
        if len(mot) < 4:
            return None
        pos = random.randint(1, len(mot) - 2)
        nouveau = mot[:pos] + mot[pos + 1:]

    elif operation == "doublement":
        # Doubler un caractere
        pos = random.randint(0, len(mot) - 1)
        nouveau = mot[:pos] + mot[pos] + mot[pos:]

    else:
        return None

    if nouveau == mot:
        return None

    return Erreur(
        position=idx,
        original=tokens[idx].forme,
        perturbe=nouveau,
        type_erreur="TYPO",
    )


# ---------------------------------------------------------------------------
# Enrichissement des tokens via lexique
# ---------------------------------------------------------------------------

def enrichir_tokens(
    tokens: list[str],
    lexique: Any,
) -> list[TokenEnrichi]:
    """Enrichit chaque token avec les infos du lexique."""
    result: list[TokenEnrichi] = []
    for i, forme in enumerate(tokens):
        te = TokenEnrichi(forme=forme, index=i)

        if not re.match(r"^[a-zàâäéèêëïîôùûüÿçœæ'-]+$", forme.lower()):
            # Ponctuation ou caractere special
            result.append(te)
            continue

        infos = lexique.info(forme)
        if infos:
            te.dans_lexique = True
            # Prendre l'entree la plus frequente
            best = max(infos, key=lambda e: float(e.get("freq", e.get("freq_opensubs", 0)) or 0))
            te.lemme = best.get("lemme", "")
            te.pos = best.get("cgram", "")
            te.genre = best.get("genre", "")
            te.nombre = best.get("nombre", "")
            te.phone = best.get("phone", "")
            te.freq = float(best.get("freq", best.get("freq_opensubs", 0)) or 0)
            te.mode = best.get("mode", "")
            te.temps = best.get("temps", "")
            te.personne = best.get("personne", "")
        else:
            te.dans_lexique = False
            # Essayer quand meme de recuperer le phone
            phone = lexique.phone_de(forme)
            if phone:
                te.phone = phone

        result.append(te)
    return result


# ---------------------------------------------------------------------------
# Orchestrateur : appliquer des perturbations a une phrase
# ---------------------------------------------------------------------------

# Probabilites de selection de chaque type (sommees a 1.0)
DISTRIBUTION_ERREURS: dict[str, float] = {
    "HOMO":   0.22,   # homophones grammaticaux (a/a, et/est...)
    "ACC":    0.20,   # accord nombre + genre
    "PP":     0.18,   # participe passe / infinitif (-e/-er)
    "CONJ":   0.15,   # conjugaison (personne/nombre)
    "PHON":   0.12,   # homophones phonetiques
    "ACCENT": 0.10,   # accents
    "TYPO":   0.03,   # fautes de frappe
}


def _choisir_type_erreur() -> str:
    """Tire un type d'erreur selon la distribution."""
    r = random.random()
    cumul = 0.0
    for type_err, prob in DISTRIBUTION_ERREURS.items():
        cumul += prob
        if r <= cumul:
            return type_err
    return "TYPO"  # fallback


def _indices_par_type(tokens: list[TokenEnrichi]) -> dict[str, list[int]]:
    """Pre-calcule les indices de mots candidats pour chaque type d'erreur."""
    result: dict[str, list[int]] = {t: [] for t in DISTRIBUTION_ERREURS}

    for i, tok in enumerate(tokens):
        if not tok.dans_lexique or len(tok.forme) < 2:
            continue
        mot = tok.forme.lower()

        # HOMO : mot present dans la table d'homophones grammaticaux
        if mot in _HOMO_INDEX:
            result["HOMO"].append(i)

        # ACC : noms et adjectifs
        if tok.pos in ("NOM", "ADJ"):
            result["ACC"].append(i)

        # CONJ : verbes conjugues (pas infinitifs/participes)
        if tok.pos in ("VER", "AUX") and tok.lemme:
            if not mot.endswith(("er", "ir", "re", "ant", "é", "ée", "és", "ées")):
                result["CONJ"].append(i)

        # PP : mots en -e/-er/-ee/-es (confusion PP/infinitif)
        if (mot.endswith("é") or mot.endswith("ée") or mot.endswith("és")
                or (mot.endswith("er") and len(mot) > 3)):
            result["PP"].append(i)

        # PHON : mots de 3+ chars, pas fonctionnels
        if len(mot) >= 3 and mot not in _PHON_EXCLUS and tok.phone:
            result["PHON"].append(i)

        # ACCENT : mots avec au moins un accent
        if any(c in CONFUSIONS_ACCENT for c in mot):
            result["ACCENT"].append(i)

        # TYPO : mots de 3+ chars
        if len(mot) >= 3:
            result["TYPO"].append(i)

    return result


def appliquer_perturbations(
    tokens: list[TokenEnrichi],
    lexique: Any,
    *,
    n_erreurs: int = 1,
    max_tentatives: int = 30,
) -> tuple[list[str], list[Erreur]]:
    """Applique N erreurs aleatoires a une phrase enrichie.

    Strategie : choisir d'abord le type d'erreur, puis un mot adapte
    a ce type (plutot que l'inverse). Cela garantit une distribution
    d'erreurs plus proche de la cible.

    Args:
        tokens: Phrase enrichie (via enrichir_tokens)
        lexique: Instance du lexique Lectura
        n_erreurs: Nombre d'erreurs a injecter
        max_tentatives: Nombre max de tentatives avant abandon

    Returns:
        (tokens_perturbes, liste_erreurs)
    """
    formes_out = [t.forme for t in tokens]
    erreurs: list[Erreur] = []
    positions_modifiees: set[int] = set()

    # Pre-calculer les indices candidats par type
    indices_par_type = _indices_par_type(tokens)

    tentatives = 0
    while len(erreurs) < n_erreurs and tentatives < max_tentatives:
        tentatives += 1

        # 1) Choisir le type d'erreur
        type_err = _choisir_type_erreur()

        # 2) Verifier qu'on a des candidats pour ce type
        candidats = [i for i in indices_par_type.get(type_err, [])
                     if i not in positions_modifiees]
        if not candidats:
            continue

        # 3) Choisir un mot candidat
        idx = random.choice(candidats)

        # 4) Appliquer la perturbation
        erreur: Erreur | None = None

        if type_err == "HOMO":
            erreur = perturber_homophone(tokens, idx)
        elif type_err == "ACC":
            if random.random() < 0.6:
                erreur = perturber_accord_nombre(tokens, idx, lexique)
            else:
                erreur = perturber_accord_genre(tokens, idx, lexique)
        elif type_err == "CONJ":
            erreur = perturber_conjugaison(tokens, idx, lexique)
        elif type_err == "PP":
            erreur = perturber_pp_infinitif(tokens, idx, lexique)
        elif type_err == "PHON":
            erreur = perturber_phonetique(tokens, idx, lexique)
        elif type_err == "ACCENT":
            erreur = perturber_accent(tokens, idx)
        elif type_err == "TYPO":
            erreur = perturber_typo(tokens, idx)

        if erreur is not None:
            formes_out[erreur.position] = erreur.perturbe
            erreurs.append(erreur)
            positions_modifiees.add(erreur.position)

    return formes_out, erreurs
