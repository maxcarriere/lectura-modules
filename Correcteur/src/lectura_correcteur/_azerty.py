"""Matrice d'adjacence clavier AZERTY francais.

Fournit ratio_adjacence_azerty() pour le scoring : quand un candidat ne differe
que par des substitutions de touches AZERTY-voisines, il est favorise.

Fournit generer_variantes_azerty() pour la generation de candidats : pour chaque
position du mot, substitue le caractere par chaque voisin AZERTY.
"""

from __future__ import annotations

# Clavier AZERTY francais (3 rangees de lettres)
# Rangee 2 : a z e r t y u i o p
# Rangee 3 : q s d f g h j k l m
# Rangee 4 : w x c v b n
#
# Chaque touche est adjacente a ses voisines horizontales + diagonales.

AZERTY_ADJACENCE: dict[str, frozenset[str]] = {
    # Rangee 2
    "a": frozenset({"z", "q"}),
    "z": frozenset({"a", "e", "q", "s"}),
    "e": frozenset({"z", "r", "s", "d"}),
    "r": frozenset({"e", "t", "d", "f"}),
    "t": frozenset({"r", "y", "f", "g"}),
    "y": frozenset({"t", "u", "g", "h"}),
    "u": frozenset({"y", "i", "h", "j"}),
    "i": frozenset({"u", "o", "j", "k"}),
    "o": frozenset({"i", "p", "k", "l"}),
    "p": frozenset({"o", "l", "m"}),
    # Rangee 3
    "q": frozenset({"a", "z", "s", "w", "x"}),
    "s": frozenset({"z", "e", "q", "d", "x", "w"}),
    "d": frozenset({"e", "r", "s", "f", "x", "c"}),
    "f": frozenset({"r", "t", "d", "g", "c", "v"}),
    "g": frozenset({"t", "y", "f", "h", "v", "b"}),
    "h": frozenset({"y", "u", "g", "j", "b", "n"}),
    "j": frozenset({"u", "i", "h", "k", "n"}),
    "k": frozenset({"i", "o", "j", "l"}),
    "l": frozenset({"o", "p", "k", "m"}),
    "m": frozenset({"p", "l"}),
    # Rangee 4
    "w": frozenset({"q", "s", "x"}),
    "x": frozenset({"w", "q", "s", "d", "c"}),
    "c": frozenset({"x", "d", "f", "v"}),
    "v": frozenset({"c", "f", "g", "b"}),
    "b": frozenset({"v", "g", "h", "n"}),
    "n": frozenset({"b", "h", "j"}),
}


def _sont_adjacentes(c1: str, c2: str) -> bool:
    """Verifie si deux caracteres sont sur des touches AZERTY adjacentes."""
    c1_low = c1.lower()
    c2_low = c2.lower()
    voisins = AZERTY_ADJACENCE.get(c1_low)
    if voisins is None:
        return False
    return c2_low in voisins


def ratio_adjacence_azerty(original: str, candidat: str) -> float:
    """Calcule le ratio de substitutions AZERTY-adjacentes entre deux mots.

    Aligne les deux mots caractere par caractere. Compte les positions ou
    les caracteres different. Parmi celles-ci, compte combien sont des
    substitutions de touches AZERTY adjacentes.

    Retourne :
    - 0.5 si les mots sont identiques (neutre)
    - nb_adjacentes / nb_substitutions sinon (0.0 a 1.0)

    Pour les mots de longueurs differentes (insertion/deletion), seules
    les positions alignees sont comparees.
    """
    orig_low = original.lower()
    cand_low = candidat.lower()

    if orig_low == cand_low:
        return 0.5  # neutre

    min_len = min(len(orig_low), len(cand_low))
    if min_len == 0:
        return 0.0

    n_subs = 0
    n_adjacentes = 0

    for i in range(min_len):
        if orig_low[i] != cand_low[i]:
            n_subs += 1
            if _sont_adjacentes(orig_low[i], cand_low[i]):
                n_adjacentes += 1

    # Compter les caracteres excedentaires comme des non-adjacentes
    n_subs += abs(len(orig_low) - len(cand_low))

    if n_subs == 0:
        return 0.5  # identiques

    return n_adjacentes / n_subs


def generer_variantes_azerty(mot: str) -> list[str]:
    """Genere toutes les variantes par substitution AZERTY d'un caractere.

    Pour chaque position du mot, remplace le caractere par chaque voisin
    AZERTY. Retourne ~5*len(mot) variantes (moy ~5 voisins par touche).
    """
    mot_low = mot.lower()
    variantes: list[str] = []
    for i, c in enumerate(mot_low):
        voisins = AZERTY_ADJACENCE.get(c)
        if voisins is None:
            continue
        for v in voisins:
            variantes.append(mot_low[:i] + v + mot_low[i + 1:])
    return variantes
