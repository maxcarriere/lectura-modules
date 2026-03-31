"""Constantes pour les regles grammaticales."""

from __future__ import annotations

# Determinants pluriels / singuliers
PLUR_DET = frozenset({
    "les", "des", "ces", "ses", "mes", "tes",
    "nos", "vos", "aux", "plusieurs", "quelques",
})

SING_FEM_DET = frozenset({
    "la", "une", "cette", "sa", "ma", "ta",
})

# Pronoms sujets -> personne attendue
PRONOM_PERSONNE: dict[str, tuple[str, str]] = {
    # pronom -> (personne, nombre)
    "je": ("1", ""), "j'": ("1", ""),
    "tu": ("2", ""),
    "il": ("3", "s"), "elle": ("3", "s"), "on": ("3", "s"),
    "nous": ("1", "p"),
    "vous": ("2", "p"),
    "ils": ("3", "p"), "elles": ("3", "p"),
}

# Pronoms sujets 3e personne du pluriel
SUJETS_3PL = frozenset({"ils", "elles"})

# Mots invariables
INVARIABLES = frozenset({
    "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
    "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "vingt", "trente", "quarante", "cinquante", "soixante",
    "cent", "mille", "chose",
})
