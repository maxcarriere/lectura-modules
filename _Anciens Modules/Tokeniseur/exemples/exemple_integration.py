"""Exemple d'intégration — Lectura Tokeniseur.

Montre comment utiliser le tokeniseur dans un pipeline NLP.
"""

from lectura_tokeniseur import (
    LecturaTokeniseur,
    Mot,
    Nombre,
    Ponctuation,
    Separateur,
    Sigle,
)


tok = LecturaTokeniseur()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Pré-traitement pour un modèle NLP
# ══════════════════════════════════════════════════════════════════════════════

print("=== Pipeline NLP ===\n")

text = "L'intelligence artificielle transforme le monde en 2025."
result = tok.analyze(text)

# Extraire les features par token
for t in result.tokens:
    if isinstance(t, Mot):
        features = {
            "word": t.ortho,
            "is_title": t.text[0].isupper(),
            "length": len(t.text),
            "position": t.span[0],
        }
        print(f"  {features}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Reconstruction du texte depuis les tokens (round-trip)
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Round-trip ===\n")

text = "L'enfant mange-t-il du chocolat ?"
result = tok.analyze(text)

# Les spans permettent de reconstruire le texte exact
reconstructed = ""
for t in result.tokens:
    reconstructed += t.text
print(f"  Original    : {result.texte_normalise!r}")
print(f"  Reconstruit : {reconstructed!r}")
print(f"  Identique   : {reconstructed == result.texte_normalise}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Statistiques textuelles
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Statistiques ===\n")

texte = """
Les enfants jouent dans la cour. Ils sont 25 et ils adorent
le chocolat ! Le SNCF transporte 3 millions de voyageurs.
C'est-à-dire qu'il y a beaucoup de monde...
"""

result = tok.analyze(texte)
n_mots = sum(1 for t in result.tokens if isinstance(t, Mot))
n_nombres = sum(1 for t in result.tokens if isinstance(t, Nombre))
n_sigles = sum(1 for t in result.tokens if isinstance(t, Sigle))
n_ponct = sum(1 for t in result.tokens if isinstance(t, Ponctuation))

print(f"  Mots        : {n_mots}")
print(f"  Nombres     : {n_nombres}")
print(f"  Sigles      : {n_sigles}")
print(f"  Ponctuation : {n_ponct}")
print(f"  Total tokens: {result.nb_tokens}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Chaînage avec Lectura Syllabeur (si disponible)
# ══════════════════════════════════════════════════════════════════════════════

# from lectura_syllabeur import LecturaSyllabeur
#
# syl = LecturaSyllabeur()
# tok = LecturaTokeniseur()
#
# text = "Le chat mange la souris"
# result = tok.analyze(text)
#
# for t in result.mots:
#     analyse = syl.analyze(t.text)
#     print(f"  {t.text:12s} → {analyse.format_simple()}")
