"""Exemple 1 — Tokeniser et normaliser du texte francais.

pip install lectura-tokeniseur
"""

from lectura_tokeniseur import (
    LecturaTokeniseur,
    normalise,
    tokenise,
    Mot,
    Formule,
)

# --- Utilisation rapide (fonctions) ---

texte = 'Le prof.  a dit:  "Lisez les pages 12 à 42" .'

# Normaliser le texte (espaces, guillemets, ponctuation)
texte_propre = normalise(texte)
print("Normalise :", texte_propre)

# Tokeniser en une liste de tokens types
tokens = tokenise(texte_propre)
for t in tokens:
    print(f"  {t.type.value:12s}  {t.text!r}")

# --- Utilisation via la classe (recommande) ---

tk = LecturaTokeniseur()
resultat = tk.analyze("J'ai appele le 06 12 34 56 78 le 01/01/2025.")

print(f"\nMots      : {resultat.nb_mots}")
print(f"Formules  : {len(resultat.formules)}")
print(f"Mots extraits : {resultat.words()}")

for f in resultat.formules:
    print(f"  Formule {f.formule_type.value} : {f.text!r} (valeur={f.valeur})")
