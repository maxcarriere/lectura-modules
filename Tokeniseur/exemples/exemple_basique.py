"""Exemple basique — Lectura Tokeniseur Complet.

Montre les cas d'usage les plus courants, incluant la detection de formules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_tokeniseur import (
    LecturaTokeniseur, Mot, Separateur, Formule, FormuleType,
)

tok = LecturaTokeniseur()

# -- Normalisation seule --
print("=== Normalisation ===\n")

exemples = [
    "L'enfant  mange...du  chocolat",
    '"Bonjour" dit-il.',
    "Il a 1 000 000 euros.",
    "C'est-a-dire ( oui ) !",
]
for text in exemples:
    print(f"  {text!r}")
    print(f"  -> {tok.normalize(text)!r}")
    print()


# -- Tokenisation complete --
print("=== Tokenisation ===\n")

result = tok.analyze("L'enfant mange-t-il du chocolat ?")
print(result.format_table())
print()

# -- Donnees structurees --
print("=== Donnees structurees ===\n")

for t in result.tokens:
    if isinstance(t, Mot):
        print(f"  MOT: <<{t.text}>> ortho={t.ortho!r} span={t.span}")
    elif isinstance(t, Separateur):
        print(f"  SEP: <<{t.text}>> type={t.sep_type} span={t.span}")
    elif isinstance(t, Formule):
        print(f"  FOR: <<{t.text}>> sous-type={t.formule_type.value} val={t.valeur!r} span={t.span}")

# -- Detection de formules --
print(f"\n=== Formules detectees ===\n")

text = "Appeler le SNCF au 06 12 34 56 78 avant le 15/03/2024 pour le 42e billet a 3/4 du prix."
result = tok.analyze(text)
for f in result.formules:
    print(f"  {f.text:20s}  {f.formule_type.value:15s}  val={f.valeur!r}")

# -- Formules mathematiques --
print(f"\n=== Formules mathematiques ===\n")

for expr in ["2x+3=0", "sin(x)", "3.14e-5", "x-2=0"]:
    result = tok.analyze(f"Calculer {expr} maintenant.")
    for f in result.formules:
        print(f"  {f.text:15s}  {f.formule_type.value:15s}")

# -- Extraction simple des mots --
print(f"\n=== Mots extraits ===\n")
print(tok.extract_words("La SNCF transporte 3 millions de voyageurs."))

# -- Extraction des formules --
print(f"\n=== Liste des formules ===\n")
formules = tok.extract_formules("Le 15/03/2024, la SNCF a vendu 42 billets pour 3/4 du tarif.")
for f in formules:
    print(f"  {f.text:15s}  -> {f.formule_type.value}")
