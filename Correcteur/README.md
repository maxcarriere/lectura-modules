# Lectura Correcteur

Correcteur orthographique et grammatical du francais. Pipeline a base de regles linguistiques avec support optionnel de modeles statistiques (BiLSTM, n-gram).

## Installation

```bash
pip install lectura-correcteur
```

Dependance : `lectura-lexique` (installe automatiquement).

## Utilisation rapide

```python
from lectura_lexique import Lexique
from lectura_correcteur import Correcteur, CorrecteurConfig

lex = Lexique("lexique.db")
correcteur = Correcteur(lex)

result = correcteur.corriger("Les enfant mange des pomme.")
print(result.phrase_corrigee)  # "Les enfants mangent des pommes."

for c in result.corrections:
    print(f"  {c.original} -> {c.corrige} ({c.type_correction.value})")
```

## Types de corrections

| Type | Exemples |
|------|----------|
| **Orthographe** | Mots hors lexique, distance d'edition 1-2, fautes AZERTY |
| **Accords** | Determinant-nom, adjectif-nom (genre et nombre) |
| **Conjugaison** | Accord sujet-verbe, terminaisons verbales |
| **Homophones** | a/a, est/et, son/sont, ou/ou, ce/se, ces/ses... |
| **Participes passes** | Accord avec avoir/etre, COD anteposes |
| **Resegmentation** | Apostrophes SMS (jai → j'ai), agglutinations |

## Configuration

```python
config = CorrecteurConfig(
    activer_orthographe=True,     # Verification lexicale (OOV)
    activer_grammaire=True,       # Accords, conjugaison, homophones
    activer_resegmentation=True,  # Apostrophes et agglutinations
    activer_azerty=True,          # Corrections specifiques clavier AZERTY
    max_suggestions=5,            # Nombre max de suggestions par mot
    activer_editeur_homophones=True,  # BiLSTM (si modele present)
    activer_lm=True,              # Modele de langue n-gram (si present)
)

correcteur = Correcteur(lex, config=config)
```

## Mode sans modeles

Le correcteur fonctionne sans fichiers de modeles (mode regles uniquement).
Les modeles optionnels (BiLSTM, n-gram) ameliorent la precision sur les homophones
mais ne sont pas necessaires. Si les fichiers sont absents, le correcteur
se rabat automatiquement sur les regles linguistiques.

## Dependances

- `lectura-lexique` : acces au lexique francais (formes, frequences, POS, morphologie)

## Licence

Ce module est distribue sous licence **AGPL-3.0** (non commerciale) — voir [LICENCE.txt](LICENCE.txt).

Pour un usage commercial, contacter [contact@lec-tu-ra.com](mailto:contact@lec-tu-ra.com).
