# Lectura Correcteur

Correcteur orthographique et grammatical du français. Pipeline à base de règles linguistiques avec support optionnel de modèles statistiques (BiLSTM, n-gram).

## Installation

```bash
pip install lectura-correcteur
```

Dépendance : `lectura-lexique` (installée automatiquement).

## Utilisation rapide

```python
from lectura_correcteur import creer_correcteur

# Mode automatique : local si lexique disponible, sinon API
correcteur = creer_correcteur()

result = correcteur.corriger("Les enfant mange des pomme.")
print(result.phrase_corrigee)  # "Les enfants mangent des pommes."

for c in result.corrections:
    print(f"  {c.original} -> {c.corrige} ({c.type_correction.value})")
```

Avec un lexique local :

```python
from lectura_lexique import Lexique
from lectura_correcteur import Correcteur

lex = Lexique("lexique.db")
correcteur = Correcteur(lex)
result = correcteur.corriger("Les enfant mange des pomme.")
```

## Types de corrections

| Type | Exemples |
|------|----------|
| **Orthographe** | Mots hors lexique, distance d'édition 1-2, fautes AZERTY |
| **Accords** | Déterminant-nom, adjectif-nom (genre et nombre) |
| **Conjugaison** | Accord sujet-verbe, terminaisons verbales |
| **Homophones** | à/a, est/et, son/sont, où/ou, ce/se, ces/ses... |
| **Participes passés** | Accord avec avoir/être, COD antéposés |
| **Resegmentation** | Apostrophes SMS (jai → j'ai), agglutinations |

## Benchmark comparatif

Évaluation sur 800 phrases issues de Wicopaco (erreurs réelles Wikipedia français).

| Correcteur | Précision | Rappel | F1 | F0.5 | Vitesse |
|------------|-----------|--------|----|------|---------|
| **Lectura** (règles + modèles) | **0.94** | **0.73** | **0.82** | **0.89** | ~55 ms/phrase |
| **Lectura** (règles seules) | **0.93** | **0.65** | **0.77** | **0.86** | ~15 ms/phrase |
| Grammalecte | 0.54 | 0.26 | 0.35 | 0.44 | ~40 ms/phrase |
| LanguageTool | 0.30 | 0.37 | 0.33 | 0.31 | ~12 600 ms/phrase |

## Configuration

```python
from lectura_correcteur import CorrecteurConfig

config = CorrecteurConfig(
    activer_orthographe=True,     # Vérification lexicale (OOV)
    activer_grammaire=True,       # Accords, conjugaison, homophones
    activer_resegmentation=True,  # Apostrophes et agglutinations
    activer_azerty=True,          # Corrections spécifiques clavier AZERTY
    max_suggestions=5,            # Nombre max de suggestions par mot
    activer_editeur_homophones=True,  # BiLSTM (si modèle présent)
    activer_lm=True,              # Modèle de langue n-gram (si présent)
)

correcteur = Correcteur(lex, config=config)
```

## Mode sans modèles

Le correcteur fonctionne sans fichiers de modèles (mode règles uniquement).
Les modèles optionnels (BiLSTM, n-gram) améliorent la précision sur les homophones
mais ne sont pas nécessaires. Si les fichiers sont absents, le correcteur
se rabat automatiquement sur les règles linguistiques.

## Compatibilité lexique

Le correcteur fonctionne avec n'importe quelle base lexicale chargée via `lectura-lexique` (Lexique383, GLAFF, LeXiK, ou tout lexique au format compatible).

## Dépendances

- `lectura-lexique` : accès au lexique français (formes, fréquences, POS, morphologie)

## Licence

Ce module est distribué sous licence **AGPL-3.0** (non commerciale) — voir [LICENCE.txt](LICENCE.txt).

Pour un usage commercial, contacter [contact@lec-tu-ra.com](mailto:contact@lec-tu-ra.com).
