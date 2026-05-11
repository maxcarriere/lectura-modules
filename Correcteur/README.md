# Lectura Correcteur

Correcteur orthographique et grammatical du français. Pipeline à base de règles linguistiques avec support optionnel de modèles statistiques (BiLSTM, n-gram).

## Installation

```bash
pip install lectura-correcteur
```

Trois modes de fonctionnement :

1. **Lexique complet** (`lectura-lexique`) — meilleure couverture, ~900 Mo
2. **Lexique léger** (SQLite intégré) — autonome, ~50 Mo, inclus dans le wheel privé
3. **API** — aucune dépendance locale, requiert un serveur Lectura

La factory `creer_correcteur()` détecte automatiquement le mode disponible.

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

Évaluation GEC débiaisée sur 180 phrases (158 erronées, 22 correctes).

| Correcteur | Précision | Rappel | F0.5 | F1 |
|------------|-----------|--------|------|-----|
| **Lectura** (règles) | **0.790** | 0.599 | **0.742** | 0.681 |
| **Lectura** (règles + scoring) | 0.782 | **0.633** | 0.747 | **0.700** |
| Grammalecte | 0.465 | 0.388 | 0.447 | 0.423 |
| Baseline (ne rien faire) | 1.000 | 0.000 | 0.000 | 0.000 |

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

- `lectura-lexique` (optionnel) : lexique complet (~900 Mo) — `pip install lectura-correcteur[sqlite]`
- Sans dépendance : le correcteur utilise le lexique léger intégré ou l'API Lectura

## Licence

Ce module est distribué sous licence **AGPL-3.0** (non commerciale) — voir [LICENCE.txt](LICENCE.txt).

Pour un usage commercial, contacter [contact@lec-tu-ra.com](mailto:contact@lec-tu-ra.com).
