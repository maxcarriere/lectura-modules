# Évaluation — Lectura POS Tagger v1.0

## Protocole

- **Modèle** : CRF (Conditional Random Fields) + mini-lexique de correction (31 entrées)
- **Données de test** : Split test des corpus Universal Dependencies pour le français
- **Métriques** : Accuracy globale, Précision / Rappel / F1 par tag
- **Tagset** : 18 catégories (hors PUNCT et SPACE)

## Résultats globaux

| Corpus | Phrases | Tokens | Accuracy |
|--------|---------|--------|----------|
| **UD French-GSD** (textes web/wiki) | 416 | 8 831 | **97.7%** |
| **UD French-Sequoia** (textes européens) | 456 | 8 960 | **98.3%** |
| **UD French-Rhapsodie** (oral transcrit) | 840 | 10 084 | **96.6%** |
| **Fusionné** | **1 712** | **27 875** | **97.5%** |

## Résultats par catégorie (test fusionné, 27 875 tokens)

| Tag | Précision | Rappel | F1 | Support |
|-----|-----------|--------|----|---------|
| ART:def | 99.7% | 99.9% | 99.8% | 2 865 |
| ART:ind | 97.5% | 97.1% | 97.3% | 851 |
| PRO:per | 99.7% | 98.8% | 99.2% | 1 497 |
| PRO:rel | 92.2% | 90.6% | 91.4% | 363 |
| PRO:dem | 99.5% | 99.8% | 99.6% | 403 |
| PRO:ind | 91.4% | 95.3% | 93.3% | 235 |
| PRO:int | 66.7% | 54.5% | 60.0% | 22 |
| ADJ:pos | 100.0% | 99.7% | 99.9% | 356 |
| ADJ:dem | 100.0% | 99.5% | 99.8% | 218 |
| ADJ:int | 100.0% | 100.0% | 100.0% | 2 |
| NOM | 98.2% | 98.2% | 98.2% | 7 541 |
| ADJ | 93.8% | 92.0% | 92.9% | 1 741 |
| VER | 96.0% | 94.7% | 95.4% | 2 761 |
| AUX | 93.9% | 97.9% | 95.9% | 1 203 |
| ADV | 96.8% | 96.0% | 96.4% | 1 817 |
| PRE | 99.0% | 99.4% | 99.2% | 4 342 |
| CON | 96.1% | 97.2% | 96.6% | 1 303 |
| INTJ | 93.3% | 97.7% | 95.5% | 355 |

## Confusions principales

| Confusion | Occurrences | Commentaire |
|-----------|-------------|-------------|
| VER → AUX | 75 | Confusion être/avoir |
| ADJ ↔ NOM | 128 | Ambiguïté classique du français |
| ADJ → VER | 46 | Participes passés |
| PRO:rel ↔ CON | 51 | "que" relatif vs conjonctif |

## Données d'entraînement

| Corpus | Licence | Usage |
|--------|---------|-------|
| UD French-GSD | CC BY-SA 4.0 | Train + Dev + Test |
| UD French-Sequoia | LGPL-LR | Train + Dev + Test |
| UD French-Rhapsodie | CC BY-SA 4.0 | Train + Dev + Test |
