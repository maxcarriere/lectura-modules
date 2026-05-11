# Évaluation détaillée

Résultats d'évaluation du modèle unifié v1.0.0 sur le jeu de test (UD French-GSD test set enrichi, 1 712 phrases, ~28K tokens).

## G2P — Graphème vers Phonème

| Métrique | Score |
|----------|-------|
| **Word Accuracy** | **98.5%** |
| **PER** (Phone Error Rate) | **0.54%** |
| Mots évalués | 25 113 |

Le G2P est évalué en contexte phrastique : le modèle voit la phrase entière, ce qui améliore la désambiguïsation des homographes.

## POS — Étiquetage morpho-syntaxique

| Métrique | Score |
|----------|-------|
| **Accuracy** | **98.2%** |
| Tokens évalués | 27 875 |

Jeu de 19 tags projet : ART:def, ART:ind, PRO:per, NOM, VER, ADJ, ADV, PRE, CON, AUX, etc.

## Liaison

| Classe | Précision | Rappel | F1 | TP | FP | FN |
|--------|-----------|--------|-----|-----|-----|-----|
| **Lz** (les‿enfants) | 96.6% | 99.1% | **97.9%** | 570 | 20 | 5 |
| **Lt** (est‿il) | 96.7% | 98.7% | **97.7%** | 236 | 8 | 3 |
| **Ln** (bon‿ami) | 97.5% | 99.0% | **98.3%** | 197 | 5 | 2 |
| **Lr** (premier‿étage) | 85.7% | 100% | **92.3%** | 6 | 1 | 0 |
| **Lp** (trop‿aimable) | 50.0% | 100% | **66.7%** | 1 | 1 | 0 |
| **Macro** | **85.3%** | **99.4%** | **90.6%** | — | — | — |

Note : Lp est extrêmement rare dans le corpus (1 seul exemple de test), le score n'est pas significatif.

## Morphologie

| Trait | Accuracy | Tokens évalués |
|-------|----------|----------------|
| **Number** | **97.0%** | 14 661 |
| **Gender** | **95.1%** | 5 782 |
| **VerbForm** | **98.8%** | 3 942 |
| **Mood** | **97.7%** | 2 370 |
| **Tense** | **97.8%** | 2 387 |
| **Person** | **99.2%** | 4 489 |

## Comparaison avec le pipeline séparé

| Aspect | Unifié v1.0 | Pipeline séparé |
|--------|-------------|-----------------|
| G2P Word Acc | **98.5%** | 80.8%* |
| POS Accuracy | **98.2%** | 97.9-98.1% |
| Morphologie | **95-99%** | non disponible |
| Liaison | **90.6% F1** | règles manuelles |
| Taille totale | **1.8 Mo** | ~8 Mo |
| Passes d'inférence | **1** | 3 |
| Dépendances min. | **0** | CRF + ONNX Runtime |

*Le G2P séparé est évalué sur dictionnaire (281K mots isolés), non comparable directement.

## Taille du modèle

| Format | Taille |
|--------|--------|
| Paramètres | 1 747 108 |
| ONNX INT8 | **1.8 Mo** |
| JSON weights | 17.3 Mo |
| Vocabulaire | 5 Ko |

## Méthodologie

- **Données** : UD French-GSD enrichi avec phonèmes (GLAFF + Lexique) et liaisons
- **Entraînement** : Phase 1 (890K mots lexique, G2P seul) + Phase 2 (18K phrases multi-tâche)
- **Split** : train/dev/test standard UD French-GSD
- **Vérification croisée** : les 3 backends (ONNX, NumPy, Pure Python) produisent des résultats identiques
