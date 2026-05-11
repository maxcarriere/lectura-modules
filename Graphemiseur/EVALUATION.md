# Evaluation detaillée

Résultats d'évaluation du modèle unifié v1.0.0 (architecture v2 avec word feedback) sur le jeu de test (UD French-GSD + Sequoia + Rhapsodie, test set enrichi, 1 712 phrases, ~25K mots).

## P2G — Phonème vers Graphème

| Métrique | Score |
|----------|-------|
| **Word Accuracy** | **93.1%** |
| **CER** (Character Error Rate) | **2.2%** |
| Mots évalués | 25 218 |

Le P2G est évalué en contexte phrastique : le modèle voit la phrase IPA entière, ce qui améliore la désambiguïsation des homophones et l'accord morphologique.

## POS — Étiquetage morpho-syntaxique

| Métrique | Score |
|----------|-------|
| **Accuracy** | **97.0%** |
| Tokens évalués | 27 875 |

Jeu de 19 tags projet : ART:def, ART:ind, PRO:per, NOM, VER, ADJ, ADV, PRE, CON, AUX, etc.

## Morphologie

| Trait | Accuracy | Tokens évalués |
|-------|----------|----------------|
| **Number** | **92.8%** | 14 661 |
| **Gender** | **92.0%** | 5 782 |
| **VerbForm** | **96.2%** | 3 942 |
| **Mood** | **93.5%** | 2 370 |
| **Tense** | **94.1%** | 2 387 |
| **Person** | **96.6%** | 4 489 |

## Comparaison v1 / v2

| Config | Word Acc | CER | Paramètres |
|--------|----------|-----|------------|
| v1 (sans word feedback) | 88.9% | 3.49% | 2.10M |
| v1 + post-traitement morpho | 89.5% | 3.31% | 2.10M |
| **v2 (word feedback)** | **93.1%** | **2.19%** | **2.56M** |
| v2 + post-traitement contextuel | 93.2% | — | 2.56M |

Le word feedback (+4.2 points) est la principale amélioration : les représentations de mots du BiLSTM word-level sont diffusées aux positions char, permettant à la tête P2G d'utiliser le contexte syntaxique pour les marques morphologiques muettes.

## Analyse des erreurs restantes (v2)

| Type d'erreur | Proportion | Exemple |
|---------------|-----------|---------|
| Marques morpho muettes (-s, -e, -nt) | ~30% | /medsɛ̃/ → "médecin" au lieu de "médecins" |
| Homophones vrais | ~10% | /e/ → "et" au lieu de "est" |
| Graphies complexes/irrégulières | ~40% | Mots rares, emprunts |
| Autres | ~20% | — |

L'accord déterminant-nom est correct dans 98.4% des cas (53 désaccords sur 3 373 paires analysées), confirmant que le modèle exploite bien le contexte inter-mots.

## Taille du modèle

| Format | v1 | v2 |
|--------|----|----|
| Paramètres | 2 102 433 | 2 562 465 |
| ONNX INT8 | 2.1 Mo | **2.6 Mo** |
| JSON weights | 21 Mo | 25.4 Mo |
| Vocabulaire | 20 Ko | 20 Ko |

## Méthodologie

- **Données** : UD French-GSD + Sequoia + Rhapsodie enrichis avec phonèmes IPA (GLAFF + Lexique + G2P)
- **Entraînement** : Phase 1 (1.06M mots lexique, P2G seul) + Phase 2 (18K phrases multi-tâche)
- **Split** : train/dev/test standard
- **Vérification croisée** : ONNX vérifié (max diff = 0.000010 vs PyTorch)
