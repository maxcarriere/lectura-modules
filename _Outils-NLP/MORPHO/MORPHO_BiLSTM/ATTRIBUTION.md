# Attribution

## Lectura Morpho Tagger (BiLSTM)

Copyright (c) 2025 Lectura.

Distribué sous licence [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Données d'entraînement

Le modèle a été entraîné sur les corpus suivants du projet
[Universal Dependencies](https://universaldependencies.org/) :

### UD French-GSD

- **Licence** : CC BY-SA 4.0
- **Source** : https://github.com/UniversalDependencies/UD_French-GSD
- **Citation** :
  > Guillaume, B., de Marneffe, M.-C., & Perrier, G. (2019).
  > Conversion et améliorations de corpus du français annotés en
  > Universal Dependencies. *Traitement Automatique des Langues*, 60(2).

### UD French-Sequoia

- **Licence** : LGPL-LR
- **Source** : https://github.com/UniversalDependencies/UD_French-Sequoia
- **Citation** :
  > Candito, M., & Seddah, D. (2012).
  > Le corpus Sequoia : annotation syntaxique et exploitation pour
  > l'adaptation d'analyseur par pont lexical.
  > *TALN 2012*, Grenoble.

### UD French-Rhapsodie

- **Licence** : CC BY-SA 4.0
- **Source** : https://github.com/UniversalDependencies/UD_French-Rhapsodie
- **Citation** :
  > Lacheret, A., Kahane, S., Beliao, J., Dister, A., Gerdes, K.,
  > Goldman, J.-P., Obin, N., Pietrandrea, P., & Tchobanov, A. (2014).
  > Rhapsodie: a Prosodic-Syntactic Treebank for Spoken French.
  > *LREC 2014*, Reykjavik.

## Bibliothèques utilisées

- **ONNX Runtime** (MIT) — Inférence du modèle BiLSTM
- **NumPy** (BSD) — Manipulation de tableaux numériques
- **PyTorch** (BSD) — Entraînement du modèle (développement uniquement)
