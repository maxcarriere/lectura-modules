# Licence des Modeles Pre-entraines Lectura

## Fichiers concernes

Ce document couvre les fichiers de modeles pre-entraines distribues
avec les modules Lectura NLP :

- `Phonemiseur/modeles/*.onnx` — Modele unifie G2P+POS+Morpho+Liaison
- `Phonemiseur/modeles/*_weights.json` — Poids pour les backends NumPy/Pure
- `Phonemiseur/modeles/*_vocab.json` — Vocabulaires
- `Phonemiseur/modeles/*_corrections*.json` — Tables de corrections
- `Graphemiseur/modeles/*.onnx` — Modele unifie P2G+POS+Morpho
- `Graphemiseur/modeles/*_weights.json` — Poids pour les backends NumPy/Pure
- `Graphemiseur/modeles/*_vocab.json` — Vocabulaires

## Conditions d'utilisation

### Usage autorise

Ces modeles sont fournis pour etre utilises **exclusivement** via les
packages Lectura NLP (`lectura-phonemiseur`, `lectura-graphemiseur`),
sous les termes de :

- L'**AGPL-3.0** (usage open-source avec copyleft), ou
- La **Licence Commerciale Lectura** (usage proprietaire).

### Usage interdit sans licence commerciale

- Extraction et utilisation des modeles independamment du code Lectura.
- Integration des fichiers de modeles dans un autre logiciel ou service.
- Redistribution separée des modeles hors du package Lectura.
- Fine-tuning, distillation, ou creation de modeles derives.
- Utilisation des modeles pour entrainer d'autres modeles.

### Licence commerciale

Une licence commerciale couvrant les modeles est disponible.
Voir `LICENCE-COMMERCIALE.md` ou https://www.lec-tu-ra.com/solutions/services/

---

Copyright (C) 2025 Max Carriere — Tous droits reserves.
