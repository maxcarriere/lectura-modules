# Changelog

## v1.0.0 — 2026

Première version publique du modèle unifié P2G+POS+Morpho.

### Modèle

- Architecture BiLSTM char-level multi-tête avec word feedback, 2.56M paramètres
- Export ONNX INT8 (2.6 Mo)
- 3 backends d'inférence : ONNX Runtime, NumPy, pur Python

### Performances (test set)

- **P2G** : 93.1% word accuracy, 2.2% CER
- **POS** : 97.0% accuracy (19 tags)
- **Morphologie** : 92.0-96.6% selon le trait

### Architecture v2 (word feedback)

- Les représentations mot (BiLSTM word-level) sont diffusées aux positions char avant la tête P2G
- Gain de +4.2 points de Word Accuracy par rapport à v1 (88.9% → 93.1%)
- Implémentation vectorisée (torch.gather) compatible ONNX

### Entraînement

- Phase 1 : pré-entraînement P2G sur 1.06M mots du lexique (GLAFF + Lexique)
- Phase 2 : fine-tuning multi-tâche sur 18K phrases (UD French-GSD + Sequoia + Rhapsodie)
- Optimisations : label smoothing, LR warmup cosine, early stopping

### Post-traitement

- Post-traitement contextuel inter-mots (accord dét-nom, sujet-verbe) avec filtre lexique
- Gain modeste sur v2 (+0.1%) car le word feedback capture déjà l'essentiel

### Package

- Package Python `lectura-p2g` avec dépendances optionnelles
- CLI interactive (`demo_cli.py`)
- Licence CC BY-SA 4.0
