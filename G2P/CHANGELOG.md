# Changelog

## v1.0.0 — 2025

Première version publique du modèle unifié G2P+POS+Morpho+Liaison.

### Modèle

- Architecture BiLSTM char-level multi-tête, 1.75M paramètres
- Export ONNX INT8 (1.8 Mo)
- 3 backends d'inférence : ONNX Runtime, NumPy, pur Python

### Performances (test set)

- **G2P** : 98.5% word accuracy, 0.54% PER
- **POS** : 98.2% accuracy (19 tags)
- **Liaison** : 90.6% macro F1
- **Morphologie** : 95.1-99.2% selon le trait

### Entraînement

- Phase 1 : pré-entraînement G2P sur 890K mots du lexique (GLAFF + Lexique)
- Phase 2 : fine-tuning multi-tâche sur 18K phrases UD French-GSD enrichi
- Optimisations : label smoothing, LR warmup, class weights liaison, early stopping

### Package

- Package Python `lectura-g2p` avec dépendances optionnelles
- CLI interactive (`demo_cli.py`)
- Exemples d'intégration
- Licence CC BY-SA 4.0
