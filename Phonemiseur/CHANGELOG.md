# Changelog

## v4.0.0 — 2026

**Renommage et refactorisation architecturale.**

### Breaking changes

- **Renommage** : `lectura-g2p` → `lectura-phonemiseur`, `lectura_nlp` → `lectura_phonemiseur`
- Les anciens imports `from lectura_nlp import ...` ne fonctionnent plus
- Un package de transition `lectura-g2p` 3.4.0 est disponible sur PyPI pour faciliter la migration

### Ajouts

- **Groupes de lecture** : `construire_groupes_lecture()` transfere depuis l'Aligneur
- **Schwa pedagogique** : `ajouter_schwa_final()` transfere depuis l'Aligneur
- **Pipeline G2P** : nouveau package `lectura-g2p` (couche 2) orchestre tokeniseur + formules + phonemiseur

### Migration

```python
# Avant (v3.x)
from lectura_nlp import creer_engine

# Apres (v4.0.0+)
from lectura_phonemiseur import creer_engine
```

---

## v1.0.0 — 2025

Premiere version publique du modele unifie G2P+POS+Morpho+Liaison.

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

- Package Python `lectura-phonemiseur` avec dépendances optionnelles
- CLI interactive (`demo_cli.py`)
- Exemples d'intégration
- Licence CC BY-SA 4.0
