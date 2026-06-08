# Changelog

## v4.3.0 — 2026-06

**Modele V7 + nettoyage.**

### Modele

- Modele V7 : 3.2M params, attention cross word-char, POS 98%, morpho 95-98%
- Lex_select : selection lexique par tete neuronale (95% word accuracy)
- Fallback V6 automatique si V7 absent

### Post-traitement

- Benchmark POS/morpho : les regles etendues (ces/ses, morpho lex) sont neutres/regressives — non activees
- Homophones a/a et ou/ou restent inconditionnels (valides)

### Nettoyage

- Archivage des modeles v3-v5, checkpoints .pt, fichiers d'erreurs vers Modeles/
- Retrait dist/, scripts/, demo/ du module publie

---

## v4.2.0 — 2026-06

**Integration lex_select + correcteur P2G dans le pipeline.**

- Lex_select V6 : selection de candidats lexique par tete neuronale
- Correcteur P2G : modele correcteur seq2seq (encoder char + decodeur auto-regressif)
- Tolerance STT (formule_tolerance) propagee dans le pipeline

---

## v4.0.0 — 2026

**Renommage et refactorisation architecturale.**

### Breaking changes

- **Renommage** : `lectura-p2g` → `lectura-graphemiseur`, `lectura_p2g` → `lectura_graphemiseur`
- Les anciens imports `from lectura_p2g import ...` ne fonctionnent plus
- Un package de transition `lectura-p2g` 3.2.0 est disponible sur PyPI pour faciliter la migration

### Migration

```python
# Avant (v3.x)
from lectura_p2g import creer_engine

# Apres (v4.0.0+)
from lectura_graphemiseur import creer_engine
```

---

## v1.0.0 — 2026

Premiere version publique du modele unifie P2G+POS+Morpho.

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

- Package Python `lectura-graphemiseur` avec dépendances optionnelles
- CLI interactive (`demo_cli.py`)
- Licence CC BY-SA 4.0
