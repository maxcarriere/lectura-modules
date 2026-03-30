# Entraînement du modèle unifié

Instructions pour reproduire l'entraînement du modèle G2P+POS+Morpho+Liaison.

## Prérequis

```bash
pip install torch>=2.0 onnx>=1.14 onnxruntime>=1.16
```

GPU recommandé (CUDA). L'entraînement complet prend environ 1h30 sur GPU.

## Données requises

Les données ne sont pas incluses dans la distribution. Vous devez les obtenir séparément :

### Sources

| Source | Téléchargement | Licence |
|--------|---------------|---------|
| **GLAFF 1.2.1** | http://redac.univ-tlse2.fr/lexicons/glaff.html | CC BY-SA 3.0 |
| **Lexique 3.83** | http://www.lexique.org/ | CC BY-SA 4.0 |
| **UD French-GSD** | https://universaldependencies.org/treebanks/fr_gsd/ | CC BY-SA 4.0 |

### Préparation

1. Télécharger les sources ci-dessus
2. Enrichir les fichiers CoNLL-U avec les phonèmes et liaisons (cf. scripts du projet Lectura)
3. Placer les fichiers préparés dans `donnees/` :

```
donnees/
├── train.conllu          # UD-GSD train enrichi (Phone, Liaison, Denas)
├── dev.conllu            # UD-GSD dev enrichi
├── test.conllu           # UD-GSD test enrichi
├── lexique_train.csv     # GLAFF+Lexique : ortho,phone (entraînement)
├── lexique_eval.csv      # GLAFF+Lexique : ortho,phone (évaluation)
├── phone_to_graphemes.csv # Table phonème→graphèmes pour l'aligneur
└── h_aspire.txt          # Liste des mots à h aspiré
```

## Pipeline d'entraînement

### 1. Préparer les données

```bash
python preparer_donnees.py --donnees donnees/
```

Produit les fichiers JSON alignés pour l'entraînement.

### 2. Entraîner le modèle

```bash
python entrainer.py --donnees donnees/
```

- **Phase 1** : pré-entraînement G2P sur le lexique (~890K mots, ~60 min GPU)
- **Phase 2** : fine-tuning multi-tâche sur les phrases (~18K phrases, ~30 min GPU)

Le meilleur checkpoint est sauvegardé automatiquement (early stopping sur dev loss).

### 3. Évaluer

```bash
python evaluer.py --split test
```

Produit les métriques pour toutes les tâches (G2P, POS, Liaison, Morphologie).

### 4. Exporter

```bash
python exporter.py
```

Produit :
- `modeles/unifie_int8.onnx` — ONNX quantifié INT8
- `modeles/unifie_vocab.json` — Vocabulaire
- `modeles/unifie_weights.json` — Poids JSON pour backend NumPy/Pure Python

### 5. Construire la table de corrections

```bash
python construire_table_corrections.py
```

Produit `modeles/g2p_corrections_unifie.json` en croisant les prédictions du modèle avec le lexique de référence.

## Hyperparamètres clés

| Paramètre | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Epochs | 30 | 80 (early stop ~70) |
| Batch size | 128 | 32 |
| Learning rate | 1e-3 | 5e-4 |
| LR schedule | CosineAnnealing | Warmup 5ep + Cosine |
| Label smoothing | — | 0.1 |
| Loss weights | G2P=1.0 | G2P=1.0, POS=1.0, Morpho=0.8, Liaison=3.0 |

## Architecture

- Char Embedding : 64d
- Shared BiLSTM : 2 couches, 160 hidden (bidirectionnel → 320d)
- Word BiLSTM : 1 couche, 128 hidden (bidirectionnel → 256d)
- G2P Head : Linear(320 → n_phones)
- Têtes mot : Linear(256 → n_classes) × 8 (POS + 6 morpho + Liaison)
