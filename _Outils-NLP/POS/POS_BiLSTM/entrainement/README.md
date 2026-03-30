# Entraînement — Lectura POS Tagger (BiLSTM)

Ce dossier contient les scripts pour **réentraîner** le modèle BiLSTM POS.

## Pré-requis

```bash
pip install torch onnx onnxruntime
```

## Contenu

```
entrainement/
├── entrainer_bilstm.py     Script d'entraînement BiLSTM + export ONNX
├── preparer_corpus.py       Télécharge les corpus UD depuis GitHub
└── README.md                Ce fichier
```

## Données

Les données CoNLL-U sont partagées avec POS_CRF. Si elles sont déjà présentes
dans `../POS_CRF/entrainement/donnees/`, le script les trouvera automatiquement.

Sinon, préparer les données :

```bash
python preparer_corpus.py
```

### Format CoNLL-U

Chaque phrase est un bloc de lignes tabulées, séparées par une ligne vide :

```
# text = Le chat mange.
1	Le	le	DET	_	Definite=Def	_	_	_	_
2	chat	chat	NOUN	_	_	_	_	_	_
3	mange	manger	VERB	_	_	_	_	_	_
4	.	.	PUNCT	_	_	_	_	_	_

```

Les colonnes utilisées sont :
- **Colonne 1** : ID du token
- **Colonne 2** : Forme du mot
- **Colonne 4** : Tag UPOS
- **Colonne 6** : Features morphologiques

## Réentraîner le modèle

```bash
# Entraînement standard
python entrainer_bilstm.py

# Avec corpus personnalisé
python entrainer_bilstm.py \
    --corpus donnees/pos_train_merged.conllu \
    --dev donnees/pos_dev_merged.conllu \
    --output ../modele/pos_model_bilstm.onnx \
    --vocab-output ../modele/pos_vocab_bilstm.json

# Ajuster les hyperparamètres
python entrainer_bilstm.py --epochs 50 --lr 0.0005 --hidden-dim 256
```

## Architecture

| Composant | Détail |
|-----------|--------|
| Char embedding | 32d → CNN (kernel 3) → 64d |
| Word embedding | 128d |
| BiLSTM | 2 couches × 128 hidden |
| Dropout | 0.3 |
| Sortie | Linear → 18 tags |
| Export | ONNX opset 14 + quantisation INT8 |

## Données d'entraînement

| Corpus | Licence | Phrases (train) |
|--------|---------|-----------------|
| [UD French-GSD](https://github.com/UniversalDependencies/UD_French-GSD) | CC BY-SA 4.0 | ~14 500 |
| [UD French-Sequoia](https://github.com/UniversalDependencies/UD_French-Sequoia) | LGPL-LR | ~2 200 |
| [UD French-Rhapsodie](https://github.com/UniversalDependencies/UD_French-Rhapsodie) | CC BY-SA 4.0 | ~1 000 |

## Tagset

Le mapping Universal Dependencies → tags projet est identique à celui du CRF.
Voir `POS_CRF/entrainement/README.md` pour la table complète.

PUNCT et SPACE sont ignorés.
