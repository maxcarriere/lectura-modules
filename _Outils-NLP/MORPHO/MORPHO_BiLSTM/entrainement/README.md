# Entraînement — Lectura Morpho Tagger (BiLSTM)

Ce dossier contient les scripts pour **réentraîner** le modèle BiLSTM morphologique.

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
dans `POS/POS_CRF/entrainement/donnees/`, le script les trouvera automatiquement.

Sinon, préparer les données :

```bash
python preparer_corpus.py
```

### Format CoNLL-U

Chaque phrase est un bloc de lignes tabulées, séparées par une ligne vide :

```
# text = Le chat mange.
1	Le	le	DET	_	Definite=Def|Number=Sing|PronType=Art	_	_	_	_
2	chat	chat	NOUN	_	Gender=Masc|Number=Sing	_	_	_	_
3	mange	manger	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	_	_	_	_
4	.	.	PUNCT	_	_	_	_	_	_

```

Les colonnes utilisées sont :
- **Colonne 1** : ID du token
- **Colonne 2** : Forme du mot
- **Colonne 3** : Lemme
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
    --output ../modele/morpho_model_bilstm.onnx \
    --vocab-output ../modele/morpho_vocab_bilstm.json

# Ajuster les hyperparamètres
python entrainer_bilstm.py --epochs 50 --lr 0.0005 --hidden-dim 256

# Activer la pondération des classes (aide sur tags rares)
python entrainer_bilstm.py --class-weights

# Changer le seuil de repli des tags rares
python entrainer_bilstm.py --min-tag-count 10
```

## Architecture

| Composant | Détail |
|-----------|--------|
| Char embedding | 32d → CNN (kernel 3) → 96d |
| Word embedding | 192d |
| BiLSTM | 2 couches × 192 hidden |
| Dropout | 0.3 |
| Sortie | Linear → ~200 tags composites |
| Export | ONNX opset 14 + quantisation INT8 |

## Tags composites

Le modèle prédit un tag composite unique par token, encodant POS + traits :

| Catégorie | Format | Exemple |
|-----------|--------|---------|
| Verbe fini | `VER\|Mood\|Tense\|Person\|Number` | `VER\|Ind\|Pres\|3\|Plur` |
| Verbe participe | `VER\|Part\|Gender\|Number` | `VER\|Part\|Masc\|Sing` |
| Verbe infinitif | `VER\|Inf` | `VER\|Inf` |
| Nom | `NOM[\|Gender][\|Number]` | `NOM\|Masc\|Plur` |
| Adjectif | `ADJ[\|Gender][\|Number]` | `ADJ\|Fem\|Plur` |
| Invariable | POS seul | `PRE`, `ADV`, `CON` |

Les tags avec < 5 occurrences dans le train sont repliés sur leur POS de base.

## Données d'entraînement

| Corpus | Licence | Phrases (train) |
|--------|---------|-----------------|
| [UD French-GSD](https://github.com/UniversalDependencies/UD_French-GSD) | CC BY-SA 4.0 | ~14 500 |
| [UD French-Sequoia](https://github.com/UniversalDependencies/UD_French-Sequoia) | LGPL-LR | ~2 200 |
| [UD French-Rhapsodie](https://github.com/UniversalDependencies/UD_French-Rhapsodie) | CC BY-SA 4.0 | ~1 000 |
