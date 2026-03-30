# Entraînement — Lectura POS Tagger

Ce dossier contient tout le nécessaire pour **réentraîner** ou **fine-tuner** le modèle CRF.

## Pré-requis

```bash
pip install sklearn-crfsuite
```

## Contenu

```
entrainement/
├── entrainer.py                     Script d'entraînement autonome
├── preparer_corpus.py               Télécharge les corpus UD depuis GitHub
├── README.md                        Ce fichier
└── donnees/
    ├── pos_train_merged.conllu      Entraînement (29 Mo, ~20K phrases)
    ├── pos_dev_merged.conllu        Validation (5 Mo, ~3K phrases)
    └── pos_test_merged.conllu       Test (3 Mo, ~2K phrases)
```

## Réentraîner le modèle (identique)

Pour reproduire le modèle fourni :

```bash
python entrainer.py
```

Le modèle est sauvegardé dans `../modele/pos_model_crf.json`.

## Fine-tuning sur vos données

### 1. Préparer vos données au format CoNLL-U

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
- **Colonne 4** : Tag UPOS (NOUN, VERB, DET, ADJ, ADV, ADP, PRON, CCONJ, SCONJ, AUX, PUNCT, etc.)
- **Colonne 6** : Features morphologiques (Definite=Def, PronType=Prs, etc.)

Les autres colonnes peuvent être `_`.

### 2. Lancer le fine-tuning

```bash
# Ajouter vos données au corpus existant
python entrainer.py --extra mon_corpus_juridique.conllu

# Ou entraîner uniquement sur vos données
python entrainer.py --corpus mon_corpus.conllu --dev mon_dev.conllu

# Sauvegarder sous un autre nom
python entrainer.py --extra mon_corpus.conllu --output ../modele/pos_model_juridique.json
```

### 3. Ajuster les hyperparamètres

```bash
# Moins de régularisation (plus de mémorisation, risque d'overfitting)
python entrainer.py --c1 0.01 --c2 0.01

# Plus de régularisation (plus de généralisation)
python entrainer.py --c1 0.5 --c2 0.5

# Plus d'itérations
python entrainer.py --max-iter 200
```

### 4. Utiliser le nouveau modèle

```python
from lectura_pos import PosTagger

tagger = PosTagger("modele/pos_model_juridique.json",
                    lexicon_path="modele/mini_lexique.json")

result = tagger.tag("Attendu que le prévenu a commis...")
```

## Mettre à jour les corpus UD

Pour télécharger la dernière version des corpus Universal Dependencies :

```bash
python preparer_corpus.py
```

Cela clone les dépôts UD depuis GitHub et fusionne les splits train/dev/test.

## Tagset

Le mapping Universal Dependencies → tags projet :

| UPOS | Features | → Tag projet |
|------|----------|-------------|
| NOUN, PROPN | — | NOM |
| VERB | — | VER |
| AUX | — | AUX |
| ADJ | — | ADJ |
| ADV | — | ADV |
| ADP | — | PRE |
| CCONJ, SCONJ | — | CON |
| INTJ | — | INTJ |
| DET | Definite=Def | ART:def |
| DET | Definite=Ind | ART:ind |
| DET | Poss=Yes | ADJ:pos |
| DET | PronType=Dem | ADJ:dem |
| DET | PronType=Int | ADJ:int |
| PRON | PronType=Prs | PRO:per |
| PRON | PronType=Rel | PRO:rel |
| PRON | PronType=Dem | PRO:dem |
| PRON | Poss=Yes | PRO:pos |
| PRON | PronType=Int | PRO:int |
| PRON | PronType=Ind | PRO:ind |

PUNCT et SPACE sont ignorés.

## Données d'entraînement

| Corpus | Licence | Phrases (train) |
|--------|---------|-----------------|
| [UD French-GSD](https://github.com/UniversalDependencies/UD_French-GSD) | CC BY-SA 4.0 | ~14 500 |
| [UD French-Sequoia](https://github.com/UniversalDependencies/UD_French-Sequoia) | LGPL-LR | ~2 200 |
| [UD French-Rhapsodie](https://github.com/UniversalDependencies/UD_French-Rhapsodie) | CC BY-SA 4.0 | ~1 000 |
