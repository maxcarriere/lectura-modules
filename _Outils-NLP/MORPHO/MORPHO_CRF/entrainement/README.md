# Entraînement — Lectura Morpho Tagger (CRF)

Ce dossier contient les scripts pour **entraîner** le modèle CRF morphologique.

## Pré-requis

```bash
pip install sklearn-crfsuite
```

## Contenu

```
entrainement/
├── entrainer_crf.py        Script d'entraînement CRF + export JSON
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

## Entraîner le modèle

```bash
# Entraînement standard
python entrainer_crf.py

# Avec corpus personnalisé
python entrainer_crf.py \
    --corpus donnees/pos_train_merged.conllu \
    --dev donnees/pos_dev_merged.conllu \
    --output ../modele/morpho_model_crf.json

# Ajuster les hyperparamètres
python entrainer_crf.py --c1 0.5 --c2 0.05 --max-iter 200

# Changer le seuil de repli des tags rares
python entrainer_crf.py --min-tag-count 10

# Changer le seuil de filtrage des poids (taille du modèle)
python entrainer_crf.py --weight-threshold 1e-3
```

## Hyperparamètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| c1 | 0.3 | Régularisation L1 (sparsification) |
| c2 | 0.1 | Régularisation L2 |
| max-iter | 150 | Itérations maximum L-BFGS |
| min-tag-count | 5 | Seuil de repli des tags rares |
| weight-threshold | 1e-4 | Seuil de filtrage des poids JSON |

## Features

| Feature | Description |
|---------|-------------|
| word | Forme minuscule du mot |
| suf2/suf3 | Suffixes 2 et 3 caractères |
| suf4/suf5 | Suffixes 4 et 5 caractères (terminaisons verbales) |
| pre2/pre3 | Préfixes 2 et 3 caractères |
| is_upper/is_title/is_digit | Casse du mot |
| BOS/EOS | Début/fin de phrase |
| w-1/w+1 | Mots contextuels (bigramme) |

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
