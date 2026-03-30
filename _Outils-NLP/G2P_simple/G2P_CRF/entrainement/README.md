# Entrainement — Lectura G2P (CRF)

Ce dossier contient les scripts pour **reentrainer** le modele G2P CRF.

## Pre-requis

```bash
pip install sklearn-crfsuite
```

## Contenu

```
entrainement/
├── entrainer_crf.py        Script d'entrainement CRF
├── preparer_dataset.py     Preparation du dataset d'alignement
└── README.md               Ce fichier
```

## Donnees

Le modele est entraine sur des alignements grapheme-phoneme derives de :
- **dico.csv** : dictionnaire de prononciation (~80 000 mots, derive du GLAFF)
- **Lexique 3.83** : frequences lexicales pour le poids des mots

### Format du dataset

Le fichier CSV d'entrainement contient au minimum :

```csv
ortho,phone,aligned_labels
bonjour,bɔ̃ʒuʁ,"b,ɔ̃,_CONT,ʒ,u,ʁ"
```

Pour le CRF (sequence-labeling), un fichier avec labels alignes est utilise.

## Reentrainer le CRF

```bash
python entrainer_crf.py \
    --train donnees/train_g2p.csv \
    --eval donnees/eval_g2p.csv \
    --output ../modele/g2p_model_crf.json
```

## Construire les corrections

Les corrections sont construites en comparant les predictions du modele
avec le dictionnaire gold (dico.csv). Tout mot mal predit jusqu'a 99%
de couverture frequentielle est ajoute a la table :

```bash
python build_corrections.py \
    --model ../modele/g2p_model_crf.json \
    --gold dico.csv \
    --freqs Lexique383.tsv \
    --coverage 0.99 \
    --output ../modele/g2p_corrections_crf.json
```

## Sources des donnees

| Source | Licence | Usage |
|--------|---------|-------|
| [GLAFF](http://redac.univ-tlse2.fr/lexiques/glaff.html) | CC BY-SA 3.0 | Alignements G2P |
| [Lexique 3.83](http://www.lexique.org/) | CC BY-SA 4.0 | Frequences lexicales |
