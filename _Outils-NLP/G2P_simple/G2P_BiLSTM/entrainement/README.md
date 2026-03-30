# Entrainement — Lectura G2P (BiLSTM)

Ce dossier contient les scripts pour **reentrainer** le modele G2P BiLSTM.

## Pre-requis

```bash
pip install torch onnx onnxruntime
```

## Contenu

```
entrainement/
├── entrainer_bilstm.py     Script d'entrainement BiLSTM + export ONNX
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

Pour le BiLSTM (sequence-labeling), un fichier avec labels alignes est utilise.

## Reentrainer le BiLSTM

```bash
python entrainer_bilstm.py \
    --train donnees/train_g2p.csv \
    --eval donnees/eval_g2p.csv \
    --output ../modele/g2p_model_bilstm.onnx \
    --quantize
```

## Construire les corrections

Les corrections sont construites en comparant les predictions du modele
avec le dictionnaire gold (dico.csv). Tout mot mal predit jusqu'a 99%
de couverture frequentielle est ajoute a la table :

```bash
python build_corrections.py \
    --model ../modele/g2p_model_bilstm_int8.onnx \
    --gold dico.csv \
    --freqs Lexique383.tsv \
    --coverage 0.99 \
    --output ../modele/g2p_corrections_bilstm.json
```

## Sources des donnees

| Source | Licence | Usage |
|--------|---------|-------|
| [GLAFF](http://redac.univ-tlse2.fr/lexiques/glaff.html) | CC BY-SA 3.0 | Alignements G2P |
| [Lexique 3.83](http://www.lexique.org/) | CC BY-SA 4.0 | Frequences lexicales |
