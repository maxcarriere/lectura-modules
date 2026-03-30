# Entrainement — Lectura G2P (Seq2Seq)

Ce dossier contient les scripts pour **reentrainer** le modele G2P Seq2Seq.

## Pre-requis

```bash
pip install torch onnx onnxruntime
```

## Contenu

```
entrainement/
├── entrainer_seq2seq.py    Script d'entrainement Seq2Seq + export ONNX
└── README.md               Ce fichier
```

## Donnees

Le modele est entraine sur des paires (ortho, phone) derivees de :
- **dico.csv** : dictionnaire de prononciation (~80 000 mots, derive du GLAFF)
- **Lexique 3.83** : frequences lexicales pour le poids des mots

### Format du dataset

Le fichier CSV d'entrainement contient au minimum :

```csv
ortho,phone
bonjour,bɔ̃ʒuʁ
```

Le Seq2Seq utilise directement les paires (ortho, phone) sans alignement prealable.

## Reentrainer le Seq2Seq

```bash
python entrainer_seq2seq.py \
    --train donnees/train_g2p.csv \
    --eval donnees/eval_g2p.csv \
    --output ../modele/g2p_seq2seq \
    --quantize
```

Cela produit :
- `g2p_seq2seq_encoder.onnx` + `g2p_seq2seq_encoder_int8.onnx`
- `g2p_seq2seq_decoder.onnx` + `g2p_seq2seq_decoder_int8.onnx`
- `g2p_seq2seq_vocab.json`

## Construire les corrections

Les corrections sont construites en comparant les predictions du modele
avec le dictionnaire gold (dico.csv). Tout mot mal predit jusqu'a 99%
de couverture frequentielle est ajoute a la table :

```bash
python build_corrections.py \
    --model ../modele/g2p_seq2seq_encoder_int8.onnx \
    --gold dico.csv \
    --freqs Lexique383.tsv \
    --coverage 0.99 \
    --output ../modele/g2p_corrections_seq2seq.json
```

## Sources des donnees

| Source | Licence | Usage |
|--------|---------|-------|
| [GLAFF](http://redac.univ-tlse2.fr/lexiques/glaff.html) | CC BY-SA 3.0 | Alignements G2P |
| [Lexique 3.83](http://www.lexique.org/) | CC BY-SA 4.0 | Frequences lexicales |
