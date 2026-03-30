# Entrainement P2G Seq2Seq

## Pre-requis

```bash
pip install torch onnx onnxruntime
```

## Etapes

### 1. Preparer le dataset

```bash
python preparer_dataset.py --dico ../../../data/prepared/train_g2p_v3.csv --output train_p2g.csv
```

Le script extrait les paires (phone, ortho) pour l'entrainement Seq2Seq.

### 2. Entrainer le modele

```bash
python entrainer_seq2seq.py --train train_p2g.csv --output ../modele/p2g_seq2seq
```

Options :
- `--eval eval_p2g.csv` : evaluer la word accuracy apres entrainement
- `--embed-dim 128` : dimension des embeddings
- `--hidden-dim 256` : dimension cachee de l'encoder BiLSTM
- `--dec-hidden-dim 512` : dimension cachee du decoder LSTM
- `--num-layers 2` : nombre de couches
- `--dropout 0.3` : taux de dropout
- `--epochs 40` : nombre d'epoques
- `--teacher-forcing 0.5` : ratio initial de teacher forcing (decroit progressivement)
- `--batch-size 64` : taille des batchs

Le modele est exporte en ONNX float32 et quantifie INT8 automatiquement.

### 3. Utiliser le modele

```bash
cd ..
python lectura_p2g.py --model-dir modele/ bɔ̃ʒuʁ pɛʃœʁ
```
