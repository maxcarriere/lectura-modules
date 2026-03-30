# Entrainement P2G BiLSTM

## Pre-requis

```bash
pip install torch onnx onnxruntime
```

## Etapes

### 1. Preparer le dataset

```bash
python preparer_dataset.py --dico ../../../data/prepared/train_g2p_v3.csv --output train_p2g.csv
```

Le script aligne chaque paire (phone, ortho) pour produire des labels graphemiques
alignes sur les caracteres IPA.

### 2. Entrainer le modele

```bash
python entrainer_bilstm.py --train train_p2g.csv --output ../modele/p2g_bilstm.onnx
```

Options :
- `--eval eval_p2g.csv` : evaluer la word accuracy apres entrainement
- `--embed-dim 64` : dimension des embeddings
- `--hidden-dim 128` : dimension cachee du LSTM
- `--num-layers 2` : nombre de couches BiLSTM
- `--dropout 0.3` : taux de dropout
- `--epochs 30` : nombre d'epoques
- `--batch-size 64` : taille des batchs
- `--scheduler` : activer le scheduler cosine

Le modele est exporte en ONNX float32 et quantifie INT8 automatiquement.

### 3. Utiliser le modele

```bash
cd ..
python lectura_p2g.py --model modele/p2g_bilstm_int8.onnx --vocab modele/p2g_vocab.json bɔ̃ʒuʁ pɛʃœʁ
```
