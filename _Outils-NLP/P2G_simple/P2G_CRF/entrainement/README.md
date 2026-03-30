# Entrainement P2G CRF

## Pre-requis

```bash
pip install sklearn-crfsuite
```

## Etapes

### 1. Preparer le dataset

```bash
python preparer_dataset.py --dico ../../../data/prepared/train_g2p_v3.csv --output train_p2g.csv
```

Le script aligne chaque paire (phone, ortho) pour produire des labels graphemiques
alignes sur les phonemes IPA.

### 2. Entrainer le modele

```bash
python entrainer_crf.py --train train_p2g.csv --output ../modele/p2g_model_crf.json
```

Options :
- `--eval eval_p2g.csv` : evaluer la word accuracy apres entrainement
- `--c1 0.1` : regularisation L1
- `--c2 0.1` : regularisation L2
- `--max-iter 100` : nombre max d'iterations LBFGS

### 3. Utiliser le modele

```bash
cd ..
python lectura_p2g.py --model modele/p2g_model_crf.json bɔ̃ʒuʁ pɛʃœʁ
```
