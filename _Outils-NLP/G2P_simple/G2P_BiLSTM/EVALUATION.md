# Evaluation — Lectura G2P (BiLSTM)

## Methodologie

Le modele est evalue sur un dictionnaire gold (dico.csv, ~80 000 mots)
derive du GLAFF. La couverture est mesuree par tranches de frequence
(Lexique 3.83, freqlivres).

Un mot est **correct** si la prediction (apres post-traitement R1-R13
et corrections) correspond exactement a l'une des prononciations gold.

Deux niveaux de tolerance supplementaires sont mesures :
- **o/ɔ-tolerant** : neutralise la distinction /o/ vs /ɔ/
- **o/ɔ+e/ɛ-tolerant** : neutralise aussi /e/ vs /ɛ/

## Resultats

### Precision intrinseque du modele (sans corrections)

| Modele | Acc @99% (exacte) |
|--------|-------------------|
| BiLSTM INT8 | 90.8% |

### Modele + corrections + post-traitement @99% couverture

| Modele | Corrections | Acc (exacte) | Acc (o/ɔ) | Acc (o/ɔ+e/ɛ) |
|--------|-------------|-------------|-----------|---------------|
| **BiLSTM INT8** | 3 351 g2p + 402 g2p_pos | 100% | 100% | 100% |

### Details

| Taille modele | Taille corrections | Dependances | Acc @99% |
|--------------|-------------------|-------------|----------|
| 614 Ko + 1.2 Ko | 142 Ko | onnxruntime | 100% |

## Comment reproduire

```bash
python evaluer.py --gold chemin/vers/dico.csv
```

## Notes

- Le modele BiLSTM atteint 100% de precision a 99% de couverture frequentielle
  grace a sa table de corrections (3 753 entrees)
- Le post-traitement R1-R13 corrige des patterns systematiques
- Les corrections g2p_pos gerent 402 homographes POS-dependants
