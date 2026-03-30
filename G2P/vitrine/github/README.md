# Lectura G2P Unifié

[![Licence: CC BY-SA 4.0](https://img.shields.io/badge/Licence-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![ONNX INT8](https://img.shields.io/badge/ONNX-INT8%20(1.8%20Mo)-green.svg)](#)

**Un modèle unique pour 4 tâches NLP françaises** : phonétique (G2P), POS-tagging, morphologie et liaison.

## Performances

| Tâche | Score |
|-------|-------|
| G2P (word acc) | **98.5%** |
| POS (accuracy) | **98.2%** |
| Liaison (macro F1) | **90.6%** |
| Morphologie | **95-99%** |

## Exemple

```python
from lectura_nlp.inference_numpy import NumpyInferenceEngine
from lectura_nlp.tokeniseur import tokeniser

engine = NumpyInferenceEngine("modeles/unifie_weights.json",
                              "modeles/unifie_vocab.json")
result = engine.analyser(tokeniser("Les enfants jouent."))

# result["g2p"]     → ['le', 'ɑ̃fɑ̃', 'ʒu']
# result["pos"]     → ['ART:def', 'NOM', 'VER']
# result["liaison"] → ['Lz', 'none', 'none']
```

## Points forts

- **1.8 Mo** ONNX INT8 — 4x plus petit que le pipeline séparé
- **Zéro dépendance** possible (backend pur Python)
- **3 backends** : ONNX Runtime (~2ms), NumPy (~50ms), pur Python (~200ms)
- **1 seule passe** d'inférence pour les 4 tâches

## Installation

```bash
pip install lectura-g2p-unifie[numpy]
```

## Licence

CC BY-SA 4.0 — Entraîné sur GLAFF (CC BY-SA 3.0), Lexique (CC BY-SA 4.0), UD French-GSD (CC BY-SA 4.0).
