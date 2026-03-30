# Lectura P2G — Backend BiLSTM

**Convertisseur phoneme-grapheme BiLSTM ONNX pour le francais**

Convertit des transcriptions IPA en orthographe francaise avec un modele BiLSTM
(Bidirectional LSTM) via ONNX Runtime.

---

## Demarrage rapide

```python
from lectura_p2g import LecturaP2G

# Avec modele BiLSTM
p2g = LecturaP2G("modele/p2g_bilstm_int8.onnx",
                   vocab_path="modele/p2g_vocab.json")
p2g.predict("bɔ̃ʒuʁ")          # → "bonjour"
p2g.predict_candidates("vɛʁ", k=5)
# → [("vert", 0.45), ("verre", 0.20), ...]

# Sans modele (table + regles, zero dependance)
p2g = LecturaP2G()
p2g.predict("bɔ̃ʒuʁ")          # → "bonjour"
p2g.predict_syllable("kɑ̃")     # → "quand"
```

### Pre-requis

- Python 3.10+
- `onnxruntime >= 1.15`
- `numpy >= 1.24`

### Contenu

```
P2G_BiLSTM/
├── lectura_p2g.py              ← Module principal
├── demo_cli.py                 ← Demo en ligne de commande
├── modele/                     ← p2g_bilstm_int8.onnx + p2g_vocab.json
├── entrainement/
│   ├── preparer_dataset.py     ← Preparation des donnees
│   ├── entrainer_bilstm.py     ← Entrainement BiLSTM → ONNX
│   └── README.md
├── exemples/
│   └── exemple_basique.py
├── README.md
├── LICENCE.txt
├── ATTRIBUTION.md
└── pyproject.toml
```

---

## Trois strategies

| Strategie | Precision | Vitesse | Multi-candidats |
|-----------|-----------|---------|-----------------|
| **BiLSTM + ONNX** | Haute | ~2 ms/mot | Oui (softmax variantes) |
| **Table** | Moyenne | <0.1 ms | Non |
| **Regles** | Basse | <0.1 ms | Non |

### Modele BiLSTM

- Char embedding (64d) → BiLSTM 2 couches (128h) → Linear
- Sequence labeling : un grapheme par caractere IPA
- Inference ONNX Runtime (CPU, INT8 quantifie)

---

## API

### `LecturaP2G(model_path=None, vocab_path=None, table_path=None)`

- `model_path` : chemin vers le modele ONNX
- `vocab_path` : chemin vers p2g_vocab.json
- `table_path` : chemin vers p2g_table.json externe

### `p2g.predict(ipa) → str`

Meilleure orthographe.

### `p2g.predict_candidates(ipa, k=5) → list[tuple[str, float]]`

Top-K orthographes avec probabilites normalisees.

### `p2g.predict_syllable(ipa_syllable) → str`

Lookup rapide pour une syllabe IPA.

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0.*
