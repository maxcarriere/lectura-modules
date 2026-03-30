# Lectura P2G — Backend Seq2Seq

**Convertisseur phoneme-grapheme Seq2Seq ONNX pour le francais**

Convertit des transcriptions IPA en orthographe francaise avec un modele
Seq2Seq (encoder-decoder + attention) via ONNX Runtime.

---

## Demarrage rapide

```python
from lectura_p2g import LecturaP2G

# Avec modele Seq2Seq
p2g = LecturaP2G("modele/p2g_seq2seq_v4_encoder_int8.onnx",
                   decoder_path="modele/p2g_seq2seq_v4_decoder_int8.onnx",
                   vocab_path="modele/p2g_seq2seq_v4_vocab.json")
p2g.predict("bɔ̃ʒuʁ")          # → "bonjour"
p2g.predict_candidates("vɛʁ", k=5)
# → [("vert", 0.35), ("verre", 0.25), ...]

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
P2G_Seq2Seq/
├── lectura_p2g.py              ← Module principal
├── demo_cli.py                 ← Demo en ligne de commande
├── modele/                     ← encoder + decoder + vocab
├── entrainement/
│   ├── preparer_dataset.py     ← Preparation des donnees
│   ├── entrainer_seq2seq.py    ← Entrainement Seq2Seq → ONNX
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
| **Seq2Seq + beam search** | Tres haute | ~5 ms/mot | Oui (beam search natif) |
| **Table** | Moyenne | <0.1 ms | Non |
| **Regles** | Basse | <0.1 ms | Non |

### Modele Seq2Seq

- Encoder : embedding IPA (128d) → BiLSTM 2 couches (256h)
- Decoder : embedding ortho (128d) → LSTM 2 couches (512h) + attention
- Beam search natif pour multi-candidats avec probabilites
- Inference ONNX Runtime (CPU, INT8 quantifie)

---

## API

### `LecturaP2G(encoder_path=None, decoder_path=None, vocab_path=None, table_path=None)`

- `encoder_path` : chemin vers l'encoder ONNX
- `decoder_path` : chemin vers le decoder ONNX
- `vocab_path` : chemin vers p2g_seq2seq_vocab.json
- `table_path` : chemin vers p2g_table.json externe

### `p2g.predict(ipa) → str`

Meilleure orthographe (decodage greedy).

### `p2g.predict_candidates(ipa, k=5) → list[tuple[str, float]]`

Top-K orthographes avec probabilites normalisees (beam search).

### `p2g.predict_syllable(ipa_syllable) → str`

Lookup rapide pour une syllabe IPA.

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0.*
