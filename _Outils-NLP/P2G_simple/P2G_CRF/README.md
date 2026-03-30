# Lectura P2G — Backend CRF

**Convertisseur phoneme-grapheme CRF pour le francais — zero dependance**

Convertit des transcriptions IPA en orthographe francaise avec un modele CRF
(Conditional Random Field) et decodage Viterbi pur Python.

---

## Demarrage rapide

```python
from lectura_p2g import LecturaP2G

# Avec modele CRF
p2g = LecturaP2G("modele/p2g_model_crf.json")
p2g.predict("bɔ̃ʒuʁ")          # → "bonjour"
p2g.predict_candidates("vɛʁ", k=5)
# → [("vert", 0.35), ("verre", 0.25), ...]

# Sans modele (table + regles, zero dependance)
p2g = LecturaP2G()
p2g.predict("bɔ̃ʒuʁ")          # → "bonjour"
p2g.predict_syllable("kɑ̃")     # → "quand"
```

### Pre-requis

- Python 3.10+ (zero dependance)

### Contenu

```
P2G_CRF/
├── lectura_p2g.py              ← Module principal
├── demo_cli.py                 ← Demo en ligne de commande
├── modele/                     ← p2g_model_crf.json (apres entrainement)
├── entrainement/
│   ├── preparer_dataset.py     ← Preparation des donnees
│   ├── entrainer_crf.py        ← Entrainement CRF
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
| **CRF + Viterbi** | Haute | ~1 ms/mot | Oui (beam Viterbi) |
| **Table** | Moyenne | <0.1 ms | Non |
| **Regles** | Basse | <0.1 ms | Non |

### Modele CRF

- Features par caractere IPA : position, contexte n-gramme, type phonologique
- Decodage Viterbi (top-1) et beam Viterbi (top-K)
- Zero dependance a l'inference (JSON pur)

---

## API

### `LecturaP2G(model_path=None, table_path=None)`

- `model_path` : chemin vers le modele CRF JSON
- `table_path` : chemin vers p2g_table.json externe

### `p2g.predict(ipa) → str`

Meilleure orthographe.

### `p2g.predict_candidates(ipa, k=5) → list[tuple[str, float]]`

Top-K orthographes avec probabilites normalisees.

### `p2g.predict_syllable(ipa_syllable) → str`

Lookup rapide pour une syllabe IPA.

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0.*
