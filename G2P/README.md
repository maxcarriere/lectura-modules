# Lectura G2P

**Modele unifie G2P + POS + Morphologie + Liaison pour le francais**

Un seul modele BiLSTM char-level multi-tete (1.75M parametres) qui predit simultanement :

- **G2P** : transcription phonemique IPA (98.5% word accuracy)
- **POS** : etiquetage morpho-syntaxique — 19 tags (98.2% accuracy)
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale (95-99%)
- **Liaison** : prediction des liaisons obligatoires/facultatives (F1 90.6%)

Quatre backends d'inference : API (zero config), ONNX Runtime, NumPy, ou pur Python (zero dependance).

## Demarrage rapide

```bash
pip install lectura-g2p
```

```python
from lectura_nlp import creer_engine

engine = creer_engine()    # mode API par defaut (zero config)

result = engine.analyser(["Les", "enfants", "sont", "arrives", "a", "la", "maison"])

print(result["g2p"])      # ['le', 'ɑ̃fɑ̃', 'sɔ̃', 'aʁive', 'a', 'la', 'mɛzɔ̃']
print(result["pos"])      # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM']
print(result["liaison"])  # ['Lz', 'none', 'Lt', 'none', 'none', 'none', 'none']
print(result["morpho"])   # {'Number': ['Plur', ...], 'Gender': [...], ...}
```

## Backends d'inference

| Backend | Dependances | Vitesse | Usage |
|---------|------------|---------|-------|
| **API** | aucune | ~100 ms (reseau) | Par defaut, zero config |
| **ONNX Runtime** | `onnxruntime` | ~2 ms/phrase | Production locale |
| **NumPy** | `numpy` | ~50 ms/phrase | Leger |
| **Pur Python** | aucune | ~200 ms/phrase | Embarque, portabilite max |

```python
# Forcer un backend specifique
engine = creer_engine(mode="onnx")    # ONNX local
engine = creer_engine(mode="api")     # API serveur
engine = creer_engine(mode="auto")    # local si modeles presents, sinon API
```

Les backends locaux (ONNX, NumPy, Pure) necessitent les modeles — disponibles sur demande.

## Benchmarks (test set)

| Tache | Metrique | Score |
|-------|----------|-------|
| **G2P** | Word Accuracy | **98.5%** |
| **G2P** | PER (Phone Error Rate) | **0.5%** |
| **POS** | Accuracy | **98.2%** |
| **Liaison** | Macro F1 | **90.6%** |
| **Morpho** — Number | Accuracy | **97.0%** |
| **Morpho** — Gender | Accuracy | **95.1%** |
| **Morpho** — VerbForm | Accuracy | **98.8%** |

## API

### `creer_engine(mode="auto") -> engine`

Factory pour creer un engine d'inference. Modes : `"auto"`, `"api"`, `"local"`, `"onnx"`, `"numpy"`, `"pure"`.

### `engine.analyser(tokens) -> dict`

Analyse une liste de tokens et retourne :
- `g2p` : transcription IPA par token
- `pos` : etiquette POS par token
- `liaison` : label liaison par token (`none`, `Lz`, `Lt`, `Ln`, `Lr`, `Lp`)
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

## Licence

Double licence :
- **AGPL-3.0** — usage libre (voir [LICENCE.txt](LICENCE.txt))
- **Licence commerciale** — usage proprietaire (voir [LICENCE-COMMERCIALE.md](LICENCE-COMMERCIALE.md))
