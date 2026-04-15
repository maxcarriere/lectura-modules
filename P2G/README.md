# Lectura P2G

**Modele unifie P2G + POS + Morphologie pour le francais (IPA → orthographe)**

Un seul modele BiLSTM char-level multi-tete avec word feedback (2.56M parametres) qui predit simultanement :

- **P2G** : transcription IPA vers orthographe (93.1% word accuracy, 2.2% CER)
- **POS** : etiquetage morpho-syntaxique — 19 tags (97.0% accuracy)
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale (92-97%)

Quatre backends d'inference : API (zero config), ONNX Runtime, NumPy, ou pur Python (zero dependance).

## Demarrage rapide

```bash
pip install lectura-p2g
```

```python
from lectura_p2g import creer_engine

engine = creer_engine()    # mode API par defaut (zero config)

result = engine.analyser(["le", "ɑ̃fɑ̃", "sɔ̃", "aʁive", "a", "la", "mɛzɔ̃"])

print(result["ortho"])   # ['les', 'enfants', 'sont', 'arrives', 'a', 'la', 'maison']
print(result["pos"])     # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM']
```

## Backends d'inference

| Backend | Dependances | Vitesse | Usage |
|---------|------------|---------|-------|
| **API** | aucune | ~100 ms (reseau) | Par defaut, zero config |
| **ONNX Runtime** | `onnxruntime` | ~2 ms/phrase | Production locale |
| **NumPy** | `numpy` | ~50 ms/phrase | Leger |
| **Pur Python** | aucune | ~200 ms/phrase | Embarque, portabilite max |

```python
engine = creer_engine(mode="onnx")    # ONNX local
engine = creer_engine(mode="api")     # API serveur
engine = creer_engine(mode="auto")    # local si modeles presents, sinon API
```

Les backends locaux (ONNX, NumPy, Pure) necessitent les modeles — disponibles sur demande.

## Benchmarks (test set)

| Tache | Metrique | Score |
|-------|----------|-------|
| **P2G** | Word Accuracy | **93.1%** |
| **P2G** | CER (Character Error Rate) | **2.2%** |
| **POS** | Accuracy | **97.0%** |
| **Morpho** — Number | Accuracy | **92.8%** |
| **Morpho** — Gender | Accuracy | **92.0%** |

## API

### `creer_engine(mode="auto") -> engine`

Factory pour creer un engine d'inference. Modes : `"auto"`, `"api"`, `"local"`, `"onnx"`, `"numpy"`, `"pure"`.

### `engine.analyser(ipa_words) -> dict`

Analyse une liste de mots IPA et retourne :
- `ortho` : orthographe reconstruite par mot
- `pos` : etiquette POS par mot
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

## Licence

Double licence :
- **AGPL-3.0** — usage libre (voir [LICENCE.txt](LICENCE.txt))
- **Licence commerciale** — usage proprietaire (voir [LICENCE-COMMERCIALE.md](LICENCE-COMMERCIALE.md))
