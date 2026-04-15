# Lectura G2P Unifié

**Modèle unifié G2P + POS + Morphologie + Liaison pour le français**

Un seul modèle BiLSTM char-level multi-tête (1.75M paramètres, ONNX INT8 = 1.8 Mo) qui prédit simultanément :

- **G2P** : transcription phonémique IPA (98.5% word accuracy)
- **POS** : étiquetage morpho-syntaxique — 19 tags (98.2% accuracy)
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale (95-99%)
- **Liaison** : prédiction des liaisons obligatoires/facultatives (F1 90.6%)

Trois backends d'inférence : ONNX Runtime, NumPy, ou pur Python (zéro dépendance).

## Démarrage rapide

### Installation

```bash
pip install lectura-g2p             # zéro dépendance (backend pur Python)
pip install lectura-g2p[numpy]      # backend NumPy
pip install lectura-g2p[onnx]       # backend ONNX Runtime (le plus rapide)
```

### Utilisation minimale (ONNX — recommande)

```python
from lectura_nlp import get_model_path
from lectura_nlp.inference_onnx import OnnxInferenceEngine
from lectura_nlp.tokeniseur import tokeniser

engine = OnnxInferenceEngine(get_model_path("unifie_int8.onnx"),
                              get_model_path("unifie_vocab.json"))

tokens = tokeniser("Les enfants sont arrives a la maison.")
result = engine.analyser(tokens)

print(result["g2p"])      # ['le', 'ɑ̃fɑ̃', 'sɔ̃', 'aʁive', 'a', 'la', 'mɛzɔ̃']
print(result["pos"])      # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM']
print(result["liaison"])  # ['Lz', 'none', 'Lt', 'none', 'none', 'none', 'none']
print(result["morpho"])   # {'Number': ['Plur', ...], 'Gender': [...], ...}
```

## Poids NumPy / Pure Python (optionnel)

Le package pip inclut uniquement le modele ONNX INT8 (1.8 Mo).
Pour utiliser les backends **NumPy** ou **Pure Python**, il faut telecharger
le fichier de poids JSON (18 Mo) depuis GitHub :

```bash
# Telecharger les poids NumPy/Pure
curl -L -o unifie_weights.json \
  https://github.com/maxcarriere/lectura-modules/raw/main/G2P/modeles_numpy/unifie_weights.json
```

```python
from lectura_nlp.inference_numpy import NumpyInferenceEngine
from lectura_nlp import get_model_path

engine = NumpyInferenceEngine("unifie_weights.json",
                              get_model_path("unifie_vocab.json"))
result = engine.analyser(tokeniser("Bonjour le monde."))
```

## Backends d'inférence

| Backend | Dépendances | Vitesse | Usage |
|---------|------------|---------|-------|
| **ONNX Runtime** | `onnxruntime` | ~2 ms/phrase | Production |
| **NumPy** | `numpy` | ~50 ms/phrase | Léger, recommandé |
| **Pur Python** | aucune | ~200 ms/phrase | Embarqué, portabilité max |

Les trois backends produisent des résultats identiques (vérification croisée dans les tests).

## Benchmarks (test set)

| Tâche | Métrique | Score |
|-------|----------|-------|
| **G2P** | Word Accuracy | **98.5%** |
| **G2P** | PER (Phone Error Rate) | **0.5%** |
| **POS** | Accuracy | **98.2%** |
| **Liaison** | Macro F1 | **90.6%** |
| **Morpho** — Number | Accuracy | **97.0%** |
| **Morpho** — Gender | Accuracy | **95.1%** |
| **Morpho** — VerbForm | Accuracy | **98.8%** |
| **Morpho** — Mood | Accuracy | **97.7%** |
| **Morpho** — Tense | Accuracy | **97.8%** |
| **Morpho** — Person | Accuracy | **99.2%** |

Voir [EVALUATION.md](EVALUATION.md) pour les résultats détaillés.

## API

### `tokeniser(text) -> list[str]`

Tokenise une phrase française (gestion apostrophes, ponctuation, contractions).

### `engine.analyser(tokens) -> dict`

Analyse une liste de tokens et retourne un dictionnaire :
- `tokens` : liste des tokens d'entrée
- `g2p` : transcription IPA par token
- `pos` : étiquette POS par token
- `liaison` : label liaison par token (`none`, `Lz`, `Lt`, `Ln`, `Lr`, `Lp`)
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

### `corriger_g2p(mot, ipa) -> str`

Applique la table de corrections puis les règles (ex+consonne, ex+voyelle, yod).

### `appliquer_liaison(tokens, phones, liaisons) -> list[str]`

Applique les consonnes de liaison entre tokens consécutifs.

## Architecture du modèle

```
Phrase → Char Embedding (64d) → Shared BiLSTM (2×160h → 320d)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ↓                                       ↓
              G2P Head (per-char)              Word BiLSTM (128h → 256d)
              Linear(320→40)                        │
                                    ┌───┬───┬───┬───┬───┬───┬───┐
                                   POS Num Gen VF  Mood Tns Per Liaison
```

- **Entrée** : séquence de caractères avec `<BOS>`, `<SEP>`, `<EOS>`
- **G2P** : prédiction par caractère avec labels `_CONT` (continuation)
- **Têtes mot** : représentation mot = fwd[dernier_char] || bwd[premier_char]
- **Paramètres** : 1,747,108 (~1.75M)

## Limites connues

- Le G2P est évalué en contexte phrastique ; la performance sur mots isolés hors-vocabulaire est plus basse (~92%)
- La liaison `Lp` est très rare dans les données d'entraînement (F1 = 66.7%)
- Les noms propres et néologismes peuvent produire des transcriptions approximatives
- Le modèle ne gère pas les homographes contextuels complexes (ex: "fils" = /fis/ vs /fil/)

## Licence

Double licence :
- **AGPL-3.0** — usage libre (voir [LICENCE.txt](LICENCE.txt))
- **Licence commerciale** — usage proprietaire (voir [LICENCE-COMMERCIALE.md](LICENCE-COMMERCIALE.md))

Voir aussi [ATTRIBUTION.md](ATTRIBUTION.md) pour les credits.
