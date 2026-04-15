# Lectura P2G Unifié

**Modèle unifié P2G + POS + Morphologie pour le français (IPA → orthographe)**

Un seul modèle BiLSTM char-level multi-tête avec word feedback (2.56M paramètres, ONNX INT8 = 2.6 Mo) qui prédit simultanément :

- **P2G** : transcription IPA vers orthographe (93.1% word accuracy, 2.2% CER)
- **POS** : étiquetage morpho-syntaxique — 19 tags (97.0% accuracy)
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale (92-97%)

Trois backends d'inférence : ONNX Runtime, NumPy, ou pur Python (zéro dépendance).

## Démarrage rapide

### Installation

```bash
pip install lectura-p2g             # zéro dépendance (backend pur Python)
pip install lectura-p2g[numpy]      # backend NumPy
pip install lectura-p2g[onnx]       # backend ONNX Runtime (le plus rapide)
```

### Utilisation minimale (ONNX — recommande)

```python
from lectura_p2g import get_model_path
from lectura_p2g.inference_onnx import OnnxInferenceEngine

engine = OnnxInferenceEngine(get_model_path("unifie_p2g_v2_int8.onnx"),
                              get_model_path("unifie_p2g_v2_vocab.json"))

result = engine.analyser(["le", "ɑ̃fɑ̃", "sɔ̃", "aʁive", "a", "la", "mɛzɔ̃"])

print(result["ortho"])   # ['les', 'enfants', 'sont', 'arrives', 'a', 'la', 'maison']
print(result["pos"])     # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM']
print(result["morpho"])  # {'Number': ['Plur', 'Plur', ...], 'Gender': [...], ...}
```

## Poids NumPy / Pure Python (optionnel)

Le package pip inclut uniquement le modele ONNX INT8 (2.6 Mo).
Pour utiliser les backends **NumPy** ou **Pure Python**, il faut telecharger
le fichier de poids JSON (26 Mo) depuis GitHub :

```bash
curl -L -o unifie_p2g_v2_weights.json \
  https://github.com/maxcarriere/lectura-modules/raw/main/P2G/modeles_numpy/unifie_p2g_v2_weights.json
```

```python
from lectura_p2g.inference_numpy import NumpyInferenceEngine
from lectura_p2g import get_model_path

engine = NumpyInferenceEngine("unifie_p2g_v2_weights.json",
                              get_model_path("unifie_p2g_v2_vocab.json"))
result = engine.analyser(["bɔ̃ʒuʁ", "lə", "mɔ̃d"])
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
| **P2G** | Word Accuracy | **93.1%** |
| **P2G** | CER (Character Error Rate) | **2.2%** |
| **POS** | Accuracy | **97.0%** |
| **Morpho** — Number | Accuracy | **92.8%** |
| **Morpho** — Gender | Accuracy | **92.0%** |
| **Morpho** — VerbForm | Accuracy | **96.2%** |
| **Morpho** — Mood | Accuracy | **93.5%** |
| **Morpho** — Tense | Accuracy | **94.1%** |
| **Morpho** — Person | Accuracy | **96.6%** |

Voir [EVALUATION.md](EVALUATION.md) pour les résultats détaillés et la comparaison v1/v2.

## API

### `engine.analyser(ipa_words) -> dict`

Analyse une liste de mots IPA et retourne un dictionnaire :
- `ipa_words` : liste des mots IPA d'entrée
- `ortho` : orthographe reconstruite par mot
- `pos` : étiquette POS par mot
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

### `tokeniser_ipa(text) -> list[str]`

Tokenise une phrase IPA (split sur espaces).

### `corriger_phrase_v2(ortho_words, pos_tags, lexique) -> list[str]`

Post-traitement contextuel inter-mots : accord déterminant-nom, sujet-verbe.

## Architecture du modèle (v2)

```
Phrase IPA → Char Embedding (64d) → Shared BiLSTM (2×160h → 320d)
                                          │
                  ┌───────────────────────┼────────────────────┐
                  ↓                                             ↓
        Word representations                          Word BiLSTM (192h → 384d)
        (fwd[last] || bwd[first])                          │
                                            ┌──────────────┼──────────────┐
                                            ↓              ↓              ↓
                                           POS        Morpho (×6)    Word Feedback
                                                                    (broadcast → char)
                                                                         │
                                                                         ↓
                                                              P2G Head (704d → 1198)
                                                              char_out + word_out
```

- **Entrée** : séquence de caractères IPA avec `<BOS>`, `<SEP>`, `<EOS>`
- **Word Feedback** : les représentations mot sont diffusées aux positions char correspondantes
- **P2G** : prédiction par caractère IPA avec labels `_CONT` (continuation pour marques combinantes)
- **Paramètres** : 2 562 465 (~2.56M)

## Limites connues

- Le P2G est intrinsèquement ambigu pour les homophones (est/et, a/à, ses/ces) — résolution partielle par le contexte phrastique
- Les marques morphologiques muettes (-s pluriel, -e féminin) restent la principale source d'erreur (~30%)
- Le modèle ne gère pas la ponctuation ni la casse (entrée = IPA pur)
- Performance sur mots hors-vocabulaire plus basse qu'en contexte

## Licence

Double licence :
- **AGPL-3.0** — usage libre (voir [LICENCE.txt](LICENCE.txt))
- **Licence commerciale** — usage proprietaire (voir [LICENCE-COMMERCIALE.md](LICENCE-COMMERCIALE.md))

Voir aussi [ATTRIBUTION.md](ATTRIBUTION.md) pour les credits.
