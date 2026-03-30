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
pip install lectura-p2g-unifie             # zéro dépendance (backend pur Python)
pip install lectura-p2g-unifie[numpy]      # backend NumPy
pip install lectura-p2g-unifie[onnx]       # backend ONNX Runtime (le plus rapide)
```

### Utilisation minimale

```python
from lectura_p2g.inference_onnx import OnnxInferenceEngine

# Charger le modèle
engine = OnnxInferenceEngine("modeles/unifie_p2g_v2_int8.onnx",
                              "modeles/unifie_p2g_v2_vocab.json")

# Analyser une phrase IPA
result = engine.analyser(["le", "ɑ̃fɑ̃", "sɔ̃", "aʁive", "a", "la", "mɛzɔ̃"])

# Résultats
print(result["ortho"])   # ['les', 'enfants', 'sont', 'arrivés', 'à', 'la', 'maison']
print(result["pos"])     # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM']
print(result["morpho"])  # {'Number': ['Plur', 'Plur', ...], 'Gender': [...], ...}
```

## Contenu de l'archive

```
├── pyproject.toml              Packaging Python (pip install)
├── README.md                   Ce fichier
├── LICENCE.txt                 CC BY-SA 4.0
├── ATTRIBUTION.md              Crédits (GLAFF, Lexique, UD-GSD)
├── CHANGELOG.md                Historique des versions
├── EVALUATION.md               Benchmarks détaillés
├── demo_cli.py                 CLI interactive
├── src/lectura_p2g/            Package Python
│   ├── __init__.py             API publique
│   ├── inference_onnx.py       Backend ONNX Runtime
│   ├── inference_numpy.py      Backend NumPy
│   ├── inference_pure.py       Backend pur Python
│   ├── tokeniseur.py           Tokenisation IPA
│   ├── posttraitement.py       Post-traitement contextuel + morpho
│   ├── modele.py               Définition PyTorch (entraînement)
│   └── utils/                  Utilitaires (aligneur, IPA, labels P2G)
├── modeles/
│   ├── unifie_p2g_v2_int8.onnx    Modèle ONNX INT8 (2.6 Mo)
│   ├── unifie_p2g_v2_vocab.json   Vocabulaire (20 Ko)
│   ├── unifie_p2g_v2_weights.json Poids JSON pour backends NumPy/Pure
│   └── metrics_p2g_test.json      Métriques d'évaluation
├── entrainement/               Scripts d'entraînement
└── tests/                      Tests unitaires
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

CC BY-SA 4.0 — Voir [LICENCE.txt](LICENCE.txt) et [ATTRIBUTION.md](ATTRIBUTION.md).
