# Lectura P2G Unifié

**Modèle unifié P2G + POS + Morphologie pour le français (IPA → orthographe)**

Un seul modèle BiLSTM char-level multi-tête avec word feedback et features lexicales (ONNX INT8) qui prédit simultanément :

- **P2G** : transcription IPA vers orthographe (93.1% word accuracy, 2.2% CER)
- **POS** : étiquetage morpho-syntaxique — 19 tags (98.3% accuracy)
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale (94.7-99.7%)

Quatre backends d'inférence : ONNX Runtime, NumPy, pur Python (zéro dépendance), ou API serveur.

## Démarrage rapide

### Installation

```bash
pip install lectura-p2g             # zéro dépendance (backend pur Python)
pip install lectura-p2g[numpy]      # backend NumPy
pip install lectura-p2g[onnx]       # backend ONNX Runtime (le plus rapide)
```

### Utilisation rapide (factory — recommande)

```python
from lectura_p2g import creer_engine

# Mode auto : utilise les modeles locaux si presents, sinon l'API
engine = creer_engine()

result = engine.analyser(["le", "ɑ̃fɑ̃", "sɔ̃", "aʁive", "a", "la", "mɛzɔ̃"])

print(result["ortho"])   # ['les', 'enfants', 'sont', 'arrives', 'a', 'la', 'maison']
print(result["pos"])     # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM']
print(result["morpho"])  # {'Number': ['Plur', 'Plur', ...], 'Gender': [...], ...}
```

Modes disponibles : `"auto"` (defaut), `"local"`, `"api"`, `"onnx"`, `"numpy"`, `"pure"`.

## Mode API (zero config)

Sans modeles locaux, `creer_engine()` utilise automatiquement l'API Lectura :

```python
engine = creer_engine()  # mode="auto" → API si pas de modeles locaux
# ou explicitement :
engine = creer_engine(mode="api", api_url="https://api.lec-tu-ra.com")
```

Variables d'environnement : `LECTURA_API_URL`, `LECTURA_API_KEY`.

## Poids NumPy / Pure Python (optionnel)

Les backends **NumPy** et **Pure Python** necessitent les poids JSON depuis GitHub :

```bash
curl -L -o unifie_p2g_v3_weights.json \
  https://github.com/maxcarriere/lectura-modules/raw/main/P2G/modeles_numpy/unifie_p2g_v3_weights.json
```

```python
engine = creer_engine(mode="numpy")
result = engine.analyser(["bɔ̃ʒuʁ", "lə", "mɔ̃d"])
```

## Backends d'inférence

| Backend | Dépendances | Vitesse | Usage |
|---------|------------|---------|-------|
| **API** | aucune | ~100 ms/phrase | Defaut (Niveau 1), zero config |
| **ONNX Runtime** | `onnxruntime` | ~2 ms/phrase | Production locale |
| **NumPy** | `numpy` | ~50 ms/phrase | Leger |
| **Pur Python** | aucune | ~200 ms/phrase | Embarque, portabilite max |

Les backends locaux (ONNX, NumPy, Pure) produisent des résultats identiques.

## Benchmarks (dev set, modele V3 avec features lexicales)

| Tâche | Métrique | Score |
|-------|----------|-------|
| **P2G** | Word Accuracy | **93.1%** |
| **P2G** | CER (Character Error Rate) | **2.2%** |
| **POS** | Accuracy | **98.3%** |
| **Morpho** — Number | Accuracy | **94.7%** |
| **Morpho** — Gender | Accuracy | **97.6%** |
| **Morpho** — VerbForm | Accuracy | **99.5%** |
| **Morpho** — Mood | Accuracy | **99.7%** |
| **Morpho** — Tense | Accuracy | **99.7%** |
| **Morpho** — Person | Accuracy | **99.6%** |

Voir [EVALUATION.md](EVALUATION.md) pour les résultats détaillés et la comparaison v1/v2/v3.

## API

### `creer_engine(mode, api_url, api_key, lexicon_path)`

Factory pour creer un engine d'inference. Modes : `"auto"`, `"local"`, `"api"`, `"onnx"`, `"numpy"`, `"pure"`.

### `engine.analyser(ipa_words, *, use_lex=True) -> dict`

Analyse une liste de mots IPA et retourne un dictionnaire :
- `ipa_words` : liste des mots IPA d'entrée
- `ortho` : orthographe reconstruite par mot
- `pos` : étiquette POS par mot
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

Le parametre `use_lex=False` desactive les features lexicales (utile pour le benchmarking).

### `tokeniser_ipa(text) -> list[str]`

Tokenise une phrase IPA (split sur espaces).

### `corriger_phrase_v2(ortho_words, pos_tags, lexique) -> list[str]`

Post-traitement contextuel inter-mots : accord déterminant-nom, sujet-verbe.

## Architecture du modèle (V3)

```
Phrase IPA → Char Embedding (64d) → Shared BiLSTM (2×160h → 320d)
                                          │
                  ┌───────────────────────┼────────────────────┐
                  ↓                                             ↓
        Word representations                Word repr (320d) + Lex Features (24d)
        (fwd[last] || bwd[first])                          │
                                                 Word BiLSTM (192h → 384d)
                                                       │
                                         ┌─────────────┼──────────────┐
                                         ↓             ↓              ↓
                                        POS       Morpho (×6)    Word Feedback
                                                                 (broadcast → char)
                                                                      │
                                                                      ↓
                                                           P2G Head (704d → 1198)
                                                           char_out + word_out
```

- **Entrée** : séquence de caractères IPA avec `<BOS>`, `<SEP>`, `<EOS>`
- **Lex Features** : 24d par mot (21 POS one-hot + known + n_candidates + unambiguous)
- **Word Feedback** : les représentations mot sont diffusées aux positions char correspondantes
- **P2G** : prédiction par caractère IPA avec labels `_CONT` (continuation pour marques combinantes)

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
