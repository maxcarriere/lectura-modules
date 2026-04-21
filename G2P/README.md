# Lectura G2P Unifié

**Modèle unifié G2P + POS + Morphologie + Liaison pour le français**

Un seul modèle BiLSTM char-level multi-tête avec features lexicales (ONNX INT8) qui prédit simultanément :

- **G2P** : transcription phonémique IPA (98.5% word accuracy)
- **POS** : étiquetage morpho-syntaxique — 19 tags (98.5% accuracy)
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale (96-99.8%)
- **Liaison** : prédiction des liaisons obligatoires/facultatives (F1 90.6%)

Quatre backends d'inférence : ONNX Runtime, NumPy, pur Python (zéro dépendance), ou API serveur.

## Démarrage rapide

### Installation

```bash
pip install lectura-g2p             # zéro dépendance (backend pur Python)
pip install lectura-g2p[numpy]      # backend NumPy
pip install lectura-g2p[onnx]       # backend ONNX Runtime (le plus rapide)
```

### Utilisation rapide (factory — recommande)

```python
from lectura_nlp import creer_engine
from lectura_nlp.tokeniseur import tokeniser

# Mode auto : utilise les modeles locaux si presents, sinon l'API
engine = creer_engine()

tokens = tokeniser("Les enfants sont arrives a la maison.")
result = engine.analyser(tokens)

print(result["g2p"])      # ['le', 'ɑ̃fɑ̃', 'sɔ̃', 'aʁive', 'a', 'la', 'mɛzɔ̃']
print(result["pos"])      # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM']
print(result["liaison"])  # ['Lz', 'none', 'Lt', 'none', 'none', 'none', 'none']
print(result["morpho"])   # {'Number': ['Plur', ...], 'Gender': [...], ...}
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

## Modeles locaux (licence commerciale)

Pour utiliser l'inference locale sans API, achetez les modeles sur
https://www.lec-tu-ra.com/solutions/services/

Installez les modeles dans `~/.lectura/models/g2p/` :

```bash
mkdir -p ~/.lectura/models/g2p
cp unifie_v2_int8.onnx unifie_v2_vocab.json ~/.lectura/models/g2p/
```

Ou via variable d'environnement :

```bash
export LECTURA_MODELS_DIR=/path/to/models
```

`creer_engine()` detecte automatiquement les modeles locaux.

## Poids NumPy / Pure Python (optionnel)

Les backends **NumPy** et **Pure Python** necessitent les poids JSON depuis GitHub :

```bash
curl -L -o unifie_v2_weights.json \
  https://github.com/maxcarriere/lectura-modules/raw/main/G2P/modeles_numpy/unifie_v2_weights.json
```

```python
engine = creer_engine(mode="numpy")
result = engine.analyser(tokeniser("Bonjour le monde."))
```

## Backends d'inférence

| Backend | Dépendances | Vitesse | Usage |
|---------|------------|---------|-------|
| **API** | aucune | ~100 ms/phrase | Defaut (Niveau 1), zero config |
| **ONNX Runtime** | `onnxruntime` | ~2 ms/phrase | Production locale |
| **NumPy** | `numpy` | ~50 ms/phrase | Leger |
| **Pur Python** | aucune | ~200 ms/phrase | Embarque, portabilite max |

Les backends locaux (ONNX, NumPy, Pure) produisent des résultats identiques.

## Features lexicales (optionnel)

Le modele V2 accepte en entree optionnelle un vecteur de 24 dimensions par mot, construit a partir d'un lexique de candidats POS. Cela ameliore la prediction POS, la morphologie et les liaisons.

Le lexique est detecte automatiquement si present dans le dossier modeles (`lexique_pos_candidates.json`), ou via le parametre `lexicon_path` de `creer_engine()`. Sans lexique, le modele fonctionne normalement (features = zeros).

```python
# Avec lexique (automatique si disponible)
engine = creer_engine()

# Desactiver les features lexicales
result = engine.analyser(tokens, use_lex=False)
```

## Benchmarks (dev set, modele V2 avec features lexicales)

| Tâche | Métrique | Score |
|-------|----------|-------|
| **G2P** | Word Accuracy | **98.5%** |
| **G2P** | PER (Phone Error Rate) | **0.5%** |
| **POS** | Accuracy | **98.5%** |
| **Liaison** | Macro F1 | **90.6%** |
| **Morpho** — Number | Accuracy | **96.4%** |
| **Morpho** — Gender | Accuracy | **98.3%** |
| **Morpho** — VerbForm | Accuracy | **99.6%** |
| **Morpho** — Mood | Accuracy | **99.8%** |
| **Morpho** — Tense | Accuracy | **99.8%** |
| **Morpho** — Person | Accuracy | **99.7%** |

Voir [EVALUATION.md](EVALUATION.md) pour les résultats détaillés.

## API

### `tokeniser(text) -> list[str]`

Tokenise une phrase française (gestion apostrophes, ponctuation, contractions).

### `creer_engine(mode, models_dir, api_url, api_key, lexicon_path)`

Factory pour creer un engine d'inference. Modes : `"auto"`, `"local"`, `"api"`, `"onnx"`, `"numpy"`, `"pure"`.
`models_dir` permet de specifier le dossier des modeles (sinon cascade automatique).

### `engine.analyser(tokens, *, use_lex=True) -> dict`

Analyse une liste de tokens et retourne un dictionnaire :
- `tokens` : liste des tokens d'entrée
- `g2p` : transcription IPA par token
- `pos` : étiquette POS par token
- `liaison` : label liaison par token (`none`, `Lz`, `Lt`, `Ln`, `Lr`, `Lp`)
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

Le parametre `use_lex=False` desactive les features lexicales (utile pour le benchmarking).

### `corriger_g2p(mot, ipa) -> str`

Applique la table de corrections puis les règles (ex+consonne, ex+voyelle, yod).

### `appliquer_liaison(tokens, phones, liaisons) -> list[str]`

Applique les consonnes de liaison entre tokens consécutifs.

## Architecture du modèle (V2)

```
Phrase → Char Embedding (64d) → Shared BiLSTM (2×160h → 320d)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ↓                                       ↓
              G2P Head (per-char)         Word repr (320d) + Lex Features (24d)
              Linear(320→40)                        │
                                          Word BiLSTM (128h → 256d)
                                                │
                                    ┌───┬───┬───┬───┬───┬───┬───┐
                                   POS Num Gen VF  Mood Tns Per Liaison
```

- **Entrée** : séquence de caractères avec `<BOS>`, `<SEP>`, `<EOS>`
- **Lex Features** : 24d par mot (21 POS one-hot + known + n_candidates + unambiguous)
- **G2P** : prédiction par caractère avec labels `_CONT` (continuation)
- **Têtes mot** : représentation mot = fwd[dernier_char] || bwd[premier_char] + lex_proj(24→24)

## Limites connues

- Le G2P est évalué en contexte phrastique ; la performance sur mots isolés hors-vocabulaire est plus basse (~92%)
- La liaison `Lp` est très rare dans les données d'entraînement (F1 = 66.7%)
- Les noms propres et néologismes peuvent produire des transcriptions approximatives
- Le modèle ne gère pas les homographes contextuels complexes (ex: "fils" = /fis/ vs /fil/)

## Licence

Ce module est distribue sous licence **AGPL-3.0** (non commerciale) — voir [LICENCE.txt](LICENCE.txt).

Pour un usage commercial, contacter [contact@lec-tu-ra.com](mailto:contact@lec-tu-ra.com).

Voir aussi [ATTRIBUTION.md](ATTRIBUTION.md) pour les credits.
