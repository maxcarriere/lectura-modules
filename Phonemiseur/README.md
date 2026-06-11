# Lectura Phonemiseur

**Phonemiseur neural du francais : G2P + POS + Morphologie + Liaison + Groupes de lecture**

Un seul modele BiLSTM char-level multi-tete (1.75M parametres, ONNX INT8 = 1.8 Mo) qui predit simultanement :

- **G2P** : transcription phonemique IPA (98.5% word accuracy)
- **POS** : etiquetage morpho-syntaxique — 19 tags (98.5% accuracy)
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale (96-99.8%)
- **Liaison** : prediction des liaisons obligatoires/facultatives (F1 90.6%)

Quatre backends d'inference : ONNX Runtime, NumPy, pur Python (zero dependance), ou API serveur.

## Demarrage rapide

### Installation

```bash
pip install lectura-phonemiseur             # zero dependance (backend pur Python)
pip install lectura-phonemiseur[numpy]      # backend NumPy
pip install lectura-phonemiseur[onnx]       # backend ONNX Runtime (le plus rapide)
```

### Utilisation rapide (factory — recommande)

```python
from lectura_phonemiseur import creer_engine
from lectura_phonemiseur.tokeniseur import tokeniser

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

### Separateurs pour mots composes

Le modele V4 gere nativement les separateurs dans les mots composes et les elisions :

```python
# IPA plat (defaut)
result = engine.analyser(["abat-jour", "d'abord"])
print(result["g2p"])  # ['abaʒuʁ', 'dabɔʁ']

# Avec separateurs preserves
result = engine.analyser(["abat-jour", "d'abord"], sep_hyphen=True, sep_apos=True)
print(result["g2p"])  # ['aba-ʒuʁ', "d'abɔʁ"]
```

## Mode API (zero config)

Sans modeles locaux, `creer_engine()` utilise automatiquement l'API Lectura :

```python
engine = creer_engine()  # mode="auto" → API si pas de modeles locaux
# ou explicitement :
engine = creer_engine(mode="api", api_url="https://api.lectura.world")
```

Variables d'environnement : `LECTURA_API_URL`, `LECTURA_API_KEY`.

## Modeles locaux (licence commerciale)

Pour utiliser l'inference locale sans API, achetez les modeles sur
https://www.lectura.world/solutions/services/

Installez les modeles dans `~/.lectura/models/g2p/` :

```bash
mkdir -p ~/.lectura/models/g2p
cp unifie_v4_int8.onnx unifie_v4_vocab.json ~/.lectura/models/g2p/
```

Ou via variable d'environnement :

```bash
export LECTURA_MODELS_DIR=/path/to/models
```

`creer_engine()` detecte automatiquement les modeles locaux.

## Backends d'inference

| Backend | Dependances | Vitesse | Usage |
|---------|------------|---------|-------|
| **API** | aucune | ~100 ms/phrase | Defaut (Niveau 1), zero config |
| **ONNX Runtime** | `onnxruntime` | ~2 ms/phrase | Production locale |
| **NumPy** | `numpy` | ~50 ms/phrase | Leger |
| **Pur Python** | aucune | ~200 ms/phrase | Embarque, portabilite max |

Les backends locaux (ONNX, NumPy, Pure) produisent des resultats identiques.

## Features lexicales (optionnel)

Le modele accepte en entree optionnelle un vecteur de 24 dimensions par mot, construit a partir d'un lexique de candidats POS. Cela ameliore la prediction POS, la morphologie et les liaisons.

Le lexique est detecte automatiquement si `lectura-lexique` est installe, ou via le parametre `lexicon_path` de `creer_engine()`. Sans lexique, le modele fonctionne normalement (features = zeros).

```python
# Avec lexique (automatique si lectura-lexique est installe)
engine = creer_engine()

# Desactiver les features lexicales
result = engine.analyser(tokens, use_lex=False)
```

## Corrections G2P

Le post-traitement applique automatiquement trois niveaux de corrections :

1. **Homographes POS-aware** (~2 700 entrees) : desambiguisation par categorie grammaticale (ex: jean/NOM = dʒin vs Jean/NOM PROPRE = ʒɑ̃)
2. **Table plate** (~3 500 corrections manuelles) : corrections ciblees avec separateurs
3. **Corrections lexique** (optionnel, ~30 000 entrees) : corrections etendues depuis le lexique de reference

```python
# Activer les corrections etendues du lexique
engine = creer_engine(corrections_lexique=True)
```

Les corrections stockent les separateurs dans l'IPA (`aba-ʒuʁ`, `d'abɔʁ`). Quand `sep_hyphen`/`sep_apos` ne sont pas actifs, les separateurs sont automatiquement retires.

## Benchmarks (dev set, modele V4 avec features lexicales)

| Tache | Metrique | Score |
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

## API

### `tokeniser(text) -> list[str]`

Tokenise une phrase francaise (gestion apostrophes, ponctuation, contractions).

### `creer_engine(mode, models_dir, api_url, api_key, lexicon_path, corrections_lexique)`

Factory pour creer un engine d'inference. Modes : `"auto"`, `"local"`, `"api"`, `"onnx"`, `"numpy"`, `"pure"`.
`models_dir` permet de specifier le dossier des modeles (sinon cascade automatique).
`corrections_lexique=True` charge les corrections etendues du lexique (~30K entrees).

### `engine.analyser(tokens, *, use_lex=True, sep_hyphen=False, sep_apos=False) -> dict`

Analyse une liste de tokens et retourne un dictionnaire :
- `g2p` : transcription IPA par token
- `pos` : etiquette POS par token
- `liaison` : label liaison par token (`none`, `Lz`, `Lt`, `Ln`, `Lr`, `Lp`)
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

Options :
- `use_lex=False` : desactive les features lexicales
- `sep_hyphen=True` : conserve les tirets dans l'IPA des mots composes
- `sep_apos=True` : conserve les apostrophes dans l'IPA des elisions

### `corriger_g2p(mot, ipa, pos, *, keep_sep=False) -> str`

Applique la table de corrections puis les regles (ex+consonne, ex+voyelle, yod).
`keep_sep=True` conserve les separateurs `-` et `'` dans l'IPA retourne.

### `appliquer_liaison(tokens, phones, liaisons) -> list[str]`

Applique les consonnes de liaison entre tokens consecutifs.

## Architecture du modele (V4)

```
Phrase → Char Embedding (64d) → Shared BiLSTM (2x160h → 320d)
                                        |
                    +-------------------+-------------------+
                    v                                       v
              G2P Head (per-char)         Word repr (320d) + Lex Features (24d)
              Linear(320→40)                        |
                                          Word BiLSTM (128h → 256d)
                                                |
                                    +---+---+---+---+---+---+---+
                                   POS Num Gen VF  Mood Tns Per Liaison
```

- **Entree** : sequence de caracteres avec `<BOS>`, `<SEP>`, `<EOS>`
- **Lex Features** : 24d par mot (21 POS one-hot + known + n_candidates + unambiguous)
- **G2P** : prediction par caractere avec labels `_CONT` (continuation) et separateurs `-`/`'`
- **Tetes mot** : representation mot = fwd[dernier_char] || bwd[premier_char] + lex_proj(24→24)

## Limites connues

- Le G2P est evalue en contexte phrastique ; la performance sur mots isoles hors-vocabulaire est plus basse (~92%)
- La liaison `Lp` est tres rare dans les donnees d'entrainement (F1 = 66.7%)
- Les noms propres et neologismes peuvent produire des transcriptions approximatives

## Changelog

### 4.1.1
- Fix pipeline_formules : elisions clitiques, propagation sep_hyphen aux compounds, neutralisation liaison avant ponctuation

### 4.1.0
- Fix separateurs : `corriger_g2p()` avec `keep_sep`, corrections enrichies avec separateurs
- Simplification : V4 uniquement, suppression cascades V2/V3
- Ajout `g2p_corrections_lexique.json` (30K corrections avec separateurs)
- Suppression dependance `lexique_pos_candidates.json` (fallback `lectura-lexique`)

### 4.0.2
- Corrections G2P etendues, homographes NOM/NOM PROPRE

### 4.0.0
- Renommage `lectura-g2p` → `lectura-phonemiseur`
- Modele V4 avec separateurs natifs

## Licence

Ce module est distribue sous licence **AGPL-3.0** (non commerciale) — voir [LICENCE.txt](LICENCE.txt).

Pour un usage commercial, contacter [admin@lectura.world](mailto:admin@lectura.world).

Voir aussi [ATTRIBUTION.md](ATTRIBUTION.md) pour les credits.
