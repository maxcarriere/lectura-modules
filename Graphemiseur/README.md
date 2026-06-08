# Lectura Graphemiseur

**Graphemiseur neural du francais : P2G + POS + Morphologie (IPA → orthographe)**

> Anciennement `lectura-p2g` (pip) / `lectura_p2g` (import).
> Renomme `lectura-graphemiseur` / `lectura_graphemiseur` a partir de la v4.0.0.

Un seul modele BiLSTM char-level multi-tete V7 avec attention cross word-char, lex_select et phone_lex_features (ONNX INT8) qui predit simultanement :

- **P2G** : transcription IPA vers orthographe (~95% word accuracy core, ~96% pipeline complet)
- **POS** : etiquetage morpho-syntaxique — 19 tags (98.3% accuracy)
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale (94.7-99.7%)

Le modele V7 ajoute un mecanisme d'attention cross word-char et le lex_select (selection lexicale parmi candidats phonetiques).

Quatre backends d'inference : ONNX Runtime, NumPy, pur Python (zero dependance), ou API serveur.

## Demarrage rapide

### Installation

```bash
pip install lectura-graphemiseur             # zero dependance (backend pur Python)
pip install lectura-graphemiseur[numpy]      # backend NumPy
pip install lectura-graphemiseur[onnx]       # backend ONNX Runtime (le plus rapide)
```

### Utilisation rapide (factory — recommande)

```python
from lectura_graphemiseur import creer_engine

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

## Modeles locaux (licence commerciale)

Pour utiliser l'inference locale sans API, achetez les modeles sur
https://www.lec-tu-ra.com/solutions/services/

Installez les modeles dans `~/.lectura/models/p2g/` :

```bash
mkdir -p ~/.lectura/models/p2g
cp unifie_p2g_v7_int8.onnx unifie_p2g_v7_vocab.json phone_lexicon.db ~/.lectura/models/p2g/
```

Ou via variable d'environnement :

```bash
export LECTURA_MODELS_DIR=/path/to/models
```

`creer_engine()` detecte automatiquement les modeles locaux. Le fichier `phone_lexicon.db` (lexique phonetique) est utilise par le modele V7 pour les phone_lex_features et le lex_select. Sans ce fichier, le modele fonctionne en mode degrade (features = zeros).

## Poids NumPy / Pure Python (optionnel)

Les backends **NumPy** et **Pure Python** necessitent les poids JSON depuis GitHub :

```bash
curl -L -o unifie_p2g_v3_weights.json \
  https://github.com/maxcarriere/lectura-modules/raw/main/Graphemiseur/modeles_numpy/unifie_p2g_v3_weights.json
```

```python
engine = creer_engine(mode="numpy")
result = engine.analyser(["bɔ̃ʒuʁ", "lə", "mɔ̃d"])
```

## Backends d'inference

| Backend | Dependances | Vitesse | Usage |
|---------|------------|---------|-------|
| **API** | aucune | ~100 ms/phrase | Defaut (Niveau 1), zero config |
| **ONNX Runtime** | `onnxruntime` | ~2 ms/phrase | Production locale |
| **NumPy** | `numpy` | ~50 ms/phrase | Leger |
| **Pur Python** | aucune | ~200 ms/phrase | Embarque, portabilite max |

Les backends locaux (ONNX, NumPy, Pure) produisent des resultats identiques.

## Pipeline V7 (raw → lex_select → post-traitement)

Le pipeline complet applique trois etapes successives :

1. **Raw** : prediction brute du modele BiLSTM (82.32% word accuracy)
2. **Lex_select** : selection lexicale par phone_lexicon — le modele choisit parmi les candidats phonetiquement compatibles (87.33%)
3. **Post-traitement** :
   - **Formules** : reconnaissance deterministe de nombres, sigles, dates via `lectura_formules`
   - **Coherence morpho** : accord ortho-morpho (pluriel, feminin, conjugaison)
   - **Accents** : correction a/a, ou/ou par POS
   - → **90.95%** word accuracy (pipeline complet)

## Phone_lex_features (V7)

Le modele V7 utilise un vecteur de 28 dimensions par mot (`phone_lex_features`), construit a partir du `phone_lexicon.db` :

- 19d : POS one-hot (candidats POS du lexique phonetique)
- 3d : features morphologiques (genre, nombre)
- 6d : features lexicales (known, n_candidates, unambiguous, top_freq, has_verb, has_nom)

Le `phone_lexicon.db` est detecte automatiquement dans le dossier modeles. Sans lexique, le modele fonctionne normalement (features = zeros).

## Benchmarks (dev set, modele V7 pipeline complet)

| Tache | Metrique | Score |
|-------|----------|-------|
| **P2G** | Word Accuracy (core + lex_select) | **~95%** |
| **POS** | Accuracy | **98.3%** |
| **Morpho** — Number | Accuracy | **94.7%** |
| **Morpho** — Gender | Accuracy | **97.6%** |
| **Morpho** — VerbForm | Accuracy | **99.5%** |
| **Morpho** — Mood | Accuracy | **99.7%** |
| **Morpho** — Tense | Accuracy | **99.7%** |
| **Morpho** — Person | Accuracy | **99.6%** |

Voir [EVALUATION.md](EVALUATION.md) pour les resultats detailles et la comparaison v1/v2/v3/v6/v7.

## API

### `creer_engine(mode, models_dir, api_url, api_key, lexicon_path)`

Factory pour creer un engine d'inference. Modes : `"auto"`, `"local"`, `"api"`, `"onnx"`, `"numpy"`, `"pure"`.
`models_dir` permet de specifier le dossier des modeles (sinon cascade automatique).

### `engine.analyser(ipa_words, *, use_lex=True) -> dict`

Analyse une liste de mots IPA et retourne un dictionnaire :
- `ipa_words` : liste des mots IPA d'entree
- `ortho` : orthographe reconstruite par mot
- `pos` : etiquette POS par mot
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

Le parametre `use_lex=False` desactive les features lexicales (utile pour le benchmarking).

### `tokeniser_ipa(text) -> list[str]`

Tokenise une phrase IPA (split sur espaces).

### `corriger_phrase_v3(ortho_words, pos_tags, morpho_features, ..., ipa_words=None) -> list[str]`

Pipeline post-traitement V6 complet : formules (via `ipa_words`) + coherence morpho + accents.
Le parametre `ipa_words` active la reconnaissance de formules (nombres, sigles).

### `corriger_phrase_v2(ortho_words, pos_tags, lexique) -> list[str]`

Post-traitement contextuel inter-mots : accord determinant-nom, sujet-verbe.

## Architecture du modele (V7)

```
Phrase IPA → Char Embedding (64d) → Shared BiLSTM (2x192h → 384d)
                                          |
                  +-----------------------+--------------------+
                  v                                             v
        Word representations              Word repr (384d) + Phone Lex Features (28d)
        (fwd[last] || bwd[first])                          |
                                                 Word BiLSTM (192h → 384d)
                                                       |
                                            +--------------+--------------+
                                           POS        Morpho (x6)    Attention Cross
                                                                    → P2G Head
                                                                    → Lex_Select Head
```

- **Entree** : sequence de caracteres IPA avec `<BOS>`, `<SEP>`, `<EOS>`
- **Phone Lex Features** : 28d par mot (19 POS one-hot + 3 morpho + 6 lex features)
- **Lex Select** : selection parmi candidats phonetiques du phone_lexicon
- **Word Feedback** : les representations mot sont diffusees aux positions char correspondantes
- **P2G** : prediction par caractere IPA avec labels `_CONT` (continuation pour marques combinantes)

## Limites connues

- Le P2G est intrinsequement ambigu pour les homophones (est/et, a/a, ses/ces) — resolution partielle par le contexte phrastique
- Les marques morphologiques muettes (-s pluriel, -e feminin) restent la principale source d'erreur (~30%)
- Le modele ne gere pas la ponctuation ni la casse (entree = IPA pur)
- Performance sur mots hors-vocabulaire plus basse qu'en contexte

## Licence

Ce module est distribue sous licence **AGPL-3.0** (non commerciale) — voir [LICENCE.txt](LICENCE.txt).

Pour un usage commercial, contacter [contact@lec-tu-ra.com](mailto:contact@lec-tu-ra.com).

Voir aussi [ATTRIBUTION.md](ATTRIBUTION.md) pour les credits.
