# Lectura Morpho Tagger — BiLSTM

**Analyseur morphologique complet pour le français — backend BiLSTM**

Un module pour l'analyse morphologique du français utilisant un modèle BiLSTM via
ONNX Runtime. Prédit en une seule passe : POS + Genre + Nombre + Temps + Mode +
Personne, puis lemmatise par règles.

---

## Démarrage rapide

```python
from lectura_morpho import MorphoTagger

tagger = MorphoTagger("modele/morpho_model_bilstm_int8.onnx",
                       vocab_path="modele/morpho_vocab_bilstm.json")
result = tagger.tag("Les chats mangent les souris")
# [{"mot": "Les", "pos": "ART:def", "tag_complet": "ART:def|Plur",
#   "genre": None, "nombre": "Plur", "lemme": "le"}, ...]
```

### Pré-requis

- Python 3.10+
- `onnxruntime` + `numpy`

```bash
pip install onnxruntime numpy
```

### Contenu de l'archive

```
MORPHO_BiLSTM/
├── lectura_morpho.py                       ← Fichier principal (copier dans votre projet)
├── demo_cli.py                             ← Démo en ligne de commande
├── evaluer.py                              ← Script d'évaluation
├── modele/
│   ├── morpho_model_bilstm_int8.onnx      ← Modèle BiLSTM quantisé (~10 Mo)
│   ├── morpho_vocab_bilstm.json            ← Vocabulaire BiLSTM
│   └── glaff_lookup.json                   ← Lexique GLAFF optionnel (fallback lemmes)
├── exemples/
│   ├── exemple_basique.py
│   └── exemple_integration.py
├── entrainement/
│   ├── entrainer_bilstm.py
│   ├── preparer_corpus.py
│   └── README.md
├── README.md
├── EVALUATION.md
├── LICENCE.txt
├── ATTRIBUTION.md
└── pyproject.toml
```

---

## Utilisation

### Dans votre code

Copiez `lectura_morpho.py` et `modele/` dans votre projet :

```python
from lectura_morpho import MorphoTagger

tagger = MorphoTagger("chemin/vers/morpho_model_bilstm_int8.onnx",
                       vocab_path="chemin/vers/morpho_vocab_bilstm.json")

# Analyse morphologique complète
result = tagger.tag("Les chats mangent les souris")
for r in result:
    print(f"{r['mot']:12} {r['tag_complet']:20} → {r['lemme']}")

# Affichage formaté
print(tagger.tag_formatted("Les enfants jouent dans la cour"))

# Mots déjà tokenisés
result = tagger.tag_words(["Le", "chat", "mange"])
```

### Sortie de `tag()`

```python
[{"mot": "mangent", "pos": "VER", "tag_complet": "VER|Ind|Pres|3|Plur",
  "genre": None, "nombre": "Plur", "temps": "Pres", "mode": "Ind",
  "personne": "3", "lemme": "manger"},
 {"mot": "les", "pos": "ART:def", "tag_complet": "ART:def|Plur",
  "genre": None, "nombre": "Plur", "temps": None, "mode": None,
  "personne": None, "lemme": "le"}]
```

### Démo en ligne de commande

```bash
# Phrase en argument
python demo_cli.py "Les chats mangent les souris"

# Mode interactif
python demo_cli.py
```

---

## Tags composites

Le tagger prédit un tag composite unique par token :

| Catégorie | Format | Exemple |
|-----------|--------|---------|
| Verbe fini | `VER\|Mood\|Tense\|Person\|Number` | `VER\|Ind\|Pres\|3\|Plur` |
| Verbe participe | `VER\|Part\|Gender\|Number` | `VER\|Part\|Masc\|Sing` |
| Verbe infinitif | `VER\|Inf` | `VER\|Inf` |
| Auxiliaire | `AUX\|Mood\|Tense\|Person\|Number` | `AUX\|Ind\|Pres\|3\|Sing` |
| Nom | `NOM[\|Gender][\|Number]` | `NOM\|Masc\|Plur` |
| Adjectif | `ADJ[\|Gender][\|Number]` | `ADJ\|Fem\|Plur` |
| Article | `ART:def[\|Gender][\|Number]` | `ART:def\|Masc\|Sing` |
| Pronom | `PRO:per[\|Gender][\|Number][\|Person]` | `PRO:per\|Masc\|Sing\|3` |
| Invariable | POS seul | `PRE`, `CON`, `ADV`, `INTJ` |

### POS de base (18 catégories)

| Tag | Description |
|-----|-------------|
| ART:def | Article défini (le, la, les) |
| ART:ind | Article indéfini (un, une, des) |
| PRO:per | Pronom personnel (je, tu, il) |
| PRO:rel | Pronom relatif (qui, que, dont) |
| PRO:dem | Pronom démonstratif (ce, ceci) |
| PRO:ind | Pronom indéfini (quelqu'un, rien) |
| PRO:int | Pronom interrogatif (qui, quoi) |
| ADJ:pos | Adjectif possessif (mon, ton, son) |
| ADJ:dem | Adjectif démonstratif (ce, cette) |
| ADJ:int | Adjectif interrogatif (quel, quelle) |
| NOM | Nom commun |
| ADJ | Adjectif qualificatif |
| VER | Verbe |
| AUX | Auxiliaire (être, avoir) |
| ADV | Adverbe |
| PRE | Préposition (à, de, en, par) |
| CON | Conjonction (et, ou, mais, car) |
| INTJ | Interjection (oh, ah, hélas) |

---

## Lemmatisation

Trois niveaux, dans l'ordre :
1. **Lexique GLAFF** (si chargé) : lookup exact forme|POS → lemme
2. **Table irréguliers intégrée** (~120 formes) : être, avoir, aller, faire, pouvoir, vouloir, etc.
3. **Règles de suffixation** : pluriel (-aux→-al, -s→ø), féminin (-euse→-eux, -ive→-if), conjugaisons

---

## Détails techniques

### Architecture

- **Char CNN** : embedding 32d → Conv1D (kernel 3) → 96d (max-pool)
- **Word embedding** : 192d
- **BiLSTM** : 2 couches × 192 hidden (bidirectionnel → 384d)
- **Sortie** : Linear → ~200 tags composites
- **Décodage** : argmax sur les émissions
- **Lemmatisation** : règles de suffixation + table irréguliers

### Modèle

| Propriété | Valeur |
|-----------|--------|
| Architecture | BiLSTM (CharCNN + WordEmbed → BiLSTM → Linear) |
| Taille du modèle | ~10 Mo (ONNX INT8 quantisé) |
| Tags | ~200 catégories composites |
| Dépendances runtime | onnxruntime, numpy |
| Python minimum | 3.10 |
| Lemmatisation | Règles + irréguliers (+ GLAFF optionnel) |

---

## API de référence

### `MorphoTagger(model_path, vocab_path, lexicon_path=None)`

Crée un analyseur morphologique BiLSTM.

### `tagger.tag(text) → list[dict]`

Tokenise et analyse morphologiquement un texte brut. Retourne des dicts avec clés :
`mot`, `pos`, `tag_complet`, `genre`, `nombre`, `temps`, `mode`, `personne`, `lemme`.

### `tagger.tag_words(words) → list[dict]`

Analyse une liste de mots déjà tokenisés.

### `tagger.tag_formatted(text) → str`

Retourne un résultat formaté lisible avec POS, traits et lemmes.

---

## Support

Pour toute question : [contact à définir]

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0. Voir LICENCE.txt et ATTRIBUTION.md.*
