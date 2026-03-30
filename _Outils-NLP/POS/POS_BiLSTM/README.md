# Lectura POS Tagger — BiLSTM

**Étiqueteur grammatical pour le français — backend BiLSTM**

Un module pour l'étiquetage grammatical (POS tagging) du français utilisant un modèle
BiLSTM via ONNX Runtime. Haute précision avec un modèle compact (6 Mo).

---

## Démarrage rapide

```python
from lectura_pos import PosTagger

tagger = PosTagger("modele/pos_model_bilstm_int8.onnx",
                    vocab_path="modele/pos_vocab_bilstm.json",
                    lexicon_path="modele/mini_lexique.json")
result = tagger.tag("Le chat mange la souris")
# [("Le", "ART:def"), ("chat", "NOM"), ("mange", "VER"),
#  ("la", "ART:def"), ("souris", "NOM")]
```

### Pré-requis

- Python 3.10+
- `onnxruntime` + `numpy`

```bash
pip install onnxruntime numpy
```

### Contenu de l'archive

```
POS_BiLSTM/
├── lectura_pos.py                       ← Fichier principal (copier dans votre projet)
├── demo_cli.py                          ← Démo en ligne de commande
├── evaluer.py                           ← Script d'évaluation
├── modele/
│   ├── pos_model_bilstm_int8.onnx      ← Modèle BiLSTM quantisé (6 Mo)
│   ├── pos_vocab_bilstm.json            ← Vocabulaire BiLSTM (868 Ko)
│   └── mini_lexique.json                ← Mini-lexique de correction (31 entrées)
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
└── ATTRIBUTION.md
```

---

## Utilisation

### Dans votre code

Copiez `lectura_pos.py` et `modele/` dans votre projet :

```python
from lectura_pos import PosTagger

tagger = PosTagger("chemin/vers/pos_model_bilstm_int8.onnx",
                    vocab_path="chemin/vers/pos_vocab_bilstm.json",
                    lexicon_path="chemin/vers/mini_lexique.json")

# Étiquetage simple
result = tagger.tag("Le chat mange la souris")
# [("Le", "ART:def"), ("chat", "NOM"), ("mange", "VER"), ...]

# Étiquetage détaillé
details = tagger.tag_detailed("Le chat mange")
# [{"mot": "Le", "tag": "ART:def", "description": "Article défini (le, la, les)"}, ...]

# Affichage formaté
print(tagger.tag_formatted("Le chat mange la souris"))

# Mots déjà tokenisés
result = tagger.tag_words(["Le", "chat", "mange"])
# [("Le", "ART:def"), ("chat", "NOM"), ("mange", "VER")]
```

### Démo en ligne de commande

```bash
# Phrase en argument
python demo_cli.py "Le chat mange la souris"

# Mode interactif
python demo_cli.py
```

---

## Tagset

Le tagger utilise 18 catégories grammaticales :

| Tag | Description | Exemples |
|-----|-------------|----------|
| ART:def | Article défini | le, la, les |
| ART:ind | Article indéfini | un, une, des |
| PRO:per | Pronom personnel | je, tu, il, nous |
| PRO:rel | Pronom relatif | qui, que, dont |
| PRO:dem | Pronom démonstratif | ce, ceci, cela |
| PRO:ind | Pronom indéfini | quelqu'un, rien |
| PRO:int | Pronom interrogatif | qui, quoi, lequel |
| ADJ:pos | Adjectif possessif | mon, ton, son |
| ADJ:dem | Adjectif démonstratif | ce, cette, ces |
| ADJ:int | Adjectif interrogatif | quel, quelle |
| NOM | Nom commun | chat, maison |
| ADJ | Adjectif qualificatif | grand, petit |
| VER | Verbe | mange, court |
| AUX | Auxiliaire | être, avoir |
| ADV | Adverbe | très, bien |
| PRE | Préposition | à, de, en, par |
| CON | Conjonction | et, ou, mais, car |
| INTJ | Interjection | oh, ah, hélas |

---

## Détails techniques

### Architecture

- **Char CNN** : embedding 32d → Conv1D (kernel 3) → 64d (max-pool)
- **Word embedding** : 128d
- **BiLSTM** : 2 couches × 128 hidden (bidirectionnel → 256d)
- **Sortie** : Linear → 18 tags
- **Décodage** : argmax sur les émissions
- **Post-traitement** : mini-lexique optionnel (31 corrections)

### Modèle

| Propriété | Valeur |
|-----------|--------|
| Architecture | BiLSTM (CharCNN + WordEmbed → BiLSTM → Linear) |
| Taille du modèle | 6 Mo (ONNX INT8 quantisé) |
| Vocabulaire | 42 363 mots, 240 caractères |
| Tags | 18 catégories |
| Dépendances runtime | onnxruntime, numpy |
| Python minimum | 3.10 |
| Post-traitement | Mini-lexique de 31 corrections |

---

## API de référence

### `PosTagger(model_path, vocab_path, lexicon_path=None)`

Crée un étiqueteur grammatical BiLSTM.

### `tagger.tag(text) → list[tuple[str, str]]`

Tokenise et étiquète un texte brut.

### `tagger.tag_words(words) → list[tuple[str, str]]`

Étiquète une liste de mots déjà tokenisés.

### `tagger.tag_detailed(text) → list[dict]`

Retourne des dicts `{"mot", "tag", "description"}`.

### `tagger.tag_formatted(text) → str`

Retourne un résultat formaté lisible.

---

## Support

Pour toute question : [contact à définir]

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0. Voir LICENCE.txt et ATTRIBUTION.md.*
