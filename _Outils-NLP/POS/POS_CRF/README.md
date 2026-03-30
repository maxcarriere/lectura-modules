# Lectura POS Tagger

**Étiqueteur grammatical CRF pour le français**

Un modèle léger (1.8 Mo) et autonome pour l'analyse morphosyntaxique du français.
Zéro dépendance externe — un seul fichier Python, prêt à l'emploi.

---

## Démarrage rapide

```python
from lectura_pos import PosTagger

tagger = PosTagger("modele/pos_model_crf.json")
result = tagger.tag("Le chat mange la souris")

# [("Le", "ART:def"), ("chat", "NOM"), ("mange", "VER"),
#  ("la", "ART:def"), ("souris", "NOM")]
```

### Prérequis

- Python 3.10+
- Aucune bibliothèque externe requise

### Contenu de l'archive

```
lectura-pos-tagger-v1.0/
├── lectura_pos.py              ← Fichier principal (copier dans votre projet)
├── demo_cli.py                 ← Démo en ligne de commande
├── demo_web.py                 ← Interface web (nécessite Gradio)
├── modele/
│   └── pos_model_crf.json      ← Modèle CRF entraîné (1.8 Mo)
├── exemples/
│   ├── exemple_basique.py      ← Utilisation simple
│   └── exemple_integration.py  ← Intégration dans un pipeline
├── README.md                   ← Ce fichier
└── LICENCE.txt                 ← Conditions d'utilisation
```

---

## Utilisation

### Dans votre code

Copiez `lectura_pos.py` et `modele/pos_model_crf.json` dans votre projet :

```python
from lectura_pos import PosTagger

tagger = PosTagger("chemin/vers/pos_model_crf.json")

# Étiquetage simple
pairs = tagger.tag("Je suis allé au marché")
# [("Je", "PRO:per"), ("suis", "AUX"), ("allé", "VER"),
#  ("au", "PRE"), ("marché", "NOM")]

# Avec descriptions
details = tagger.tag_detailed("Je suis allé au marché")
# [{"mot": "Je", "tag": "PRO:per", "description": "Pronom personnel"}, ...]

# Mots déjà tokenisés
tags = tagger.tag_words(["Le", "chat", "mange"])
# [("Le", "ART:def"), ("chat", "NOM"), ("mange", "VER")]

# Affichage formaté
print(tagger.tag_formatted("Les enfants jouent dans la cour"))
#   Les       ART:def  Article défini
#   enfants   NOM      Nom commun
#   jouent    VER      Verbe
#   dans      PRE      Préposition
#   la        ART:def  Article défini
#   cour      NOM      Nom commun
```

### Démo en ligne de commande

```bash
# Phrase en argument
python demo_cli.py "Le chat mange la souris"

# Mode interactif
python demo_cli.py
```

### Interface web

```bash
pip install gradio
python demo_web.py
# → Ouvre http://localhost:7860
```

---

## Tagset (18 catégories)

| Tag | Description | Exemples |
|-----|-------------|----------|
| `ART:def` | Article défini | le, la, les, l' |
| `ART:ind` | Article indéfini | un, une, des |
| `PRO:per` | Pronom personnel | je, tu, il, nous, vous, ils |
| `PRO:rel` | Pronom relatif | qui, que, dont, où |
| `PRO:dem` | Pronom démonstratif | ce, ceci, cela, celui |
| `PRO:ind` | Pronom indéfini | quelqu'un, rien, tout |
| `PRO:int` | Pronom interrogatif | qui, quoi, lequel |
| `ADJ:pos` | Adjectif possessif | mon, ton, son, notre |
| `ADJ:dem` | Adjectif démonstratif | ce, cette, ces |
| `ADJ:int` | Adjectif interrogatif | quel, quelle |
| `NOM` | Nom commun | chat, maison, idée |
| `ADJ` | Adjectif qualificatif | grand, beau, rouge |
| `VER` | Verbe | mange, parle, court |
| `AUX` | Auxiliaire | est, a, sera, avait |
| `ADV` | Adverbe | très, bien, souvent |
| `PRE` | Préposition | à, de, en, par, pour |
| `CON` | Conjonction | et, ou, mais, car |
| `INTJ` | Interjection | oh, ah, hélas |

---

## Détails techniques

### Architecture

- **Modèle** : CRF (Conditional Random Fields) entraîné sur les corpus Universal Dependencies
- **Décodage** : Algorithme de Viterbi implémenté en Python pur
- **Features** : Morphologiques (préfixes, suffixes, casse) + contextuelles (mots voisins)
- **Entraînement** : UD French-GSD, UD French-Sequoia, UD French-Rhapsodie

### Performances

Le modèle est évalué sur le split test des corpus Universal Dependencies.

### Limites

- Modèle sans lexique : se base uniquement sur la forme des mots et le contexte
- Optimisé pour le français contemporain standard
- Les phrases très courtes (1-2 mots) peuvent donner des résultats moins fiables
- Les noms propres ne sont pas distingués des noms communs

---

## API de référence

### `PosTagger(model_path)`

Crée un tagger en chargeant le modèle CRF.

### `tagger.tag(text) → list[tuple[str, str]]`

Tokenise et étiquète un texte brut. Retourne une liste de `(mot, tag)`.

### `tagger.tag_words(words) → list[tuple[str, str]]`

Étiquète une liste de mots déjà tokenisés.

### `tagger.tag_detailed(text) → list[dict]`

Comme `tag()` mais retourne des dicts `{"mot", "tag", "description"}`.

### `tagger.tag_formatted(text) → str`

Retourne un texte formaté lisible avec un mot par ligne.

### `TAGSET → dict[str, str]`

Dictionnaire `{tag: description}` des 18 catégories.

---

## Support

Pour toute question : [contact à définir]

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0. Voir LICENCE.txt et ATTRIBUTION.md.*
