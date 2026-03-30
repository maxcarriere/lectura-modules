# Lectura POS Tagger

**Étiqueteur grammatical CRF pour le français — 97.5% d'accuracy, 1.8 Mo, zéro dépendance.**

Un modèle CRF (Conditional Random Fields) léger et autonome pour l'analyse morphosyntaxique du français. Un seul fichier Python, aucune bibliothèque externe requise.

---

## Exemple

```python
from lectura_pos import PosTagger

tagger = PosTagger("modele/pos_model_crf.json",
                    lexicon_path="modele/mini_lexique.json")

tagger.tag("Le chat mange la souris")
# [("Le", "ART:def"), ("chat", "NOM"), ("mange", "VER"),
#  ("la", "ART:def"), ("souris", "NOM")]

tagger.tag("Je suis allé au marché acheter des pommes")
# [("Je", "PRO:per"), ("suis", "AUX"), ("allé", "VER"),
#  ("au", "PRE"), ("marché", "NOM"), ("acheter", "VER"),
#  ("des", "ART:ind"), ("pommes", "NOM")]
```

## Performances

Évalué sur le split test des corpus Universal Dependencies (27 875 tokens) :

| Corpus | Accuracy |
|--------|----------|
| UD French-GSD (textes web/wiki) | **97.7%** |
| UD French-Sequoia (textes institutionnels) | **98.3%** |
| UD French-Rhapsodie (oral transcrit) | **96.6%** |
| **Fusionné** | **97.5%** |

### Par catégorie (top 10)

| Tag | F1 | Description |
|-----|----|-------------|
| ART:def | 99.8% | Article défini (le, la, les) |
| ADJ:pos | 99.9% | Adjectif possessif (mon, ton, son…) |
| ADJ:dem | 99.8% | Adjectif démonstratif (ce, cette) |
| PRO:dem | 99.6% | Pronom démonstratif (ce, ceci, cela) |
| PRO:per | 99.2% | Pronom personnel (je, tu, il…) |
| PRE | 99.2% | Préposition (à, de, en, par…) |
| NOM | 98.2% | Nom commun |
| ART:ind | 97.3% | Article indéfini (un, une, des) |
| CON | 96.6% | Conjonction (et, ou, mais…) |
| ADV | 96.4% | Adverbe |

[Voir l'évaluation complète](EVALUATION.md)

## Caractéristiques

| Propriété | Valeur |
|-----------|--------|
| Architecture | CRF + Viterbi (Python pur) |
| Taille du modèle | 1.8 Mo |
| Dépendances runtime | **Aucune** |
| Python minimum | 3.10 |
| Tagset | 18 catégories grammaticales |
| Post-traitement | Mini-lexique de corrections (31 entrées) |
| Entraînement | UD French-GSD + Sequoia + Rhapsodie |

## Tagset (18 catégories)

```
ART:def   Article défini          PRO:per   Pronom personnel
ART:ind   Article indéfini        PRO:rel   Pronom relatif
ADJ:pos   Adjectif possessif      PRO:dem   Pronom démonstratif
ADJ:dem   Adjectif démonstratif   PRO:ind   Pronom indéfini
ADJ:int   Adjectif interrogatif   PRO:int   Pronom interrogatif
NOM       Nom commun              VER       Verbe
ADJ       Adjectif qualificatif   AUX       Auxiliaire
ADV       Adverbe                 PRE       Préposition
CON       Conjonction             INTJ      Interjection
```

## Pourquoi ce modèle ?

- **Léger** : 1.8 Mo vs des centaines de Mo pour spaCy ou Stanza
- **Autonome** : un seul fichier Python, aucune dépendance (`pip install` rien)
- **Tagset fin** : 18 catégories avec distinctions articles/pronoms utiles pour le TTS, la phonétisation, l'analyse syntaxique
- **Rapide** : inférence en Python pur, pas besoin de GPU
- **Documenté** : évaluation complète, exemples, interface de test incluse

## Contenu de l'archive

```
lectura-pos-tagger-v1.0/
├── lectura_pos.py              Fichier principal (copier dans votre projet)
├── demo_cli.py                 Démo en ligne de commande
├── demo_web.py                 Interface web (Gradio)
├── evaluer.py                  Script d'évaluation reproductible
├── modele/
│   ├── pos_model_crf.json      Modèle CRF entraîné (1.8 Mo)
│   └── mini_lexique.json       Corrections post-CRF (31 entrées)
├── exemples/
│   ├── exemple_basique.py      Utilisation simple
│   └── exemple_integration.py  Batch, filtrage, export JSON/CSV
├── EVALUATION.md               Résultats détaillés
├── README.md                   Mode d'emploi
└── LICENCE.txt                 Licence commerciale
```

## Obtenir le modèle

**[Acheter sur Gumroad — 19 €](https://TODO_LIEN_GUMROAD)**

Licence commerciale incluse : intégrez le modèle dans vos projets sans restriction.

---

## Licence

Ce projet est distribué sous [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Vous pouvez l'utiliser, le modifier et le redistribuer (y compris à des fins commerciales), à condition de créditer Lectura et de partager vos modifications sous la même licence.

## Contact

[TODO : email ou lien de contact]

---

*Développé par Lectura.*
