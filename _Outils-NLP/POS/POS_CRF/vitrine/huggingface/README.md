---
language:
  - fr
license: cc-by-sa-4.0
tags:
  - pos-tagging
  - french
  - crf
  - token-classification
  - nlp
  - lightweight
datasets:
  - universal_dependencies
metrics:
  - accuracy
  - f1
pipeline_tag: token-classification
model-index:
  - name: lectura-pos-tagger-crf-french
    results:
      - task:
          type: token-classification
          name: POS Tagging
        dataset:
          name: UD French (GSD + Sequoia + Rhapsodie)
          type: universal_dependencies
          split: test
        metrics:
          - type: accuracy
            value: 0.975
            name: Accuracy
---

# Lectura POS Tagger — CRF French

**Étiqueteur grammatical CRF pour le français — 97.5% d'accuracy, 1.8 Mo, zéro dépendance.**

## Description

Modèle CRF (Conditional Random Fields) léger pour le POS-tagging du français contemporain. Décodage Viterbi implémenté en Python pur : aucune dépendance externe à l'exécution.

- **18 catégories grammaticales** avec distinctions fines (articles définis/indéfinis, pronoms personnels/relatifs/démonstratifs…)
- **1.8 Mo** — deux ordres de grandeur plus léger que spaCy ou Stanza
- **Aucune dépendance** — un seul fichier Python à copier dans votre projet
- **97.5% d'accuracy** sur le test set Universal Dependencies

## Utilisation

```python
from lectura_pos import PosTagger

tagger = PosTagger("modele/pos_model_crf.json",
                    lexicon_path="modele/mini_lexique.json")

tagger.tag("Le chat mange la souris")
# [("Le", "ART:def"), ("chat", "NOM"), ("mange", "VER"),
#  ("la", "ART:def"), ("souris", "NOM")]
```

## Obtenir le modèle

Ce modèle est disponible sous licence **CC BY-SA 4.0**.

**[Télécharger l'archive complète sur Gumroad — 29 €](https://TODO_LIEN_GUMROAD)**

L'archive inclut : modèle entraîné, code d'inférence, démos (CLI + web), exemples, données et script d'entraînement, évaluation complète.

*Vous payez pour le packaging, la documentation et le support. La licence CC BY-SA 4.0 autorise l'usage commercial et la redistribution avec attribution.*

## Performances

Évalué sur le split test des corpus Universal Dependencies (27 875 tokens, 1 712 phrases) :

| Corpus | Tokens | Accuracy |
|--------|--------|----------|
| UD French-GSD | 8 831 | **97.7%** |
| UD French-Sequoia | 8 960 | **98.3%** |
| UD French-Rhapsodie | 10 084 | **96.6%** |
| **Fusionné** | **27 875** | **97.5%** |

### Par catégorie

| Tag | F1 | Support | Description |
|-----|----|---------|-------------|
| ART:def | 99.8% | 2 865 | Article défini |
| ADJ:pos | 99.9% | 356 | Adjectif possessif |
| ADJ:dem | 99.8% | 218 | Adjectif démonstratif |
| PRO:dem | 99.6% | 403 | Pronom démonstratif |
| PRO:per | 99.2% | 1 497 | Pronom personnel |
| PRE | 99.2% | 4 342 | Préposition |
| NOM | 98.2% | 7 541 | Nom commun |
| ART:ind | 97.3% | 851 | Article indéfini |
| CON | 96.6% | 1 303 | Conjonction |
| ADV | 96.4% | 1 817 | Adverbe |
| AUX | 95.9% | 1 203 | Auxiliaire |
| VER | 95.4% | 2 761 | Verbe |
| INTJ | 95.5% | 355 | Interjection |
| PRO:ind | 93.3% | 235 | Pronom indéfini |
| ADJ | 92.9% | 1 741 | Adjectif qualificatif |
| PRO:rel | 91.4% | 363 | Pronom relatif |
| PRO:int | 60.0% | 22 | Pronom interrogatif |
| ADJ:int | 100.0% | 2 | Adjectif interrogatif |

## Tagset

18 catégories grammaticales, plus fines que le tagset UPOS standard :

| Tag | Description | Exemples |
|-----|-------------|----------|
| `ART:def` | Article défini | le, la, les, l' |
| `ART:ind` | Article indéfini | un, une, des |
| `PRO:per` | Pronom personnel | je, tu, il, nous, vous |
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

## Architecture

| Propriété | Valeur |
|-----------|--------|
| Type | CRF (Conditional Random Fields) |
| Décodage | Viterbi (Python pur) |
| Features | Morphologiques (préfixes, suffixes, casse) + contextuelles (mots voisins) |
| Post-traitement | Mini-lexique de 31 corrections |
| Taille | 1.8 Mo (JSON) |
| Dépendances runtime | Aucune |
| Python | 3.10+ |

## Données d'entraînement

| Corpus | Licence |
|--------|---------|
| [UD French-GSD](https://github.com/UniversalDependencies/UD_French-GSD) | CC BY-SA 4.0 |
| [UD French-Sequoia](https://github.com/UniversalDependencies/UD_French-Sequoia) | LGPL-LR |
| [UD French-Rhapsodie](https://github.com/UniversalDependencies/UD_French-Rhapsodie) | CC BY-SA 4.0 |

## Limites

- Modèle sans lexique complet : repose sur la forme des mots et le contexte
- Optimisé pour le français contemporain standard
- Les pronoms interrogatifs (PRO:int) sont rares et mal couverts (F1 = 60%)
- L'oral transcrit est plus difficile (96.6%) que l'écrit formel (98.3%)

## Citation

```bibtex
@misc{lectura-pos-tagger-2025,
  title={Lectura POS Tagger: A Lightweight CRF French POS Tagger},
  author={Lectura},
  year={2025},
  url={https://huggingface.co/lectura-nlp/pos-tagger-crf-french}
}
```

## Contact

[TODO : email ou lien de contact]
