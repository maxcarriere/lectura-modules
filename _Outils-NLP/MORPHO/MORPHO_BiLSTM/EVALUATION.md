# Évaluation — Lectura Morpho Tagger BiLSTM v1.0

## Protocole

- **Modèle** : BiLSTM (CharCNN 96d + WordEmbed 192d → BiLSTM 2×192h → ~200 tags)
- **Format** : ONNX INT8 quantisé (~10 Mo)
- **Données de test** : Split test des corpus Universal Dependencies pour le français (1 712 phrases, 27 875 tokens hors PUNCT/SPACE)
- **Métriques** : Accuracy tag complet, Accuracy POS, P/R/F1 par tag, Accuracy par trait, Accuracy lemmes
- **Tags évalués** : 144 catégories composites (hors PUNCT et SPACE)
- **Dépendances** : onnxruntime, numpy

## Résultats globaux

| Métrique | Valeur |
|----------|--------|
| **Accuracy tag complet** | 87,71 % (24 449 / 27 875) |
| **Accuracy POS seul** | 96,13 % (26 796 / 27 875) |
| **Accuracy lemmes** | 81,54 % (22 729 / 27 875) |

## Accuracy par trait

| Trait | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Genre | 5 582 | 5 775 | 96,66 % |
| Nombre | 14 231 | 14 649 | 97,15 % |
| Temps | 2 312 | 2 358 | 98,05 % |
| Mode | 3 779 | 3 872 | 97,60 % |
| Personne | 4 424 | 4 474 | 98,88 % |

## Résultats par tag (top 20)

| Tag | Prec | Rappel | F1 | Support |
|-----|------|--------|----|---------|
| PRE | 99,4 % | 97,1 % | 98,2 % | 4 342 |
| NOM | 86,1 % | 52,5 % | 65,2 % | 3 240 |
| NOM\|Sing | 72,6 % | 87,2 % | 79,2 % | 2 814 |
| ADV | 96,5 % | 94,5 % | 95,5 % | 1 817 |
| CON | 95,9 % | 92,4 % | 94,1 % | 1 303 |
| NOM\|Plur | 77,4 % | 88,8 % | 82,7 % | 1 158 |
| ART:def\|Masc\|Sing | 99,7 % | 99,8 % | 99,7 % | 949 |
| ART:def\|Fem\|Sing | 99,9 % | 100,0 % | 99,9 % | 701 |
| ART:def\|Plur | 99,9 % | 99,9 % | 99,9 % | 686 |
| AUX\|Ind\|Pres\|3\|Sing | 95,3 % | 99,2 % | 97,2 % | 610 |
| VER\|Ind\|Pres\|3\|Sing | 93,2 % | 90,0 % | 91,6 % | 610 |
| VER\|Inf | 95,0 % | 98,2 % | 96,6 % | 543 |
| ART:def\|Sing | 99,4 % | 99,6 % | 99,5 % | 529 |
| PRO:per\|Sing\|1 | 99,0 % | 100,0 % | 99,5 % | 380 |
| ADJ\|Masc\|Sing | 75,1 % | 88,6 % | 81,3 % | 360 |
| INTJ | 91,8 % | 98,3 % | 95,0 % | 355 |
| PRO:rel | 80,4 % | 94,9 % | 87,1 % | 351 |
| PRO:dem\|Masc\|Sing\|3 | 95,3 % | 84,8 % | 89,7 % | 335 |
| PRO:per\|Masc\|Sing\|3 | 98,8 % | 98,8 % | 98,8 % | 332 |
| ART:ind\|Masc\|Sing | 98,2 % | 100,0 % | 99,1 % | 322 |

## Points forts

- **Personne verbale** : 98,88 % — le trait le mieux prédit
- **Temps verbal** : 98,05 % — très fiable
- **Mode verbal** : 97,60 % — excellent
- **Articles définis** : F1 > 99 % pour toutes les formes (singulier, pluriel, masculin, féminin)
- **Pronoms personnels** : F1 > 97 % en général
- **POS seul** : 96,13 % — très bon niveau de catégorisation grammaticale

## Confusions principales

| Vrai | Prédit | Nb |
|------|--------|----|
| NOM | NOM\|Sing | 878 |
| NOM | NOM\|Plur | 273 |
| NOM\|Sing | NOM | 104 |
| ADJ | ADJ\|Sing | 97 |
| NOM | NOM\|Masc\|Sing | 83 |
| CON | PRO:rel | 80 |
| NOM\|Plur | NOM | 80 |
| VER\|Part | VER\|Part\|Masc\|Sing | 76 |
| NOM | NOM\|Fem\|Sing | 61 |
| NOM\|Sing | NOM\|Masc\|Sing | 53 |

La majorité des confusions sont des erreurs **intra-catégorie** (NOM vs NOM\|Sing, ADJ vs ADJ\|Sing) : le modèle hésite sur le niveau de détail morphologique, pas sur la POS elle-même. La confusion CON / PRO:rel reflète l'ambiguïté classique de "que/qui" en français.

## Caractéristiques techniques

| Propriété | Valeur |
|-----------|--------|
| Architecture | BiLSTM (CharCNN + WordEmbed → BiLSTM 2 couches) |
| Paramètres | 9 834 288 |
| Décodage | Argmax sur émissions |
| Taille du modèle | 9,5 Mo (ONNX INT8 quantisé) |
| Dépendances runtime | onnxruntime, numpy |
| Python minimum | 3.10 |
| Lemmatisation | Règles + irréguliers (+ GLAFF optionnel) |

## Données d'entraînement

| Corpus | Licence | Phrases |
|--------|---------|---------|
| UD French-GSD | CC BY-SA 4.0 | ~16 000 |
| UD French-Sequoia | LGPL-LR | ~3 000 |
| UD French-Rhapsodie | CC BY-SA 4.0 | ~1 300 |
| **Total entraînement** | | **17 968 phrases** |
| **Total dev** | | **2 969 phrases** |
| **Total test** | | **1 712 phrases** |
