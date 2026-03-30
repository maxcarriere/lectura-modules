# Évaluation — Lectura Morpho Tagger CRF v1.0

## Protocole

- **Modèle** : CRF (features : word, suf2-5, pre2-3, casse, BOS/EOS, contexte w-1/w+1)
- **Format** : JSON pur Python (zéro dépendance)
- **Données de test** : Split test des corpus Universal Dependencies pour le français (1 712 phrases, 27 875 tokens hors PUNCT/SPACE)
- **Métriques** : Accuracy tag complet, Accuracy POS, P/R/F1 par tag, Accuracy par trait, Accuracy lemmes
- **Tags évalués** : 144 catégories composites (hors PUNCT et SPACE)
- **Dépendances** : aucune (stdlib Python uniquement)

## Résultats globaux

| Métrique | Valeur |
|----------|--------|
| **Accuracy tag complet** | 88,47 % (24 660 / 27 875) |
| **Accuracy POS seul** | 97,47 % (27 169 / 27 875) |
| **Accuracy lemmes** | 81,87 % (22 820 / 27 875) |

## Accuracy par trait

| Trait | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Genre | 5 513 | 5 775 | 95,46 % |
| Nombre | 14 206 | 14 649 | 96,98 % |
| Temps | 2 317 | 2 358 | 98,26 % |
| Mode | 3 791 | 3 872 | 97,91 % |
| Personne | 4 401 | 4 474 | 98,37 % |

## Résultats par tag (top 20)

| Tag | Prec | Rappel | F1 | Support |
|-----|------|--------|----|---------|
| PRE | 99,0 % | 99,4 % | 99,2 % | 4 342 |
| NOM | 83,2 % | 56,0 % | 66,9 % | 3 240 |
| NOM\|Sing | 69,9 % | 93,4 % | 80,0 % | 2 814 |
| ADV | 96,9 % | 96,3 % | 96,6 % | 1 817 |
| CON | 96,3 % | 96,9 % | 96,6 % | 1 303 |
| NOM\|Plur | 76,8 % | 89,6 % | 82,7 % | 1 158 |
| ART:def\|Masc\|Sing | 99,7 % | 99,8 % | 99,7 % | 949 |
| ART:def\|Fem\|Sing | 99,7 % | 99,9 % | 99,8 % | 701 |
| ART:def\|Plur | 99,9 % | 99,9 % | 99,9 % | 686 |
| AUX\|Ind\|Pres\|3\|Sing | 93,8 % | 98,9 % | 96,2 % | 610 |
| VER\|Ind\|Pres\|3\|Sing | 94,1 % | 90,8 % | 92,4 % | 610 |
| VER\|Inf | 98,0 % | 99,1 % | 98,5 % | 543 |
| ART:def\|Sing | 99,6 % | 99,8 % | 99,7 % | 529 |
| PRO:per\|Sing\|1 | 100,0 % | 100,0 % | 100,0 % | 380 |
| ADJ\|Masc\|Sing | 78,5 % | 89,2 % | 83,5 % | 360 |
| INTJ | 93,3 % | 97,7 % | 95,5 % | 355 |
| PRO:rel | 91,0 % | 91,7 % | 91,3 % | 351 |
| PRO:dem\|Masc\|Sing\|3 | 93,9 % | 91,3 % | 92,6 % | 335 |
| PRO:per\|Masc\|Sing\|3 | 98,8 % | 98,2 % | 98,5 % | 332 |
| ART:ind\|Masc\|Sing | 98,8 % | 98,8 % | 98,8 % | 322 |

## Points forts

- **Temps verbal** : 98,26 % — très fiable
- **Personne verbale** : 98,37 % — le trait le mieux prédit
- **Mode verbal** : 97,91 % — excellent
- **Articles définis** : F1 > 99 % pour toutes les formes
- **Pronoms personnels** : F1 > 98 % en général
- **POS seul** : 97,47 % — meilleur que le BiLSTM (96,13 %)
- **Tag complet** : 88,47 % — meilleur que le BiLSTM (87,71 %)

## Confusions principales

| Vrai | Prédit | Nb |
|------|--------|----|
| NOM | NOM\|Sing | 1 038 |
| NOM | NOM\|Plur | 268 |
| ADJ | ADJ\|Sing | 130 |
| NOM\|Sing | NOM | 121 |
| VER\|Part | VER\|Part\|Masc\|Sing | 96 |
| NOM\|Plur | NOM | 94 |
| NOM\|Masc\|Sing | NOM | 71 |
| ADJ\|Fem | ADJ\|Fem\|Sing | 58 |
| VER\|Part | VER\|Part\|Fem\|Sing | 39 |
| VER\|Ind\|Pres\|3\|Sing | AUX\|Ind\|Pres\|3\|Sing | 38 |

La majorité des confusions sont des erreurs **intra-catégorie** (NOM vs NOM\|Sing, ADJ vs ADJ\|Sing) : le modèle hésite sur le niveau de détail morphologique, pas sur la POS elle-même. Le CRF surpasse le BiLSTM sur cette tâche grâce aux features de suffixe étendues (suf4/suf5).

## Comparaison avec MORPHO_BiLSTM

| Métrique | CRF | BiLSTM |
|----------|-----|--------|
| Accuracy tag complet | **88,47 %** | 87,71 % |
| Accuracy POS seul | **97,47 %** | 96,13 % |
| Accuracy lemmes | 81,87 % | 81,54 % |
| Genre | 95,46 % | **96,66 %** |
| Nombre | 96,98 % | **97,15 %** |
| Temps | **98,26 %** | 98,05 % |
| Mode | **97,91 %** | 97,60 % |
| Personne | 98,37 % | **98,88 %** |
| Dépendances | aucune | onnxruntime, numpy |
| Taille modèle | 2,9 Mo | 9,5 Mo |

## Caractéristiques techniques

| Propriété | Valeur |
|-----------|--------|
| Architecture | CRF (Viterbi pur Python) |
| Features | word, suf2-5, pre2-3, casse, BOS/EOS, w-1/w+1 |
| Régularisation | L1=0.3, L2=0.1 |
| Taille du modèle | 2,9 Mo (JSON) |
| State features | 72 782 poids |
| Tags | 144 catégories composites |
| Dépendances runtime | aucune |
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
