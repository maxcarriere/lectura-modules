# Évaluation — Lectura POS Tagger BiLSTM v1.0

## Protocole

- **Modèle** : BiLSTM (CharCNN 64d + WordEmbed 128d → BiLSTM 2×128h → 18 tags)
- **Format** : ONNX INT8 quantisé (6 Mo)
- **Données de test** : Split test des corpus Universal Dependencies pour le français
- **Métriques** : Accuracy globale, Précision / Rappel / F1 par tag
- **Tags évalués** : 18 catégories (hors PUNCT et SPACE)
- **Dépendances** : onnxruntime, numpy

## Résultats globaux

| Corpus | Phrases | Tokens | Accuracy (BiLSTM) | Accuracy (BiLSTM + lex) |
|--------|---------|--------|--------------------|-------------------------|
| **Fusionné (GSD + Sequoia + Rhapsodie)** | 1 712 | 27 875 | **97.88%** | **97.88%** |

Le mini-lexique n'a pas d'effet mesurable ici : le BiLSTM prédit déjà correctement
les 31 mots-outils couverts par le lexique.

## Résultats par catégorie (test fusionné)

| Tag | Précision | Rappel | F1 | Support |
|-----|-----------|--------|----|---------|
| ART:def | 100.0% | 99.9% | 99.9% | 2 865 |
| ART:ind | 96.5% | 98.1% | 97.3% | 851 |
| PRO:per | 99.7% | 99.6% | 99.6% | 1 497 |
| PRO:rel | 93.9% | 92.6% | 93.2% | 363 |
| PRO:dem | 99.8% | 99.8% | 99.8% | 403 |
| PRO:ind | 95.4% | 97.0% | 96.2% | 235 |
| PRO:int | 81.8% | 81.8% | 81.8% | 22 |
| ADJ:pos | 100.0% | 99.4% | 99.7% | 356 |
| ADJ:dem | 100.0% | 100.0% | 100.0% | 218 |
| ADJ:int | 100.0% | 100.0% | 100.0% | 2 |
| NOM | 98.9% | 97.5% | 98.2% | 7 541 |
| ADJ | 94.9% | 93.1% | 94.0% | 1 741 |
| VER | 94.0% | 97.5% | 95.8% | 2 761 |
| AUX | 96.5% | 98.2% | 97.3% | 1 203 |
| ADV | 97.9% | 96.9% | 97.4% | 1 817 |
| PRE | 99.3% | 99.3% | 99.3% | 4 342 |
| CON | 96.7% | 97.9% | 97.3% | 1 303 |
| INTJ | 92.7% | 97.2% | 94.9% | 355 |

## Points forts

- **Articles définis** : F1 = 99.9% — quasi parfait
- **Pronoms personnels** : F1 = 99.6%
- **Adjectifs possessifs/démonstratifs** : F1 > 99.7%
- **Prépositions** : F1 = 99.3%
- **Noms** : F1 = 98.2%
- **Adverbes** : F1 = 97.4% — meilleur que le CRF (96.4%)
- **Auxiliaires** : F1 = 97.3% — meilleur que le CRF (95.9%)
- **Pronoms interrogatifs** : F1 = 81.8% — nettement meilleur que le CRF (60.0%)

## Confusions principales

| Vrai | Prédit | Occurrences | Commentaire |
|------|--------|-------------|-------------|
| NOM → VER | | 82 | Homographes nom/verbe |
| NOM → ADJ | | 69 | Ambiguïté classique |
| ADJ → VER | | 62 | Participes passés |
| VER → AUX | | 39 | être/avoir |
| ADJ → NOM | | 38 | Substantivation |
| PRO:rel → CON | | 23 | "que" relatif vs conjonctif |

## Comparaison avec le CRF

| Métrique | CRF | BiLSTM |
|----------|-----|--------|
| Accuracy | 97.5% | **97.9%** |
| Taille modèle | 1.8 Mo | 6 Mo |
| Dépendances | Aucune | onnxruntime, numpy |
| PRO:int F1 | 60.0% | **81.8%** |
| ADV F1 | 96.4% | **97.4%** |
| AUX F1 | 95.9% | **97.3%** |
| PRO:rel F1 | 91.4% | **93.2%** |

Le BiLSTM offre un gain de +0.4 point d'accuracy globale par rapport au CRF,
avec des améliorations notables sur les catégories difficiles (pronoms interrogatifs,
adverbes, auxiliaires, pronoms relatifs).

## Caractéristiques techniques

| Propriété | Valeur |
|-----------|--------|
| Architecture | BiLSTM (CharCNN + WordEmbed → BiLSTM 2 couches) |
| Décodage | Argmax sur émissions + mini-lexique optionnel |
| Taille du modèle | 6 Mo (ONNX INT8 quantisé) |
| Vocabulaire | 42 363 mots, 240 caractères |
| Dépendances runtime | onnxruntime, numpy |
| Python minimum | 3.10 |
| Post-traitement | Mini-lexique de 31 corrections |

## Données d'entraînement

| Corpus | Licence | Phrases |
|--------|---------|---------|
| UD French-GSD | CC BY-SA 4.0 | ~16 000 |
| UD French-Sequoia | LGPL-LR | ~3 000 |
| UD French-Rhapsodie | CC BY-SA 4.0 | ~1 300 |
