# Évaluation — Lectura POS Tagger v1.0

## Protocole

- **Modèle** : CRF (Conditional Random Fields) + mini-lexique de correction
- **Données de test** : Split test des corpus Universal Dependencies pour le français
- **Métriques** : Accuracy globale, Précision / Rappel / F1 par tag
- **Tags évalués** : 18 catégories (hors PUNCT et SPACE)

## Résultats globaux

| Corpus | Phrases | Tokens | Accuracy |
|--------|---------|--------|----------|
| **UD French-GSD** (textes web/wiki) | 416 | 8 831 | **97.7%** |
| **UD French-Sequoia** (textes européens) | 456 | 8 960 | **98.3%** |
| **UD French-Rhapsodie** (oral transcrit) | 840 | 10 084 | **96.6%** |
| **Fusionné (GSD + Sequoia + Rhapsodie)** | 1 712 | 27 875 | **97.5%** |

## Résultats par catégorie (test fusionné)

| Tag | Précision | Rappel | F1 | Support |
|-----|-----------|--------|----|---------|
| ART:def | 99.7% | 99.9% | 99.8% | 2 865 |
| ART:ind | 97.5% | 97.1% | 97.3% | 851 |
| PRO:per | 99.7% | 98.8% | 99.2% | 1 497 |
| PRO:rel | 92.2% | 90.6% | 91.4% | 363 |
| PRO:dem | 99.5% | 99.8% | 99.6% | 403 |
| PRO:ind | 91.4% | 95.3% | 93.3% | 235 |
| PRO:int | 66.7% | 54.5% | 60.0% | 22 |
| ADJ:pos | 100.0% | 99.7% | 99.9% | 356 |
| ADJ:dem | 100.0% | 99.5% | 99.8% | 218 |
| ADJ:int | 100.0% | 100.0% | 100.0% | 2 |
| NOM | 98.2% | 98.2% | 98.2% | 7 541 |
| ADJ | 93.8% | 92.0% | 92.9% | 1 741 |
| VER | 96.0% | 94.7% | 95.4% | 2 761 |
| AUX | 93.9% | 97.9% | 95.9% | 1 203 |
| ADV | 96.8% | 96.0% | 96.4% | 1 817 |
| PRE | 99.0% | 99.4% | 99.2% | 4 342 |
| CON | 96.1% | 97.2% | 96.6% | 1 303 |
| INTJ | 93.3% | 97.7% | 95.5% | 355 |

## Points forts

- **Articles et déterminants** : F1 > 97% — excellente distinction ART:def / ART:ind
- **Pronoms personnels** : F1 = 99.2% — très fiable
- **Adjectifs possessifs/démonstratifs** : F1 > 99.5%
- **Prépositions** : F1 = 99.2%
- **Noms** : F1 = 98.2% — solide sur la catégorie la plus fréquente
- **Taille du modèle** : 1.8 Mo — léger et rapide

## Confusions principales

| Vrai | Prédit | Occurrences | Commentaire |
|------|--------|-------------|-------------|
| VER → AUX | | 75 | Confusion être/avoir (VER vs AUX) |
| ADJ ↔ NOM | | 128 | Ambiguïté classique français |
| ADJ → VER | | 46 | Participes passés (adjectif ou verbe) |
| PRO:rel ↔ CON | | 51 | "que" relatif vs conjonctif |

## Limites connues

- **PRO:int** (pronoms interrogatifs) : F1 = 60% — catégorie rare (22 occurrences), souvent confondue avec PRO:rel
- **Oral transcrit** : 96.6% sur Rhapsodie vs 98.3% sur Sequoia — le registre oral est plus difficile
- **Modèle sans lexique** : se base uniquement sur la forme et le contexte, pas de dictionnaire de formes
- **Homographes** : "est" (AUX/NOM), "a" (AUX/NOM), "les" (ART/PRO) dépendent du contexte

## Données d'entraînement

| Corpus | Licence | Phrases |
|--------|---------|---------|
| UD French-GSD | CC BY-SA 4.0 | ~16 000 |
| UD French-Sequoia | LGPL-LR | ~3 000 |
| UD French-Rhapsodie | CC BY-SA 4.0 | ~1 300 |

## Caractéristiques techniques

| Propriété | Valeur |
|-----------|--------|
| Architecture | CRF (Conditional Random Fields) |
| Décodage | Viterbi (Python pur) |
| Taille du modèle | 1.8 Mo (JSON) |
| Features | Morphologiques + contextuelles (20+) |
| Dépendances runtime | Aucune |
| Python minimum | 3.10 |
| Post-traitement | Mini-lexique de 31 corrections |
