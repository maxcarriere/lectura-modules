# Évaluation — Lectura Syllabeur v1.0

## Protocole

- **Module** : Analyseur syllabique purement algorithmique (modèle de sonorité)
- **Données de test** : Jeu de test intégré — 82 mots français avec phonémisation IPA et syllabation de référence
- **Métriques** : Mots corrects (syllabation exacte) et frontières syllabiques correctes
- **Évaluation IPA** : le test utilise `syllabify_ipa()` directement (pas besoin d'eSpeak-NG)

## Résultats

| Métrique | Correct | Total | Score |
|----------|---------|-------|-------|
| Mots corrects (syllabation exacte) | 82 | 82 | **100%** |
| Frontières syllabiques correctes | 89 | 89 | **100%** |

## Couverture du jeu de test

| Catégorie | Nb mots | Exemples |
|-----------|---------|----------|
| Mots simples (1 syllabe) | 10 | chat, bon, mer, pain |
| Mots 2 syllabes | 10 | maison, bonjour, jardin |
| Mots 3 syllabes | 10 | chocolat, animal, cinéma |
| Mots 4+ syllabes | 7 | université, communication |
| Clusters consonantiques | 10 | train, spectacle, structure |
| Semi-voyelles / diphtongues | 10 | pied, lui, loi, mouette |
| Voyelles nasales | 6 | chanson, parfum, invention |
| Hiatus | 5 | chaos, aéré, poète, naïf |
| Codas complexes | 4 | arbre, monstre, texte |
| Mots courants divers | 10 | musique, école, fromage |

## Points forts

- **Modèle de sonorité** : découpage basé sur l'échelle de sonorité (voyelles > semi-voyelles > liquides > fricatives > occlusives)
- **Clusters consonantiques** : gère correctement les attaques complexes `/tʁ/`, `/pl/`, `/bʁ/`, `/spɛk/`
- **Semi-voyelles** : `/j/`, `/w/`, `/ɥ/` correctement rattachées à la syllabe suivante
- **Voyelles nasales** : les combining marks (`◌̃`) sont correctement groupées avec la voyelle de base
- **Hiatus** : deux voyelles consécutives forment bien deux syllabes distinctes

## Limites connues

- **Dépendance phonémique** : la syllabation opère sur l'IPA, pas sur l'orthographe — la qualité de la phonémisation en amont est déterminante
- **Cas ambigus** : certains mots savants (ex. « extraordinaire ») ont des découpages discutables selon les conventions
- **Pas de contexte prosodique** : le découpage est purement phonologique, sans information d'accent ou de rythme

## Caractéristiques techniques

| Propriété | Valeur |
|-----------|--------|
| Architecture | Modèle de sonorité + clusters |
| Dépendances runtime | Aucune (Python pur) |
| Phonémiseur par défaut | eSpeak-NG (optionnel, pré-requis système) |
| Python minimum | 3.10 |

## Lancer l'évaluation

```bash
python evaluer.py           # résumé
python evaluer.py --verbose  # détail des erreurs
```
