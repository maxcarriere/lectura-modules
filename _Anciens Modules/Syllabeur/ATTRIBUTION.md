# Attribution

## Lectura Syllabeur

Copyright (c) 2025 Lectura.

Distribue sous licence [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Algorithmes

### Syllabation par sonorité

L'algorithme de syllabation est basé sur le modèle géométrique à sonorité,
avec frontières convergentes et absorption alternée unitaire.

### Alignement graphème-phonème

L'algorithme d'alignement utilise un parcours DFS avec segmentation
phonétique et gestion des lettres muettes, développé spécifiquement
pour les particularités orthographiques du français.

## Données

### Table phonème → graphèmes

La table de correspondances phonème-graphèmes est dérivée d'analyses
du corpus GLAFF :

- **GLAFF** (Grand Lexique Analysé du Français Fléchionnel)
  - **Licence** : CC BY-SA 3.0
  - **Source** : http://redac.univ-tlse2.fr/lexiques/glaff.html
  - **Citation** :
    > Hathout, N., & Sajous, F. (2016).
    > GLAFF, un Gros Lexique À tout Faire du Français.
    > *Traitement Automatique des Langues*, 57(2), 11-34.

## Logiciels tiers (optionnels)

- **eSpeak-NG** (GPL v3) — Phonémiseur par défaut (non redistribué)
  - Source : https://github.com/espeak-ng/espeak-ng
