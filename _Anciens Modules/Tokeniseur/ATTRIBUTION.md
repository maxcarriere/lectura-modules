# Attribution

## Lectura Tokeniseur

Copyright (c) 2025 Lectura.

Distribué sous licence [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Algorithmes

### Normalisation typographique

Règles de normalisation développées spécifiquement pour le français :
espaces, apostrophes, guillemets (droits → français), ellipses,
ponctuation faible/forte, nombres avec séparateurs, parenthèses, tirets.

### Tokenisation

Tokeniseur à parcours linéaire avec reconnaissance de 5 types de tokens
(Mot, Ponctuation, Séparateur, Nombre, Sigle) et calcul de spans.
Gestion des spécificités françaises : apostrophes (l', d', qu'),
mots composés (trait d'union), sigles (SNCF, ONU).
