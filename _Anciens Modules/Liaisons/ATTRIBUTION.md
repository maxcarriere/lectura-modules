# Attribution

## Lectura Liaisons

Copyright (c) 2025 Lectura.

Distribué sous licence [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Algorithmes

### Classification des liaisons grammaticales

Moteur à règles basé sur les catégories morphosyntaxiques (POS tags)
et les contraintes phonologiques du français. Implémente les règles
de liaison obligatoire, facultative et interdite selon la grammaire
normative du français standard.

### Enchaînements phonétiques

Détection des enchaînements consonantiques (consonne finale prononcée
+ voyelle initiale) avec cas spéciaux lexicaux (neuf → /v/).

### Élisions

Détection et fusion des élisions (Mot + apostrophe + Mot → groupe phonétique).

### Mots composés

Détection et fusion des mots composés (Mot + trait d'union + Mot → groupe phonétique).

### Pipeline de jonctions

Application séquentielle des 4 types de jonctions (composés, élisions,
liaisons, enchaînements) avec itération jusqu'à stabilisation pour les liaisons.

### Liste h aspiré

Liste de 863 formes fléchies à h aspiré, compilée à partir de
sources lexicographiques françaises de référence.

## Références linguistiques

- Delattre, P. (1966). *Studies in French and Comparative Phonetics*.
- Encrevé, P. (1988). *La liaison avec et sans enchaînement*.
- Durand, J. & Lyche, C. (2008). French liaison in the light of corpus data.
