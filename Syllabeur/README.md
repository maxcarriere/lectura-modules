# Lectura Syllabeur

**Analyseur syllabique complet du français**

Module autonome, zéro dépendance Python. Découpe les mots en syllabes,
identifie les groupes de lecture, gère les élisions, liaisons et enchaînements.

## Installation

```bash
pip install lectura-syllabeur
```

## Utilisation

```python
from lectura_syllabeur import LecturaSyllabeur

syllabeur = LecturaSyllabeur()
resultat = syllabeur.syllabifier("Les enfants jouent dans la cour.")

for mot in resultat.mots:
    syllabes = "-".join(s.texte for s in mot.syllabes)
    print(f"{mot.forme:15s}  {syllabes}")
```

```
Les               Les
enfants           en-fants
jouent            jouent
dans              dans
la                la
cour              cour
```

## Fonctionnalités

- **Syllabation** : découpage en syllabes selon les règles phonologiques du français
- **Groupes de lecture** : regroupement des syllabes pour la lecture assistée
- **Phénomènes de liaison** : élisions, liaisons obligatoires/facultatives, enchaînements
- **Phonémiseur pluggable** : backend eSpeak-NG par défaut, compatible avec Lectura G2P
- **Formules** : gestion des spans de formules (nombres lus, dates, etc.)

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt)
Licence commerciale disponible — voir [LICENCE-COMMERCIALE.md](LICENCE-COMMERCIALE.md)
