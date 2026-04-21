# Lectura Tokeniseur

**Normalisateur et tokeniseur complet pour le français**

Module autonome, zéro dépendance externe. Détecte et classifie les formules
(nombres, sigles, dates, téléphones, numéros, ordinaux, fractions, notations
scientifiques, expressions mathématiques).

## Installation

```bash
pip install lectura-tokeniseur
```

## Utilisation

```python
from lectura_tokeniseur import tokenise

resultat = tokenise("Le 25 décembre 2024, il faisait -3°C à Paris.")

for phrase in resultat.phrases:
    for token in phrase:
        print(f"{token.texte:20s}  {token.type.name}")
```

```
Le                    MOT
25 décembre 2024      FORMULE
,                     PONCTUATION
il                    MOT
faisait               MOT
-3°C                  FORMULE
à                     MOT
Paris                 MOT
.                     PONCTUATION
```

## Fonctionnalités

- **Normalisation** : typographie, espaces, Unicode
- **Tokenisation** : mots, ponctuation, séparateurs
- **Détection de formules** : nombres (entiers, décimaux, négatifs), dates,
  heures, téléphones, sigles, ordinaux, fractions, pourcentages, monnaies,
  unités de mesure, expressions mathématiques, chiffres romains
- **API simple** : `tokenise(texte)` renvoie un objet structuré

## Licence

Ce module est distribue sous licence **AGPL-3.0** (non commerciale) — voir [LICENCE.txt](LICENCE.txt).

Pour un usage commercial, contacter [contact@lec-tu-ra.com](mailto:contact@lec-tu-ra.com).
