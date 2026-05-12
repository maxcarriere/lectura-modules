# Évaluation — Lectura Tokeniseur v1.0

## Protocole

- **Module** : Normalisateur + tokeniseur purement algorithmique (zéro dépendance)
- **Données de test** : Jeu de test intégré (121 assertions) couvrant normalisation, tokenisation, round-trip et cas spéciaux
- **Métriques** : Taux de réussite par catégorie + score global

## Résultats

| Catégorie | Correct | Total | Score |
|-----------|---------|-------|-------|
| Normalisation | 18 | 18 | **100%** |
| Tokenisation (tokens) | 69 | 69 | **100%** |
| Tokenisation (phrases exactes) | 11 | 11 | **100%** |
| Round-trip (tokens → texte) | 28 | 28 | **100%** |
| Couverture cas spéciaux | 6 | 6 | **100%** |
| **Global** | **121** | **121** | **100%** |

## Détail des tests

### Normalisation (18 cas)

Vérifie les transformations typographiques :
- Espaces multiples → espace unique
- Apostrophes avec espaces → collées
- Ellipses `...` → `…` avec espacement
- Guillemets droits → chevrons français `« »`
- Nombres avec espaces → formatés (séparateur `'`)
- Virgules décimales → points
- Tirets composés vs tirets de dialogue
- Ponctuation forte/faible : espacement correct
- Parenthèses/crochets : suppression espaces internes

### Tokenisation (11 phrases, 69 tokens)

Vérifie la segmentation en types :
- `Mot` : séquences de lettres (y compris accentuées)
- `Nombre` : séquences de chiffres (avec `.` et `'`)
- `Sigle` : 2+ majuscules consécutives (ex. FBI)
- `Separateur` : espace, apostrophe entre lettres, tiret entre lettres
- `Ponctuation` : virgule, point, guillemets, etc.

### Round-trip (28 cas)

Vérifie que `"".join(t.text for t in tokens) == texte_normalisé` pour tout texte.

### Couverture cas spéciaux (6 cas)

Vérifie l'extraction correcte des mots dans des cas variés :
- Mots composés avec tirets (`C'est-à-dire`)
- Abréviations (`Mme`)
- Énumérations avec virgules
- Phrases basiques, exclamations, prénoms composés

## Points forts

- **Zéro dépendance** : fonctionne en Python pur
- **Spans exacts** : chaque token porte sa position `(début, fin)` dans le texte normalisé
- **Normalisation idempotente** : `normalize(normalize(x)) == normalize(x)`
- **Round-trip parfait** : la concaténation des tokens reconstitue toujours le texte normalisé

## Limites connues

- **Sigles avec points** : `O.N.U.` est tokenisé lettre par lettre (pas comme un sigle unique)
- **Nombres formatés** : le seuil de regroupement (6 chiffres) peut surprendre pour les nombres courts
- **Pas de détokenisation sémantique** : le round-trip fonctionne sur le texte normalisé, pas sur le texte original

## Lancer l'évaluation

```bash
python evaluer.py           # résumé
python evaluer.py --verbose  # détail des erreurs
```
