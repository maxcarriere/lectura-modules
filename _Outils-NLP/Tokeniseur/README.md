# Lectura Tokeniseur

**Normalisateur et tokeniseur pour le français — zéro dépendance**

Un module léger et autonome qui normalise et tokenise du texte français
en tokens typés avec spans. Un seul fichier Python, prêt à l'emploi.

---

## Démarrage rapide

```python
from lectura_tokeniseur import LecturaTokeniseur

tok = LecturaTokeniseur()
result = tok.analyze("L'enfant mange du chocolat.")

for t in result.tokens:
    print(f"{t.text:12s}  {t.type.value:12s}  span={t.span}")
# L            mot           span=(0, 1)
# '            separateur    span=(1, 2)
# enfant       mot           span=(2, 8)
# ...
```

### Prérequis

- Python 3.10+
- Aucune bibliothèque externe requise

### Contenu de l'archive

```
lectura-tokeniseur-v1.0/
├── lectura_tokeniseur.py       ← Fichier principal (copier dans votre projet)
├── demo_cli.py                 ← Démo en ligne de commande
├── exemples/
│   ├── exemple_basique.py      ← Utilisation simple
│   └── exemple_integration.py  ← Intégration dans un pipeline NLP
├── README.md
├── LICENCE.txt
└── ATTRIBUTION.md
```

---

## Utilisation

### Normalisation seule

```python
tok = LecturaTokeniseur()

tok.normalize("L'enfant  mange...du  chocolat")
# → "L'enfant mange … du chocolat"

tok.normalize('"Bonjour" dit-il.')
# → '« Bonjour » dit-il.'

tok.normalize("Il a 1 000 000 euros.")
# → "Il a 1'000'000 euros."

tok.normalize("C'est-à-dire ( oui ) !")
# → "C'est-à-dire (oui) !"
```

### Tokenisation

```python
tokens = tok.tokenize("L'enfant mange-t-il du chocolat ?")
for t in tokens:
    print(t.type.value, t.text, t.span)
```

### Analyse complète

```python
result = tok.analyze("Le SNCF transporte 3 millions de voyageurs.")

print(result.texte_normalise)    # texte après normalisation
print(result.nb_mots)            # nombre de mots
print(result.words())            # ['le', 'sncf', 'transporte', ...]
print(result.format_table())     # affichage tabulaire
```

### Extraction rapide des mots

```python
words = tok.extract_words("L'enfant mange du chocolat.")
# → ['l', 'enfant', 'mange', 'du', 'chocolat']
```

### Démo en ligne de commande

```bash
# Analyse complète
python demo_cli.py "L'enfant mange du chocolat."

# Mots uniquement
python demo_cli.py --words "L'enfant mange du chocolat."

# Normalisation uniquement
python demo_cli.py --normalize "L'enfant  mange...du  chocolat"

# Mode interactif
python demo_cli.py
```

---

## Normalisation

La normalisation applique dans l'ordre :

| Étape | Transformation | Exemple |
|-------|---------------|---------|
| Espaces | Espaces multiples → un seul | `"a  b"` → `"a b"` |
| Apostrophes | Supprimer espaces autour | `"l' enfant"` → `"l'enfant"` |
| Ellipses | `...` → `…` avec espacement | `"mange...du"` → `"mange … du"` |
| Nombres | Groupement par `'` | `"1 000 000"` → `"1'000'000"` |
| Virgule décimale | `,` → `.` entre chiffres | `"3,14"` → `"3.14"` |
| Ponctuation faible | Pas d'espace avant `,;.` | `"mot , mot"` → `"mot, mot"` |
| Ponctuation forte | Espace autour `!?;:` | `"mot!"` → `"mot !"` |
| Guillemets | Droits → français | `'"mot"'` → `'« mot »'` |
| Parenthèses | Pas d'espace interne | `"( mot )"` → `"(mot)"` |
| Tirets | Espace si pas composé | `"- oui"` → `"- oui"` |

---

## Types de tokens

| Type | Classe | Exemples | Attributs |
|------|--------|----------|-----------|
| `mot` | `Mot` | le, enfant, mange | `.ortho` (minuscule) |
| `ponctuation` | `Ponctuation` | `.` `,` `!` `?` `«` `»` | |
| `separateur` | `Separateur` | `'` `-` ` ` | `.sep_type` (apostrophe / hyphen / space) |
| `nombre` | `Nombre` | 42, 3.14 | |
| `sigle` | `Sigle` | SNCF, ONU | `.children` (liste de Mot) |

Tous les tokens ont : `.type`, `.text`, `.span` (position dans le texte normalisé).

---

## Détails techniques

### Algorithme de tokenisation

Parcours linéaire en une passe (O(n)) :
1. Lettres consécutives → `Mot` (ou `Sigle` si 2+ majuscules)
2. Chiffres consécutifs → `Nombre`
3. Apostrophe/tiret entre lettres → `Separateur`
4. Espace → `Separateur(sep_type="space")`
5. Tout le reste → `Ponctuation`

### Spécificités françaises

- **Apostrophes** : `l'enfant` → `Mot("L")` + `Separateur("'")` + `Mot("enfant")`
- **Mots composés** : `peut-être` → `Mot` + `Separateur("-")` + `Mot`
- **Sigles** : `SNCF` → `Sigle` avec `.children` = 4 `Mot`
- **Guillemets** : droits `"..."` normalisés en français `« ... »`

### Limites

- Tokenisation naïve (pas de modèle statistique)
- Les abréviations avec point (`M.`, `etc.`) ne sont pas gérées spécialement
- Les URLs et adresses email sont tokenisées comme du texte normal
- Pas de détection de phrases

---

## API de référence

### `LecturaTokeniseur()`

Crée un normalisateur/tokeniseur.

### `tok.normalize(text) → str`

Normalise un texte brut.

### `tok.tokenize(text, normalize=True) → list[Token]`

Tokenise un texte. Si `normalize=True` (défaut), normalise d'abord.

### `tok.analyze(text) → ResultatTokenisation`

Normalisation + tokenisation avec résultat structuré.

### `tok.extract_words(text) → list[str]`

Raccourci : retourne la liste des formes orthographiques des mots.

### `ResultatTokenisation`

- `.texte_original` : texte brut d'entrée
- `.texte_normalise` : texte après normalisation
- `.tokens` : liste de `Token`
- `.mots` : uniquement les `Mot`
- `.nb_mots`, `.nb_tokens` : compteurs
- `.words()` : liste des formes orthographiques
- `.format_table()` : affichage tabulaire

### Classes de tokens

- `Token(type, text, span)` — base
- `Mot(ortho)` — mot avec forme normalisée
- `Ponctuation` — signe de ponctuation
- `Separateur(sep_type)` — apostrophe, tiret, espace
- `Nombre` — séquence numérique
- `Sigle(children)` — acronyme avec lettres individuelles

### Fonctions bas-niveau

- `normalise(text) → str` : normalisation seule
- `tokenise(text) → list[Token]` : tokenisation seule

---

## Support

Pour toute question : [contact à définir]

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0. Voir LICENCE.txt et ATTRIBUTION.md.*
