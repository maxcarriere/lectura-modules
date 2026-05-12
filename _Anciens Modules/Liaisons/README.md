# Lectura Liaisons

**Moteur de liaisons et jonctions pour le français — classification et fusion phonétique**

Un module léger et autonome qui gère les 4 types de jonctions
entre mots français :
- **Liaisons grammaticales** (les‿enfants → /lezɑ̃fɑ̃/)
- **Enchaînements phonétiques** (avec‿elle → /avɛkɛl/)
- **Élisions** (l'enfant → /lɑ̃fɑ̃/)
- **Mots composés** (peut-être → /pøɛtʁ/)

Zéro dépendance — un seul fichier Python avec liste h aspiré embarquée.

---

## Démarrage rapide

### API par paires (deux mots)

```python
from lectura_liaisons import LecturaLiaisons, MotInfo

lia = LecturaLiaisons()

# Classifier une liaison
decision = lia.classify(
    MotInfo("les", "le", ["ART:def"]),
    MotInfo("enfants", "ɑ̃fɑ̃", ["NOM"]),
)
print(decision.kind)             # "grammaticale"
print(decision.typ)              # "obligatoire"
print(decision.latent_phoneme)   # "z"

# Fusionner la phonétique
phone = lia.merge("le", "ɑ̃fɑ̃", decision)
print(phone)                     # "lezɑ̃fɑ̃"
```

### Pipeline complet (phrase entière)

```python
from lectura_liaisons import (
    LecturaLiaisons, TokenMot, TokenSep, JonctionOptions,
)

lia = LecturaLiaisons()

tokens = [
    TokenMot("L'", "l", ["ART:def"], (0, 2)),
    TokenSep("'", "apostrophe", (1, 2)),
    TokenMot("enfant", "ɑ̃fɑ̃", ["NOM"], (2, 8)),
    TokenSep(" ", "space", (8, 9)),
    TokenMot("est", "ɛ", ["AUX"], (9, 12)),
    TokenSep(" ", "space", (12, 13)),
    TokenMot("peut", "pø", ["VER"], (13, 17)),
    TokenSep("-", "hyphen", (17, 18)),
    TokenMot("être", "ɛtʁ", ["VER"], (18, 22)),
]

groups = lia.apply_jonctions(tokens)
# → [GroupeJonction("l'enfant", elision), GroupeJonction("est", simple),
#    GroupeJonction("peut-être", compose)]
```

### Prérequis

- Python 3.10+
- Aucune bibliothèque externe requise

### Contenu de l'archive

```
lectura-liaisons-v1.0/
├── lectura_liaisons.py         ← Fichier principal (copier dans votre projet)
├── demo_cli.py                 ← Démo en ligne de commande
├── donnees/
│   └── h_aspire.txt            ← Liste h aspiré (optionnel, embarqué dans le .py)
├── exemples/
│   ├── exemple_basique.py
│   └── exemple_integration.py
├── README.md
├── LICENCE.txt
└── ATTRIBUTION.md
```

---

## Utilisation

### MotInfo — Décrire un mot (API paires)

```python
from lectura_liaisons import MotInfo

# Forme orthographique, phonétique IPA, POS tags possibles
w = MotInfo(ortho="les", phone="le", pos=["ART:def"])
```

### Classifier une liaison

```python
decision = lia.classify(w1, w2)

# decision.kind : "grammaticale" | "enchainement" | "none"
# decision.typ  : "obligatoire" | "facultative" | "interdite" | "none"
```

### Fusionner la phonétique

```python
merged = lia.merge(w1.phone, w2.phone, decision)
```

### Raccourci : classify + merge

```python
decision, merged = lia.analyze_pair(w1, w2)
```

### TokenMot, TokenSep — Décrire des tokens (API pipeline)

```python
from lectura_liaisons import TokenMot, TokenSep, TokenPonct

mot = TokenMot(ortho="les", phone="le", pos=["ART:def"], span=(0, 3))
sep = TokenSep(text=" ", sep_type="space", span=(3, 4))
# sep_type : "space" | "apostrophe" | "hyphen"
```

### apply_jonctions — Pipeline complet

```python
from lectura_liaisons import JonctionOptions

# Options par défaut : élisions + composés + liaisons gram (enchaînements off)
groups = lia.apply_jonctions(tokens)

# Tout activer
groups = lia.apply_jonctions(tokens, JonctionOptions(enchainements=True))

# Liaisons seules
groups = lia.apply_jonctions(tokens, JonctionOptions(
    elisions=False, mots_composes=False,
    liaisons_gram=True, enchainements=False,
))
```

### Vérifier h aspiré

```python
lia.is_h_aspire("haricot")   # True
lia.is_h_aspire("homme")     # False
```

### Démo en ligne de commande

```bash
# Exemples intégrés
python demo_cli.py

# Vérifier un h aspiré
python demo_cli.py --check "haricot"

# Mode interactif
python demo_cli.py --interactive
```

---

## Types de jonctions

### Liaisons grammaticales

| Cas | Exemple | Type |
|-----|---------|------|
| ART + NOM | les‿enfants | obligatoire |
| ART:ind + NOM | un‿ami | obligatoire (+ dénasalisation) |
| ADJ + NOM | petit‿oiseau | obligatoire |
| PRO:per + VER | ils‿ont | obligatoire |
| "est" + ... | est‿arrivé | obligatoire |
| ADV + ADJ/VER | très‿important | obligatoire |
| PRE + ... | dans‿un | obligatoire |
| NOM + ADJ | enfants‿adorables | facultative |
| "et" + ... | et // alors | interdite |
| ... + h aspiré | les // héros | bloquée |
| ... + "onze" | les // onze | bloquée |

### Enchaînements phonétiques

| Cas | Exemple | Réalisé |
|-----|---------|---------
| C finale + V initiale | avec‿elle | /k/ → /k/ |
| "neuf" + V | neuf‿heures | /f/ → /v/ |

### Élisions

| Cas | Exemple | Résultat |
|-----|---------|----------|
| Mot + apostrophe + Mot | l'enfant | phone1 + phone2 |
| Mot + apostrophe + Mot | j'ai | phone1 + phone2 |

### Mots composés

| Cas | Exemple | Résultat |
|-----|---------|----------|
| Mot + trait d'union + Mot | peut-être | phone1 + phone2 |
| Mot + trait d'union + Mot | peut-on | phone1 + phone2 |

### Phonèmes latents

| Lettre finale | Phonème | Exemple |
|--------------|---------|---------|
| s, x, z | /z/ | les‿enfants |
| t, d | /t/ | petit‿ami |
| n (mon, bon...) | /n/ | bon‿ami |
| r (premier...) | /r/ | premier‿étage |

---

## Entrées requises

### API paires (classify / merge)

| Champ | Description | Source possible |
|-------|-------------|----------------|
| `ortho` | Forme orthographique | Tokeniseur |
| `phone` | Transcription IPA | G2P / eSpeak |
| `pos` | Catégorie(s) grammaticale(s) | POS tagger |

### API pipeline (apply_jonctions)

| Token | Champs | Description |
|-------|--------|-------------|
| `TokenMot` | ortho, phone, pos, span | Mot avec phonétique et POS |
| `TokenSep` | text, sep_type, span | Séparateur (space/apostrophe/hyphen) |
| `TokenPonct` | text, span | Ponctuation (ignorée) |

Ces informations peuvent venir de n'importe quel outil :
Lectura Tokeniseur + Lectura POS + Lectura G2P, ou tout autre pipeline.

---

## Ordre de traitement (apply_jonctions)

Le pipeline applique les jonctions dans cet ordre :

1. **Composés** : Mot + `-` + Mot → groupe
2. **Élisions** : Mot + `'` + Mot → groupe
3. **Liaisons** : groupe/mot + espace + groupe/mot → fusion (itéré jusqu'à stabilisation)
4. **Emballage** : mots isolés restants → groupes simples

---

## H aspiré

La liste embarquée contient 863 formes fléchies de mots français
à h aspiré (haricot, héros, honte, etc.). Elle bloque automatiquement
les liaisons et élisions devant ces mots.

Pour fournir une liste custom :

```python
lia = LecturaLiaisons(h_aspire_path="mon_h_aspire.txt")
```

Format : un mot par ligne, encodage UTF-8.

---

## API de référence

### `LecturaLiaisons(h_aspire_path=None)`

Crée un moteur de jonctions. Sans argument, utilise la liste h aspiré embarquée.

### API paires

#### `lia.classify(w1, w2) → LiaisonDecision`

Classifie la liaison entre deux mots adjacents.

#### `lia.merge(phone1, phone2, decision) → str`

Calcule la phonétique IPA combinée.

#### `lia.analyze_pair(w1, w2) → (LiaisonDecision, str)`

Raccourci : classify + merge.

#### `lia.format_decision(w1, w2) → str`

Affichage lisible de la décision.

### API pipeline

#### `lia.apply_jonctions(tokens, options=None) → list[GroupeJonction]`

Applique composés, élisions et liaisons sur une liste de tokens.
Retourne une liste de `GroupeJonction`.

### Vérification h aspiré

#### `lia.is_h_aspire(word) → bool`

Vérifie si un mot a un h aspiré.

### Types de données

#### `MotInfo(ortho, phone, pos)`

Description d'un mot : orthographe, IPA, liste de POS.

#### `TokenMot(ortho, phone, pos, span)`

Mot pour le pipeline : orthographe, IPA, POS, position.

#### `TokenSep(text, sep_type, span)`

Séparateur : texte, type (space/apostrophe/hyphen), position.

#### `TokenPonct(text, span)`

Ponctuation (ignorée par le pipeline).

#### `LiaisonDecision`

- `.kind` : "grammaticale" | "enchainement" | "none"
- `.typ` : "obligatoire" | "facultative" | "interdite" | "none"
- `.latent_phoneme` : phonème latent (/z/, /t/, /n/...)
- `.latent_ortho` : lettre source (s, t, n...)
- `.phone_patch` : (old, new) pour dénasalisation
- `.realized_phoneme` : phonème réalisé (enchaînement)

#### `GroupeJonction`

- `.components` : liste de tokens composants
- `.phone` : phonétique IPA combinée
- `.span` : position (début, fin)
- `.jonction_type` : "compose" | "elision" | "liaison_gram" | "enchainement" | ""

#### `JonctionOptions`

- `.elisions` : bool (défaut True)
- `.mots_composes` : bool (défaut True)
- `.liaisons_gram` : bool (défaut True)
- `.enchainements` : bool (défaut False)

### Fonction utilitaire

- `merge_phones(phone1, phone2, decision) → str` : fusion bas-niveau

---

## Support

Pour toute question : [contact à définir]

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0. Voir LICENCE.txt et ATTRIBUTION.md.*
