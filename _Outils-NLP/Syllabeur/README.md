# Lectura Syllabeur

**Analyseur syllabique du français — phonémisation, syllabation et alignement graphème-phonème**

Un module léger et autonome qui transforme du texte français en syllabes structurées
avec correspondances orthographe ↔ phonétique. Zéro dépendance Python — un seul
fichier, prêt à l'emploi.

---

## Démarrage rapide

```python
from lectura_syllabeur import LecturaSyllabeur

syl = LecturaSyllabeur()
result = syl.analyze("chocolat")

for s in result.syllabes:
    print(f"{s.ortho:6s} /{s.phone}/  span={s.span}")
# cho    /ʃo/   span=(0, 3)
# co     /ko/   span=(3, 5)
# lat    /la/   span=(5, 8)
```

### Prérequis

- Python 3.10+
- eSpeak-NG installé sur le système (phonémiseur par défaut)
- Aucune bibliothèque Python requise

```bash
# Linux
sudo apt install espeak-ng

# macOS
brew install espeak

# Windows
choco install espeak-ng
```

### Contenu de l'archive

```
lectura-syllabeur-v1.0/
├── lectura_syllabeur.py        ← Fichier principal (copier dans votre projet)
├── demo_cli.py                 ← Démo en ligne de commande
├── exemples/
│   ├── exemple_basique.py      ← Utilisation simple
│   └── exemple_integration.py  ← Intégration dans un pipeline
├── README.md
├── LICENCE.txt
└── ATTRIBUTION.md
```

---

## Utilisation

### Dans votre code

Copiez `lectura_syllabeur.py` dans votre projet :

```python
from lectura_syllabeur import LecturaSyllabeur

syl = LecturaSyllabeur()

# Analyse complète d'un mot
result = syl.analyze("extraordinaire")
print(result.format_detail())
# extraordinaire → /ɛkstʁaɔʁdinɛʁ/
#   σ1: /ɛk/ «ex» [0:2] att=- noy=ɛ cod=k
#   σ2: /stʁa/ «tra» [2:5] att=stʁ noy=a cod=-
#   ...

# Accès structuré aux données
for s in result.syllabes:
    print(f"Syllabe «{s.ortho}» [{s.span[0]}:{s.span[1]}]")
    print(f"  Attaque : {[(p.ipa, p.grapheme) for p in s.attaque.phonemes]}")
    print(f"  Noyau   : {[(p.ipa, p.grapheme) for p in s.noyau.phonemes]}")
    print(f"  Coda    : {[(p.ipa, p.grapheme) for p in s.coda.phonemes]}")

# Analyse d'une phrase
for r in syl.analyze_text("Les enfants jouent dans la cour"):
    print(r.format_simple())
```

### Phonétique manuelle

Si vous connaissez la prononciation exacte, passez-la directement :

```python
# Bypass le phonémiseur — utile pour corriger ou pour les mots irréguliers
result = syl.analyze("fils", phone="fis")     # le fil
result = syl.analyze("fils", phone="fis")     # le garçon
```

### Mode IPA direct (sans alignement)

```python
# Syllabation pure sur de l'IPA — aucune dépendance
sylls = syl.syllabify_ipa("ɛkstʁaɔʁdinɛʁ")
# → ['ɛk', 'stʁa', 'ɔʁ', 'di', 'nɛʁ']
```

### Démo en ligne de commande

```bash
# Phrase en argument
python demo_cli.py "Le chat mange la souris"

# Affichage simplifié
python demo_cli.py --simple "Le chat mange la souris"

# Entrée IPA directe
python demo_cli.py --ipa "ʃɔkɔla"

# Mode interactif
python demo_cli.py
```

---

## Phonémiseur pluggable

Par défaut, le syllabeur utilise **eSpeak-NG** pour la phonémisation.
Vous pouvez brancher n'importe quel phonémiseur en passant un objet
avec une méthode `phonemize(word) → str` :

### Phonémiseur custom

```python
class MonPhonemiseur:
    def phonemize(self, word: str) -> str:
        return mon_lexique.get(word.lower(), "")

syl = LecturaSyllabeur(phonemizer=MonPhonemiseur())
```

### Avec Lectura G2P (vendu séparément)

Pour une meilleure précision phonémique, utilisez Lectura G2P :

```python
from lectura_g2p import LecturaG2P

g2p = LecturaG2P("modele/g2p_model_crf.json",
                   corrections_path="modele/g2p_corrections_crf.json")

# LecturaG2P est détecté automatiquement (méthode .predict)
syl = LecturaSyllabeur(phonemizer=g2p)
result = syl.analyze("extraordinaire")
```

---

## Structure de sortie

### ResultatAnalyse

```
ResultatAnalyse
├── mot: str                    "chocolat"
├── phone: str                  "ʃɔkɔla"
├── nb_syllabes: int            3
└── syllabes: list[Syllabe]
    ├── Syllabe
    │   ├── phone: str          "ʃɔ"
    │   ├── ortho: str          "cho"
    │   ├── span: (int, int)    (0, 3)
    │   ├── attaque: GroupePhonologique
    │   │   └── phonemes: [Phoneme(ipa="ʃ", grapheme="ch")]
    │   ├── noyau: GroupePhonologique
    │   │   └── phonemes: [Phoneme(ipa="ɔ", grapheme="o")]
    │   └── coda: GroupePhonologique
    │       └── phonemes: []
    ├── Syllabe ...
    └── Syllabe ...
```

### Spans

Les `span` indiquent les positions (start, end) dans le mot original,
permettant le surlignage dans une interface :

```python
result = syl.analyze("bonjour")
for s in result.syllabes:
    print(f"  «{'bonjour'[s.span[0]:s.span[1]]}»")
# → «bon»
# → «jour»
```

---

## Détails techniques

### Trois algorithmes intégrés

| Composant | Rôle |
|-----------|------|
| **Syllabeur** | Découpage par modèle de sonorité (5 classes : O/N/L/Y/V) |
| **Aligneur** | Correspondance ortho ↔ phonème par DFS + lettres muettes |
| **IPA utils** | Parsing Unicode (combining marks), classification phonétique |

### Pipeline

```
mot → phonémiseur → IPA → syllabation → alignement → Syllabe(attaque/noyau/coda + spans)
```

### Modèle de sonorité

Les phonèmes sont classés en 5 niveaux de sonorité croissante :

```
O (Obstruantes) < N (Nasales) < L (Liquides) < Y (Semi-voyelles) < V (Voyelles)
```

Les frontières syllabiques sont placées aux creux de sonorité
entre deux pics vocaliques.

### Limites

- Le phonémiseur eSpeak-NG peut faire des erreurs sur certains mots
  irréguliers — utilisez le paramètre `phone` pour corriger manuellement,
  ou passez à Lectura G2P pour une meilleure précision.
- L'alignement dépend de la qualité de la transcription phonémique.
- Optimisé pour le français contemporain standard.
- Les mots composés avec trait d'union sont traités mot par mot via `analyze_text`.

---

## API de référence

### `LecturaSyllabeur(phonemizer=None)`

Crée un analyseur syllabique. Sans argument, utilise eSpeak-NG.

### `LecturaSyllabeur.with_espeak(lang="fr")`

Constructeur alternatif explicite avec eSpeak-NG.

### `syl.analyze(word, phone=None) → ResultatAnalyse`

Analyse syllabique complète d'un mot. Si `phone` est fourni,
il est utilisé à la place du phonémiseur.

### `syl.analyze_text(text) → list[ResultatAnalyse]`

Analyse chaque mot d'un texte (tokenisation par espaces/ponctuation).

### `syl.syllabify_ipa(phone) → list[str]`

Syllabation bas-niveau sur de l'IPA brut (sans alignement).

### `ResultatAnalyse`

- `.mot` : mot original
- `.phone` : transcription IPA complète
- `.syllabes` : liste de `Syllabe`
- `.nb_syllabes` : nombre de syllabes
- `.format_simple()` : "mot → /syll.abes/ (n syll.)"
- `.format_detail()` : affichage multi-lignes avec attaque/noyau/coda

### `Syllabe`

- `.phone` : IPA de la syllabe ("ʃɔ")
- `.ortho` : graphie correspondante ("cho")
- `.span` : (start, end) dans le mot original
- `.attaque` / `.noyau` / `.coda` : `GroupePhonologique`

### `GroupePhonologique`

- `.phonemes` : liste de `Phoneme`
- `.phone` : IPA concaténée
- `.grapheme` : graphie concaténée

### `Phoneme`

- `.ipa` : symbole IPA ("ʃ")
- `.grapheme` : graphie correspondante ("ch")

### Fonctions utilitaires

- `iter_phonemes(ipa) → list[str]` : découpe une chaîne IPA en phonèmes
- `est_voyelle(ph)`, `est_consonne(ph)`, `est_semi_voyelle(ph)` : classification

### Protocol Phonemizer

Toute classe implémentant `phonemize(word: str) → str` est compatible.
Les objets avec `.predict(word)` (comme LecturaG2P) sont adaptés automatiquement.

---

## Support

Pour toute question : [contact à définir]

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0. Voir LICENCE.txt et ATTRIBUTION.md.*
