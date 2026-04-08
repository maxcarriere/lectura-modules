# Lectura Pseudo-Ortho

Convertisseur IPA vers pseudo-orthographe lisible pour le francais.

Module standalone, zero dependance, table embarquee (3493 entrees Lexique383).

## Installation

```bash
pip install lectura-pseudo-ortho
```

Ou copier `lectura_pseudo_ortho.py` dans votre projet (fichier unique, autonome).

## Usage

```python
from lectura_pseudo_ortho import LecturaPseudoOrtho

p2g = LecturaPseudoOrtho()

# Lookup lexical (3493 syllabes connues)
p2g.predict("kɑ̃")           # → "quand"
p2g.predict("a")             # → "a"

# Regles fallback (syllabes hors corpus)
p2g.predict("tʁa")           # → "tra"

# Mot multi-syllabes
p2g.predict_word(["pa", "pi", "jɔ̃"])  # → "pas pi yon"
```

## API

### `LecturaPseudoOrtho(table_path=None)`

- Si `table_path` est `None`, utilise la table embarquee (base85, ~24 Ko)
- Sinon, charge un fichier JSON externe

### `.predict(ipa_syllable) → str`

Convertit une syllabe IPA en pseudo-orthographe.
Priorite : overrides > lookup lexical > regles fallback.

### `.predict_word(syllables_ipa) → str`

Concatene les predictions avec espace.

### `.set_overrides(dict)`

Definit des mappings prioritaires sur tout (overrides > lookup > regles).

### `iter_phonemes(ipa) → list[str]`

Fonction utilitaire : segmente une chaine IPA en phonemes individuels
(gere les combining marks Unicode).

## Strategie de conversion

1. **Overrides** : mappings utilisateur prioritaires
2. **Lookup lexical** : 3493 vrais mots francais monosyllabiques (Lexique383),
   classes par frequence — le mot le plus frequent est retenu comme graphie
3. **Regles fallback** : 36 regles deterministes phoneme → grapheme
   (`ʃ→ch`, `ɑ̃→an`, `ʁ→r`, etc.)

## Demo CLI

```bash
python demo_cli.py "kɑ̃"
python demo_cli.py "tʁa" "bɔ̃"
python demo_cli.py          # mode interactif
```

## Licence

CC BY-SA 4.0 — Voir LICENCE.txt.
