# lectura-lexique

French lexical database access — 359k lemmas, 1.5M inflected forms with IPA phonetics, syllabification, and morphology.

*Module d'acces au lexique Lectura : 359 000 lemmes, 1 518 000 formes flechies avec phonetique IPA, syllabation et morphologie.*

## Installation

```bash
pip install lectura-lexique
```

## Usage

```python
from lectura_lexique import Lexique

lex = Lexique("lexique_lectura_v7.db")

# Verifier l'existence d'un mot
lex.existe("bonjour")  # True

# Phonetique IPA
lex.phone_de("bonjour")  # 'bon.ZuR'

# Conjugaison
lex.conjuguer("manger")  # {mode: {temps: {personne: forme}}}

# Formes flechies
lex.formes_de("chat")  # [chat, chats, chatte, chattes]

# Definitions
lex.definitions("chat")  # [{definition: "...", ...}, ...]
```

## Contenu de la base

- **359 303 lemmes** (noms, verbes, adjectifs, noms propres)
- **1 518 155 formes flechies** avec phonetique IPA et decoupage syllabique
- **456 335 definitions** (Wiktionnaire)
- **2 483 597 entites nommees** (Wikidata)
- **873 categories semantiques** hierarchiques
- Relations : 81k synonymes, 69k derives, 76k apparentes, 19k hyperonymes, 11k antonymes

## Licence

CC-BY-SA-4.0 — voir les sources dans [ATTRIBUTION.md](ATTRIBUTION.md).

**Licence commerciale et acces API disponibles** — contacter [admin@lectura.world](mailto:admin@lectura.world).
