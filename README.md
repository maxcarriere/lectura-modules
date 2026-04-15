# Lectura NLP — Modules de traitement du langage naturel pour le francais

Briques logicielles autonomes pour le traitement du francais : tokenisation,
phonetique, syllabes, formules. Installez tout d'un coup avec `pip install lectura`
ou chaque module independamment.

## Modules

| Module | Description | Version | pip install |
|--------|-------------|---------|-------------|
| **[Tokeniseur](Tokeniseur/)** | Normalisation et tokenisation du francais, detection de formules | 2.2.1 | `pip install lectura-tokeniseur` |
| **[G2P](G2P/)** | Grapheme-to-Phoneme unifie + POS + Morpho + Liaison | 2.0.0 | `pip install lectura-g2p` |
| **[P2G](P2G/)** | Phoneme-to-Grapheme unifie + POS + Morpho (IPA vers orthographe) | 2.0.0 | `pip install lectura-p2g` |
| **[Aligneur-Syllabeur](Aligneur/)** | Alignement grapheme-phoneme, groupes de lecture, syllabation | 3.0.0 | `pip install lectura-aligneur` |
| **[Formules](Formules/)** | Lecture algorithmique des formules (nombres, dates, heures...) | 3.0.0 | `pip install lectura-formules` |
| **[Lexique](Lexique/)** | Acces generique au lexique francais | 1.3.0 | `pip install lectura-lexique` |

## Caracteristiques

- **Zero dependance** sur les modules de base (Tokeniseur, Formules, Lexique)
- **Mode API** par defaut pour G2P, P2G, Aligneur — zero config, fonctionne immediatement
- **4 backends d'inference** pour G2P/P2G : API, ONNX Runtime, NumPy, Pure Python
- **Type hints complets** (Python 3.10+, PEP-561)

## Installation rapide

```bash
# Tous les modules d'un coup
pip install lectura

# Un seul module
pip install lectura-g2p

# G2P avec backend ONNX local (optionnel)
pip install lectura-g2p[onnx]
```

Par defaut, G2P, P2G et Aligneur utilisent l'API Lectura (`api.lec-tu-ra.com`).
Aucune configuration necessaire.

## Exemple

```python
from lectura_nlp import creer_engine

engine = creer_engine()    # mode API (zero config)
result = engine.analyser(["bonjour", "le", "monde"])

print(result["g2p"])      # ['bɔ̃ʒuʁ', 'lə', 'mɔ̃d']
print(result["pos"])      # ['INTJ', 'ART:def', 'NOM']
print(result["liaison"])  # ['none', 'none', 'none']
```

## Licence

Les modules Lectura NLP sont distribues sous **double licence** :

- **[AGPL-3.0-or-later](LICENCE.txt)** — libre, avec obligation de publication
  du code source pour tout logiciel derive.
- **[Licence Commerciale](LICENCE-COMMERCIALE.md)** — payante, pour integration
  dans des logiciels proprietaires sans obligation de publication.

Pour obtenir une licence commerciale : **https://www.lec-tu-ra.com/solutions/services/**

## Auteur

Max Carriere — [lec-tu-ra.com](https://www.lec-tu-ra.com)
