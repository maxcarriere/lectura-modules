# Lectura NLP — Modules de traitement du langage naturel pour le francais

Briques logicielles autonomes pour le traitement du francais : tokenisation,
phonetique, syllabes, formules. Installez tout d'un coup avec `pip install lectura`
ou chaque module independamment.

## Modules

| Module | Description | Version | pip install |
|--------|-------------|---------|-------------|
| **[Tokeniseur](Tokeniseur/)** | Normalisation et tokenisation du francais, detection de formules | 2.0.0 | `pip install lectura-tokeniseur` |
| **[G2P](G2P/)** | Grapheme-to-Phoneme unifie + POS + Morpho + Liaison | 3.0.0 | `pip install lectura-g2p` |
| **[P2G](P2G/)** | Phoneme-to-Grapheme unifie + POS + Morpho (IPA vers orthographe) | 3.0.0 | `pip install lectura-p2g` |
| **[Aligneur-Syllabeur](Aligneur/)** | Alignement grapheme-phoneme, groupes de lecture, syllabation | 2.2.0 | `pip install lectura-aligneur` |
| **[Formules](Formules/)** | Lecture algorithmique des formules (nombres, dates, heures...) | 2.0.0 | `pip install lectura-formules` |

## Caracteristiques

- **Zero dependance** sur les modules de base (Tokeniseur, Formules, Aligneur)
- **4 backends d'inference** pour G2P/P2G : API, ONNX Runtime, NumPy, Pure Python
- **Type hints complets** (Python 3.10+, PEP-561)
- **Modeles compacts** : G2P = 1.8 Mo, P2G = 2.6 Mo (ONNX INT8)

## Installation rapide

```bash
# Tous les modules d'un coup
pip install lectura

# Avec backends ONNX pour G2P/P2G (recommande)
pip install lectura[onnx]

# Un seul module
pip install lectura-tokeniseur

# G2P avec backend ONNX
pip install lectura-g2p[onnx]
```

## Exemple

```python
from lectura_tokeniseur import tokenise
from lectura_formules import lire_formule

# Tokeniser du texte francais
tokens = tokenise("Le 1er janvier 2025, j'ai lu 42 pages.")

# Lire une formule
result = lire_formule("NOMBRE", "42")
print(result.display_fr)  # "quarante-deux"
```

## Licence

Les modules Lectura sont distribues sous licence **[AGPL-3.0](LICENCE.txt)** (non commerciale).

Les modeles pre-entraines (.onnx) sont soumis a des conditions specifiques :
voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

Pour un usage commercial, contacter **[contact@lec-tu-ra.com](mailto:contact@lec-tu-ra.com)**.

## Auteur

Max Carriere — [lec-tu-ra.com](https://www.lec-tu-ra.com)
