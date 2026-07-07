# Lectura — French NLP & Speech Toolkit

**Suite complete de traitement linguistique et vocal du francais.**
Tokenisation, phonetique, syllabation, correction orthographique, synthese vocale et plus.

*A comprehensive French natural language processing and speech toolkit.
Tokenization, phonetics, syllabification, spell-checking, text-to-speech, and more.*

```bash
pip install lectura            # meta-package (tout installer)
pip install lectura-phonemiseur  # ou un seul module
```

---

## Modules atomiques (couche 1)

| Module | Description | Version | pip install |
|--------|-------------|---------|-------------|
| **[Tokeniseur](Tokeniseur/)** | Normalisation et tokenisation du francais, detection de formules | 2.3.3 | `pip install lectura-tokeniseur` |
| **[Formules](Formules/)** | Lecture algorithmique des formules (nombres, dates, heures...) | 3.7.6 | `pip install lectura-formules` |
| **[Phonemiseur](Phonemiseur/)** | G2P neural : grapheme-to-phoneme + POS + Morpho + Liaison | 4.1.6 | `pip install lectura-phonemiseur` |
| **[Graphemiseur](Graphemiseur/)** | P2G neural : phoneme-to-grapheme + POS + Morpho (IPA → orthographe) | 4.3.5 | `pip install lectura-graphemiseur` |
| **[Aligneur-Syllabeur](Aligneur/)** | Alignement grapheme-phoneme et syllabation / syllabification | 4.0.1 | `pip install lectura-aligneur` |
| **[Correcteur](Correcteur/)** | Correcteur orthographique et grammatical du francais | 1.1.1 | `pip install lectura-correcteur` |
| **[Decodeur](Decodeur/)** | Decodeur phonetique CTC : audio → phones IPA (CNN-BiGRU-CTC) | 3.0.1 | `pip install lectura-decodeur` |
| **[Lexique](Lexique/)** | Acces au lexique Lectura (359k lemmes, 1.5M formes) | 1.5.0 | `pip install lectura-lexique` |

## Pipelines (couche 2)

| Module | Description | Version | pip install |
|--------|-------------|---------|-------------|
| **[G2P-Pipeline](G2P-Pipeline/)** | Pipeline complet texte → phonetique (tokeniseur + formules + phonemiseur) | 4.1.1 | `pip install lectura-g2p` |
| **[P2G-Pipeline](P2G-Pipeline/)** | Pipeline complet phonetique → texte | 4.6.2 | `pip install lectura-p2g` |
| **[STT](STT/)** | Pipeline audio → texte (CTC + P2G) | 3.2.2 | `pip install lectura-stt` |

## Synthese vocale / Text-to-Speech

| Module | Description | Version | pip install |
|--------|-------------|---------|-------------|
| **[Monospeaker](Monospeaker/)** | TTS neural monospeaker francais (FastPitch + HiFi-GAN) | 4.0.0 | `pip install lectura-monospeaker` |
| **[MultiSpeaker](MultiSpeaker/)** | TTS neural multispeaker francais | 4.0.0 | `pip install lectura-multispeaker` |
| **[Diphone](Diphone/)** | TTS par concatenation de diphones WORLD (prosodie reglee) | 2.0.1 | `pip install lectura-diphone` |

## Conversion vocale / Voice Conversion

| Module | Description | Version | pip install |
|--------|-------------|---------|-------------|
| **[VC-ZeroShot](VC-ZeroShot/)** | Conversion vocale zero-shot | 1.2.0 | `pip install lectura-vc-zeroshot` |
| **[VC-Locuteurs](VC-Locuteurs/)** | Conversion vocale par locuteurs (RVC) | 1.0.0 | `pip install lectura-vc-locuteurs` |

---

## Caracteristiques / Key Features

- **Zero dependance** sur les modules de base (Tokeniseur, Formules, Aligneur) — *Zero dependencies for core modules*
- **4 backends d'inference** pour G2P/P2G : API, ONNX Runtime, NumPy, Pure Python — *4 inference backends*
- **Type hints complets** (Python 3.10+, PEP-561) — *Full type hints*
- **Modeles compacts** : G2P = 1.8 Mo, P2G = 2.6 Mo (ONNX INT8) — *Compact models*
- **Syllabation explicite** par alignement grapheme-phoneme — *Explicit syllabification via grapheme-phoneme alignment*

## Installation rapide / Quick Start

```bash
# Tous les modules d'un coup / Install everything
pip install lectura

# Avec backends ONNX pour G2P/P2G (recommande / recommended)
pip install lectura[onnx]

# Un seul module / Single module
pip install lectura-tokeniseur

# Phonemiseur avec backend ONNX
pip install lectura-phonemiseur[onnx]
```

## Exemple / Example

```python
from lectura_tokeniseur import tokenise
from lectura_formules import lire_formule

# Tokeniser du texte francais
tokens = tokenise("Le 1er janvier 2025, j'ai lu 42 pages.")

# Lire une formule (nombre → texte)
result = lire_formule("NOMBRE", "42")
print(result.display_fr)  # "quarante-deux"
```

## Licence

Code source sous licence **[AGPL-3.0](LICENCE.txt)**.
Modeles pre-entraines (.onnx) : voir **[MODEL_LICENCE.md](MODEL_LICENCE.md)**.

**Licence commerciale et modeles locaux disponibles** — contacter **[admin@lectura.world](mailto:admin@lectura.world)**.

## Auteur

Max Carriere — [lectura.world](https://www.lectura.world)
