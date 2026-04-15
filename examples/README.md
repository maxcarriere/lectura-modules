# Exemples Lectura NLP

Exemples d'utilisation des modules Lectura NLP pour le traitement du francais.

## Liste des exemples

| Fichier | Description | Modules utilises |
|---------|-------------|-----------------|
| `01_tokeniser_un_texte.py` | Normalisation et tokenisation | Tokeniseur |
| `02_lire_des_formules.py` | Lecture de nombres, dates, heures... | Formules |
| `03_phonetiser_avec_g2p.py` | Conversion texte vers IPA | G2P |
| `04_analyser_syllabes.py` | Analyse syllabique et groupes de lecture | Aligneur |
| `05_pipeline_complet.py` | Enchainement de tous les modules | Tous |

## Installation

```bash
# Minimum (exemples 1-2)
pip install lectura-tokeniseur lectura-formules

# Complet (tous les exemples)
pip install lectura-tokeniseur lectura-formules lectura-g2p[onnx] lectura-aligneur
```

## Lancer un exemple

```bash
python examples/01_tokeniser_un_texte.py
```
