# lectura-g2p

Complete French grapheme-to-phoneme pipeline — text to IPA phonetics in one call.
Combines tokenizer, formula reader, and neural phonemizer.

*Pipeline G2P complet du francais — texte vers phonetique IPA en un appel.
Combine tokeniseur, lecteur de formules et phonemiseur neural.*

## Installation

```bash
pip install lectura-g2p              # pipeline complet
pip install lectura-g2p[aligneur]    # + alignement grapheme-phoneme et syllabation
```

## Usage

```python
from lectura_g2p import analyser, creer_engine

engine = creer_engine()

# Texte → phonetique IPA
result = analyser("Bonjour, il est 14h30.", engine=engine)
print(result["ipa"])
```

### Avec alignement et syllabation

```python
from lectura_g2p import analyser, creer_engine
from lectura_aligneur.lectura_aligneur import LecturaSyllabeur

engine = creer_engine()
syl = LecturaSyllabeur()

result = analyser("Bonjour !", engine=engine, aligner=syl)
for a in result["alignments"]:
    print(a.syllabes)
```

## Architecture

- **Couche 1** : `lectura-phonemiseur` — modele G2P neural BiLSTM (1.8 Mo ONNX INT8, 98.5% word accuracy)
- **Couche 2** : `lectura-g2p` — pipeline complet (tokeniseur + formules + phonemiseur)
- **4 backends** : API, ONNX Runtime, NumPy, Pure Python (zero dependance)

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt).

**Licence commerciale et modeles locaux disponibles** — contacter [admin@lectura.world](mailto:admin@lectura.world).
