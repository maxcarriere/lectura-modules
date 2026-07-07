# lectura-p2g

Pipeline P2G complet du francais : Graphemiseur + Formules + Noms propres (IPA -> orthographe).

Couche 2 du pipeline P2G, en miroir de `lectura-g2p` pour le G2P.

## Installation

```bash
pip install lectura-p2g              # pipeline complet
pip install lectura-p2g[aligneur]    # + alignement grapheme-phoneme
```

## Usage

```python
from lectura_p2g import analyser, creer_engine

engine = creer_engine()
result = analyser(["le", "sha", "eh", "bon"], engine=engine)
print(result["ortho"])
```

### Avec alignement

```python
from lectura_p2g import analyser, creer_engine
from lectura_aligneur.lectura_aligneur import LecturaSyllabeur

engine = creer_engine()
syl = LecturaSyllabeur()
result = analyser(["bɔ̃ʒuʁ"], engine=engine, aligner=syl)
print(result["alignments"][0].syllabes)
```

## Architecture

- **Couche 1** : `lectura-graphemiseur` — modele P2G core (lex_select + coherence morpho + accents)
- **Couche 2** : `lectura-p2g` — pipeline complet (graphemiseur + formules + noms propres)

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt).

**Licence commerciale et modeles locaux disponibles** — contacter [admin@lectura.world](mailto:admin@lectura.world).
