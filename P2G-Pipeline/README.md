# lectura-p2g

Pipeline P2G complet du francais : Graphemiseur + Formules + Noms propres (IPA -> orthographe).

Couche 2 du pipeline P2G, en miroir de `lectura-g2p` pour le G2P.

## Installation

```bash
pip install lectura-p2g
```

## Usage

```python
from lectura_p2g import analyser, creer_engine

engine = creer_engine()
result = analyser(["le", "sha", "eh", "bon"], engine=engine)
print(result["ortho"])
```

## Architecture

- **Couche 1** : `lectura-graphemiseur` — modele P2G core (lex_select + coherence morpho + accents)
- **Couche 2** : `lectura-p2g` — pipeline complet (graphemiseur + formules + noms propres)

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt)
