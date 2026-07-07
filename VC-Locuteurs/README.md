# lectura-vc-locuteurs

RVC voice conversion with 6 pre-trained French voices — pure ONNX inference, no PyTorch required.

*Conversion vocale RVC avec 6 voix francaises pre-entrainees — inference ONNX pure, sans PyTorch.*

## Installation

```bash
pip install lectura-vc-locuteurs
```

## Usage

```python
from lectura_vc_locuteurs import convertir, lister_locuteurs

# Lister les voix disponibles
locuteurs = lister_locuteurs()

# Convertir un audio avec une voix
audio_converti = convertir(audio_source, locuteur="voix_1")
```

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt).
Modeles pre-entraines : voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

**Licence commerciale et modeles locaux disponibles** — contacter [admin@lectura.world](mailto:admin@lectura.world).
