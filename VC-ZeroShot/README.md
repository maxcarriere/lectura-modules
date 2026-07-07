# lectura-vc-zeroshot

Zero-shot voice conversion for French via OpenVoice — pure ONNX inference, no PyTorch required.
Transfer any speaker's voice characteristics to synthesized speech.

*Conversion vocale zero-shot pour le francais via OpenVoice — inference ONNX pure, sans PyTorch.
Transfert des caracteristiques vocales de n'importe quel locuteur vers la parole synthetisee.*

## Installation

```bash
pip install lectura-vc-zeroshot
```

## Usage

```python
from lectura_vc_zeroshot import convertir

# Convertir un audio avec la voix d'un locuteur cible
audio_converti = convertir(audio_source, audio_cible)
```

## Presets

Des presets de voix sont inclus pour une utilisation sans audio de reference.

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt).
Modeles pre-entraines : voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

**Licence commerciale et modeles locaux disponibles** — contacter [admin@lectura.world](mailto:admin@lectura.world).
