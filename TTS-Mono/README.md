# lectura-tts-mono

French text-to-speech pipeline (monospeaker) — text to audio in one call.
Combines G2P pipeline with neural TTS (Conformer or FastPitch + HiFi-GAN).

*Pipeline TTS monospeaker du francais — texte vers audio en un appel.
Combine le pipeline G2P avec la synthese neuronale (Conformer ou FastPitch + HiFi-GAN).*

## Installation

```bash
pip install lectura-tts-mono                 # pipeline
pip install lectura-tts-mono[retimbre]       # + conversion vocale zero-shot
```

## Usage

```python
from lectura_tts_mono import synthetiser

audio = synthetiser("Bonjour, comment allez-vous ?")
# audio : numpy array (int16, 22050 Hz)
```

## Architecture

- **G2P** : `lectura-g2p` — texte vers phonetique IPA
- **TTS** : `lectura-monospeaker` — phonetique vers audio (Conformer ou FastPitch + HiFi-GAN, ONNX)
- **Retimbre** (optionnel) : `lectura-vc-zeroshot` — conversion vocale zero-shot

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt).
Modeles pre-entraines : voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

**Licence commerciale et modeles locaux disponibles** — contacter [admin@lectura.world](mailto:admin@lectura.world).
