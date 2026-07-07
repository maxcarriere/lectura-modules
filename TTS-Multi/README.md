# lectura-tts-multi

French text-to-speech pipeline (multi-speaker) — text to audio with speaker selection.
Combines G2P pipeline with neural multi-speaker TTS (Conformer or FastPitch + HiFi-GAN).

*Pipeline TTS multi-speaker du francais — texte vers audio avec choix de voix.
Combine le pipeline G2P avec la synthese neuronale multi-speaker (Conformer ou FastPitch + HiFi-GAN).*

## Installation

```bash
pip install lectura-tts-multi                 # pipeline
pip install lectura-tts-multi[retimbre]       # + conversion vocale zero-shot
```

## Usage

```python
from lectura_tts_multi import synthetiser

audio = synthetiser("Bonjour, comment allez-vous ?", speaker_id=0)
# audio : numpy array (int16, 22050 Hz)
```

## Architecture

- **G2P** : `lectura-g2p` — texte vers phonetique IPA
- **TTS** : `lectura-multispeaker` — phonetique vers audio multi-speaker (Conformer ou FastPitch, ONNX)
- **Retimbre** (optionnel) : `lectura-vc-zeroshot` — conversion vocale zero-shot

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt).
Modeles pre-entraines : voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

**Licence commerciale et modeles locaux disponibles** — contacter [admin@lectura.world](mailto:admin@lectura.world).
