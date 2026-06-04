# Lectura STT — Pipeline STT complet du francais

Pipeline de transcription automatique du francais : audio vers texte.
Chaine le decodeur CTC (audio → phones IPA) avec le pipeline P2G (phones → orthographe).

## Installation

```bash
# Mode minimal (CTC uniquement, transcription phonetique)
pip install lectura-stt

# Avec pipeline P2G complet (formules + noms propres)
pip install lectura-stt[p2g]

# Avec backend ONNX (inference locale rapide)
pip install lectura-stt[onnx]

# Avec support micro
pip install lectura-stt[micro]
```

## Exemple

```python
import numpy as np
from lectura_stt import creer_engine

engine = creer_engine()

# Charger un fichier WAV
import wave
with wave.open("bonjour.wav", "rb") as wf:
    sr = wf.getframerate()
    audio = np.frombuffer(
        wf.readframes(wf.getnframes()), dtype=np.int16
    ).astype(np.float32) / 32768.0

result = engine.transcrire(audio, sr=sr)
print(result.ipa)    # "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d ."
print(result.texte)  # "Bonjour le monde."
```

## Architecture

```
Audio 16kHz mono
     |
     v
[lectura-ctc]  --> IPA phones "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d ."
     |
     v
[_parse_ctc]   --> mots IPA ["bɔ̃ʒuʁ", "lə", "mɔ̃d"] + ponctuation ["."]
     |
     v
[lectura-p2g]  --> ortho ["bonjour", "le", "monde"]
     |
     v
[_assembler]   --> "Bonjour le monde."
```

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt).
Licence commerciale disponible — voir [LICENCE-COMMERCIALE.md](LICENCE-COMMERCIALE.md).
