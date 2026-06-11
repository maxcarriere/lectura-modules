# lectura-ctc — Decodeur phonetique CTC du francais

Transcription audio vers phonemes IPA via un modele CNN-BiGRU-CTC medium (10.6M params, PER ~4.34%).

Le modele medium supporte les sigles, formules (nombres, dates, calculs) et la parole courante grace a un vocabulaire etendu de 59 tokens et un fine-tuning specialise.

## Installation

```bash
# Avec backend ONNX (recommande)
pip install lectura-ctc[onnx]

# Sans ONNX (mode API uniquement)
pip install lectura-ctc
```

## Utilisation

```python
import numpy as np
from lectura_ctc import creer_engine

engine = creer_engine()

# Audio PCM float32 mono 16kHz
audio = np.zeros(16000, dtype=np.float32)  # 1 seconde de silence
result = engine.transcrire(audio)
print(result)  # chaine IPA : "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d"
```

## Modele

| Caracteristique | Valeur |
|-----------------|--------|
| **Architecture** | CNN [48, 96] + BiGRU 384x4 + CTC head |
| **Parametres** | 10.6M |
| **Performance** | PER ~4.34% (formules v2) |
| **Vocabulaire** | 59 tokens (46 phones IPA + liaisons + ponctuation + speciaux) |
| **Entrainement** | Pre-train corpus complet + fine-tune formules/sigles multispeaker |

## Backends

| Backend | Dependance | Latence | Modele |
|---------|-----------|---------|--------|
| ONNX Runtime | `onnxruntime` | ~10 ms/s audio | `phone_ctc_int8.onnx` (38 Mo) |
| API | aucune | ~100 ms/s audio | serveur Lectura |

## Parametres audio

- Sample rate : 16 kHz
- Format : PCM float32 mono
- Mel : 80 bins, n_fft=512, hop=160, win=400

## Licence

Double licence : [AGPL-3.0](LICENCE.txt) (libre) + [Licence Commerciale](LICENCE-COMMERCIALE.md) (payante).

Les modeles ONNX sont distribues separement — voir [MODEL_LICENCE.md](../MODEL_LICENCE.md).

## Auteur

Max Carriere — [lectura.world](https://www.lectura.world)
