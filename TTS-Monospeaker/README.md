# lectura-tts-monospeaker

Synthese vocale neuronale monospeaker pour le francais.

## Installation

```bash
# Version minimale (API distante, zero deps)
pip install lectura-tts-monospeaker

# Version locale (inference ONNX)
pip install lectura-tts-monospeaker[onnx]

# Avec G2P integre (texte → audio)
pip install lectura-tts-monospeaker[all]
```

## Utilisation

### Depuis du texte (necessite lectura-g2p)

```python
from lectura_tts_monospeaker import synthetiser

audio = synthetiser("Bonjour le monde")
# audio: numpy array float32, 22050 Hz
```

### Depuis des phonemes IPA

```python
from lectura_tts_monospeaker import creer_engine

engine = creer_engine(mode="local")
result = engine.synthesize_phonemes(
    "bɔ̃ʒuʁ",
    phrase_type=0,
    pitch_range=1.3,
)
# result.samples: numpy float32
# result.sample_rate: 22050
# result.phoneme_timings: list[PhonemeTiming]
```

### Via l'API distante

```python
from lectura_tts_monospeaker import creer_engine

engine = creer_engine(mode="api", api_url="https://api.lec-tu-ra.com")
result = engine.synthesize("Bonjour")
```

## Controles prosodiques

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| duration_scale | 1.0 | Vitesse globale |
| pitch_shift | 0.0 | Decalage F0 (demi-tons) |
| pitch_range | 1.3 | Variation F0 (1.0 = neutre) |
| energy_scale | 1.0 | Intensite |
| pause_scale | 1.0 | Duree des pauses |
| phrase_type | 0 | 0=decl, 1=inter, 2=excl, 3=susp |

## Architecture

- **FastPitch-Lite** : phonemes → mel-spectrogramme (~5M params)
- **HiFi-GAN V1** : mel → audio 22050 Hz (~3.5M params)
- **Runtime** : ONNX (pas de dependance PyTorch)

## Licence

Licence proprietaire Lectura. Voir LICENCE.txt.
