# lectura-monospeaker

Moteur de synthese vocale neuronale monospeaker pour le francais — Matcha-Conformer + HiFi-GAN (ONNX).

## Installation

```bash
# Version minimale (API distante, zero deps)
pip install lectura-monospeaker

# Version locale (inference ONNX)
pip install lectura-monospeaker[onnx]
```

> Pour le pipeline complet texte → audio, utilisez `pip install lectura-tts-mono` (inclut le G2P).

## Utilisation

### Depuis des phonemes IPA

```python
from lectura_monospeaker import creer_engine

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
from lectura_monospeaker import creer_engine

engine = creer_engine(mode="api", api_url="https://api.lectura.world")
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

- **Matcha-Conformer** : phonemes → mel-spectrogramme via flow matching ODE
- **HiFi-GAN V1** : mel → audio 22050 Hz
- **Runtime** : ONNX (pas de dependance PyTorch)

## Emplacements des modeles

Recherche dans l'ordre :
1. Parametre `models_dir` explicite
2. `$LECTURA_MODELS_DIR/tts_mono/`
3. `~/.lectura/models/tts_mono/`
4. Modeles embarques dans le package (version privee)

## Licence

Licence proprietaire Lectura. Voir LICENCE.txt.
