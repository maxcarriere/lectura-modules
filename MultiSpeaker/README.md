# lectura-multispeaker

Moteur de synthese vocale neuronale multi-speaker pour le francais — Matcha-Conformer + HiFi-GAN (ONNX).

Supporte 6 voix : siwis, ezwa, nadine, bernard, gilles, zeckou.

## Installation

```bash
# Version minimale (API distante, zero deps)
pip install lectura-multispeaker

# Version locale (inference ONNX)
pip install lectura-multispeaker[onnx]
```

> Pour le pipeline complet texte → audio, utilisez `pip install lectura-tts-multi` (inclut le G2P).

## Utilisation

### Depuis des phonemes IPA

```python
from lectura_multispeaker import creer_engine

engine = creer_engine(mode="local")
engine.set_speaker("siwis")
result = engine.synthesize_phonemes(
    "bɔ̃ʒuʁ",
    phrase_type=0,
)
# result.samples: numpy float32
# result.sample_rate: 22050
# result.phoneme_timings: list[PhonemeTiming]
```

### Lister les speakers disponibles

```python
from lectura_multispeaker import liste_speakers
print(liste_speakers())  # ['siwis', 'ezwa', 'nadine', 'bernard', 'gilles', 'zeckou']
```

## Controles prosodiques

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| duration_scale | 1.0 | Vitesse globale |
| pitch_shift | 0.0 | Decalage F0 (demi-tons) |
| pitch_range | 1.0 | Variation F0 |
| energy_scale | 1.0 | Intensite |
| pause_scale | 1.0 | Duree des pauses |
| phrase_type | 0 | 0=decl, 1=inter, 2=excl, 3=susp |
| n_ode_steps | 4 | Pas ODE Matcha (plus = meilleur) |

## Architecture

- **Matcha-Conformer** : phonemes → mel-spectrogramme via flow matching ODE (encodeur par speaker)
- **HiFi-GAN V1** : mel → audio 22050 Hz
- **Runtime** : ONNX (pas de dependance PyTorch)

## Emplacements des modeles

Recherche dans l'ordre :
1. Parametre `models_dir` explicite
2. `$LECTURA_MODELS_DIR/tts_multispeaker/`
3. `~/.lectura/models/tts_multispeaker/`
4. Modeles embarques dans le package (version privee)

## Licence

Licence proprietaire Lectura. Voir LICENCE.txt.
