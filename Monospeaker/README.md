# lectura-monospeaker

Moteur de synthese vocale neuronale monospeaker pour le francais — deux modeles au choix : **high** (Matcha-Conformer) et **light** (FastPitch) + HiFi-GAN (ONNX).

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

# Modele high (Conformer, meilleure qualite) — par defaut
engine = creer_engine(mode="local")
result = engine.synthesize_phonemes(
    "bɔ̃ʒuʁ",
    phrase_type=0,
    pitch_range=1.3,
)

# Modele light (FastPitch, plus rapide/leger)
engine_light = creer_engine(mode="local", model="light")
result = engine_light.synthesize_phonemes("bɔ̃ʒuʁ")
```

### Raccourci synthetiser()

```python
from lectura_monospeaker import synthetiser

# High (defaut)
audio = synthetiser("Bonjour le monde.")

# Light
audio = synthetiser("Bonjour le monde.", model="light")
```

### Via l'API distante

```python
from lectura_monospeaker import creer_engine

engine = creer_engine(mode="api", api_url="https://api.lectura.world")
result = engine.synthesize("Bonjour")
```

## Modeles

| Modele | Architecture | Taille (INT8) | Qualite | Vitesse |
|--------|-------------|---------------|---------|---------|
| **high** (defaut) | Matcha-Conformer + HiFi-GAN | ~29 Mo | Meilleure | ~30x temps-reel |
| **light** | FastPitch + HiFi-GAN | ~28 Mo | Bonne | ~50x temps-reel |

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

- **high** : Matcha-Conformer — phonemes → mel-spectrogramme via flow matching ODE
- **light** : FastPitch — phonemes → mel-spectrogramme via FFT decoder
- **HiFi-GAN V1** : mel → audio 22050 Hz (partage)
- **Runtime** : ONNX (pas de dependance PyTorch)

## Emplacements des modeles

Recherche dans l'ordre :
1. Parametre `models_dir` explicite
2. `$LECTURA_MODELS_DIR/tts_mono/`
3. `~/.lectura/models/tts_mono/`
4. Modeles embarques dans le package (version privee)

Chaque emplacement peut contenir des sous-repertoires `conformer/` et `fastpitch/` (nouveau layout) ou les fichiers directement (ancien layout, retrocompatible).

## Licence

Licence proprietaire Lectura. Voir LICENCE.txt.
