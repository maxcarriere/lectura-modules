# lectura-multispeaker

Moteur de synthese vocale neuronale multi-speaker pour le francais — deux modeles au choix : **high** (Matcha-Conformer) et **light** (FastPitch) + HiFi-GAN (ONNX).

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

# Modele high (Conformer, meilleure qualite) — par defaut
engine = creer_engine(mode="local", speaker="siwis")
result = engine.synthesize_phonemes("bɔ̃ʒuʁ", phrase_type=0)

# Modele light (FastPitch, plus rapide/leger)
engine_light = creer_engine(mode="local", speaker="siwis", model="light")
result = engine_light.synthesize_phonemes("bɔ̃ʒuʁ")
```

### Raccourci synthetiser()

```python
from lectura_multispeaker import synthetiser

# High (defaut)
audio = synthetiser("Bonjour.", speaker="bernard")

# Light
audio = synthetiser("Bonjour.", speaker="bernard", model="light")
```

### Lister les speakers disponibles

```python
from lectura_multispeaker import liste_speakers
print(liste_speakers())  # ['siwis', 'ezwa', 'nadine', 'bernard', 'gilles', 'zeckou']
```

## Modeles

| Modele | Architecture | Taille (INT8) | Qualite | Vitesse |
|--------|-------------|---------------|---------|---------|
| **high** (defaut) | Matcha-Conformer (d=384) + HiFi-GAN | ~40 Mo | Meilleure | ~30x temps-reel |
| **light** | FastPitch (d=256) + HiFi-GAN | ~40 Mo | Bonne | ~50x temps-reel |

## Controles prosodiques

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| duration_scale | 1.0 | Vitesse globale |
| pitch_shift | 0.0 | Decalage F0 (demi-tons) |
| pitch_range | 1.0 | Variation F0 |
| energy_scale | 1.0 | Intensite |
| pause_scale | 1.0 | Duree des pauses |
| phrase_type | 0 | 0=decl, 1=inter, 2=excl, 3=susp |
| n_ode_steps | 4 | Pas ODE Matcha (plus = meilleur, high uniquement) |

## Architecture

- **high** : Matcha-Conformer (d_model=384) — phonemes → mel via flow matching ODE (encodeur par speaker)
- **light** : FastPitch (d_model=256) — phonemes → mel via FFT decoder (encodeur par speaker)
- **HiFi-GAN V1** : mel → audio 22050 Hz (partage)
- **Runtime** : ONNX (pas de dependance PyTorch)

## Emplacements des modeles

Recherche dans l'ordre :
1. Parametre `models_dir` explicite
2. `$LECTURA_MODELS_DIR/tts_multispeaker/`
3. `~/.lectura/models/tts_multispeaker/`
4. Modeles embarques dans le package (version privee)

Chaque emplacement peut contenir des sous-repertoires `conformer/` et `fastpitch/` (nouveau layout) ou les fichiers directement (ancien layout, retrocompatible).

## Licence

Code source sous licence **AGPL-3.0** — voir [LICENCE.txt](LICENCE.txt).
Modeles pre-entraines : voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

**Licence commerciale et modeles locaux disponibles** — contacter [admin@lectura.world](mailto:admin@lectura.world).
