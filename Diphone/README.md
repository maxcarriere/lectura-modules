# lectura-diphone

Synthese vocale francaise par concatenation de diphones dans le domaine WORLD.

## Installation

```bash
# Sans dependances (import seul)
pip install lectura-diphone

# Inference locale (pyworld + numpy + scipy)
pip install "lectura-diphone[local]"
```

> Pour le pipeline complet texte → audio, utilisez `pip install lectura-tts-dipho` (inclut le G2P).


## Utilisation

### Depuis du texte (necessite lectura-phonemiseur)

```python
from lectura_diphone import synthetiser

audio = synthetiser("Bonjour le monde")
# audio: numpy array float32, 44100 Hz
```

### Depuis des phonemes IPA

```python
from lectura_diphone import creer_engine

engine = creer_engine()
audio = engine.synthesize_groups([
    {"phones": ["b", "ɔ̃", "ʒ", "u", "ʁ"], "boundary": "none"},
    {"phones": ["l", "ə", "m", "ɔ̃", "d"], "boundary": "period"},
])
```

## Controles prosodiques

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| duration_scale | 1.0 | Vitesse globale (>1 = plus lent) |
| pause_scale | 1.0 | Duree des pauses inter-groupes |
| macro_expressivity | 2.0 | Gestes prosodiques (0=neutre, 4=exagere) |
| micro_expressivity | 5.0 | Micro-variations (0=robot, 10=tres expressif) |
| spectral_contrast | 1.5 | Contraste spectral (1.0=off, 2.0=fort) |
| prosody_style | "auto" | "declaratif", "question", "exclamation", "suspensif", "neutre" |
| seed | None | Graine pour micro-prosodie reproductible |

## Modes de synthese

- **FLUIDE** : lecture naturelle, enchainement continu
- **MOT_A_MOT** : lecture mot par mot avec pauses
- **SYLLABES** : lecture syllabe par syllabe

## Architecture

```
Texte → [G2P] → Phonemes IPA → Diphone chain
                                      ↓
                              WORLD params (F0 + SP + AP)
                                      ↓
                              Stretch + Concat (overlap)
                                      ↓
                              Prosodie (F0 contour + durees)
                                      ↓
                              GV compensation (contraste spectral)
                                      ↓
                              pw.synthesize → Audio 44100 Hz
```

Les diphones sont des parametres WORLD (F0 + spectral envelope + aperiodicity) extraits du corpus SIWIS et moyennes par type de transition phonetique.

## Emplacements des modeles

Recherche dans l'ordre :
1. Parametre `models_dir` explicite
2. `$LECTURA_MODELS_DIR/tts_diphone/`
3. `~/.lectura/models/tts_diphone/`
4. Modeles embarques dans le package

Fichier requis : `diphones.dpk.gz` (ou `.dpk.gz.enc` chiffre)
Fichier optionnel : `diphone_statistics.pkl`

## Licence

Double licence : AGPL-3.0 (code) + [Licence Commerciale](mailto:admin@lectura.world) (modeles).
