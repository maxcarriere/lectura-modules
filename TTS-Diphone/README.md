# lectura-tts-diphone

Synthese vocale francaise par concatenation de diphones dans le domaine WORLD.

## Installation

```bash
# Sans dependances (import seul, API distante)
pip install lectura-tts-diphone

# Inference locale (pyworld + numpy + scipy)
pip install "lectura-tts-diphone[local]"

# Avec G2P integre
pip install "lectura-tts-diphone[all]"
```

## Usage rapide

```python
from lectura_tts_diphone import creer_engine, synthetiser

# Synthese directe texte → audio
audio = synthetiser("Bonjour le monde")

# Ou via engine pour controle fin
engine = creer_engine()
audio = engine.synthesize_groups([
    {"phones": ["b", "o~", "Z", "u", "R"], "boundary": "none"},
])
```

## Modes de synthese

- **FLUIDE** : lecture naturelle, enchainement continu
- **MOT_A_MOT** : lecture mot par mot avec pauses
- **SYLLABES** : lecture syllabe par syllabe

## Architecture

Pipeline : texte → G2P → phones IPA → diphone chain → WORLD concat → pw.synthesize → audio 44.1 kHz

Les diphones sont des parametres WORLD (F0 + spectral envelope + aperiodicity) extraits du corpus SIWIS et moyennes par type de transition phonetique.

## Emplacements des modeles

Recherche dans l'ordre :
1. Parametre `models_dir` explicite
2. `$LECTURA_MODELS_DIR/tts_diphone/`
3. `~/.lectura/models/tts_diphone/`
4. Modeles embarques dans le package

Fichier requis : `diphones.dpk.gz` (ou `.dpk.gz.enc` chiffre)
Fichier optionnel : `diphone_statistics.pkl`
