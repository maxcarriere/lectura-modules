# Lectura STT — Pipeline STT complet du francais

Pipeline de transcription automatique du francais : audio vers texte.
Chaine le decodeur CTC medium (10.6M params, PER ~4.34%) avec le pipeline P2G (phones → orthographe).

WER benchmark : ~23.5% (all) / ~19.7% (parole courante).

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

### Pipeline optimal (avec PhoneLexicon)

Lorsqu'un `PhoneLexicon` est disponible (via le graphemiseur), le pipeline
optimal est active automatiquement :

```
Audio 16kHz mono
     |
     v
[lectura-ctc]       --> IPA phones "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d ."
     |
     v
[parse_ctc_v2]      --> segments enrichis (mots, liaisons, composes, ponctuation)
     |
     v
[strip_liaisons]    --> supprime les liaisons erronees (via lexique phonetique)
     |
     v
[split_elisions]    --> separe les clitiques elides (l'ami → l + ami)
     |
     v
[split_merged_words] --> decoupe les mots sur-segmentes
     |
     v
[P2G analyser_v2]   --> conversion IPA → orthographe avec lex_select
     |
     v
[merge_and_rescore]  --> fusionne les mots sur-segmentes (rescoring lexical)
     |
     v
[try_elision_merges] --> fusionne les clitiques elides adjacents
     |
     v
[rejoin_elisions]    --> reconstruction texte final avec apostrophes et tirets
     |
     v
"Bonjour le monde."
```

### Pipeline simplifie (sans PhoneLexicon)

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
