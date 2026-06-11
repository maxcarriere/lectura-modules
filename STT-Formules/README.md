# lectura-stt-formules

STT dedie formules — modele CTC autonome avec vocabulaire semantique
(~87 tokens : nombres atomiques, mois, lettres, marqueurs) au lieu de
phonemes IPA.

## Phase 1 — Generateur de corpus

Ce module fournit :

- Un vocabulaire de 87 tokens semantiques (`_vocab.py`)
- Un tokenizer events → token sequence (`_tokenizer.py`)
- Un generateur de corpus synthetique (`scripts/generate_corpus.py`)

## Installation

```bash
pip install lectura-stt-formules

# Pour la generation de corpus (necessite TTS)
pip install lectura-stt-formules[corpus]
```

## Utilisation

### Vocabulaire et tokenizer

```python
from lectura_stt_formules import VOCAB, events_to_token_sequence, token_ids_to_names
from lectura_formules import lire_nombre

result = lire_nombre("42")
tokens = events_to_token_sequence(result)
print(tokens)           # [22, 4]
print(token_ids_to_names(tokens))  # ['QUARANTE', 'DEUX']
```

### Generation de corpus

```bash
python scripts/generate_corpus.py \
    --output-dir /data/voix_ssd/formula_corpus/ \
    --n-base 16000 \
    --n-augmentations 3 \
    --seed 42 \
    --num-workers 4
```

## Licence

AGPL-3.0-or-later — voir [LICENCE.txt](LICENCE.txt)
