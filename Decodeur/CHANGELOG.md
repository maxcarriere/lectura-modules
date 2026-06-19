# Changelog — lectura-ctc

## v2.0.0 (2026-06-08)

- Modele medium (10.6M params) remplace le small (3.5M)
- Architecture CNN [48, 96] + BiGRU 384x4 (vs [32, 64] + 256x3)
- PER ~4.34% (vs ~6% pour le small)
- Support des sigles et formules (fine-tuning specialise v2)
- Vocabulaire 59 tokens (inchange)
- ONNX INT8 : 38 Mo (vs 13 Mo)

## v1.0.1 (2026-06-04)

- Ajout CLI : `python -m lectura_ctc` (fichier WAV ou micro)
- Fix mel spectrogram edge cases

## v1.0.0 (2026-05-11)

- Version initiale
- Modele small (3.5M params, PER ~6%)
- Backends ONNX Runtime et API
- Mel spectrogram numpy pur
- Decodage CTC greedy
