# Benchmark GEC debiaise — Rapport

Corpus : 180 phrases (158 erronees, 22 correctes)

## Scores globaux (micro-averaged)

| Outil | Precision | Rappel | F0.5 | F1 |
|-------|-----------|--------|------|-----|
| Lectura | 0.805 | 0.612 | 0.757 | 0.695 |
| Lec+Score | 0.782 | 0.633 | 0.747 | 0.700 |
| Grammalecte | 0.465 | 0.388 | 0.447 | 0.423 |
| Baseline | 1.000 | 0.000 | 0.000 | 0.000 |

## Faux positifs

| Outil | Modifiees | Total OK | % |
|-------|-----------|----------|---|
| Lectura | 10 | 22 | 45.5% |
| Lec+Score | 10 | 22 | 45.5% |
| Grammalecte | 9 | 22 | 40.9% |
| Baseline | 0 | 22 | 0.0% |

## Par categorie (F1)

| Categorie | Nb | Lectura | Lec+Score | Grammalecte | Baseline |
|-----------|---:|------:|------:|------:|------:|
| ORTH | 62 | 0.698 | 0.702 | 0.402 | 0.000 |
| ACCORD | 38 | 0.724 | 0.718 | 0.612 | 0.000 |
| CONJ | 24 | 0.788 | 0.824 | 0.388 | 0.000 |
| HOMO | 17 | 0.630 | 0.630 | 0.196 | 0.000 |
| AUTRE | 17 | 0.412 | 0.400 | 0.229 | 0.000 |
| OK | 22 | 0.833 | 0.833 | 0.545 | 0.000 |

## Temps d'execution

| Outil | Total | Par phrase |
|-------|-------|------------|
| Lectura | 74.3s | 413ms |
| Lec+Score | 83.9s | 466ms |
| Grammalecte | 17.3s | 96ms |
| Baseline | 0.0s | 0ms |
