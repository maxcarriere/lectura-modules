# Attribution

## Lectura P2G

Copyright (c) 2025 Lectura.

Distribue sous licence [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Donnees d'entrainement

Le modele a ete entraine sur des paires phoneme-grapheme derivees des
ressources suivantes :

### GLAFF (Grand Lexique Analyse du Francais Flexionnel)

- **Licence** : CC BY-SA 3.0
- **Source** : http://redac.univ-tlse2.fr/lexiques/glaff.html
- **Citation** :
  > Hathout, N., & Sajous, F. (2016).
  > GLAFF, un Gros Lexique A tout Faire du Francais.
  > *Traitement Automatique des Langues*, 57(2), 11-34.

### Lexique 3.83

- **Licence** : CC BY-SA 4.0
- **Source** : http://www.lexique.org/
- **Citation** :
  > New, B., Pallier, C., Brysbaert, M., & Ferrand, L. (2004).
  > Lexique 2: A New French Lexical Database.
  > *Behavior Research Methods, Instruments, & Computers*, 36, 516-524.

## Table P2G

La table P2G embarquee (3 493 entrees) est derivee de Lexique 3.83.
Elle associe les syllabes IPA a leur orthographe la plus frequente.

## Bibliotheques utilisees

- **PyTorch** (BSD) — Entrainement Seq2Seq
- **ONNX Runtime** (MIT) — Inference Seq2Seq
- **NumPy** (BSD) — Calculs numeriques
