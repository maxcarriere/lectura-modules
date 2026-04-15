# Attribution et crédits

Ce projet repose sur les ressources linguistiques et logiciels suivants.

## Données linguistiques

### GLAFF 1.2.1

- **Description** : dictionnaire flexionnel du français avec transcriptions phonémiques
- **Auteurs** : Franck Sajous, Nabil Hathout, Basilio Calderone (CLLE-ERSS, CNRS & Université Toulouse Jean Jaurès)
- **Licence** : CC BY-SA 3.0
- **URL** : http://redac.univ-tlse2.fr/lexicons/glaff.html
- **Usage** : données G2P (graphème → phonème) pour l'entraînement Phase 1 (lexique)

### Lexique 3.83

- **Description** : base lexicale du français avec fréquences, phonologie, morphologie
- **Auteurs** : Boris New, Christophe Pallier
- **Licence** : CC BY-SA 4.0
- **URL** : http://www.lexique.org/
- **Usage** : données G2P complémentaires et table de corrections

### Universal Dependencies — French-GSD 2.x

- **Description** : corpus annoté morpho-syntaxique du français
- **Auteurs** : Marie-Catherine de Marneffe, Bruno Guillaume, Ryan McDonald, et al.
- **Licence** : CC BY-SA 4.0
- **URL** : https://universaldependencies.org/treebanks/fr_gsd/index.html
- **Usage** : données d'entraînement Phase 2 (POS, morphologie, contexte phrastique), enrichies avec phonèmes et liaisons

## Logiciels et frameworks

### PyTorch

- **Licence** : BSD-3-Clause
- **URL** : https://pytorch.org/
- **Usage** : entraînement du modèle (non requis pour l'inférence)

### ONNX / ONNX Runtime

- **Licence** : Apache 2.0 / MIT
- **URL** : https://onnxruntime.ai/
- **Usage** : export et inférence optimisée (optionnel)

### NumPy

- **Licence** : BSD-3-Clause
- **URL** : https://numpy.org/
- **Usage** : backend d'inférence léger (optionnel)
