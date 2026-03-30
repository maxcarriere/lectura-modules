---
title: Lectura G2P Unifié
emoji: "\U0001F1EB\U0001F1F7"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: cc-by-sa-4.0
---

# Lectura G2P Unifié

Démonstration interactive du modèle unifié G2P + POS + Morphologie + Liaison pour le français.

## Déploiement

Ce Space nécessite les fichiers suivants dans le répertoire :

```
app.py                      # Application Gradio
requirements.txt            # Dépendances
lectura_nlp/                # Copie du package src/lectura_nlp/
modeles/                    # Copie des fichiers modèle
  unifie_weights.json
  unifie_vocab.json
  g2p_corrections_unifie.json
```

Pour déployer, copier `src/lectura_nlp/` et `modeles/` dans ce répertoire.
