---
language:
  - fr
license: cc-by-sa-4.0
tags:
  - g2p
  - grapheme-to-phoneme
  - pos-tagging
  - morphology
  - liaison
  - french
  - bilstm
  - onnx
pipeline_tag: token-classification
metrics:
  - accuracy
model-index:
  - name: lectura-g2p-unifie
    results:
      - task:
          type: token-classification
          name: Grapheme-to-Phoneme
        dataset:
          type: universal-dependencies
          name: UD French-GSD (enrichi)
        metrics:
          - type: accuracy
            value: 98.5
            name: Word Accuracy
          - type: per
            value: 0.54
            name: Phone Error Rate
      - task:
          type: token-classification
          name: POS Tagging
        dataset:
          type: universal-dependencies
          name: UD French-GSD
        metrics:
          - type: accuracy
            value: 98.2
      - task:
          type: token-classification
          name: Liaison Prediction
        dataset:
          type: universal-dependencies
          name: UD French-GSD (enrichi)
        metrics:
          - type: f1
            value: 90.6
            name: Macro F1
---

# Lectura G2P Unifié

Modèle unifié **G2P + POS + Morphologie + Liaison** pour le français.

## Description

BiLSTM char-level multi-tête (1.75M paramètres) qui prédit simultanément :

- **G2P** : transcription phonémique IPA (98.5% word accuracy)
- **POS** : étiquetage morpho-syntaxique (98.2% accuracy)
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale (95-99%)
- **Liaison** : prédiction des liaisons obligatoires/facultatives (F1 90.6%)

## Taille

| Format | Taille |
|--------|--------|
| ONNX INT8 | **1.8 Mo** |
| Paramètres | 1.75M |

## Utilisation

```python
from lectura_nlp.inference_numpy import NumpyInferenceEngine
from lectura_nlp.tokeniseur import tokeniser

engine = NumpyInferenceEngine("modeles/unifie_weights.json",
                              "modeles/unifie_vocab.json")
result = engine.analyser(tokeniser("Les enfants sont arrivés."))

print(result["g2p"])      # ['le', 'ɑ̃fɑ̃', 'sɔ̃', 'aʁive']
print(result["pos"])      # ['ART:def', 'NOM', 'AUX', 'VER']
print(result["liaison"])  # ['Lz', 'none', 'Lt', 'none']
```

## Backends d'inférence

Trois backends au choix (résultats identiques) :

| Backend | Dépendances | Vitesse |
|---------|------------|---------|
| ONNX Runtime | `onnxruntime` | ~2 ms/phrase |
| NumPy | `numpy` | ~50 ms/phrase |
| Pur Python | aucune | ~200 ms/phrase |

## Entraînement

- **Phase 1** : pré-entraînement G2P sur 890K mots (GLAFF + Lexique)
- **Phase 2** : fine-tuning multi-tâche sur 18K phrases UD French-GSD enrichi

## Données d'entraînement

- [GLAFF 1.2.1](http://redac.univ-tlse2.fr/lexicons/glaff.html) (CC BY-SA 3.0)
- [Lexique 3.83](http://www.lexique.org/) (CC BY-SA 4.0)
- [UD French-GSD](https://universaldependencies.org/treebanks/fr_gsd/) (CC BY-SA 4.0)

## Limites

- Performance réduite sur mots isolés hors-vocabulaire (~92%)
- Liaison `Lp` rare dans les données (F1 = 66.7%)
- Noms propres et néologismes : transcriptions approximatives

## Licence

CC BY-SA 4.0
