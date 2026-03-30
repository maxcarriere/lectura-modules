# Outils NLP — Lectura

Collection de modèles et outils NLP pour le français, développés dans le cadre du projet Lectura.

## Modèles simples

Modules spécialisés pour une seule tâche, avec différentes architectures (CRF, BiLSTM, Seq2Seq).

| Module | Description | Architectures |
|--------|-------------|---------------|
| [G2P](Modeles_simples/G2P/) | Graphème → Phonème (IPA) | CRF, BiLSTM, Seq2Seq |
| [P2G](Modeles_simples/P2G/) | Phonème → Graphème | CRF, BiLSTM, Seq2Seq |
| [POS](Modeles_simples/POS/) | Étiquetage morpho-syntaxique | CRF, BiLSTM |
| [MORPHO](Modeles_simples/MORPHO/) | Analyse morphologique | CRF, BiLSTM |
| [Liaisons](Modeles_simples/Liaisons/) | Prédiction des liaisons | Règles + modèle |
| [Syllabeur](Modeles_simples/Syllabeur/) | Syllabification | Règles |
| [Tokeniseur](Modeles_simples/Tokeniseur/) | Tokenisation française | Règles |

## Modèles complets

Modèles multi-tâches combinant plusieurs capacités en un seul réseau.

| Module | Description | Taille |
|--------|-------------|--------|
| [G2P Unifié](Modeles_complets/G2P/) | G2P + POS + Morpho + Liaison (BiLSTM multi-tête) | 1.8 Mo (INT8) |
| [P2G Complet](Modeles_complets/P2G/) | Phonème → Graphème complet | — |

## Licence

CC BY-SA 4.0 — Voir les fichiers LICENCE.txt dans chaque module.
