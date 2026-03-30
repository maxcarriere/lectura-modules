# Lectura G2P — CRF

**Convertisseur grapheme-phoneme pour le francais — backend CRF**

Un module leger et autonome pour la transcription phonetique IPA du francais.
Zero dependance externe — un seul fichier Python, pret a l'emploi.

---

## Demarrage rapide

```python
from lectura_g2p import LecturaG2P

g2p = LecturaG2P("modele/g2p_model_crf.json",
                   corrections_path="modele/g2p_corrections_crf.json")
phone = g2p.predict("bonjour")    # → "bɔ̃ʒuʁ"
phone = g2p.predict("maison")     # → "mɛzɔ̃"
phone = g2p.predict("ordinateur") # → "ɔʁdinatœʁ"
```

### Pre-requis

- Python 3.10+
- Aucune bibliotheque externe requise

### Contenu de l'archive

```
G2P_CRF/
├── lectura_g2p.py                       ← Fichier principal (copier dans votre projet)
├── demo_cli.py                          ← Demo en ligne de commande
├── evaluer.py                           ← Script d'evaluation
├── modele/
│   ├── g2p_model_crf.json              ← Modele CRF (747 Ko)
│   └── g2p_corrections_crf.json        ← Corrections CRF (7 996 g2p + 402 g2p_pos)
├── exemples/
│   ├── exemple_basique.py
│   └── exemple_integration.py
├── entrainement/
│   ├── entrainer_crf.py
│   ├── preparer_dataset.py
│   └── README.md
├── README.md
├── EVALUATION.md
├── LICENCE.txt
└── ATTRIBUTION.md
```

---

## Utilisation

### Dans votre code

Copiez `lectura_g2p.py` et `modele/` dans votre projet :

```python
from lectura_g2p import LecturaG2P

g2p = LecturaG2P("chemin/vers/g2p_model_crf.json",
                   corrections_path="chemin/vers/g2p_corrections_crf.json")

# Transcription simple
phone = g2p.predict("chat")        # → "ʃa"

# Avec POS (desambiguisation des homographes)
phone = g2p.predict("est", pos="AUX")   # → "ɛ"
phone = g2p.predict("est", pos="NOM")   # → "ɛst"

# Traitement par lot
phones = g2p.predict_batch(["Le", "chat", "mange"])
# → ["lə", "ʃa", "mɑ̃ʒ"]

# Affichage formate
print(g2p.predict_formatted("extraordinaire"))
# → extraordinaire → /ɛkstʁaɔʁdinɛʁ/
```

### Demo en ligne de commande

```bash
# Phrase en argument
python demo_cli.py "Bonjour le monde"

# Mode interactif
python demo_cli.py
```

---

## Modele disponible

| Modele | Fichier | Taille | Dependances | Precision @99% |
|--------|---------|--------|-------------|----------------|
| **CRF** | `g2p_model_crf.json` | 747 Ko | Aucune | ~100% |

**Precision @99%** = pourcentage de mots corrects parmi les ~14 000 mots couvrant 99%
des occurrences en texte (modele + corrections + post-traitement).

### Corrections

| Table | Entrees g2p | Entrees g2p_pos |
|-------|-------------|-----------------|
| `g2p_corrections_crf.json` | 7 996 | 402 |

Les corrections `g2p_pos` gerent les homographes dont la prononciation depend
de la classe grammaticale (ex: "est" NOM vs AUX, "plus" ADV vs NOM).

---

## Post-traitement (regles R1-R13)

Le modele est suivi d'un post-traitement systematique qui corrige
les erreurs recurrentes :

| Regle | Description | Exemple |
|-------|-------------|---------|
| R1 | x → ks/ɡz | taxi → taksi, examen → ɛɡzamɛ̃ |
| R2 | -enne → ɛn | parisienne → paʁizjɛn |
| R3 | -isme → izm | realisme → ʁealizm |
| R4 | -tion → sjɔ̃ | nation → nasjɔ̃ (sauf verbes) |
| R5 | -ent verbal → muet | mangent → mɑ̃ʒ |
| R7 | ex- → ɛɡz/ɛks | inexact → inɛɡzakt |
| R8 | Hiatus /ij/ | oublier → ublije |
| R9 | bs→ps, bt→pt | observer → ɔpseʁve |
| R10 | /ɡ/ initial | grand → ɡʁɑ̃ |
| R11 | oe/oeu → œ/ø | coeur → kœʁ |
| R13 | ø → œ fermee | seul → sœl |

---

## Details techniques

### Architecture

- **CRF** : Conditional Random Fields avec decodage Viterbi pur Python
- **Entrainement** : alignements grapheme-phoneme derives du GLAFF

### Pipeline

```
mot → corrections → (si absent) → modele CRF → post-traitement R1-R13 → IPA
```

### Limites

- Optimise pour le francais contemporain standard
- Les noms propres et emprunts non-francais peuvent etre mal transcrits
- La desambiguisation POS-aware necessite un POS tagger en amont
- Les mots composes (avec tiret) ne sont pas geres directement

---

## API de reference

### `LecturaG2P(model_path, corrections_path=None)`

Cree un convertisseur G2P CRF.

### `g2p.predict(word, pos=None) → str`

Transcrit un mot en IPA. Le POS optionnel permet la desambiguisation.

### `g2p.predict_batch(words, pos_tags=None) → list[str]`

Transcrit une liste de mots en IPA.

### `g2p.predict_formatted(word, pos=None) → str`

Retourne "mot → /phone/".

### `g2p.backend → str`

Retourne le type de backend : "crf".

### Fonctions utilitaires

- `iter_phonemes(ipa) → list[str]` : decompose une chaine IPA en phonemes
- `postprocess(word, phone, pos=None) → str` : applique les regles R1-R13
- `est_voyelle(ph)`, `est_consonne(ph)`, `est_semi_voyelle(ph)` : classification

---

## Support

Pour toute question : [contact a definir]

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0. Voir LICENCE.txt et ATTRIBUTION.md.*
