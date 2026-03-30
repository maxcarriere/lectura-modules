# Lectura P2G

**Convertisseur phoneme-grapheme pour le francais — IPA vers orthographe**

Un module pour convertir des transcriptions IPA en orthographe francaise,
avec trois strategies complementaires : modele Seq2Seq + beam search,
table de lookup lexicale, et regles deterministes.

---

## Demarrage rapide

```python
from lectura_p2g import LecturaP2G

# Sans modele (table + regles, zero dependance)
p2g = LecturaP2G()
p2g.predict("bɔ̃ʒuʁ")          # → "bonjour"
p2g.predict_syllable("kɑ̃")     # → "quand"

# Avec modele Seq2Seq (beam search, top-K candidates)
p2g = LecturaP2G(model_dir="modele/")
p2g.predict("pɛʃœʁ")           # → "pêcheur"
p2g.predict_candidates("pɛʃœʁ", k=5)
# → [("pêcheur", 0.42), ("pêcheurs", 0.28), ("pécheur", 0.15), ...]
```

### Pre-requis

- Python 3.10+ (table + regles : zero dependance)
- `onnxruntime` + `numpy` (pour le modele Seq2Seq)

```bash
pip install onnxruntime numpy
```

### Contenu de l'archive

```
P2G/
├── lectura_p2g.py              ← Module principal (copier dans votre projet)
├── demo_cli.py                 ← Demo en ligne de commande
├── modele/
│   ├── p2g_seq2seq_v5_encoder_int8.onnx
│   ├── p2g_seq2seq_v5_decoder_int8.onnx
│   └── p2g_seq2seq_v5_vocab.json
├── exemples/
│   ├── exemple_basique.py      ← Utilisation table + regles
│   └── exemple_integration.py  ← Utilisation avec modele Seq2Seq
├── README.md
├── LICENCE.txt
└── ATTRIBUTION.md
```

---

## Utilisation

### Dans votre code

Copiez `lectura_p2g.py` (et optionnellement `modele/`) dans votre projet :

```python
from lectura_p2g import LecturaP2G

# Option 1 : table + regles uniquement (zero dependance)
p2g = LecturaP2G()

# Option 2 : avec modele Seq2Seq
p2g = LecturaP2G(model_dir="chemin/vers/modele/")

# Option 3 : table externe
p2g = LecturaP2G(table_path="chemin/vers/p2g_table.json")

# Prediction simple
ortho = p2g.predict("bɔ̃ʒuʁ")   # → "bonjour"

# Top-K candidates avec probabilites
candidates = p2g.predict_candidates("vɛʁ", k=5)
# → [("vert", 0.35), ("verre", 0.25), ("vers", 0.20), ("ver", 0.12), ...]

# Lookup par syllabe (rapide, table uniquement)
ortho = p2g.predict_syllable("tʁa")   # → "tra"
```

### Demo en ligne de commande

```bash
# Demo avec mots courants
python demo_cli.py

# Mots specifiques
python demo_cli.py bɔ̃ʒuʁ pɛʃœʁ ɑ̃fɑ̃

# Avec modele
python demo_cli.py --model modele/ bɔ̃ʒuʁ pɛʃœʁ

# Mode interactif
python demo_cli.py --interactive
```

---

## Trois strategies

| Strategie | Precision | Vitesse | Dependances | Multi-candidats |
|-----------|-----------|---------|-------------|-----------------|
| **Seq2Seq + beam** | Haute | ~5 ms/mot | onnxruntime | Oui (top-K) |
| **Table** | Moyenne | <0.1 ms | Aucune | Non |
| **Regles** | Basse | <0.1 ms | Aucune | Non |

### Modele Seq2Seq

- Architecture : BiLSTM encoder (256h×2) + LSTM decoder (512h) + attention
- Beam search avec normalisation par longueur
- ~6M parametres, ~2.5 Mo INT8
- Retourne K candidates avec probabilites normalisees

### Table P2G

- 3 493 entrees (syllabes IPA → orthographe la plus frequente)
- Derivee de Lexique 3.83
- Embarquee dans le module (zlib + base85, ~24 Ko)

### Regles deterministes

- Mapping phoneme → grapheme (ex: ʃ → ch, ɛ → e, ɑ̃ → an)
- Fallback universel pour les sequences hors vocabulaire

---

## API de reference

### `LecturaP2G(model_dir=None, table_path=None, beam_width=5)`

Cree un convertisseur P2G.

- `model_dir` : repertoire contenant les fichiers ONNX + vocab JSON
- `table_path` : chemin vers un fichier p2g_table.json externe
- `beam_width` : largeur du beam search par defaut

### `p2g.predict(ipa) → str`

Retourne la meilleure orthographe pour une chaine IPA.

### `p2g.predict_candidates(ipa, k=5) → list[tuple[str, float]]`

Retourne les K meilleures orthographes avec probabilites normalisees.
Necessite le modele Seq2Seq pour des resultats multiples.

### `p2g.predict_syllable(ipa_syllable) → str`

Lookup rapide pour une syllabe IPA (table ou regles, pas de modele).

### `p2g.has_model → bool`

True si un modele Seq2Seq est charge.

### Fonctions utilitaires

- `iter_phonemes(ipa) → list[str]` : segmente une chaine IPA en phonemes

---

## Cas d'usage

### Desambiguisation d'homophones

Le beam search genere plusieurs candidates pour les sequences ambigues.
En combinant P2G avec un modele de langue, on peut choisir le bon homophone :

```python
# "vɛʁ" → vert, verre, vers, ver...
candidates = p2g.predict_candidates("vɛʁ", k=5)
# Utiliser un LM pour scorer chaque candidat en contexte
```

### Sous-titrage phonetique

Convertir des transcriptions phonetiques en texte lisible pour
l'affichage de sous-titres ou l'aide a la lecture.

### Evaluation de modeles G2P

Verifier la qualite d'un modele G2P en faisant l'aller-retour :
texte → G2P → IPA → P2G → texte, et comparer avec l'original.

---

*Copyright (c) 2025 Lectura. Licence CC BY-SA 4.0. Voir LICENCE.txt et ATTRIBUTION.md.*
