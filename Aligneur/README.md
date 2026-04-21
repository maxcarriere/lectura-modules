# Lectura Aligneur-Syllabeur

**Aligneur grapheme-phoneme et syllabeur phonologique du francais**

Module autonome, **zero dependance** Python. Pivot central du pipeline Lectura, il realise l'alignement lettre-par-lettre entre orthographe et phonetique, construit les groupes de lecture en gerant les phenomenes de chaine parlee (elisions, liaisons, enchainements), et decompose chaque syllabe en attaque/noyau/coda avec correspondance grapheme-phoneme.

C'est grace a cet alignement que les corpus d'entrainement des modeles G2P et P2G ont ete prepares.

## Installation

```bash
pip install lectura-aligneur
```

## Fonctionnalites

| Fonction | Description |
|----------|-------------|
| **Alignement grapheme-phoneme** | Correspondance lettre-par-lettre entre orthographe et IPA via algorithme DFS |
| **Lettres muettes** | Detection et marquage des lettres silencieuses (e, s, t, d, h, x...) |
| **Lettres fusionnees** | Identification des graphemes multi-phonemes (x = ks/gz, y...) |
| **Groupes de lecture** | Regroupement des mots lies par elision, liaison ou enchainement |
| **Syllabation phonologique** | Decoupage en syllabes par modele de sonorite (IPA + orthographe) |
| **Decomposition attaque/noyau/coda** | Chaque syllabe decomposee en ses constituants avec phonemes distribues |
| **Spans** | Positions caractere de chaque syllabe, groupe et composant dans le texte source |
| **Phonemiseur pluggable** | eSpeak-NG (defaut), Lectura G2P, ou tout objet avec `.phonemize()` / `.predict()` |
| **Formules** | Gestion des lectures de formules (nombres, dates, etc.) avec events alignes |

## Exemples

### Analyse d'un mot (API simple)

```python
from lectura_aligneur import LecturaSyllabeur

syllabeur = LecturaSyllabeur()    # eSpeak-NG par defaut
result = syllabeur.analyze("chocolat")

print(result.format_detail())
# chocolat -> /ʃɔkɔla/
#   σ1: /ʃɔ/ <<cho>> [0:3] att=ʃ noy=ɔ cod=-
#   σ2: /kɔ/ <<co>>  [3:5] att=k noy=ɔ cod=-
#   σ3: /la/ <<lat>>  [5:8] att=l noy=a cod=-

# Chaque syllabe expose son alignement :
for s in result.syllabes:
    att = " ".join(f"{p.ipa}→{p.grapheme}" for p in s.attaque.phonemes)
    noy = " ".join(f"{p.ipa}→{p.grapheme}" for p in s.noyau.phonemes)
    cod = " ".join(f"{p.ipa}→{p.grapheme}" for p in s.coda.phonemes)
    print(f"  {s.ortho:6s} /{s.phone}/  att=[{att}] noy=[{noy}] cod=[{cod}]  span={s.span}")
# cho    /ʃɔ/  att=[ʃ→ch] noy=[ɔ→o] cod=[]  span=(0, 3)
# co     /kɔ/  att=[k→c]  noy=[ɔ→o] cod=[]  span=(3, 5)
# lat    /la/  att=[l→l]  noy=[a→a] cod=[]   span=(5, 8)
```

### Analyse complete avec groupes de lecture

L'API `analyser_complet()` prend une liste de `MotAnalyse` (produite par le G2P) et construit les groupes de lecture en appliquant liaisons, enchainements et elisions, puis syllabe chaque groupe :

```python
from lectura_aligneur import LecturaSyllabeur, MotAnalyse

syllabeur = LecturaSyllabeur()

# Mots annotes par le G2P (phone + liaison)
mots = [
    MotAnalyse(phone="lez", liaison="Lz"),   # les (liaison en z)
    MotAnalyse(phone="ɑ̃fɑ̃", liaison="none"),  # enfants
    MotAnalyse(phone="ʒu",  liaison="none"),   # jouent
]

result = syllabeur.analyser_complet(mots)

print(f"{result.nb_groupes} groupes, {result.nb_syllabes} syllabes")
print(f"Groupes : {result.format_ligne1()}")
print(f"Syllabes : {result.format_ligne2()}")

# 2 groupes, 4 syllabes
# Groupes : les enfants | jouent
# Syllabes : le.zɑ̃.fɑ̃ | ʒu

# Detail des groupes :
for rg in result.groupes:
    g = rg.groupe
    jonc = ", ".join(g.jonctions) if g.jonctions else "-"
    print(f"  Groupe: /{g.phone_groupe}/  jonctions: {jonc}")
    for s in rg.syllabes:
        print(f"    σ /{s.phone}/ <<{s.ortho}>>  att={s.attaque.phone or '-'} noy={s.noyau.phone} cod={s.coda.phone or '-'}")
```

### API IPA directe (sans phonemiseur)

```python
syllabeur = LecturaSyllabeur()
syllabes = syllabeur.syllabify_ipa("ʃɔkɔla")
print(syllabes)  # ['ʃɔ', 'kɔ', 'la']
```

### Avec Lectura G2P comme phonemiseur

```python
from lectura_aligneur import LecturaSyllabeur
from lectura_nlp.inference_onnx import OnnxInferenceEngine
from lectura_nlp import get_model_path

g2p = OnnxInferenceEngine(get_model_path("unifie_int8.onnx"),
                           get_model_path("unifie_vocab.json"))

# Tout objet avec .predict(word) est accepte
class G2PPhonemizer:
    def predict(self, word):
        return g2p.analyser([word])['g2p'][0]

syllabeur = LecturaSyllabeur(phonemizer=G2PPhonemizer())
result = syllabeur.analyze("maison")
```

## Structures de donnees

| Classe | Role |
|--------|------|
| `ResultatAnalyse` | Analyse d'un mot : `mot`, `phone`, `syllabes[]` |
| `Syllabe` | Syllabe : `phone`, `ortho`, `span`, `attaque`, `noyau`, `coda` |
| `GroupePhonologique` | Attaque/noyau/coda : liste de `Phoneme` avec `ipa` + `grapheme` |
| `Phoneme` | Phoneme individuel : `ipa` (ex: "ʃ") + `grapheme` (ex: "ch") |
| `MotAnalyse` | Mot annote : `token`, `phone`, `liaison`, `pos` |
| `GroupeLecture` | Groupe de lecture : `mots[]`, `phone_groupe`, `jonctions[]`, `span` |
| `ResultatGroupe` | Groupe syllabe : `groupe` + `syllabes[]` |
| `ResultatSyllabation` | Resultat complet : `groupes[]`, `nb_syllabes`, `nb_groupes` |
| `OptionsGroupes` | Options : `gerer_elisions`, `gerer_liaisons`, `gerer_enchainement` |

## Role dans le pipeline Lectura

L'Aligneur-Syllabeur est le **pivot central** de Lectura :

1. **Preparation des corpus** : l'alignement grapheme-phoneme a permis de constituer les donnees d'entrainement des modeles G2P et P2G
2. **Lecture assistee** : les groupes de lecture avec syllabes colorees sont la base de l'interface de lecture
3. **Synthese vocale** : l'alignement et les spans permettent la synchronisation texte-audio

## Caracteristiques techniques

- **Zero dependance** Python
- **Alignement DFS** grapheme-phoneme avec gestion des lettres muettes et fusionnees
- **Modele de sonorite** pour la syllabation (5 classes : obstruantes, nasales, liquides, semi-voyelles, voyelles)
- **Architecture E1/E2** : construction des groupes (E1) puis syllabation (E2), utilisables separement
- **Phonemiseur pluggable** : eSpeak-NG, Lectura G2P, ou tout objet compatible
- **Python 3.10+** avec type hints complets (PEP-561)
- **Licence** : AGPL-3.0 (non commerciale) — licence commerciale sur demande

## Licence

Ce module est distribue sous licence **AGPL-3.0** (non commerciale) — voir [LICENCE.txt](LICENCE.txt).

Pour un usage commercial, contacter [contact@lec-tu-ra.com](mailto:contact@lec-tu-ra.com).
