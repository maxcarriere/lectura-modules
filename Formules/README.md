# Lectura Formules

Lecture algorithmique des formules pour le francais : nombres, sigles, dates, telephones, heures, monnaies, ordinaux, fractions, scientifiques, mathematiques, GPS, etc.

Module autonome, zero dependance externe.

## Installation

```bash
pip install lectura-formules
```

## Utilisation rapide

```python
from lectura_formules import lire_formule, enrichir_formules

# Lire une formule
result = lire_formule("42")
print(result.display_fr)    # "quarante-deux"
print(result.phonetique)    # "ka.ʁɑ̃t.dø"

# Enrichir les tokens d'une phrase
tokens = [{"texte": "Il", "type_f": "MOT"}, {"texte": "a", "type_f": "MOT"},
          {"texte": "3", "type_f": "NOMBRE"}, {"texte": "chats", "type_f": "MOT"}]
enrichir_formules(tokens)
# Le token "3" est enrichi : display_fr="trois", phonetique="tʁwa"
```

## Sons (optionnel)

Les fichiers WAV (~12 Mo, 289 fichiers) ne sont **pas** inclus dans le package pip.
Ils sont disponibles sur GitHub pour la lecture audio des formules.

### Telecharger les sons depuis GitHub

```bash
# Creer le dossier de destination
mkdir -p sons_formules

# Telecharger depuis le repo GitHub
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/maxcarriere/lectura-modules.git /tmp/lectura-sons
cd /tmp/lectura-sons
git sparse-checkout set Formules/src/lectura_formules/data/sons
cp -r Formules/src/lectura_formules/data/sons/fr/wav/* sons_formules/
rm -rf /tmp/lectura-sons
```

### Configurer le chemin des sons

```python
from lectura_formules import set_sounds_dir, get_sound_path

# Indiquer ou se trouvent les WAV
set_sounds_dir("/chemin/vers/sons_formules")

# Recuperer le chemin d'un son
wav = get_sound_path("42")
if wav:
    print(f"Fichier son : {wav}")
```

## API principale

| Fonction | Description |
|---|---|
| `lire_formule(texte)` | Point d'entree principal — detecte le type et lit |
| `lire_nombre(texte)` | Nombres : "42" -> "quarante-deux" |
| `lire_date(texte)` | Dates : "25/12/2024" -> "vingt-cinq decembre..." |
| `lire_heure(texte)` | Heures : "14h30" -> "quatorze heures trente" |
| `lire_telephone(texte)` | Telephones : "06 12 34 56 78" |
| `lire_sigle(texte)` | Sigles : "SNCF" -> "esse-enne-ce-effe" |
| `lire_ordinal(texte)` | Ordinaux : "3e" -> "troisieme" |
| `lire_fraction(texte)` | Fractions : "3/4" -> "trois quarts" |
| `lire_monnaie(texte)` | Monnaies : "42 EUR" -> "quarante-deux euros" |
| `lire_pourcentage(texte)` | Pourcentages : "50%" -> "cinquante pour cent" |
| `enrichir_formules(tokens)` | Enrichit les tokens d'une phrase |
| `int_to_roman(n)` / `roman_to_int(s)` | Chiffres romains |

## Licence

Double licence :
- **AGPL-3.0** — usage libre (voir [LICENCE.txt](LICENCE.txt))
- **Licence commerciale** — usage proprietaire, contacter [contact@lec-tu-ra.com](mailto:contact@lec-tu-ra.com)
