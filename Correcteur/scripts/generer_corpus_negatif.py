#!/usr/bin/env python3
"""
Generation d'un corpus de phrases CORRECTES (negatif) pour mesurer les faux positifs.

Extrait les phrases corrigees (version "after") du corpus WiCoPaCo v2
et les filtre pour ne garder que des phrases francaises bien formees.
Ces phrases, presumees correctes apres correction par les editeurs Wikipedia,
servent de references negatives : le correcteur ne devrait rien signaler.

Usage :
    python scripts/generer_corpus_negatif.py

Entree :  data/corpus/wicopaco_v2.xml.gz  (~54 MB compresse)
Sortie :  data/negatif_wicopaco.tsv        (1000 phrases correctes)
"""

from __future__ import annotations

import gzip
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "corpus" / "wicopaco_v2.xml.gz"
OUTPUT_PATH = BASE_DIR / "data" / "negatif_wicopaco.tsv"

# ---------------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 1000
RANDOM_SEED = 42
MIN_WORDS = 5
MAX_WORDS = 30

# ---------------------------------------------------------------------------
# Caracteres francais autorises
# ---------------------------------------------------------------------------
# Lettres ASCII + lettres accentuees courantes en francais + ponctuation standard
_ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "\u00e0\u00e2\u00e4\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb\u00ee\u00ef"  # a grave, a circ, a trema, ae, c cedille, e grave/aigu/circ/trema, i circ/trema
    "\u00f4\u00f6\u00f9\u00fb\u00fc\u00ff"  # o circ/trema, u grave/circ/trema, y trema
    "\u00c0\u00c2\u00c4\u00c6\u00c7\u00c8\u00c9\u00ca\u00cb\u00ce\u00cf"  # majuscules
    "\u00d4\u00d6\u00d9\u00db\u00dc\u0178"
    "\u0152\u0153"  # OE ligature
    "0123456789"
    " .,;:!?'-\u2019\u2018\u201c\u201d\u00ab\u00bb()/"
)

# Patterns a exclure (URLs, codes, markup, etc.)
_RE_EXCLUSION = re.compile(
    r"http[s]?://"        # URL
    r"|www\."             # URL sans protocole
    r"|\[\["              # markup wiki
    r"|\]\]"              # markup wiki
    r"|\{\{"              # template wiki
    r"|\}\}"              # template wiki
    r"|<[a-zA-Z]"         # balises HTML residuelles
    r"|\|"                # pipe (tableaux wiki)
    r"|&[a-zA-Z]+;"       # entites HTML (&amp; &nbsp; etc.)
    r"|&[#]\d+"           # entites numeriques
    r"|==+"               # titres wiki
    r"|\*\*"              # gras markdown
    r"|#REDIRECT"         # redirections wiki
    r"|REDIRECT"          # redirections wiki
    r"|Category:"         # categories wiki
    r"|Cat\u00e9gorie:"   # categories wiki FR
    r"|Fichier:"          # fichiers wiki
    r"|Image:"            # images wiki
)

# Mots indicateurs qu'il s'agit d'une vraie phrase (contient au moins un
# element verbal ou modal). On cherche des prefixes/formes courants.
_MOTS_VERBAUX = re.compile(
    r"\b(?:"
    r"est|sont|sera|seront|fut|furent|soit|soient|"
    r"a|ont|aura|auront|eut|eurent|ait|aient|avait|avaient|"
    r"fait|font|fera|feront|faisait|faisaient|"
    r"peut|peuvent|pouvait|pouvaient|pourra|pourront|"
    r"doit|doivent|devait|devaient|devra|devront|"
    r"va|vont|allait|allaient|ira|iront|"
    r"dit|disent|disait|"
    r"voit|voient|voyait|"
    r"sait|savent|savait|"
    r"veut|veulent|voulait|"
    r"prend|prennent|prenait|"
    r"donne|donnent|donnait|"
    r"trouve|trouvent|trouvait|"
    r"reste|restent|restait|"
    r"vient|viennent|venait|"
    r"devient|deviennent|devenait|"
    r"comprend|comprennent|"
    r"permet|permettent|"
    r"produit|produisent|"
    r"existe|existent|"
    r"forme|forment|"
    r"constitue|constituent|"
    r"repr[e\u00e9]sente|repr[e\u00e9]sentent|"
    r"poss[e\u00e8]de|poss[e\u00e8]dent|"
    r"contient|contiennent|"
    r"cr[e\u00e9\u00e8][e\u00e9]|"
    r"[a-z]+ait|[a-z]+aient|"          # imparfait generique
    r"[a-z]+era|[a-z]+eront|"          # futur generique
    r"[a-z]+[eè]rent|"                 # passe simple generique
    r"[a-z]{3,}e(?:nt)?"               # present -e/-ent generique
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Extraction du texte brut depuis un element XML
# ---------------------------------------------------------------------------
def element_texte_brut(elem: ET.Element) -> str:
    """Extrait le texte brut d'un element, incluant le contenu des sous-elements <m>."""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        # Inclure le texte du sous-element (ex: <m>mot</m>)
        if child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Nettoyage du texte
# ---------------------------------------------------------------------------
def nettoyer_espaces(texte: str) -> str:
    """Normalise les espaces (WiCoPaCo tokenise avec des espaces autour de la ponctuation)."""
    # Supprimer les espaces avant la ponctuation de fermeture
    texte = re.sub(r"\s+([.,;:!?\)])", r"\1", texte)
    # Supprimer les espaces apres la ponctuation d'ouverture
    texte = re.sub(r"([\(])\s+", r"\1", texte)
    # Guillemets francais : espace insecable apres/avant
    texte = re.sub("\u00ab\\s*", "\u00ab ", texte)
    texte = re.sub("\\s*\u00bb", " \u00bb", texte)
    # Apostrophes : coller au mot suivant
    texte = re.sub(r"(\w)'\s+", r"\1'", texte)
    texte = re.sub("(\\w)\u2019\\s+", lambda m: m.group(1) + "\u2019", texte)
    # Espaces multiples
    texte = re.sub(r"\s{2,}", " ", texte)
    return texte.strip()


# ---------------------------------------------------------------------------
# Filtres de qualite
# ---------------------------------------------------------------------------
def contient_uniquement_chars_francais(texte: str) -> bool:
    """Verifie que la phrase ne contient que des caracteres francais standard."""
    for ch in texte:
        if ch not in _ALLOWED_CHARS:
            return False
    return True


def est_phrase_valide(texte: str) -> bool:
    """
    Verifie qu'un texte ressemble a une vraie phrase francaise :
    - Longueur entre MIN_WORDS et MAX_WORDS mots
    - Caracteres francais uniquement
    - Pas d'URL, de code, de markup
    - Commence par une majuscule ou un mot courant
    - Contient au moins un mot verbal
    """
    # Longueur en mots
    mots = texte.split()
    if len(mots) < MIN_WORDS or len(mots) > MAX_WORDS:
        return False

    # Exclusion de patterns non-textuels
    if _RE_EXCLUSION.search(texte):
        return False

    # Caracteres autorises uniquement
    if not contient_uniquement_chars_francais(texte):
        return False

    # Commence par une majuscule ou un determinant/pronom/preposition courant
    premier = mots[0]
    if not (premier[0].isupper() or premier.lower() in {
        "le", "la", "les", "un", "une", "des", "du", "de", "ce", "cette",
        "ces", "son", "sa", "ses", "mon", "ma", "mes", "ton", "ta", "tes",
        "notre", "nos", "votre", "vos", "leur", "leurs",
        "il", "elle", "ils", "elles", "on", "nous", "vous", "je", "tu",
        "en", "y", "qui", "que", "quand", "si", "mais", "ou", "et", "donc",
        "or", "ni", "car", "pour", "par", "dans", "sur", "avec", "sans",
        "sous", "entre", "vers", "chez", "depuis", "pendant", "avant",
        "apres", "alors", "ainsi", "aussi", "bien", "comme", "puis",
        "cependant", "toutefois", "neanmoins", "pourtant",
        "d'abord", "d'ailleurs", "d'autre", "d'apres",
        "l'", "c'", "s'", "n'", "j'", "qu'",
    }):
        return False

    # Contient au moins un mot a allure verbale
    if not _MOTS_VERBAUX.search(texte):
        return False

    # Pas trop de majuscules (signe de titre ou d'acronymes)
    nb_maj = sum(1 for c in texte if c.isupper())
    if nb_maj > len(mots) * 0.4:
        return False

    # Pas que des mots tres courts (signe de listes ou codes)
    mots_longs = [m for m in mots if len(m) >= 3]
    if len(mots_longs) < len(mots) * 0.4:
        return False

    return True


# ---------------------------------------------------------------------------
# Extraction principale
# ---------------------------------------------------------------------------
def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERREUR : fichier introuvable : {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Lecture de {INPUT_PATH} ...")
    print(f"(fichier de {INPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB compresse)")
    print()

    # Phase 1 : collecter toutes les phrases valides
    phrases_candidates: set[str] = set()
    compteurs = {
        "total_modifs": 0,
        "phrases_extraites": 0,
        "filtrees_longueur": 0,
        "filtrees_chars": 0,
        "filtrees_pattern": 0,
        "filtrees_qualite": 0,
        "doublons": 0,
    }

    with gzip.open(INPUT_PATH, "rb") as f:
        context = ET.iterparse(f, events=("end",))

        for event, elem in context:
            if elem.tag != "modif":
                continue

            compteurs["total_modifs"] += 1

            # Progression
            if compteurs["total_modifs"] % 100000 == 0:
                print(f"  ... {compteurs['total_modifs']:,} modifs traitees, "
                      f"{len(phrases_candidates):,} phrases retenues")

            after_elem = elem.find("after")
            if after_elem is None:
                elem.clear()
                continue

            # Extraire le texte brut de la version corrigee
            texte = element_texte_brut(after_elem)
            if not texte:
                elem.clear()
                continue

            compteurs["phrases_extraites"] += 1

            # Nettoyage des espaces
            texte = nettoyer_espaces(texte)

            # Filtre longueur (rapide)
            mots = texte.split()
            if len(mots) < MIN_WORDS or len(mots) > MAX_WORDS:
                compteurs["filtrees_longueur"] += 1
                elem.clear()
                continue

            # Filtre patterns exclus
            if _RE_EXCLUSION.search(texte):
                compteurs["filtrees_pattern"] += 1
                elem.clear()
                continue

            # Filtre caracteres
            if not contient_uniquement_chars_francais(texte):
                compteurs["filtrees_chars"] += 1
                elem.clear()
                continue

            # Filtre qualite (phrase valide)
            if not est_phrase_valide(texte):
                compteurs["filtrees_qualite"] += 1
                elem.clear()
                continue

            # Deduplication
            if texte in phrases_candidates:
                compteurs["doublons"] += 1
                elem.clear()
                continue

            phrases_candidates.add(texte)

            # Liberer la memoire
            elem.clear()

    # Convertir en liste triee pour garantir la reproductibilite
    # (l'ordre d'iteration d'un set Python est non-deterministe)
    phrases_candidates_list = sorted(phrases_candidates)

    print()
    print(f"Phrases candidates retenues : {len(phrases_candidates_list):,}")
    print()

    # Phase 2 : echantillonnage aleatoire
    random.seed(RANDOM_SEED)

    if len(phrases_candidates_list) < SAMPLE_SIZE:
        print(f"ATTENTION : seulement {len(phrases_candidates_list)} phrases disponibles "
              f"(demande : {SAMPLE_SIZE})")
        echantillon = phrases_candidates_list
    else:
        echantillon = random.sample(phrases_candidates_list, SAMPLE_SIZE)

    # Trier pour reproductibilite du fichier de sortie
    echantillon.sort()

    # Phase 3 : ecriture du TSV
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for phrase in echantillon:
            # Format : negatif\tphrase_correcte\tphrase_correcte
            # (meme phrase pour les deux colonnes : aucune correction attendue)
            out.write(f"negatif\t{phrase}\t{phrase}\n")

    # Phase 4 : statistiques
    print("=" * 60)
    print("STATISTIQUES DE GENERATION DU CORPUS NEGATIF")
    print("=" * 60)
    print()
    print(f"Fichier source          : {INPUT_PATH.name}")
    print(f"Total modifs parcourues : {compteurs['total_modifs']:,}")
    print(f"Phrases extraites       : {compteurs['phrases_extraites']:,}")
    print()
    print("Filtrage :")
    print(f"  Hors longueur ({MIN_WORDS}-{MAX_WORDS} mots) : {compteurs['filtrees_longueur']:,}")
    print(f"  Patterns exclus (URL, wiki...)  : {compteurs['filtrees_pattern']:,}")
    print(f"  Caracteres non-francais         : {compteurs['filtrees_chars']:,}")
    print(f"  Qualite insuffisante            : {compteurs['filtrees_qualite']:,}")
    print(f"  Doublons                        : {compteurs['doublons']:,}")
    print()
    print(f"Phrases candidates valides : {len(phrases_candidates_list):,}")
    print(f"Echantillon selectionne    : {len(echantillon):,}")
    print()
    print(f"Sortie : {OUTPUT_PATH}")
    print(f"Format : negatif<TAB>phrase_correcte<TAB>phrase_correcte")
    print()

    # Quelques exemples
    print("--- Exemples de phrases selectionnees ---")
    exemples = echantillon[:10]
    for i, phrase in enumerate(exemples, 1):
        nb_mots = len(phrase.split())
        if len(phrase) > 100:
            print(f"  {i:2d}. ({nb_mots} mots) {phrase[:100]}...")
        else:
            print(f"  {i:2d}. ({nb_mots} mots) {phrase}")
    print()

    # Distribution de la longueur
    longueurs = [len(p.split()) for p in echantillon]
    print("--- Distribution des longueurs (mots) ---")
    print(f"  Min : {min(longueurs)}")
    print(f"  Max : {max(longueurs)}")
    print(f"  Moy : {sum(longueurs) / len(longueurs):.1f}")
    # Histogramme par tranches de 5
    tranches = {}
    for l in longueurs:
        tranche = (l // 5) * 5
        cle = f"{tranche}-{tranche + 4}"
        tranches[cle] = tranches.get(cle, 0) + 1
    for cle in sorted(tranches.keys()):
        barre = "#" * (tranches[cle] // 5)
        print(f"  {cle:>6s} : {tranches[cle]:4d} {barre}")
    print()


if __name__ == "__main__":
    main()
