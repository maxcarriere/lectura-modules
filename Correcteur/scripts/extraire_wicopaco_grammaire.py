#!/usr/bin/env python3
"""
Extraction de paires de corrections grammaticales depuis WiCoPaCo v2.

Filtre le corpus WiCoPaCo (Wikipedia Correction Pairs Corpus) pour ne garder
que les corrections grammaticales : conjugaison, accords (genre/nombre)
et homophones.

Usage :
    python scripts/extraire_wicopaco_grammaire.py

Entree :  data/corpus/wicopaco_v2.xml.gz  (~54 MB compresse)
Sortie :  data/grammaire_wicopaco.tsv
"""

from __future__ import annotations

import gzip
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "corpus" / "wicopaco_v2.xml.gz"
OUTPUT_PATH = BASE_DIR / "data" / "grammaire_wicopaco.tsv"

MAX_PAR_CATEGORIE = 5000

# ---------------------------------------------------------------------------
# Homophones connus
# ---------------------------------------------------------------------------
HOMOPHONES = [
    {"a", "à"},
    {"est", "et"},
    {"ou", "où"},
    {"son", "sont"},
    {"ces", "ses", "c'est", "s'est"},
    {"on", "ont"},
    {"ce", "se"},
    {"la", "là", "l'a"},
    {"ma", "m'a"},
    {"mais", "mes", "met", "mets"},
    {"peu", "peut", "peux"},
    {"leur", "leurs"},
    {"quel", "quelle", "qu'elle"},
    {"quand", "quant", "qu'en"},
    {"sans", "s'en", "sent", "sens"},
    {"dans", "d'en"},
    {"ni", "n'y"},
    {"si", "s'y"},
    {"sa", "ça"},
    {"été", "étais", "était", "étaient", "étés"},
    {"ai", "aie", "aies", "ait", "aient"},
    {"er", "é", "ée", "és", "ées", "ez"},
    {"ons", "ont"},
    {"soi", "soit"},
    {"voie", "voix"},
    {"foi", "fois", "foie"},
    {"cour", "cours", "court"},
    {"tant", "temps", "tend", "tends"},
    {"par", "part"},
    {"vers", "vert", "verre", "ver"},
    {"air", "aire", "ère"},
    {"pré", "prêt", "près"},
]

# Construction d'un index rapide : mot_lower -> ensemble d'homophones
_HOMO_INDEX: dict[str, set[str]] = {}
for group in HOMOPHONES:
    for word in group:
        _HOMO_INDEX.setdefault(word.lower(), set()).update(group)


def est_homophone(mot_avant: str, mot_apres: str) -> bool:
    """Verifie si deux mots forment une paire d'homophones connue."""
    a, b = mot_avant.lower(), mot_apres.lower()
    if a == b:
        return False
    group = _HOMO_INDEX.get(a)
    if group and b in group:
        return True
    return False


# ---------------------------------------------------------------------------
# Detection d'accord (genre / nombre)
# ---------------------------------------------------------------------------
# Terminaisons indicatives de variations genre/nombre
_ACCORD_PATTERNS = [
    # nombre : singulier <-> pluriel
    (r"^(.+)s$", r"^\1$"),         # mots -> mot
    (r"^(.+)$", r"^\1s$"),         # mot -> mots
    (r"^(.+)x$", r"^\1$"),         # animaux -> animal? non, mais chevaux/chevau...
    (r"^(.+)aux$", r"^\1al$"),     # animaux -> animal
    (r"^(.+)al$", r"^\1aux$"),     # animal -> animaux
    # genre : masculin <-> feminin
    (r"^(.+)e$", r"^\1$"),         # grande -> grand
    (r"^(.+)$", r"^\1e$"),         # grand -> grande
    (r"^(.+)eur$", r"^\1euse$"),   # menteur -> menteuse
    (r"^(.+)euse$", r"^\1eur$"),   # menteuse -> menteur
    (r"^(.+)teur$", r"^\1trice$"), # directeur -> directrice
    (r"^(.+)trice$", r"^\1teur$"), # directrice -> directeur
    (r"^(.+)er$", r"^\1ère$"),     # premier -> premiere
    (r"^(.+)ère$", r"^\1er$"),
    (r"^(.+)if$", r"^\1ive$"),     # actif -> active
    (r"^(.+)ive$", r"^\1if$"),
]

# Determinants / pronoms avec variation genre/nombre
_DET_ACCORD = {
    # genre
    ("un", "une"), ("une", "un"),
    ("le", "la"), ("la", "le"),
    ("du", "de la"), ("de la", "du"),
    ("au", "à la"), ("à la", "au"),
    ("mon", "ma"), ("ma", "mon"),
    ("ton", "ta"), ("ta", "ton"),
    ("son", "sa"), ("sa", "son"),  # aussi homophone, sera classe homophone d'abord
    ("ce", "cette"), ("cette", "ce"),
    ("cet", "cette"), ("cette", "cet"),
    # nombre
    ("le", "les"), ("les", "le"),
    ("la", "les"), ("les", "la"),
    ("un", "des"), ("des", "un"),
    ("une", "des"), ("des", "une"),
    ("mon", "mes"), ("mes", "mon"),
    ("ton", "tes"), ("tes", "ton"),
    ("son", "ses"), ("ses", "son"),
    ("ce", "ces"), ("ces", "ce"),
    ("notre", "nos"), ("nos", "notre"),
    ("votre", "vos"), ("vos", "votre"),
    ("leur", "leurs"), ("leurs", "leur"),
    ("du", "des"), ("des", "du"),
    ("au", "aux"), ("aux", "au"),
    ("quel", "quels"), ("quels", "quel"),
    ("quelle", "quelles"), ("quelles", "quelle"),
    ("quel", "quelle"), ("quelle", "quel"),
}


def est_accord(mot_avant: str, mot_apres: str) -> bool:
    """Detecte si la difference est un accord genre/nombre."""
    a, b = mot_avant.lower(), mot_apres.lower()
    if a == b:
        return False

    # Determinants / pronoms connus
    if (a, b) in _DET_ACCORD:
        return True

    # Difference minimale de terminaison (1-3 caracteres)
    # indiquant un accord
    if len(a) < 2 or len(b) < 2:
        return False

    # Racine commune (au moins 3 caracteres)
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix_len += 1
        else:
            break

    if prefix_len < 3:
        return False

    suffix_a = a[prefix_len:]
    suffix_b = b[prefix_len:]

    # Variation simple : ajout/retrait de 's', 'e', 'es', 'x'
    variations_accord = {
        ("", "s"), ("s", ""),
        ("", "e"), ("e", ""),
        ("", "es"), ("es", ""),
        ("", "x"), ("x", ""),
        ("e", "es"), ("es", "e"),
        ("", "ne"), ("ne", ""),
        ("", "le"), ("le", ""),
        ("", "te"), ("te", ""),
        ("al", "aux"), ("aux", "al"),
        ("el", "elle"), ("elle", "el"),
        ("en", "enne"), ("enne", "en"),
        ("on", "onne"), ("onne", "on"),
        ("et", "ette"), ("ette", "et"),
        ("f", "ve"), ("ve", "f"),
        ("er", "ère"), ("ère", "er"),
        ("eur", "euse"), ("euse", "eur"),
        ("teur", "trice"), ("trice", "teur"),
        ("x", "se"), ("se", "x"),
    }

    if (suffix_a, suffix_b) in variations_accord:
        return True

    return False


# ---------------------------------------------------------------------------
# Detection de conjugaison
# ---------------------------------------------------------------------------
# Terminaisons verbales francaises courantes
_TERMINAISONS_VERBALES = {
    # infinitif
    "er", "ir", "re", "oir",
    # present
    "e", "es", "ons", "ez", "ent",
    "is", "it", "issons", "issez", "issent",
    "s", "t", "ons", "ez", "ent",
    "ds", "d",
    # imparfait
    "ais", "ait", "ions", "iez", "aient",
    # passe simple
    "ai", "as", "a", "âmes", "âtes", "èrent",
    # futur
    "rai", "ras", "ra", "rons", "rez", "ront",
    # conditionnel
    "rais", "rait", "rions", "riez", "raient",
    # subjonctif
    "asse", "asses", "ât", "assions", "assiez", "assent",
    # participe passe
    "é", "ée", "és", "ées",
    "i", "ie", "is", "ies",
    "u", "ue", "us", "ues",
}

# Paires de terminaisons qui indiquent un changement de conjugaison
_CONJ_VARIATIONS = {
    # present <-> present (personne)
    ("e", "es"), ("es", "e"),
    ("e", "ent"), ("ent", "e"),
    ("es", "ent"), ("ent", "es"),
    # imparfait (personne)
    ("ais", "ait"), ("ait", "ais"),
    ("ais", "aient"), ("aient", "ais"),
    ("ait", "aient"), ("aient", "ait"),
    # futur (personne)
    ("rai", "ra"), ("ra", "rai"),
    ("ras", "ra"), ("ra", "ras"),
    # passe compose accord participe
    ("é", "ée"), ("ée", "é"),
    ("é", "és"), ("és", "é"),
    ("é", "ées"), ("ées", "é"),
    ("ée", "és"), ("és", "ée"),
    ("ée", "ées"), ("ées", "ée"),
    ("és", "ées"), ("ées", "és"),
    ("i", "ie"), ("ie", "i"),
    ("i", "is"), ("is", "i"),
    ("i", "ies"), ("ies", "i"),
    ("u", "ue"), ("ue", "u"),
    ("u", "us"), ("us", "u"),
    ("u", "ues"), ("ues", "u"),
    # present <-> imparfait
    ("e", "ait"), ("ait", "e"),
    ("e", "ais"), ("ais", "e"),
    # infinitif <-> participe
    ("er", "é"), ("é", "er"),
    ("er", "ée"), ("ée", "er"),
    ("er", "és"), ("és", "er"),
    ("er", "ées"), ("ées", "er"),
    ("er", "ez"), ("ez", "er"),
    # conditionnel
    ("rais", "rait"), ("rait", "rais"),
    ("rais", "raient"), ("raient", "rais"),
    ("rait", "raient"), ("raient", "rait"),
    # subjonctif
    ("e", "ent"), ("ent", "e"),
}


def est_conjugaison(mot_avant: str, mot_apres: str) -> bool:
    """Detecte si la difference est un changement de conjugaison."""
    a, b = mot_avant.lower(), mot_apres.lower()
    if a == b:
        return False

    if len(a) < 3 or len(b) < 3:
        return False

    # Racine commune (au moins 2 caracteres pour les verbes courts)
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix_len += 1
        else:
            break

    if prefix_len < 2:
        return False

    suffix_a = a[prefix_len:]
    suffix_b = b[prefix_len:]

    if (suffix_a, suffix_b) in _CONJ_VARIATIONS:
        return True

    return False


# ---------------------------------------------------------------------------
# Extraction du texte et du mot modifie depuis un element <before>/<after>
# ---------------------------------------------------------------------------
_RE_M_TAG = re.compile(r"<m\b[^>]*>(.*?)</m>")


def extraire_texte_et_mot(xml_str: str) -> tuple[str, str, str] | None:
    """
    Depuis le contenu XML d'un <before> ou <after>, extrait :
    - le texte complet (sans balises <m>)
    - le mot modifie (contenu de <m>)
    - le contexte (texte sans le mot modifie, pour validation)

    Retourne None si pas exactement 1 balise <m> ou si num_words != 1.
    """
    # Verifier qu'il y a exactement une balise <m>
    matches = list(_RE_M_TAG.finditer(xml_str))
    if len(matches) != 1:
        return None

    m = matches[0]
    mot = m.group(1).strip()

    # Verifier num_words="1"
    if 'num_words="1"' not in xml_str[:m.end()]:
        # Extraire num_words depuis la balise ouvrante
        nw_match = re.search(r'num_words="(\d+)"', xml_str[:m.end()])
        if nw_match and nw_match.group(1) != "1":
            return None

    # Texte complet sans balises
    texte = _RE_M_TAG.sub(r"\1", xml_str).strip()

    # Contexte = texte avant + texte apres le mot modifie (pour comparer
    # que seul le mot <m> a change entre before et after)
    contexte = _RE_M_TAG.sub("<<PLACEHOLDER>>", xml_str).strip()

    return texte, mot, contexte


# ---------------------------------------------------------------------------
# Extraction du texte brut d'un element XML (inclut le texte des sous-elements)
# ---------------------------------------------------------------------------
def element_inner_xml(elem: ET.Element) -> str:
    """Reconstruit le XML interne d'un element (texte + sous-elements)."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(ET.tostring(child, encoding="unicode"))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classifier(mot_avant: str, mot_apres: str, comment: str) -> str | None:
    """
    Classifie une paire de mots. Retourne le type d'erreur ou None.
    Priorite : homophone > accord > conjugaison
    """
    # Ignorer les mots identiques (casse differente seulement = pas d'erreur grammaticale)
    if mot_avant.lower() == mot_apres.lower():
        return None

    # Ignorer les mots avec chiffres ou caracteres speciaux
    if not mot_avant.isalpha() or not mot_apres.isalpha():
        return None

    # 1. Homophones (priorite la plus haute)
    if est_homophone(mot_avant, mot_apres):
        return "homophone"

    # 2. Accord genre/nombre
    if est_accord(mot_avant, mot_apres):
        return "accord"

    # 3. Conjugaison
    if est_conjugaison(mot_avant, mot_apres):
        return "conjugaison"

    return None


# ---------------------------------------------------------------------------
# Parsing principal
# ---------------------------------------------------------------------------
def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERREUR : fichier introuvable : {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    resultats: dict[str, list[tuple[str, str, str]]] = {
        "homophone": [],
        "accord": [],
        "conjugaison": [],
    }
    compteurs = {"total_modifs": 0, "single_word": 0, "classifie": 0}
    # Pour eviter les doublons
    vus: set[tuple[str, str, str]] = set()

    print(f"Lecture de {INPUT_PATH} ...")
    print(f"(fichier de {INPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB compresse)")

    with gzip.open(INPUT_PATH, "rb") as f:
        context = ET.iterparse(f, events=("end",))

        for event, elem in context:
            if elem.tag != "modif":
                continue

            compteurs["total_modifs"] += 1

            # Progression
            if compteurs["total_modifs"] % 50000 == 0:
                print(f"  ... {compteurs['total_modifs']:,} modifs traitees "
                      f"(H:{len(resultats['homophone'])} "
                      f"A:{len(resultats['accord'])} "
                      f"C:{len(resultats['conjugaison'])})")

            # Verifier si on a atteint le max pour toutes les categories
            if all(len(v) >= MAX_PAR_CATEGORIE for v in resultats.values()):
                print("  Max atteint pour toutes les categories, arret.")
                break

            comment = elem.get("wp_comment", "")

            before_elem = elem.find("before")
            after_elem = elem.find("after")

            if before_elem is None or after_elem is None:
                elem.clear()
                continue

            before_xml = element_inner_xml(before_elem)
            after_xml = element_inner_xml(after_elem)

            result_before = extraire_texte_et_mot(before_xml)
            result_after = extraire_texte_et_mot(after_xml)

            if result_before is None or result_after is None:
                elem.clear()
                continue

            texte_avant, mot_avant, ctx_avant = result_before
            texte_apres, mot_apres, ctx_apres = result_after

            # Valider que seul le mot <m> a change (le contexte doit etre identique)
            if ctx_avant != ctx_apres:
                elem.clear()
                continue

            compteurs["single_word"] += 1

            # Classifier
            type_erreur = classifier(mot_avant, mot_apres, comment)
            if type_erreur is None:
                elem.clear()
                continue

            # Verifier la limite par categorie
            if len(resultats[type_erreur]) >= MAX_PAR_CATEGORIE:
                elem.clear()
                continue

            # Dedoublonner sur (type, mot_avant, mot_apres)
            cle = (type_erreur, mot_avant.lower(), mot_apres.lower())
            if cle in vus:
                elem.clear()
                continue
            vus.add(cle)

            resultats[type_erreur].append((type_erreur, texte_avant, texte_apres))
            compteurs["classifie"] += 1

            # Liberer la memoire
            elem.clear()

    # Ecriture du TSV
    total_ecrit = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        out.write("type_erreur\tphrase_erronee\tphrase_corrigee\n")
        for cat in ("homophone", "accord", "conjugaison"):
            for row in resultats[cat]:
                out.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                total_ecrit += 1

    # Statistiques
    print()
    print("=" * 60)
    print("STATISTIQUES D'EXTRACTION")
    print("=" * 60)
    print(f"Total modifs dans le corpus : {compteurs['total_modifs']:,}")
    print(f"Modifs single-word (1 <m>) : {compteurs['single_word']:,}")
    print(f"Paires classifiees (uniques): {compteurs['classifie']:,}")
    print()
    print(f"  homophone   : {len(resultats['homophone']):,}")
    print(f"  accord      : {len(resultats['accord']):,}")
    print(f"  conjugaison : {len(resultats['conjugaison']):,}")
    print()
    print(f"Total ecrit dans TSV : {total_ecrit:,}")
    print(f"Sortie : {OUTPUT_PATH}")
    print()

    # Exemples
    for cat in ("homophone", "accord", "conjugaison"):
        items = resultats[cat]
        if items:
            print(f"--- Exemples {cat} ({len(items)} paires) ---")
            for _, avant, apres in items[:5]:
                # Trouver la difference
                mots_a = avant.split()
                mots_b = apres.split()
                diff = ""
                for ma, mb in zip(mots_a, mots_b):
                    if ma != mb:
                        diff = f"{ma} -> {mb}"
                        break
                print(f"  [{diff}]")
                if len(avant) > 100:
                    print(f"    avant: {avant[:100]}...")
                else:
                    print(f"    avant: {avant}")
            print()


if __name__ == "__main__":
    main()
