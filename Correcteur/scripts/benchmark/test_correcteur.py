#!/usr/bin/env python3
"""Test de regression deterministe — Recall + Faux Positifs.

Valide le correcteur V6 sur deux axes :
  1. RECALL  : le correcteur corrige bien les erreurs attendues
  2. ZERO FP : le correcteur ne touche pas au texte propre naturel

Sources FP :
  - WiCoPaCo negatif (Wikipedia, 200 phrases echantillonnees)
  - Litterature classique (Hugo, Verne, Flaubert — 200 phrases)
  - Phrases ciblees existantes (~130 phrases de test_fp_propre)

Usage :
    cd Modules/Correcteur
    python scripts/benchmark/test_correcteur.py
    python scripts/benchmark/test_correcteur.py --verbose
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))

LEXIQUE_PATH = Path(
    "/data/work/projets/lectura/workspace/Lexique/bases/lexique_lectura.db"
)

WORKSPACE = ROOT.parent.parent
WICOPACO_PATH = WORKSPACE / "Corpus" / "Correcteur" / "negatif_wicopaco.tsv"
LITTERATURE_BASE = WORKSPACE / "Corpus" / "Voix" / "raw" / "fr_FR"


# ========================================================================
# A1 — Corpus de recall (phrases fautives → attendues)
# ========================================================================

@dataclass
class CasRecall:
    """Un cas de test recall : phrase fautive → phrase attendue."""
    fautive: str
    attendue: str
    categorie: str   # ORTH, HOMO, ACCORD, CONJ, NEGATION, CAPITAL
    regle: str       # ortho.accent, homophones.a_a, etc.


CORPUS_RECALL: list[CasRecall] = [
    # ------------------------------------------------------------------
    # ORTH / ortho.accent (5 cas)
    # ------------------------------------------------------------------
    CasRecall("L'ecole est fermee pour les vacances.", "L'école est fermée pour les vacances.", "ORTH", "ortho.accent"),
    CasRecall("Sa mere est partie en vacances.", "Sa mère est partie en vacances.", "ORTH", "ortho.accent"),
    CasRecall("Nous partons cet apres-midi.", "Nous partons cet après-midi.", "ORTH", "ortho.accent"),
    CasRecall("Le telephone sonne dans la cuisine.", "Le téléphone sonne dans la cuisine.", "ORTH", "ortho.accent"),
    CasRecall("Il etait une fois un petit garcon.", "Il était une fois un petit garçon.", "ORTH", "ortho.accent"),

    # ------------------------------------------------------------------
    # ORTH / ortho.distance (4 cas)
    # ------------------------------------------------------------------
    CasRecall("Le chein dort dans le jardin.", "Le chien dort dans le jardin.", "ORTH", "ortho.distance"),
    CasRecall("Le profeseur explique la lecon.", "Le professeur explique la leçon.", "ORTH", "ortho.distance"),
    CasRecall("Elle a un bycicle rouge.", "Elle a un bicycle rouge.", "ORTH", "ortho.distance"),
    CasRecall("Il a mange une pome.", "Il a mangé une pomme.", "ORTH", "ortho.distance"),

    # ------------------------------------------------------------------
    # ORTH / ortho.phonetique (3 cas)
    # ------------------------------------------------------------------
    CasRecall("Les zanimo sont dans la ferme.", "Les animaux sont dans la ferme.", "ORTH", "ortho.phonetique"),
    CasRecall("Il va a la farmasi.", "Il va à la pharmacie.", "ORTH", "ortho.phonetique"),
    CasRecall("C'est un bato magnifique.", "C'est un bateau magnifique.", "ORTH", "ortho.phonetique"),

    # ------------------------------------------------------------------
    # ORTH / ortho.resegmentation (2 cas)
    # ------------------------------------------------------------------
    CasRecall("Il fait beau aujourdhui.", "Il fait beau aujourd'hui.", "ORTH", "ortho.resegmentation"),
    CasRecall("Mon grandpere habite a la campagne.", "Mon grand-père habite à la campagne.", "ORTH", "ortho.resegmentation"),

    # ------------------------------------------------------------------
    # ORTH / ortho.fusion (4 cas)
    # ------------------------------------------------------------------
    CasRecall("Il y avait beau coup de monde.", "Il y avait beaucoup de monde.", "ORTH", "ortho.fusion"),
    CasRecall("Nous travaillons en semble.", "Nous travaillons ensemble.", "ORTH", "ortho.fusion"),
    CasRecall("Il a mal heureusement rate le bus.", "Il a malheureusement raté le bus.", "ORTH", "ortho.fusion"),
    CasRecall("Il est sur tout tres gentil.", "Il est surtout très gentil.", "ORTH", "ortho.fusion"),

    # ------------------------------------------------------------------
    # ORTH / ortho.elision (3 cas)
    # ------------------------------------------------------------------
    CasRecall("Parce que il fait beau, on sort.", "Parce qu'il fait beau, on sort.", "ORTH", "ortho.elision"),
    CasRecall("Lorsque il pleut, je reste chez moi.", "Lorsqu'il pleut, je reste chez moi.", "ORTH", "ortho.elision"),
    CasRecall("Jusque a demain, il faut attendre.", "Jusqu'à demain, il faut attendre.", "ORTH", "ortho.elision"),

    # ------------------------------------------------------------------
    # ORTH / accent.lexique (3 cas)
    # ------------------------------------------------------------------
    CasRecall("Ca fait longtemps que je ne l'ai pas vu.", "Ça fait longtemps que je ne l'ai pas vu.", "ORTH", "accent.lexique"),
    CasRecall("C'est la meme chose pour tout le monde.", "C'est la même chose pour tout le monde.", "ORTH", "accent.lexique"),
    CasRecall("Il est tres content de son resultat.", "Il est très content de son résultat.", "ORTH", "accent.lexique"),

    # ------------------------------------------------------------------
    # HOMO / homophones.a_a (3 cas)
    # ------------------------------------------------------------------
    CasRecall("Il à mange toute la tarte.", "Il a mangé toute la tarte.", "HOMO", "homophones.a_a"),
    CasRecall("Elle va a la gare.", "Elle va à la gare.", "HOMO", "homophones.a_a"),
    CasRecall("Il à toujours faim.", "Il a toujours faim.", "HOMO", "homophones.a_a"),

    # ------------------------------------------------------------------
    # HOMO / homophones.et_est (3 cas)
    # ------------------------------------------------------------------
    CasRecall("Le chat et noir.", "Le chat est noir.", "HOMO", "homophones.et_est"),
    CasRecall("Elle et grande et forte.", "Elle est grande et forte.", "HOMO", "homophones.et_est"),
    CasRecall("Le temps et magnifique aujourd'hui.", "Le temps est magnifique aujourd'hui.", "HOMO", "homophones.et_est"),

    # ------------------------------------------------------------------
    # HOMO / homophones.son_sont (2 cas)
    # ------------------------------------------------------------------
    CasRecall("Ils son partis en vacances.", "Ils sont partis en vacances.", "HOMO", "homophones.son_sont"),
    CasRecall("Les enfants son fatigues ce soir.", "Les enfants sont fatigués ce soir.", "HOMO", "homophones.son_sont"),

    # ------------------------------------------------------------------
    # HOMO / homophones.on_ont (2 cas)
    # ------------------------------------------------------------------
    CasRecall("Ils on mange a la cantine.", "Ils ont mangé à la cantine.", "HOMO", "homophones.on_ont"),
    CasRecall("Ils on termine leurs devoirs.", "Ils ont terminé leurs devoirs.", "HOMO", "homophones.on_ont"),

    # ------------------------------------------------------------------
    # HOMO / homophones.se_ce (2 cas)
    # ------------------------------------------------------------------
    CasRecall("Se garcon est tres gentil.", "Ce garçon est très gentil.", "HOMO", "homophones.se_ce"),
    CasRecall("Se livre est passionnant.", "Ce livre est passionnant.", "HOMO", "homophones.se_ce"),

    # ------------------------------------------------------------------
    # HOMO / homophones.ou_ou (2 cas)
    # ------------------------------------------------------------------
    CasRecall("Ou vas-tu ce soir ?", "Où vas-tu ce soir ?", "HOMO", "homophones.ou_ou"),
    CasRecall("Ou habites-tu maintenant ?", "Où habites-tu maintenant ?", "HOMO", "homophones.ou_ou"),

    # ------------------------------------------------------------------
    # ACCORD / accord.det_nom (4 cas)
    # ------------------------------------------------------------------
    CasRecall("Les chien jouent dans le parc.", "Les chiens jouent dans le parc.", "ACCORD", "accord.det_nom"),
    CasRecall("Trois pomme sont sur la table.", "Trois pommes sont sur la table.", "ACCORD", "accord.det_nom"),
    CasRecall("Les petit chat dorment.", "Les petits chats dorment.", "ACCORD", "accord.det_nom"),
    CasRecall("Ses ami sont venus hier.", "Ses amis sont venus hier.", "ACCORD", "accord.det_nom"),

    # ------------------------------------------------------------------
    # ACCORD / accord.attribut (3 cas)
    # ------------------------------------------------------------------
    CasRecall("Elle est intelligent.", "Elle est intelligente.", "ACCORD", "accord.attribut"),
    CasRecall("Les fleurs sont beau.", "Les fleurs sont belles.", "ACCORD", "accord.attribut"),
    CasRecall("La maison est grand.", "La maison est grande.", "ACCORD", "accord.attribut"),

    # ------------------------------------------------------------------
    # ACCORD / accord.pp_etre (3 cas)
    # ------------------------------------------------------------------
    CasRecall("Elle est arrive en retard.", "Elle est arrivée en retard.", "ACCORD", "accord.pp_etre"),
    CasRecall("Elles sont parti ce matin.", "Elles sont parties ce matin.", "ACCORD", "accord.pp_etre"),
    CasRecall("Elle est tombe dans les escaliers.", "Elle est tombée dans les escaliers.", "ACCORD", "accord.pp_etre"),

    # ------------------------------------------------------------------
    # CONJ / participe.inf_pp (4 cas)
    # ------------------------------------------------------------------
    CasRecall("J'ai manger toute la tarte.", "J'ai mangé toute la tarte.", "CONJ", "participe.inf_pp"),
    CasRecall("Il a trouver la solution.", "Il a trouvé la solution.", "CONJ", "participe.inf_pp"),
    CasRecall("Nous avons acheter du pain.", "Nous avons acheté du pain.", "CONJ", "participe.inf_pp"),
    CasRecall("Elle a terminer son travail.", "Elle a terminé son travail.", "CONJ", "participe.inf_pp"),

    # ------------------------------------------------------------------
    # CONJ / participe.pp_inf (3 cas)
    # ------------------------------------------------------------------
    CasRecall("Il doit mangé avant de partir.", "Il doit manger avant de partir.", "CONJ", "participe.pp_inf"),
    CasRecall("Elle veut trouvé la solution.", "Elle veut trouver la solution.", "CONJ", "participe.pp_inf"),
    CasRecall("Nous allons terminé bientot.", "Nous allons terminer bientôt.", "CONJ", "participe.pp_inf"),

    # ------------------------------------------------------------------
    # NEGATION / negation.ne (3 cas)
    # ------------------------------------------------------------------
    CasRecall("Il veut pas venir avec nous.", "Il ne veut pas venir avec nous.", "NEGATION", "negation.ne"),
    CasRecall("Elle sait pas quoi repondre.", "Elle ne sait pas quoi répondre.", "NEGATION", "negation.ne"),
    CasRecall("Je comprends pas cette question.", "Je ne comprends pas cette question.", "NEGATION", "negation.ne"),

    # ------------------------------------------------------------------
    # CAPITAL / capitalisation (2 cas)
    # ------------------------------------------------------------------
    CasRecall("il mange dans la cuisine.", "Il mange dans la cuisine.", "CAPITAL", "capitalisation"),
    CasRecall("les enfants jouent dehors.", "Les enfants jouent dehors.", "CAPITAL", "capitalisation"),
]


# ========================================================================
# A2 — Corpus FP (texte propre naturel)
# ========================================================================

# Phrases WiCoPaCo etiquetees "negatif" mais contenant des erreurs reelles.
# Le correcteur les corrige legitimement → exclure du comptage FP.
_WICOPACO_FAUX_NEGATIFS = {
    # Accents manquants
    "batis en cercle",           # batis -> bâtis
    "chateau fort",              # chateau -> château
    "amèricaine",                # amèricaine -> américaine
    "dialècte",                  # dialècte -> dialecte
    "chaine des magasins",       # chaine -> chaîne
    "devaient d'etre",           # etre -> être
    "remplacant a Bugz",         # remplacant -> remplaçant
    "s'élève donc a 4242",       # a -> à
    "l'hopital de Bulovka",      # hopital -> hôpital
    "on reconnait les",          # reconnait -> reconnaît
    "s'entremèle étroitement",   # entremèle -> entremêle
    "pepete le Jeune",           # pepete -> pépète, a -> à
    "annéée 50",                 # annéée -> années
    "chateaumeillant",           # chateaumeillant -> châteaumeillant
    # Mots colles / resegmentation
    "etsujets",                  # etsujets -> et sujets
    # Accords
    "plusieurs type de",         # type -> types
    "un soucis de",              # soucis -> souci (singulier)
    "style vocale",              # vocale -> vocal
    "les caractère rencontré",   # caractère -> caractères
    "bases britannique et",      # britannique -> britanniques
    # Orthographe / accents manquants (lot 3)
    "un reflexe proche",         # reflexe -> réflexe
    "heavy metal britannique",   # metal -> métal
    "sur l'aparence du",         # aparence -> apparence
    "est une confitures de",     # confitures -> confiture (accord)
    "a été arrétée le",          # arrétée -> arrêtée
    "Le medecin légiste",        # medecin -> médecin
    "pénetrent alors",           # pénetrent -> pénètrent
    # Orthographe (lot 4 — pool variable apres filtrage)
    "s'attrappe également",      # attrappe -> attrape
    "deux charge antichars",     # charge -> charges (accord)
    "royalde Blois",             # royalde -> royal de (reseg)
    "on relache le",             # relache -> relâche
    "chinois des telecoms",      # telecoms -> télécoms
    "un cimétière militaire",    # cimétière -> cimetière
    # Lot 5 — pool variable
    "il y a le chateau",         # chateau -> château
    "de nombreux logiciel",      # logiciel -> logiciels
    "pas survecu aux",           # survecu -> survécu
    "ont entrainé le plus",      # entrainé -> entraîné (accent manquant)
    "d'asteroïdes similaires",   # asteroïdes -> astéroïdes
}


def _est_faux_negatif_wicopaco(phrase: str) -> bool:
    """Verifie si une phrase WiCoPaCo contient une erreur reelle connue."""
    for marqueur in _WICOPACO_FAUX_NEGATIFS:
        if marqueur in phrase:
            return True
    return False


def charger_fp_wicopaco(n: int = 200, seed: int = 42) -> list[str]:
    """Charge n phrases du corpus WiCoPaCo negatif (Wikipedia propre).

    Format TSV : negatif<TAB>texte<TAB>texte
    Exclut les phrases contenant des erreurs reelles connues.
    """
    if not WICOPACO_PATH.exists():
        print(f"  [WARN] WiCoPaCo non trouve : {WICOPACO_PATH}")
        return []

    phrases: list[str] = []
    with open(WICOPACO_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] == "negatif":
                texte = parts[1].strip()
                if texte and not _est_faux_negatif_wicopaco(texte):
                    phrases.append(texte)

    rng = random.Random(seed)
    if len(phrases) > n:
        phrases = rng.sample(phrases, n)
    return phrases


def charger_fp_litterature(n: int = 200, seed: int = 42) -> list[str]:
    """Charge n phrases de litterature classique (metadata.csv pipe-separated).

    Format : id|texte|texte_normalise
    Filtre : ignore les lignes trop courtes (<20 chars) ou tout en majuscules.
    """
    if not LITTERATURE_BASE.exists():
        print(f"  [WARN] Corpus litterature non trouve : {LITTERATURE_BASE}")
        return []

    metadata_files = sorted(LITTERATURE_BASE.glob("**/metadata.csv"))
    if not metadata_files:
        print(f"  [WARN] Aucun metadata.csv dans {LITTERATURE_BASE}")
        return []

    phrases: list[str] = []
    for fpath in metadata_files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    texte = parts[1].strip()
                    # Filtrer les titres (ALL CAPS), phrases courtes, et artefacts OCR
                    # Filtre mots-colles : tout token >20 chars = OCR concatene
                    has_long_token = any(
                        len(tok) > 20 for tok in texte.split()
                    )
                    if (len(texte) >= 20
                            and not texte.isupper()
                            and "_" not in texte
                            and not has_long_token):
                        phrases.append(texte)

    rng = random.Random(seed)
    if len(phrases) > n:
        phrases = rng.sample(phrases, n)
    return phrases


def charger_fp_ciblees() -> list[str]:
    """Charge les phrases ciblees de test_fp_propre.SOURCES."""
    phrases: list[str] = []
    try:
        from test_fp_propre import SOURCES
        for source in SOURCES:
            phrases.extend(source.phrases)
    except ImportError:
        print("  [WARN] test_fp_propre non importable, phrases ciblees ignorees")
    return phrases


# ========================================================================
# Normalisation
# ========================================================================

def normaliser(phrase: str) -> str:
    """Normalisation legere pour comparer sortie correcteur vs attendue.

    - Strip espaces
    - Strip ponctuation finale (. ! ? …)
    - Lowercase
    - Normalise espaces multiples
    """
    phrase = phrase.strip()
    while phrase and phrase[-1] in ".!?…":
        phrase = phrase[:-1]
    return " ".join(phrase.lower().split())


# ========================================================================
# Main
# ========================================================================

@dataclass
class EchecRecall:
    """Detail d'un echec recall."""
    cas: CasRecall
    obtenu: str


@dataclass
class FPDetail:
    """Detail d'un faux positif."""
    original: str
    corrige: str


@dataclass
class ResultatSource:
    """Resultat FP pour une source."""
    nom: str
    n_phrases: int = 0
    n_fp: int = 0
    details: list[FPDetail] = field(default_factory=list)


PROFILES: dict[str, dict[str, bool]] = {
    # Tout actif (defaut)
    "all": {},
    # Regles a 0 FP supplementaire + regles avec gardes corriges
    "safe": {
        "activer_p2g_global": False,
        "activer_verbe_p2g": False,
        "activer_accent_p2g": False,
        "activer_accord_attribut": False,
        "activer_pp_etre": False,
        # ON : accords, homophones_struct, homophones_p2g,
        #      accent_lexique, negation, accord_det_nom
    },
    # Etape 3 completement off
    "baseline": {
        "activer_p2g_global": False,
        "activer_homophones_p2g": False,
        "activer_accords": False,
        "activer_accord_det_nom": False,
        "activer_accord_attribut": False,
        "activer_pp_etre": False,
        "activer_verbe_p2g": False,
        "activer_accent_p2g": False,
        "activer_accent_lexique": False,
        "activer_negation": False,
        "activer_homophones_struct": False,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test de regression — Recall + Faux Positifs",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Afficher les details des FP")
    parser.add_argument("--profile", "-p", default="all",
                        choices=list(PROFILES.keys()),
                        help="Profil de config (all, safe, baseline)")
    args = parser.parse_args()

    # --- Chargement correcteur ---
    print("Chargement du lexique...")
    from lectura_lexique import Lexique
    lexique = Lexique(str(LEXIQUE_PATH))

    print(f"Chargement du correcteur V6 (profil: {args.profile})...")
    from lectura_correcteur import CorrecteurV6, CorrecteurV6Config
    overrides = PROFILES.get(args.profile, {})
    config = CorrecteurV6Config(**overrides)
    correcteur = CorrecteurV6(lexique, config=config)

    # Afficher les regles actives
    regles_status = []
    for attr in sorted(vars(config)):
        if attr.startswith("activer_") and attr != "activer_negation":
            val = getattr(config, attr)
            nom = attr.replace("activer_", "")
            regles_status.append(f"  {nom}: {'ON' if val else 'OFF'}")
    # negation
    regles_status.append(f"  negation: {'ON' if config.activer_negation else 'OFF'}")
    print("  Regles etape 3 :")
    for s in regles_status:
        print(s)
    print()

    t0 = time.time()

    # ==================================================================
    # PARTIE 1 — RECALL
    # ==================================================================

    # Resultats par categorie et par regle
    resultats_cat: dict[str, list[bool]] = {}
    resultats_regle: dict[str, list[bool]] = {}
    echecs: list[EchecRecall] = []

    for cas in CORPUS_RECALL:
        result = correcteur.corriger(cas.fautive)
        norm_attendue = normaliser(cas.attendue)
        norm_obtenue = normaliser(result.phrase_corrigee)
        ok = norm_attendue == norm_obtenue

        resultats_cat.setdefault(cas.categorie, []).append(ok)
        resultats_regle.setdefault(cas.regle, []).append(ok)

        if not ok:
            echecs.append(EchecRecall(cas=cas, obtenu=result.phrase_corrigee))

    total_recall = len(CORPUS_RECALL)
    total_ok = sum(1 for cat_res in resultats_cat.values() for v in cat_res if v)
    pct_recall = 100 * total_ok / total_recall if total_recall > 0 else 0

    # ==================================================================
    # PARTIE 2 — FAUX POSITIFS
    # ==================================================================

    sources_fp: list[tuple[str, list[str]]] = [
        ("WiCoPaCo Wikipedia", charger_fp_wicopaco()),
        ("Litterature classique", charger_fp_litterature()),
        ("Phrases ciblees", charger_fp_ciblees()),
    ]

    resultats_fp: list[ResultatSource] = []
    total_fp_phrases = 0
    total_fp_count = 0

    for nom, phrases in sources_fp:
        if not phrases:
            continue
        res = ResultatSource(nom=nom, n_phrases=len(phrases))
        for phrase in phrases:
            result = correcteur.corriger(phrase)
            orig_norm = normaliser(phrase)
            corr_norm = normaliser(result.phrase_corrigee)
            if orig_norm != corr_norm:
                res.n_fp += 1
                res.details.append(FPDetail(original=phrase,
                                            corrige=result.phrase_corrigee))
        resultats_fp.append(res)
        total_fp_phrases += res.n_phrases
        total_fp_count += res.n_fp

    pct_fp = 100 * total_fp_count / total_fp_phrases if total_fp_phrases > 0 else 0

    duree = time.time() - t0

    # ==================================================================
    # RAPPORT
    # ==================================================================

    sep = "=" * 72
    print(sep)
    print("  TEST CORRECTEUR — Recall + Faux Positifs")
    print(sep)

    # --- Recall par categorie ---
    print()
    print("--- RECALL (corrections attendues) ---")
    print(f"  {'Categorie':<16s} | {'Reussi':>7s} | {'Total':>5s} | {'%':>6s}")
    print("  " + "-" * 47)

    ordre_cat = ["ORTH", "HOMO", "ACCORD", "CONJ", "NEGATION", "CAPITAL"]
    for cat in ordre_cat:
        if cat in resultats_cat:
            vals = resultats_cat[cat]
            n_ok = sum(vals)
            n_tot = len(vals)
            pct = 100 * n_ok / n_tot
            print(f"  {cat:<16s} | {n_ok:>7d} | {n_tot:>5d} | {pct:>5.1f}%")
    print("  " + "-" * 47)
    print(f"  {'TOTAL':<16s} | {total_ok:>7d} | {total_recall:>5d} | {pct_recall:>5.1f}%")

    # --- Detail echecs ---
    if echecs:
        print()
        print("  Detail des echecs :")
        for i, e in enumerate(echecs):
            print(f"  [{i+1}] {e.cas.categorie}/{e.cas.regle} :")
            print(f"      fautive  : \"{e.cas.fautive}\"")
            print(f"      attendu  : \"{e.cas.attendue}\"")
            print(f"      obtenu   : \"{e.obtenu}\"")

    # --- Detail recall par regle (verbose) ---
    if args.verbose:
        print()
        print("  Detail par regle :")
        for regle in sorted(resultats_regle):
            vals = resultats_regle[regle]
            n_ok = sum(vals)
            n_tot = len(vals)
            marker = "OK" if n_ok == n_tot else "ECHEC"
            print(f"    {regle:<25s}  {n_ok}/{n_tot}  {marker}")

    # --- Faux positifs ---
    print()
    print("--- FAUX POSITIFS (texte propre) ---")
    print(f"  {'Source':<25s} | {'Phrases':>8s} | {'FP':>4s} | {'%FP':>6s}")
    print("  " + "-" * 51)
    for res in resultats_fp:
        pct_src = 100 * res.n_fp / res.n_phrases if res.n_phrases > 0 else 0
        print(f"  {res.nom:<25s} | {res.n_phrases:>8d} | {res.n_fp:>4d} | {pct_src:>5.1f}%")
    print("  " + "-" * 51)
    print(f"  {'TOTAL':<25s} | {total_fp_phrases:>8d} | {total_fp_count:>4d} | {pct_fp:>5.1f}%")

    # --- Detail FP ---
    if total_fp_count > 0:
        for res in resultats_fp:
            if res.details:
                print(f"\n  FP detail : {res.nom} ({res.n_fp} FP)")
                for j, fp in enumerate(res.details):
                    print(f"    [{j+1:2d}] \"{fp.original}\"")
                    print(f"         -> \"{fp.corrige}\"")

    # --- Verdict ---
    print()
    print(f"  Duree : {duree:.1f}s")
    print()
    if pct_recall >= 80 and pct_fp == 0:
        print(f"  VERDICT : OK — Recall {pct_recall:.1f}%, FP {pct_fp:.1f}%")
    elif pct_fp == 0:
        print(f"  VERDICT : ACCEPTABLE — Recall {pct_recall:.1f}% (< 80%), FP {pct_fp:.1f}%")
    else:
        print(f"  VERDICT : ECHEC — Recall {pct_recall:.1f}%, FP {pct_fp:.1f}%")
    print()


if __name__ == "__main__":
    main()
