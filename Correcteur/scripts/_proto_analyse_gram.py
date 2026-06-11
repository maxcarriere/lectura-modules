#!/usr/bin/env python3
"""Prototype — Evaluer l'analyse grammaticale Pass 2 avec candidats elargis.

Affiche l'analyse POS+MORPHO detaillee pour 100 phrases couvrant :
- DET+NOM accord nombre (les chat, des enfant, ...)
- DET+ADJ+NOM accord genre (le petite chat, la petit fille, ...)
- AUX etre + PP accord (ils sont parti, elles sont parties, ...)
- AUX avoir + PP (il a mange, elle a mangees, ...)
- PRO+VER conjugaison (ils mange, elle mangent, ...)
- Homophones POS-dependants (a/à, ou/où, et/est, on/ont, ...)
- Phrases correctes (pas de correction attendue)

Pour chaque phrase :
1. Tokenise et analyse via le module _analyse_grammaticale
2. Affiche par mot : POS, nombre, genre, personne, temps, confiance, ancre
3. Affiche les corrections proposees (forme_corrigee) quand differentes
4. Detecte les conflits (DET Plur + NOM Sing, etc.)
5. Affiche le score PM trigramme
"""
from __future__ import annotations

import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"

# ---- 100 phrases de test ----
# (phrase_erronee_ou_correcte, type_erreur)
PHRASES = [
    # === DET + NOM : accord nombre (20) ===
    ("Les chat dort sur le lit.", "det_nom_nombre"),
    ("Des enfant jouent dans la cour.", "det_nom_nombre"),
    ("Les maison sont grandes.", "det_nom_nombre"),
    ("Les chien courent vite.", "det_nom_nombre"),
    ("Mes ami viendront demain.", "det_nom_nombre"),
    ("Ces fleur sont magnifiques.", "det_nom_nombre"),
    ("Tes livre sont sur la table.", "det_nom_nombre"),
    ("Nos voisin font du bruit.", "det_nom_nombre"),
    ("Les oiseau chantent le matin.", "det_nom_nombre"),
    ("Des voiture passent dans la rue.", "det_nom_nombre"),
    ("Les fenetre sont ouvertes.", "det_nom_nombre"),
    ("Les arbre perdent leurs feuilles.", "det_nom_nombre"),
    ("Ses parent arrivent ce soir.", "det_nom_nombre"),
    ("Les pomme sont rouges.", "det_nom_nombre"),
    ("Les eleve ecoutent le professeur.", "det_nom_nombre"),
    ("Plusieurs personne attendent.", "det_nom_nombre"),
    ("Quelques jour suffisent.", "det_nom_nombre"),
    ("Tous les garcon jouent dehors.", "det_nom_nombre"),
    ("Les premiere annee sont difficiles.", "det_nom_nombre"),
    ("Les professeur corrigent les copies.", "det_nom_nombre"),

    # === DET + ADJ + NOM : accord genre (10) ===
    ("Le petite chat dort.", "det_adj_genre"),
    ("La petit fille joue.", "det_adj_genre"),
    ("Un grande maison.", "det_adj_genre"),
    ("Une grand jardin.", "det_adj_genre"),
    ("Le belle femme chante.", "det_adj_genre"),
    ("La beau garcon court.", "det_adj_genre"),
    ("Un nouvelle voiture rouge.", "det_adj_genre"),
    ("Le premiere fois.", "det_adj_genre"),
    ("La dernier chance.", "det_adj_genre"),
    ("Le petit filles jouent.", "det_adj_genre"),

    # === AUX etre + PP accord (10) ===
    ("Ils sont parti hier soir.", "aux_pp_etre"),
    ("Elles sont arrive en retard.", "aux_pp_etre"),
    ("Elle est alle au marche.", "aux_pp_etre"),
    ("Nous sommes reste a la maison.", "aux_pp_etre"),
    ("Les filles sont monte au grenier.", "aux_pp_etre"),
    ("Ils sont tombe dans la boue.", "aux_pp_etre"),
    ("Elle est parti sans rien dire.", "aux_pp_etre"),
    ("Elles sont reste dehors.", "aux_pp_etre"),
    ("Les enfants sont sorti jouer.", "aux_pp_etre"),
    ("Il est reste toute la journee.", "aux_pp_etre"),

    # === AUX avoir + PP (5) ===
    ("Il a mangees la pomme.", "aux_pp_avoir"),
    ("Elle a mange les fruits.", "aux_pp_avoir"),
    ("Les filles ont mangees les gateaux.", "aux_pp_avoir"),
    ("Il a pris les livres.", "aux_pp_avoir"),
    ("Nous avons fini le travail.", "aux_pp_avoir"),

    # === PRO + VER conjugaison (15) ===
    ("Ils mange du pain.", "pro_ver_conj"),
    ("Elle mangent au restaurant.", "pro_ver_conj"),
    ("Je manges une pomme.", "pro_ver_conj"),
    ("Tu mange le gateau.", "pro_ver_conj"),
    ("Nous mangez ensemble.", "pro_ver_conj"),
    ("Vous mangeons le soir.", "pro_ver_conj"),
    ("Il finissent le travail.", "pro_ver_conj"),
    ("Ils finis son exercice.", "pro_ver_conj"),
    ("Les enfants joue dans le jardin.", "pro_ver_conj"),
    ("Le chat dorment sur le canape.", "pro_ver_conj"),
    ("Les oiseaux chante le matin.", "pro_ver_conj"),
    ("Mon frere jouent au football.", "pro_ver_conj"),
    ("Ma soeur mange.", "pro_ver_conj"),
    ("Les eleves travaille dur.", "pro_ver_conj"),
    ("On mangent bien ici.", "pro_ver_conj"),

    # === Homophones POS (15) ===
    ("Il a raison de partir.", "homophone_correct"),
    ("Il va a la maison.", "homophone_a_prep"),
    ("Il a manger.", "homophone_a_inf"),
    ("Elle mange ou elle dort.", "homophone_correct"),
    ("Je sais ou il habite.", "homophone_ou_rel"),
    ("Mon chat et gros.", "homophone_est"),
    ("Il est content et il part.", "homophone_correct"),
    ("On a faim.", "homophone_correct"),
    ("Ils on faim.", "homophone_ont"),
    ("Se chat est noir.", "homophone_ce"),
    ("Ce sont mes amis.", "homophone_correct"),
    ("Il ce trompe souvent.", "homophone_se"),
    ("Son chat mange.", "homophone_correct"),
    ("Ils sont arrive.", "aux_pp_etre"),
    ("Sa voiture est rouge.", "homophone_correct"),

    # === Phrases correctes — AUCUNE correction attendue (25) ===
    ("Les petits chats mangent du fromage.", "correct"),
    ("Elle est partie hier soir.", "correct"),
    ("Nous avons fini le travail.", "correct"),
    ("Les enfants jouent dans la cour.", "correct"),
    ("Il mange une pomme rouge.", "correct"),
    ("Ma soeur est grande.", "correct"),
    ("Les oiseaux chantent le matin.", "correct"),
    ("Vous etes les bienvenus.", "correct"),
    ("Ces fleurs sont magnifiques.", "correct"),
    ("Je suis content de te voir.", "correct"),
    ("Tu manges trop vite.", "correct"),
    ("Ils ont mange le gateau.", "correct"),
    ("Les filles sont arrivees en retard.", "correct"),
    ("Mon frere joue au football.", "correct"),
    ("Nous regardons la television.", "correct"),
    ("Elle lit un livre interessant.", "correct"),
    ("Le professeur corrige les copies.", "correct"),
    ("Les arbres perdent leurs feuilles.", "correct"),
    ("Il fait beau aujourd'hui.", "correct"),
    ("Les eleves ecoutent le professeur.", "correct"),
    ("On mange bien dans ce restaurant.", "correct"),
    ("Ses parents arrivent ce soir.", "correct"),
    ("La petite fille court dans le jardin.", "correct"),
    ("Ils sont partis sans rien dire.", "correct"),
    ("Elle a pris le train de huit heures.", "correct"),
]


def tokeniser_simple(phrase: str) -> list[str]:
    """Tokenise en gardant seulement les mots (pas la ponctuation)."""
    import re
    return [m.group() for m in re.finditer(r"[\w]+(?:['\u2019][\w]+)*", phrase)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prototype analyse grammaticale")
    parser.add_argument("--expand", action="store_true",
                        help="Activer les candidats elargis (morpho + homophones)")
    parser.add_argument("--no-morpho", action="store_true",
                        help="Desactiver les variantes morphologiques")
    parser.add_argument("--no-homophones", action="store_true",
                        help="Desactiver les variantes homophones")
    parser.add_argument("--penalite-morpho", type=float, default=-3.0,
                        help="Penalite emission variantes morpho (defaut: -3.0)")
    parser.add_argument("--penalite-homophone", type=float, default=-5.0,
                        help="Penalite emission homophones (defaut: -5.0)")
    parser.add_argument("--g2p", action="store_true",
                        help="Utiliser G2P Unifie V2 comme prior d'emission")
    cli_args = parser.parse_args()

    from lectura_lexique import Lexique
    from lectura_correcteur._utils import LexiqueNormalise
    from lectura_correcteur._pos_ngram import PosNgram
    from lectura_correcteur._analyse_grammaticale import (
        analyser_phrase,
        formater_analyse,
        score_pm_sequence,
    )

    # Optionnel : LM homophones
    lm_homophones = None
    try:
        from lectura_correcteur._lm_homophones import LMHomophones
        lm_db = os.path.join(
            _PROJECT_ROOT, "src", "lectura_correcteur", "data", "homophones_trigrams.db",
        )
        if os.path.exists(lm_db):
            lm_homophones = LMHomophones(lm_db)
            print(f"LM Homophones charge : {lm_db}")
    except Exception as e:
        print(f"LM Homophones non disponible : {e}")

    print("Chargement lexique...")
    lexique = Lexique(LEXIQUE_DB)
    lex = LexiqueNormalise(lexique)

    # Optionnel : G2P Unifie V2
    g2p_tagger = None
    if cli_args.g2p:
        try:
            from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
            g2p_tagger = creer_adapter_g2p_unifie()
            if g2p_tagger:
                print("G2P Unifie V2 charge")
            else:
                print("G2P Unifie V2 non disponible (fonctionnement sans prior)")
        except Exception as e:
            print(f"G2P Unifie V2 erreur: {e}")

    print("Chargement POS n-gram...")
    pos_ngram_db = os.path.join(
        _PROJECT_ROOT, "src", "lectura_correcteur", "data", "pos_ngram.db",
    )
    if not os.path.exists(pos_ngram_db):
        print(f"ERREUR: pos_ngram.db introuvable: {pos_ngram_db}")
        sys.exit(1)
    pos_ngram = PosNgram(pos_ngram_db)

    # Flags expand
    do_expand_morpho = cli_args.expand and not cli_args.no_morpho
    do_expand_homophones = cli_args.expand and not cli_args.no_homophones

    mode_label = "baseline"
    if cli_args.expand:
        parts = []
        if do_expand_morpho:
            parts.append(f"morpho(p={cli_args.penalite_morpho})")
        if do_expand_homophones:
            parts.append(f"homo(p={cli_args.penalite_homophone})")
        mode_label = " + ".join(parts) if parts else "expand (aucun)"
    if cli_args.g2p:
        mode_label += " + G2P"

    print(f"\n{'='*80}")
    print(f"  PROTOTYPE — Analyse grammaticale Pass 2 ({len(PHRASES)} phrases)")
    print(f"  Mode : {mode_label}")
    print(f"{'='*80}\n")

    # Statistiques globales
    stats = {
        "total_mots": 0,
        "ancres": 0,
        "conflits": 0,
        "mots_avec_conflit": 0,
        "mots_corriges": 0,
        "confiance_sum": 0.0,
        "par_type": {},
    }

    t0 = time.time()

    for idx, (phrase, typ) in enumerate(PHRASES):
        mots = tokeniser_simple(phrase)
        if not mots:
            continue

        analyses = analyser_phrase(
            mots, lex, pos_ngram,
            lm_homophones=lm_homophones,
            tagger=g2p_tagger,
            expand_morpho=do_expand_morpho,
            expand_homophones=do_expand_homophones,
            penalite_morpho=cli_args.penalite_morpho,
            penalite_homophone=cli_args.penalite_homophone,
        )
        pm_score = score_pm_sequence(analyses, pos_ngram)

        # Header
        print(f"[{idx+1:3d}] ({typ})")
        print(formater_analyse(analyses, phrase))
        print(f"  PM score: {pm_score:.1f}")

        # Corrections
        corrections = [a for a in analyses if a.forme_corrigee]
        if corrections:
            for a in corrections:
                print(f"  >> CORRECTION: {a.forme} -> {a.forme_corrigee} ({a.pm_tag})")

        # Conflits
        conflits = [a for a in analyses if a.conflits]
        if conflits:
            print(f"  >> {len(conflits)} conflit(s) detecte(s)")

        print()

        # Stats
        stats["total_mots"] += len(analyses)
        stats["ancres"] += sum(1 for a in analyses if a.ancre)
        stats["mots_corriges"] += len(corrections)
        n_conflits = sum(len(a.conflits) for a in analyses)
        stats["conflits"] += n_conflits
        stats["mots_avec_conflit"] += len(conflits)
        stats["confiance_sum"] += sum(a.confiance for a in analyses)

        if typ not in stats["par_type"]:
            stats["par_type"][typ] = {"phrases": 0, "conflits_detectes": 0, "corrections": 0}
        stats["par_type"][typ]["phrases"] += 1
        if n_conflits > 0:
            stats["par_type"][typ]["conflits_detectes"] += 1
        stats["par_type"][typ]["corrections"] += len(corrections)

    elapsed = time.time() - t0

    # Rapport global
    print(f"\n{'='*80}")
    print("  STATISTIQUES GLOBALES")
    print(f"{'='*80}")
    print(f"  Mode                  : {mode_label}")
    print(f"  Phrases analysees     : {len(PHRASES)}")
    print(f"  Mots analyses         : {stats['total_mots']}")
    print(f"  Ancres                : {stats['ancres']} ({100*stats['ancres']/max(stats['total_mots'],1):.1f}%)")
    print(f"  Confiance moyenne     : {stats['confiance_sum']/max(stats['total_mots'],1):.3f}")
    print(f"  Conflits detectes     : {stats['conflits']}")
    print(f"  Mots avec conflit     : {stats['mots_avec_conflit']}")
    print(f"  Mots corriges         : {stats['mots_corriges']}")
    print(f"  Temps                 : {elapsed:.2f}s")

    print(f"\n  {'Type':<25s} {'Phrases':>8s} {'Avec conflit':>12s} {'Taux':>8s} {'Corrections':>12s}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*8} {'-'*12}")
    for typ, info in sorted(stats["par_type"].items()):
        n = info["phrases"]
        c = info["conflits_detectes"]
        corr = info.get("corrections", 0)
        taux = 100 * c / max(n, 1)
        print(f"  {typ:<25s} {n:>8d} {c:>12d} {taux:>7.1f}% {corr:>12d}")

    # Ce qu'on cherche a valider
    print(f"\n{'='*80}")
    print("  QUESTIONS CLES")
    print(f"{'='*80}")

    n_mots = stats["total_mots"]
    n_ancres = stats["ancres"]
    correct_phrases = [p for p, t in PHRASES if t == "correct"]
    erreur_phrases = [p for p, t in PHRASES if t != "correct"]

    print(f"  1. Ancres fiables ? {n_ancres}/{n_mots} mots sont ancres ({100*n_ancres/max(n_mots,1):.1f}%)")
    print(f"  2. Conflits sur phrases erronees : ", end="")
    err_with_conflict = stats["par_type"].get("correct", {}).get("conflits_detectes", 0)
    err_types = [t for t in stats["par_type"] if t != "correct"]
    err_detected = sum(stats["par_type"][t]["conflits_detectes"] for t in err_types)
    err_total = sum(stats["par_type"][t]["phrases"] for t in err_types)
    print(f"{err_detected}/{err_total} phrases erronees ont des conflits ({100*err_detected/max(err_total,1):.1f}%)")

    fp_conflits = stats["par_type"].get("correct", {}).get("conflits_detectes", 0)
    fp_total = stats["par_type"].get("correct", {}).get("phrases", 0)
    print(f"  3. Faux positifs (conflits sur correct) : {fp_conflits}/{fp_total} ({100*fp_conflits/max(fp_total,1):.1f}%)")

    print()
    pos_ngram.close()


if __name__ == "__main__":
    main()
