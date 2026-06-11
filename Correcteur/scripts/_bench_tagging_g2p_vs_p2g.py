#!/usr/bin/env python3
"""Compare le tagging POS/Morpho G2P vs P2G sur phrases fautives.

Hypothese : sur une phrase avec fautes d'accord, le G2P est trompe par
l'orthographe (il voit "enfant" singulier et tague singulier) tandis que
le P2G, partant de l'IPA, doit reconstruire nombre/genre depuis le contexte
phonetique et pourrait mieux deviner la morphologie attendue.

Protocole :
  1. Phrase fautive → G2P → POS/morpho (biaise par l'ortho)
  2. Phrase fautive → G2P → IPA → P2G → POS/morpho (depuis phonetique)
  3. Phrase correcte → G2P → POS/morpho (reference)
  4. Comparer : qui a raison, G2P ou P2G ?
"""

from __future__ import annotations

import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.disable(logging.WARNING)


# Normalisation UD → FR (pour G2P qui retourne en FR via _extraire_morpho_fr)
_NB_NORM = {"p": "Plur", "s": "Sing", "Plur": "Plur", "Sing": "Sing", "_": "_", "": "_"}
_GN_NORM = {"m": "Masc", "f": "Fem", "Masc": "Masc", "Fem": "Fem", "_": "_", "": "_"}
_PS_NORM = {"1": "1", "2": "2", "3": "3", "_": "_", "": "_"}


def _norm_nb(v: str) -> str:
    return _NB_NORM.get(v, v)


def _norm_gn(v: str) -> str:
    return _GN_NORM.get(v, v)


def _norm_ps(v: str) -> str:
    return _PS_NORM.get(v, v)


# Paires : (phrase_fautive, phrase_correcte)
PAIRES = [
    # Accords nombre
    ("les enfant mange des pomme", "les enfants mangent des pommes"),
    ("les chat dort", "les chats dorment"),
    ("mes ami arrivent demain", "mes amis arrivent demain"),
    ("les voiture rouge", "les voitures rouges"),
    ("des beau jardin", "des beaux jardins"),
    ("les livre sont sur la table", "les livres sont sur la table"),

    # Participes
    ("il a manger", "il a mange"),
    ("elle a acheter un livre", "elle a achete un livre"),
    ("il a prit le train", "il a pris le train"),
    ("ils ont preparer le repas", "ils ont prepare le repas"),

    # Accords genre
    ("une bon idee", "une bonne idee"),
    ("ces petit fille", "ces petites filles"),
    ("les maison sont grande", "les maisons sont grandes"),
]


def main():
    # Charger G2P
    from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
    g2p = creer_adapter_g2p_unifie()
    if g2p is None:
        print("ERREUR: G2P indisponible")
        return

    # Charger P2G
    from lectura_correcteur._adapter_p2g import creer_adapter_p2g
    p2g = creer_adapter_p2g()
    if p2g is None:
        print("ERREUR: P2G indisponible")
        return

    p2g_engine = p2g._engine

    total_mots_cibles = 0
    g2p_nb_ok = 0
    p2g_nb_ok = 0
    g2p_gn_ok = 0
    p2g_gn_ok = 0
    g2p_ps_ok = 0
    p2g_ps_ok = 0

    for phrase_fautive, phrase_correcte in PAIRES:
        mots_fautifs = phrase_fautive.lower().split()
        mots_corrects = phrase_correcte.lower().split()

        if len(mots_fautifs) != len(mots_corrects):
            print(f"SKIP (longueur differente): {phrase_fautive}")
            continue

        diffs = [(i, mf, mc) for i, (mf, mc) in
                 enumerate(zip(mots_fautifs, mots_corrects)) if mf != mc]

        if not diffs:
            continue

        print(f"\n{'='*80}")
        print(f"  FAUTIVE:  {phrase_fautive}")
        print(f"  CORRECTE: {phrase_correcte}")
        print(f"{'='*80}")

        # 1. G2P sur phrase fautive
        tags_g2p_fautive = g2p.tag_words_rich(mots_fautifs)

        # 2. G2P sur phrase correcte (reference)
        tags_g2p_correcte = g2p.tag_words_rich(mots_corrects)

        # 3. IPA depuis G2P
        ipa_fautive = []
        for i, mot in enumerate(mots_fautifs):
            phone = tags_g2p_fautive[i].get("g2p", "") or (g2p.prononcer(mot) or "")
            ipa_fautive.append(phone)

        # 4. P2G : IPA → ortho + POS/morpho (cles UD)
        p2g_result = p2g_engine.analyser_avec_alternatives(
            ipa_fautive, ortho_words=mots_fautifs, k=5,
        )
        p2g_pos_list = p2g_result.get("pos", [])
        p2g_morpho_raw = p2g_result.get("morpho", {})
        p2g_ortho = p2g_result.get("ortho", [])
        p2g_confiance = p2g_result.get("confiance", [])
        p2g_alternatives = p2g_result.get("alternatives", [])
        p2g_morpho_scores = p2g_result.get("morpho_scores", {})

        # Extraire P2G morpho par index (cles UD)
        def p2g_nb(i):
            vals = p2g_morpho_raw.get("Number", [])
            return vals[i] if i < len(vals) else "_"

        def p2g_gn(i):
            vals = p2g_morpho_raw.get("Gender", [])
            return vals[i] if i < len(vals) else "_"

        def p2g_ps(i):
            vals = p2g_morpho_raw.get("Person", [])
            return vals[i] if i < len(vals) else "_"

        # Tableau mot par mot
        hdr = (f"  {'Mot':12s} {'IPA':10s} "
               f"{'G2P-Nb':7s} {'P2G-Nb':7s} {'REF-Nb':7s} "
               f"{'G2P-Gn':7s} {'P2G-Gn':7s} {'REF-Gn':7s} "
               f"{'P2G-ortho':12s} {'Conf':5s}")
        print(f"\n{hdr}")
        print(f"  {'-'*12} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*12} {'-'*5}")

        for i, mot in enumerate(mots_fautifs):
            # G2P fautive (retourne en FR via _extraire_morpho_fr)
            g_nb = _norm_nb(tags_g2p_fautive[i].get("nombre", "_"))
            g_gn = _norm_gn(tags_g2p_fautive[i].get("genre", "_"))

            # P2G (cles UD directement)
            p_nb = p2g_nb(i)
            p_gn = p2g_gn(i)

            # Reference (G2P sur phrase correcte)
            r_nb = _norm_nb(tags_g2p_correcte[i].get("nombre", "_"))
            r_gn = _norm_gn(tags_g2p_correcte[i].get("genre", "_"))

            ipa = ipa_fautive[i] if i < len(ipa_fautive) else ""
            p_ortho = p2g_ortho[i] if i < len(p2g_ortho) else ""
            p_conf = p2g_confiance[i] if i < len(p2g_confiance) else 0.0

            is_diff = any(idx == i for idx, _, _ in diffs)
            marker = " >>>" if is_diff else "    "

            # Marquer OK/FAUX pour nb
            g_nb_mark = "ok" if g_nb == r_nb else "!!"
            p_nb_mark = "ok" if p_nb == r_nb else "!!"
            g_gn_mark = "ok" if g_gn == r_gn else "!!"
            p_gn_mark = "ok" if p_gn == r_gn else "!!"

            print(f"{marker}{mot:12s} {ipa:10s} "
                  f"{g_nb:4s}{g_nb_mark:>3s} {p_nb:4s}{p_nb_mark:>3s} {r_nb:7s} "
                  f"{g_gn:4s}{g_gn_mark:>3s} {p_gn:4s}{p_gn_mark:>3s} {r_gn:7s} "
                  f"{p_ortho:12s} {p_conf:.3f}")

        # Analyse mots fautifs
        print(f"\n  Analyse des mots fautifs:")
        for idx, mot_fautif, mot_correct in diffs:
            total_mots_cibles += 1

            # Reference
            ref_nb = _norm_nb(tags_g2p_correcte[idx].get("nombre", "_"))
            ref_gn = _norm_gn(tags_g2p_correcte[idx].get("genre", "_"))
            ref_ps = _norm_ps(tags_g2p_correcte[idx].get("personne", "_"))

            # G2P fautive
            g_nb = _norm_nb(tags_g2p_fautive[idx].get("nombre", "_"))
            g_gn = _norm_gn(tags_g2p_fautive[idx].get("genre", "_"))
            g_ps = _norm_ps(tags_g2p_fautive[idx].get("personne", "_"))

            # P2G
            p_nb = p2g_nb(idx)
            p_gn = p2g_gn(idx)
            p_ps = p2g_ps(idx)

            # Comptage
            if g_nb == ref_nb:
                g2p_nb_ok += 1
            if p_nb == ref_nb:
                p2g_nb_ok += 1
            if g_gn == ref_gn:
                g2p_gn_ok += 1
            if p_gn == ref_gn:
                p2g_gn_ok += 1
            if g_ps == ref_ps:
                g2p_ps_ok += 1
            if p_ps == ref_ps:
                p2g_ps_ok += 1

            # P2G ortho
            p_ortho = p2g_ortho[idx] if idx < len(p2g_ortho) else ""
            p_conf = p2g_confiance[idx] if idx < len(p2g_confiance) else 0.0
            p2g_ortho_ok = p_ortho.lower() == mot_correct.lower()

            # Morpho scores P2G
            nb_scores = p2g_morpho_scores.get("Number", [])
            gn_scores = p2g_morpho_scores.get("Gender", [])
            nb_s = nb_scores[idx] if idx < len(nb_scores) else []
            gn_s = gn_scores[idx] if idx < len(gn_scores) else []

            g_mark = lambda v, r: "OK" if v == r else "FAUX"

            print(f"    {mot_fautif} → {mot_correct}")
            print(f"      REF    : nb={ref_nb:5s} gn={ref_gn:5s} ps={ref_ps}")
            print(f"      G2P    : nb={g_nb:5s} [{g_mark(g_nb, ref_nb):4s}]  "
                  f"gn={g_gn:5s} [{g_mark(g_gn, ref_gn):4s}]  "
                  f"ps={g_ps:2s} [{g_mark(g_ps, ref_ps):4s}]")
            print(f"      P2G    : nb={p_nb:5s} [{g_mark(p_nb, ref_nb):4s}]  "
                  f"gn={p_gn:5s} [{g_mark(p_gn, ref_gn):4s}]  "
                  f"ps={p_ps:2s} [{g_mark(p_ps, ref_ps):4s}]")
            if nb_s:
                nb_str = ", ".join(f"{l}={v:.2f}" for l, v in nb_s[:3])
                print(f"      P2G Nb scores: {nb_str}")
            if gn_s:
                gn_str = ", ".join(f"{l}={v:.2f}" for l, v in gn_s[:3])
                print(f"      P2G Gn scores: {gn_str}")
            print(f"      P2G ortho: {p_ortho} (conf={p_conf:.3f}) "
                  f"[{'OK' if p2g_ortho_ok else 'FAUX'}]")

            # Alternatives
            alts = p2g_alternatives[idx] if idx < len(p2g_alternatives) else []
            if alts:
                alt_strs = [f"{a}({c:.2f})" for a, c in alts[:5]]
                print(f"      P2G alts: {', '.join(alt_strs)}")

    # Bilan
    n = total_mots_cibles
    print(f"\n\n{'='*80}")
    print(f"  BILAN sur {n} mots fautifs (morpho attendu = ref G2P phrase correcte)")
    print(f"{'='*80}")
    print(f"  {'':20s}  {'G2P':>10s}  {'P2G':>10s}")
    print(f"  {'Nombre correct':20s}  {g2p_nb_ok:>3d}/{n} ({100*g2p_nb_ok/n:.0f}%)  "
          f"{p2g_nb_ok:>3d}/{n} ({100*p2g_nb_ok/n:.0f}%)")
    print(f"  {'Genre correct':20s}  {g2p_gn_ok:>3d}/{n} ({100*g2p_gn_ok/n:.0f}%)  "
          f"{p2g_gn_ok:>3d}/{n} ({100*p2g_gn_ok/n:.0f}%)")
    print(f"  {'Personne correct':20s}  {g2p_ps_ok:>3d}/{n} ({100*g2p_ps_ok/n:.0f}%)  "
          f"{p2g_ps_ok:>3d}/{n} ({100*p2g_ps_ok/n:.0f}%)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
