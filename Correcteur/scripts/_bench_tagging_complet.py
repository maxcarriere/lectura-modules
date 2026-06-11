#!/usr/bin/env python3
"""Benchmark complet POS/Morpho : G2P vs P2G sur phrases fautives.

Compare la qualite du tagging G2P (depuis l'ortho fautive) vs P2G
(depuis l'IPA) sur TOUS les mots de chaque phrase, pas seulement les
mots fautifs.

Reference = G2P sur la phrase correcte (oracle orthographique).

Mesures :
  - POS accuracy (global + par mot fautif)
  - Nombre accuracy
  - Genre accuracy
  - Personne accuracy
  - VerbForm accuracy
  - Tense accuracy
  - Mood accuracy
"""

from __future__ import annotations

import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.disable(logging.WARNING)


# Corpus elargi : (phrase_fautive, phrase_correcte)
PAIRES = [
    # --- Accords nombre (NOM, ADJ) ---
    ("les enfant mange des pomme", "les enfants mangent des pommes"),
    ("les chat dort", "les chats dorment"),
    ("mes ami arrivent demain", "mes amis arrivent demain"),
    ("les voiture rouge", "les voitures rouges"),
    ("des beau jardin", "des beaux jardins"),
    ("les livre sont sur la table", "les livres sont sur la table"),
    ("les maison sont grande", "les maisons sont grandes"),
    ("ces petit fille", "ces petites filles"),
    ("les oiseau chantent", "les oiseaux chantent"),
    ("deux pomme rouge", "deux pommes rouges"),

    # --- Participes passes (infinitif au lieu de PP) ---
    ("il a manger", "il a mangé"),
    ("elle a acheter un livre", "elle a acheté un livre"),
    ("il a prit le train", "il a pris le train"),
    ("ils ont preparer le repas", "ils ont préparé le repas"),
    ("il a chanter une chanson", "il a chanté une chanson"),
    ("elle a jouer du piano", "elle a joué du piano"),
    ("nous avons marcher longtemps", "nous avons marché longtemps"),
    ("il a oublier son sac", "il a oublié son sac"),

    # --- Conjugaison (infinitif au lieu de forme flechie) ---
    ("tu manger bien", "tu manges bien"),
    ("il partir demain", "il part demain"),
    ("nous partir demain", "nous partons demain"),
    ("vous finir le travail", "vous finissez le travail"),
    ("ils prendre le train", "ils prennent le train"),
    ("je manger une pomme", "je mange une pomme"),

    # --- Accords genre ---
    ("une bon idée", "une bonne idée"),
    ("le petit fille joue", "la petite fille joue"),
    ("un grande maison", "une grande maison"),

    # --- Homophones grammaticaux ---
    ("il et content", "il est content"),
    ("ils on raison", "ils ont raison"),

    # --- Mixtes (nombre + participe, etc.) ---
    ("les enfant ont manger", "les enfants ont mangé"),
    ("elles sont partir hier", "elles sont parties hier"),
    ("nous avons finir les exercice", "nous avons fini les exercices"),
]


# Features morpho a comparer (cle UD)
MORPHO_FEATURES = ["Number", "Gender", "Person", "VerbForm", "Tense", "Mood"]

# Mapping G2P (FR) → UD pour normalisation
_G2P_KEY_MAP = {
    "nombre": "Number", "genre": "Gender", "personne": "Person",
    "temps": "Tense", "mode": "Mood",
}
_G2P_VAL_MAP = {
    "p": "Plur", "s": "Sing",
    "m": "Masc", "f": "Fem",
    "1": "1", "2": "2", "3": "3",
    "pre": "Pres", "pas": "Past", "imp": "Imp", "fut": "Fut",
    "ind": "Ind", "sub": "Sub", "con": "Cnd",
}


def g2p_to_ud(tag: dict) -> dict[str, str]:
    """Convertit un tag G2P (cles FR, valeurs FR) en convention UD."""
    out: dict[str, str] = {}
    # POS
    out["POS"] = tag.get("pos", "_") or "_"
    # Morpho
    for fr_key, ud_key in _G2P_KEY_MAP.items():
        val = tag.get(fr_key, "_") or "_"
        out[ud_key] = _G2P_VAL_MAP.get(val, val)
    # VerbForm n'est pas dans le G2P FR standard
    out.setdefault("VerbForm", "_")
    return out


def p2g_to_ud(p2g_result: dict, idx: int) -> dict[str, str]:
    """Extrait les features UD du resultat P2G a la position idx."""
    out: dict[str, str] = {}
    pos_list = p2g_result.get("pos", [])
    out["POS"] = pos_list[idx] if idx < len(pos_list) else "_"
    morpho = p2g_result.get("morpho", {})
    for feat in MORPHO_FEATURES:
        vals = morpho.get(feat, [])
        out[feat] = vals[idx] if idx < len(vals) else "_"
    return out


def main():
    from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
    g2p = creer_adapter_g2p_unifie()
    if g2p is None:
        print("ERREUR: G2P indisponible"); return

    from lectura_correcteur._adapter_p2g import creer_adapter_p2g
    p2g = creer_adapter_p2g()
    if p2g is None:
        print("ERREUR: P2G indisponible"); return

    p2g_engine = p2g._engine

    # Compteurs globaux : tous mots
    all_features = ["POS"] + MORPHO_FEATURES
    counts_all = {f: {"g2p": 0, "p2g": 0, "total": 0} for f in all_features}
    # Compteurs : mots fautifs uniquement
    counts_err = {f: {"g2p": 0, "p2g": 0, "total": 0} for f in all_features}

    n_phrases = 0
    n_mots_total = 0
    n_mots_err = 0

    for phrase_fautive, phrase_correcte in PAIRES:
        mots_fautifs = phrase_fautive.lower().split()
        mots_corrects = phrase_correcte.lower().split()

        if len(mots_fautifs) != len(mots_corrects):
            continue

        n_phrases += 1
        diffs_set = {i for i, (a, b) in
                     enumerate(zip(mots_fautifs, mots_corrects)) if a != b}

        # G2P sur fautive et correcte
        tags_g2p_fautive = g2p.tag_words_rich(mots_fautifs)
        tags_g2p_correcte = g2p.tag_words_rich(mots_corrects)

        # IPA
        ipa = []
        for i, mot in enumerate(mots_fautifs):
            phone = tags_g2p_fautive[i].get("g2p", "") or (g2p.prononcer(mot) or "")
            ipa.append(phone)

        # P2G
        p2g_result = p2g_engine.analyser_avec_alternatives(
            ipa, ortho_words=mots_fautifs, k=3,
        )

        # Comparer mot par mot
        for i in range(len(mots_fautifs)):
            ref = g2p_to_ud(tags_g2p_correcte[i])
            g2p_tag = g2p_to_ud(tags_g2p_fautive[i])
            p2g_tag = p2g_to_ud(p2g_result, i)

            n_mots_total += 1
            is_err = i in diffs_set
            if is_err:
                n_mots_err += 1

            for feat in all_features:
                ref_v = ref.get(feat, "_")
                g2p_v = g2p_tag.get(feat, "_")
                p2g_v = p2g_tag.get(feat, "_")

                counts_all[feat]["total"] += 1
                if g2p_v == ref_v:
                    counts_all[feat]["g2p"] += 1
                if p2g_v == ref_v:
                    counts_all[feat]["p2g"] += 1

                if is_err:
                    counts_err[feat]["total"] += 1
                    if g2p_v == ref_v:
                        counts_err[feat]["g2p"] += 1
                    if p2g_v == ref_v:
                        counts_err[feat]["p2g"] += 1

    # Affichage
    def pct(n, t):
        return f"{100*n/t:.0f}%" if t > 0 else "n/a"

    def show_table(title, counts, n_mots):
        print(f"\n{'='*70}")
        print(f"  {title} ({n_mots} mots, {n_phrases} phrases)")
        print(f"{'='*70}")
        print(f"  {'Feature':12s}  {'G2P':>12s}  {'P2G':>12s}  {'Gagnant':>10s}")
        print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
        for feat in all_features:
            c = counts[feat]
            t = c["total"]
            g = c["g2p"]
            p = c["p2g"]
            if t == 0:
                continue
            gp = g / t * 100
            pp = p / t * 100
            if abs(gp - pp) < 1:
                winner = "≈"
            elif gp > pp:
                winner = "G2P"
            else:
                winner = "P2G"
            print(f"  {feat:12s}  {g:3d}/{t} ({pct(g,t):>4s})  "
                  f"{p:3d}/{t} ({pct(p,t):>4s})  {winner:>10s}")

    show_table("TOUS LES MOTS", counts_all, n_mots_total)
    show_table("MOTS FAUTIFS UNIQUEMENT", counts_err, n_mots_err)

    # Detail par type d'erreur
    print(f"\n{'='*70}")
    print(f"  DETAIL MOT PAR MOT (mots fautifs)")
    print(f"{'='*70}")
    print(f"  {'Fautif':12s} {'Correct':12s} {'Feat':8s} "
          f"{'G2P':8s} {'P2G':8s} {'REF':8s} {'Verdict':8s}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for phrase_fautive, phrase_correcte in PAIRES:
        mots_fautifs = phrase_fautive.lower().split()
        mots_corrects = phrase_correcte.lower().split()
        if len(mots_fautifs) != len(mots_corrects):
            continue

        tags_g2p_fautive = g2p.tag_words_rich(mots_fautifs)
        tags_g2p_correcte = g2p.tag_words_rich(mots_corrects)

        ipa = []
        for i, mot in enumerate(mots_fautifs):
            phone = tags_g2p_fautive[i].get("g2p", "") or (g2p.prononcer(mot) or "")
            ipa.append(phone)

        p2g_result = p2g_engine.analyser_avec_alternatives(
            ipa, ortho_words=mots_fautifs, k=3,
        )

        for i, (mf, mc) in enumerate(zip(mots_fautifs, mots_corrects)):
            if mf == mc:
                continue

            ref = g2p_to_ud(tags_g2p_correcte[i])
            g2p_tag = g2p_to_ud(tags_g2p_fautive[i])
            p2g_tag = p2g_to_ud(p2g_result, i)

            for feat in all_features:
                ref_v = ref.get(feat, "_")
                g2p_v = g2p_tag.get(feat, "_")
                p2g_v = p2g_tag.get(feat, "_")

                # Afficher uniquement les divergences
                if g2p_v == p2g_v == ref_v:
                    continue

                g_ok = "ok" if g2p_v == ref_v else "!!"
                p_ok = "ok" if p2g_v == ref_v else "!!"

                if g2p_v != ref_v and p2g_v == ref_v:
                    verdict = "P2G wins"
                elif g2p_v == ref_v and p2g_v != ref_v:
                    verdict = "G2P wins"
                else:
                    verdict = "both !!"

                print(f"  {mf:12s} {mc:12s} {feat:8s} "
                      f"{g2p_v:5s} {g_ok:2s} {p2g_v:5s} {p_ok:2s} {ref_v:8s} {verdict}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
