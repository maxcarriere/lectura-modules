#!/usr/bin/env python3
"""Stabilite du tagging : fautif vs correct, G2P vs P2G.

Question : les predictions POS/Morpho changent-elles entre la phrase
fautive et la phrase correcte ? Si un modele est stable, il predit la
meme morpho sur "les enfant mange" et "les enfants mangent" — ce qui
prouve qu'il ne se fie pas a l'orthographe.

Protocole :
  - G2P(phrase_fautive) vs G2P(phrase_correcte)  → divergences G2P
  - P2G(IPA(phrase_fautive)) vs P2G(IPA(phrase_correcte))  → divergences P2G

On mesure la stabilite = % de predictions identiques entre fautif et correct.
"""

from __future__ import annotations

import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.disable(logging.WARNING)

# Mapping G2P FR → UD
_G2P_KEY_MAP = {
    "nombre": "Number", "genre": "Gender", "personne": "Person",
    "temps": "Tense", "mode": "Mood",
}
_G2P_VAL_MAP = {
    "p": "Plur", "s": "Sing", "m": "Masc", "f": "Fem",
    "1": "1", "2": "2", "3": "3",
    "pre": "Pres", "pas": "Past", "imp": "Imp", "fut": "Fut",
    "ind": "Ind", "sub": "Sub", "con": "Cnd",
}

MORPHO_FEATURES = ["Number", "Gender", "Person", "VerbForm", "Tense", "Mood"]
ALL_FEATURES = ["POS"] + MORPHO_FEATURES


def g2p_to_ud(tag: dict) -> dict[str, str]:
    out = {"POS": tag.get("pos", "_") or "_"}
    for fr_key, ud_key in _G2P_KEY_MAP.items():
        val = tag.get(fr_key, "_") or "_"
        out[ud_key] = _G2P_VAL_MAP.get(val, val)
    out.setdefault("VerbForm", "_")
    return out


def p2g_to_ud(p2g_result: dict, idx: int) -> dict[str, str]:
    out = {}
    pos_list = p2g_result.get("pos", [])
    out["POS"] = pos_list[idx] if idx < len(pos_list) else "_"
    morpho = p2g_result.get("morpho", {})
    for feat in MORPHO_FEATURES:
        vals = morpho.get(feat, [])
        out[feat] = vals[idx] if idx < len(vals) else "_"
    return out


PAIRES = [
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
    ("il a manger", "il a mangé"),
    ("elle a acheter un livre", "elle a acheté un livre"),
    ("il a prit le train", "il a pris le train"),
    ("ils ont preparer le repas", "ils ont préparé le repas"),
    ("il a chanter une chanson", "il a chanté une chanson"),
    ("elle a jouer du piano", "elle a joué du piano"),
    ("nous avons marcher longtemps", "nous avons marché longtemps"),
    ("il a oublier son sac", "il a oublié son sac"),
    ("tu manger bien", "tu manges bien"),
    ("il partir demain", "il part demain"),
    ("nous partir demain", "nous partons demain"),
    ("vous finir le travail", "vous finissez le travail"),
    ("ils prendre le train", "ils prennent le train"),
    ("je manger une pomme", "je mange une pomme"),
    ("une bon idée", "une bonne idée"),
    ("le petit fille joue", "la petite fille joue"),
    ("un grande maison", "une grande maison"),
    ("il et content", "il est content"),
    ("ils on raison", "ils ont raison"),
    ("les enfant ont manger", "les enfants ont mangé"),
    ("elles sont partir hier", "elles sont parties hier"),
    ("nous avons finir les exercice", "nous avons fini les exercices"),
]


def main():
    from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
    g2p = creer_adapter_g2p_unifie()
    if not g2p:
        print("ERREUR: G2P indisponible"); return

    from lectura_correcteur._adapter_p2g import creer_adapter_p2g
    p2g = creer_adapter_p2g()
    if not p2g:
        print("ERREUR: P2G indisponible"); return
    p2g_engine = p2g._engine

    # Compteurs stabilite : tous mots
    stab_all = {f: {"g2p_same": 0, "p2g_same": 0, "total": 0} for f in ALL_FEATURES}
    # Compteurs stabilite : mots fautifs uniquement
    stab_err = {f: {"g2p_same": 0, "p2g_same": 0, "total": 0} for f in ALL_FEATURES}
    # Compteurs stabilite : mots NON fautifs (contexte)
    stab_ctx = {f: {"g2p_same": 0, "p2g_same": 0, "total": 0} for f in ALL_FEATURES}

    divergences_g2p: list[dict] = []
    divergences_p2g: list[dict] = []

    for phrase_f, phrase_c in PAIRES:
        mots_f = phrase_f.lower().split()
        mots_c = phrase_c.lower().split()
        if len(mots_f) != len(mots_c):
            continue

        diffs_set = {i for i, (a, b) in enumerate(zip(mots_f, mots_c)) if a != b}

        # G2P sur les deux
        tags_g2p_f = g2p.tag_words_rich(mots_f)
        tags_g2p_c = g2p.tag_words_rich(mots_c)

        # IPA depuis les deux
        ipa_f, ipa_c = [], []
        for i in range(len(mots_f)):
            pf = tags_g2p_f[i].get("g2p", "") or (g2p.prononcer(mots_f[i]) or "")
            pc = tags_g2p_c[i].get("g2p", "") or (g2p.prononcer(mots_c[i]) or "")
            ipa_f.append(pf)
            ipa_c.append(pc)

        # P2G sur les deux
        p2g_res_f = p2g_engine.analyser_avec_alternatives(ipa_f, ortho_words=mots_f, k=2)
        p2g_res_c = p2g_engine.analyser_avec_alternatives(ipa_c, ortho_words=mots_c, k=2)

        for i in range(len(mots_f)):
            is_err = i in diffs_set

            g2p_f_tag = g2p_to_ud(tags_g2p_f[i])
            g2p_c_tag = g2p_to_ud(tags_g2p_c[i])
            p2g_f_tag = p2g_to_ud(p2g_res_f, i)
            p2g_c_tag = p2g_to_ud(p2g_res_c, i)

            for feat in ALL_FEATURES:
                g_same = g2p_f_tag[feat] == g2p_c_tag[feat]
                p_same = p2g_f_tag[feat] == p2g_c_tag[feat]

                for d in [stab_all, stab_err if is_err else stab_ctx]:
                    d[feat]["total"] += 1
                    if g_same:
                        d[feat]["g2p_same"] += 1
                    if p_same:
                        d[feat]["p2g_same"] += 1

                # Log divergences
                if not g_same:
                    divergences_g2p.append({
                        "mot_f": mots_f[i], "mot_c": mots_c[i],
                        "feat": feat, "val_f": g2p_f_tag[feat],
                        "val_c": g2p_c_tag[feat], "is_err": is_err,
                        "phrase": phrase_f,
                    })
                if not p_same:
                    divergences_p2g.append({
                        "mot_f": mots_f[i], "mot_c": mots_c[i],
                        "feat": feat, "val_f": p2g_f_tag[feat],
                        "val_c": p2g_c_tag[feat], "is_err": is_err,
                        "phrase": phrase_f,
                        "ipa_f": ipa_f[i], "ipa_c": ipa_c[i],
                    })

    # Affichage
    def pct(n, t):
        return f"{100*n/t:.0f}%" if t > 0 else "n/a"

    def show(title, stab, total_label):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        print(f"  {'Feature':12s}  {'G2P stable':>14s}  {'P2G stable':>14s}  {'+ stable':>10s}")
        print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*10}")
        for feat in ALL_FEATURES:
            c = stab[feat]
            t = c["total"]
            if t == 0:
                continue
            gs = c["g2p_same"]
            ps = c["p2g_same"]
            gp = gs / t * 100
            pp = ps / t * 100
            winner = "≈" if abs(gp - pp) < 1 else ("P2G" if pp > gp else "G2P")
            print(f"  {feat:12s}  {gs:3d}/{t} ({pct(gs,t):>4s})  "
                  f"{ps:3d}/{t} ({pct(ps,t):>4s})  {winner:>10s}")

    show("STABILITE — TOUS LES MOTS", stab_all, "all")
    show("STABILITE — MOTS FAUTIFS", stab_err, "err")
    show("STABILITE — MOTS CONTEXTE (non fautifs)", stab_ctx, "ctx")

    # Detail divergences G2P
    print(f"\n{'='*70}")
    print(f"  DIVERGENCES G2P fautif≠correct ({len(divergences_g2p)} cas)")
    print(f"{'='*70}")
    print(f"  {'Mot_F':10s} {'Mot_C':10s} {'Feat':8s} {'Fautif':8s} {'Correct':8s} {'Type':5s}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    for d in divergences_g2p:
        t = "ERR" if d["is_err"] else "ctx"
        print(f"  {d['mot_f']:10s} {d['mot_c']:10s} {d['feat']:8s} "
              f"{d['val_f']:8s} {d['val_c']:8s} {t:5s}")

    # Detail divergences P2G
    print(f"\n{'='*70}")
    print(f"  DIVERGENCES P2G fautif≠correct ({len(divergences_p2g)} cas)")
    print(f"{'='*70}")
    print(f"  {'Mot_F':10s} {'Mot_C':10s} {'Feat':8s} {'Fautif':8s} {'Correct':8s} "
          f"{'Type':5s} {'IPA_F':10s} {'IPA_C':10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*5} {'-'*10} {'-'*10}")
    for d in divergences_p2g:
        t = "ERR" if d["is_err"] else "ctx"
        print(f"  {d['mot_f']:10s} {d['mot_c']:10s} {d['feat']:8s} "
              f"{d['val_f']:8s} {d['val_c']:8s} {t:5s} "
              f"{d['ipa_f']:10s} {d['ipa_c']:10s}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
