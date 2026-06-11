#!/usr/bin/env python3
"""Benchmark P2G : avec vs sans ortho_words sur phrases fautives.

Compare 3 configurations :
  1. G2P(phrase_fautive) → tagging depuis l'ortho fautive
  2. P2G(IPA, ortho_words=formes_fautives) → biaise par l'ortho
  3. P2G(IPA, ortho_words=None) → pur contexte phonetique

Reference = G2P(phrase_correcte) → tagging depuis l'ortho correcte.

On mesure l'accuracy sur POS + chaque feature morpho.
"""

from __future__ import annotations

import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.disable(logging.WARNING)

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
    # Accords nombre
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
    # Participes
    ("il a manger", "il a mangé"),
    ("elle a acheter un livre", "elle a acheté un livre"),
    ("il a prit le train", "il a pris le train"),
    ("ils ont preparer le repas", "ils ont préparé le repas"),
    ("il a chanter une chanson", "il a chanté une chanson"),
    ("elle a jouer du piano", "elle a joué du piano"),
    ("nous avons marcher longtemps", "nous avons marché longtemps"),
    ("il a oublier son sac", "il a oublié son sac"),
    # Conjugaison
    ("tu manger bien", "tu manges bien"),
    ("il partir demain", "il part demain"),
    ("nous partir demain", "nous partons demain"),
    ("vous finir le travail", "vous finissez le travail"),
    ("ils prendre le train", "ils prennent le train"),
    ("je manger une pomme", "je mange une pomme"),
    # Accords genre
    ("une bon idée", "une bonne idée"),
    ("le petit fille joue", "la petite fille joue"),
    ("un grande maison", "une grande maison"),
    # Homophones
    ("il et content", "il est content"),
    ("ils on raison", "ils ont raison"),
    ("on a vu sont film", "on a vu son film"),
    ("il a sonne a la porte", "il a sonné à la porte"),
    ("grâce a la vie", "grâce à la vie"),
    ("face a la mer", "face à la mer"),
    # Mixtes
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

    # Compteurs : tous mots
    c_all = {f: {"g2p": 0, "p2g_with": 0, "p2g_without": 0, "total": 0} for f in ALL_FEATURES}
    # Compteurs : mots fautifs
    c_err = {f: {"g2p": 0, "p2g_with": 0, "p2g_without": 0, "total": 0} for f in ALL_FEATURES}
    # Compteurs : mots contexte
    c_ctx = {f: {"g2p": 0, "p2g_with": 0, "p2g_without": 0, "total": 0} for f in ALL_FEATURES}

    # Aussi comparer les ortho P2G
    ortho_ok = {"p2g_with": 0, "p2g_without": 0, "total_err": 0}

    # Divergences P2G avec vs sans pour le detail
    divergences: list[dict] = []

    for phrase_f, phrase_c in PAIRES:
        mots_f = phrase_f.lower().split()
        mots_c = phrase_c.lower().split()
        if len(mots_f) != len(mots_c):
            continue

        diffs_set = {i for i, (a, b) in enumerate(zip(mots_f, mots_c)) if a != b}

        # Reference : G2P sur phrase correcte
        tags_ref = g2p.tag_words_rich(mots_c)
        # G2P sur phrase fautive
        tags_g2p_f = g2p.tag_words_rich(mots_f)

        # IPA depuis G2P fautive
        ipa = []
        for i in range(len(mots_f)):
            phone = tags_g2p_f[i].get("g2p", "") or (g2p.prononcer(mots_f[i]) or "")
            ipa.append(phone)

        # P2G avec ortho_words
        p2g_with = p2g_engine.analyser_avec_alternatives(ipa, ortho_words=mots_f, k=3)
        # P2G sans ortho_words
        p2g_without = p2g_engine.analyser_avec_alternatives(ipa, ortho_words=None, k=3)

        for i in range(len(mots_f)):
            ref = g2p_to_ud(tags_ref[i])
            g = g2p_to_ud(tags_g2p_f[i])
            pw = p2g_to_ud(p2g_with, i)
            pwo = p2g_to_ud(p2g_without, i)

            is_err = i in diffs_set
            targets = [c_all]
            targets.append(c_err if is_err else c_ctx)

            for feat in ALL_FEATURES:
                rv = ref.get(feat, "_")
                for c in targets:
                    c[feat]["total"] += 1
                    if g[feat] == rv:
                        c[feat]["g2p"] += 1
                    if pw[feat] == rv:
                        c[feat]["p2g_with"] += 1
                    if pwo[feat] == rv:
                        c[feat]["p2g_without"] += 1

            # Ortho P2G
            if is_err:
                ortho_ok["total_err"] += 1
                ow = p2g_with["ortho"][i] if i < len(p2g_with["ortho"]) else ""
                owo = p2g_without["ortho"][i] if i < len(p2g_without["ortho"]) else ""
                if ow.lower() == mots_c[i]:
                    ortho_ok["p2g_with"] += 1
                if owo.lower() == mots_c[i]:
                    ortho_ok["p2g_without"] += 1

                # Log divergences avec vs sans
                if ow != owo or pw["POS"] != pwo["POS"]:
                    divergences.append({
                        "mot_f": mots_f[i], "mot_c": mots_c[i],
                        "ow": ow, "owo": owo,
                        "pos_w": pw["POS"], "pos_wo": pwo["POS"],
                        "ref_pos": ref["POS"],
                        "nb_w": pw.get("Number", "_"), "nb_wo": pwo.get("Number", "_"),
                        "ref_nb": ref.get("Number", "_"),
                    })

    # Affichage
    def pct(n, t):
        return f"{100*n/t:.0f}%" if t > 0 else "n/a"

    def show(title, c):
        n = c["POS"]["total"]
        print(f"\n{'='*75}")
        print(f"  {title} ({n} mots)")
        print(f"{'='*75}")
        print(f"  {'Feature':12s}  {'G2P':>14s}  {'P2G+ortho':>14s}  {'P2G pur':>14s}  {'Meilleur':>10s}")
        print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*10}")
        for feat in ALL_FEATURES:
            t = c[feat]["total"]
            if t == 0:
                continue
            g = c[feat]["g2p"]
            pw = c[feat]["p2g_with"]
            pwo = c[feat]["p2g_without"]
            vals = [("G2P", g), ("P2G+ow", pw), ("P2G pur", pwo)]
            best_v = max(v for _, v in vals)
            winners = [n for n, v in vals if v == best_v]
            winner = " / ".join(winners) if len(winners) < 3 else "≈"
            print(f"  {feat:12s}  {g:3d}/{t} ({pct(g,t):>4s})  "
                  f"{pw:3d}/{t} ({pct(pw,t):>4s})  "
                  f"{pwo:3d}/{t} ({pct(pwo,t):>4s})  {winner:>10s}")

    show("TOUS LES MOTS", c_all)
    show("MOTS FAUTIFS UNIQUEMENT", c_err)
    show("MOTS CONTEXTE (non fautifs)", c_ctx)

    # Ortho
    t = ortho_ok["total_err"]
    print(f"\n{'='*75}")
    print(f"  ORTHO P2G CORRECTE (mots fautifs, {t} mots)")
    print(f"{'='*75}")
    print(f"  P2G+ortho_words:  {ortho_ok['p2g_with']:3d}/{t} ({pct(ortho_ok['p2g_with'], t)})")
    print(f"  P2G pur (sans):   {ortho_ok['p2g_without']:3d}/{t} ({pct(ortho_ok['p2g_without'], t)})")

    # Divergences
    print(f"\n{'='*75}")
    print(f"  DIVERGENCES P2G avec vs sans ortho_words ({len(divergences)} mots fautifs)")
    print(f"{'='*75}")
    print(f"  {'Fautif':10s} {'Correct':10s} "
          f"{'Ortho+ow':10s} {'Ortho pur':10s} "
          f"{'POS+ow':8s} {'POS pur':8s} {'POS ref':8s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for d in divergences:
        ow_mark = "ok" if d["ow"].lower() == d["mot_c"] else "!!"
        owo_mark = "ok" if d["owo"].lower() == d["mot_c"] else "!!"
        pw_mark = "ok" if d["pos_w"] == d["ref_pos"] else "!!"
        pwo_mark = "ok" if d["pos_wo"] == d["ref_pos"] else "!!"
        print(f"  {d['mot_f']:10s} {d['mot_c']:10s} "
              f"{d['ow']:7s} {ow_mark:2s} {d['owo']:7s} {owo_mark:2s} "
              f"{d['pos_w']:5s} {pw_mark:2s} {d['pos_wo']:5s} {pwo_mark:2s} {d['ref_pos']:8s}")

    print(f"{'='*75}")


if __name__ == "__main__":
    main()
