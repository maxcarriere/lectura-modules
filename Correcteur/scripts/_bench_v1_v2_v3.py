#!/usr/bin/env python3
"""Benchmark comparatif V1 vs V2 vs V3 du correcteur.

Lance les 3 versions sur le meme corpus et affiche un tableau comparatif.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field

# Paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"

# ---- Corpus de test ----

# Chaque categorie : list[(phrase_erronee, phrase_attendue)]

CORPUS: dict[str, list[tuple[str, str]]] = {
    # --- Accords en nombre ---
    "accord_nombre": [
        ("les enfant mange des pomme", "les enfants mangent des pommes"),
        ("les chat dort", "les chats dorment"),
        ("ces petit fille", "ces petites filles"),
        ("les maison sont grande", "les maisons sont grandes"),
        ("mes ami arrivent demain", "mes amis arrivent demain"),
        ("les oiseau chantent", "les oiseaux chantent"),
        ("deux pomme rouge", "deux pommes rouges"),
        ("les livre sont sur la table", "les livres sont sur la table"),
        ("les voiture rouge", "les voitures rouges"),
        ("des beau jardin", "des beaux jardins"),
    ],

    # --- Participes passes ---
    "participe": [
        ("il a manger", "il a mangé"),
        ("elle a acheter un livre", "elle a acheté un livre"),
        ("il a prit le train", "il a pris le train"),
        ("nous avons finir le travail", "nous avons fini le travail"),
        ("ils ont preparer le repas", "ils ont préparé le repas"),
        ("il a chanter une chanson", "il a chanté une chanson"),
        ("elle a jouer du piano", "elle a joué du piano"),
        ("nous avons marcher longtemps", "nous avons marché longtemps"),
    ],

    # --- Homophones grammaticaux ---
    "homophones": [
        ("il et venu", "il est venu"),
        ("ils on dit", "ils ont dit"),
        ("à chaque fois", "à chaque fois"),          # phrase correcte (guard)
        ("on a vu sont film", "on a vu son film"),
        ("il a preparé le repas", "il a préparé le repas"),  # accent sur participe
        ("grâce a la vie", "grâce à la vie"),         # a → à
        ("face a la mer", "face à la mer"),           # a → à
    ],

    # --- Accents manquants (passe 1 orthographe) ---
    "accents": [
        ("le cinema est fermé", "le cinéma est fermé"),
        ("mon frere est là", "mon frère est là"),
        ("l'ecole est grande", "l'école est grande"),
        ("garcon", "garçon"),
        ("francais", "français"),
        ("à bientot", "à bientôt"),
        ("tres bien", "très bien"),
        ("la litterature francaise", "la littérature française"),
    ],

    # --- Conjugaison (sujet + infinitif -> forme conjuguee) ---
    "conjugaison": [
        ("tu manger bien", "tu manges bien"),
        ("il partir demain", "il part demain"),
        ("nous partir demain", "nous partons demain"),
        ("vous finir le travail", "vous finissez le travail"),
        ("ils prendre le train", "ils prennent le train"),
        ("je revoir ce film", "je revois ce film"),
    ],

    # --- Faux positifs (phrases correctes, rien a changer) ---
    "faux_positifs": [
        ("le chat mange", "le chat mange"),
        ("les chats mangent", "les chats mangent"),
        ("je mange bien", "je mange bien"),
        ("tu manges bien", "tu manges bien"),
        ("il mange bien", "il mange bien"),
        ("nous mangeons bien", "nous mangeons bien"),
        ("elle est contente", "elle est contente"),
        ("il a raison", "il a raison"),
        ("je veux revoir ce film", "je veux revoir ce film"),
        ("il va manger", "il va manger"),
        ("pour revoir ce film", "pour revoir ce film"),
        ("il peut partir", "il peut partir"),
        ("elle doit finir", "elle doit finir"),
        ("un peu plus", "un peu plus"),
        ("sans doute", "sans doute"),
        ("de son côté", "de son côté"),
        ("elle est là", "elle est là"),
        ("ils sont partis avec son vélo", "ils sont partis avec son vélo"),
        ("le téléphone a sonné", "le téléphone a sonné"),
        ("il a chaque fois raison", "il a chaque fois raison"),
        ("les grandes maisons sont belles", "les grandes maisons sont belles"),
        ("elle a vu sa fille", "elle a vu sa fille"),
        ("nous avons bien travaillé", "nous avons bien travaillé"),
        ("tu as raison", "tu as raison"),
        ("je suis content", "je suis content"),
    ],
}

# Total
_TOTAL = sum(len(v) for v in CORPUS.values())


@dataclass
class CatResult:
    total: int = 0
    corrects: int = 0
    fn: int = 0          # attendait correction, rien fait
    fp: int = 0          # pas de correction attendue, correction faite
    wrong: int = 0       # correction faite mais pas la bonne
    details: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        tp = self.corrects - self._gardes
        fp = self.wrong + self.fp
        return tp / (tp + fp) if (tp + fp) > 0 else 1.0

    @property
    def _gardes(self) -> int:
        """Nombre de phrases sans correction attendue et correctement gardees."""
        return sum(1 for d in self.details if d.get("type") == "OK_GUARD")

    @property
    def recall(self) -> float:
        # Parmi les corrections attendues, combien faites correctement
        tp = self.corrects - self._gardes
        fn = self.fn
        return tp / (tp + fn) if (tp + fn) > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)


def _normaliser(texte: str) -> str:
    return " ".join(texte.strip().lower().split())


def evaluer(correcteur, label: str) -> dict[str, CatResult]:
    """Evalue un correcteur sur tout le corpus."""
    resultats: dict[str, CatResult] = {}

    for cat, paires in CORPUS.items():
        r = CatResult()
        for erronee, attendue in paires:
            r.total += 1
            res = correcteur.corriger(erronee)
            obtenu = res.phrase_corrigee

            obtenu_n = _normaliser(obtenu)
            attendu_n = _normaliser(attendue)
            erronee_n = _normaliser(erronee)

            correction_attendue = (attendu_n != erronee_n)
            correction_faite = (obtenu_n != erronee_n)

            if obtenu_n == attendu_n:
                r.corrects += 1
                if not correction_attendue:
                    r.details.append({"type": "OK_GUARD"})
            elif correction_attendue and not correction_faite:
                r.fn += 1
                r.details.append({
                    "type": "FN", "in": erronee,
                    "expected": attendue, "got": obtenu,
                })
            elif not correction_attendue and correction_faite:
                r.fp += 1
                r.details.append({
                    "type": "FP", "in": erronee,
                    "expected": attendue, "got": obtenu,
                })
            else:
                r.wrong += 1
                r.details.append({
                    "type": "WRONG", "in": erronee,
                    "expected": attendue, "got": obtenu,
                })
        resultats[cat] = r

    return resultats


def afficher(label: str, resultats: dict[str, CatResult], elapsed: float):
    """Affiche les resultats pour un correcteur."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    total_ok = total_n = total_fp = total_fn = total_wr = 0

    for cat in CORPUS:
        r = resultats[cat]
        total_ok += r.corrects
        total_n += r.total
        total_fp += r.fp
        total_fn += r.fn
        total_wr += r.wrong

        status = "OK" if r.fp == 0 and r.fn == 0 and r.wrong == 0 else "  "
        print(f"  {status} {cat:20s}  {r.corrects:2d}/{r.total:2d}  "
              f"P={r.precision:.2f} R={r.recall:.2f} F1={r.f1:.2f}  "
              f"FP={r.fp} FN={r.fn} WR={r.wrong}")

    print(f"  {'':22s}  ----")

    # Global
    total_r = CatResult(total=total_n, corrects=total_ok, fn=total_fn,
                        fp=total_fp, wrong=total_wr)
    # Count guards across all cats
    guards = sum(r._gardes for r in resultats.values())
    total_r.details = [{"type": "OK_GUARD"}] * guards

    p = total_r.precision
    r_ = total_r.recall
    f1 = total_r.f1
    print(f"     {'TOTAL':20s}  {total_ok:2d}/{total_n:2d}  "
          f"P={p:.2f} R={r_:.2f} F1={f1:.2f}  "
          f"FP={total_fp} FN={total_fn} WR={total_wr}")
    print(f"     Temps: {elapsed:.1f}s")

    # Details erreurs
    has_errors = False
    for cat in CORPUS:
        r = resultats[cat]
        for d in r.details:
            if d["type"] in ("FP", "FN", "WRONG"):
                if not has_errors:
                    print(f"\n  Details erreurs:")
                    has_errors = True
                t = d["type"]
                print(f"    [{cat}] {t}: '{d['in']}' -> attendu '{d['expected']}' obtenu '{d['got']}'")


def main():
    import logging
    logging.disable(logging.WARNING)

    print(f"Corpus: {_TOTAL} phrases dans {len(CORPUS)} categories\n")
    print(f"Chargement du lexique: {LEXIQUE_DB}")

    from lectura_lexique import Lexique
    lexique = Lexique(LEXIQUE_DB)

    # --- V1 ---
    print("\nInstanciation V1...")
    from lectura_correcteur import Correcteur
    v1 = Correcteur(lexique)

    t0 = time.time()
    r1 = evaluer(v1, "V1")
    t1 = time.time() - t0
    afficher("V1 — Correcteur (7 etapes, regles)", r1, t1)

    # --- V2 ---
    print("\nInstanciation V2...")
    from lectura_correcteur import CorrecteurV2
    v2 = CorrecteurV2(lexique)

    t0 = time.time()
    r2 = evaluer(v2, "V2")
    t2 = time.time() - t0
    afficher("V2 — 3 passes (Ortho + POS Viterbi + Morpho Viterbi)", r2, t2)

    # --- V3 ---
    print("\nInstanciation V3...")
    from lectura_correcteur import CorrecteurV3
    v3 = CorrecteurV3(lexique)

    t0 = time.time()
    r3 = evaluer(v3, "V3")
    t3 = time.time() - t0
    afficher("V3 — 2 passes (Ortho + G2P/P2G roundtrip)", r3, t3)

    # --- V4 ---
    print("\nInstanciation V4...")
    from lectura_correcteur import CorrecteurV4
    v4 = CorrecteurV4(lexique)

    t0 = time.time()
    r4 = evaluer(v4, "V4")
    t4 = time.time() - t0
    afficher("V4 — 2 passes (Ortho + P2G sans ortho_words)", r4, t4)

    # --- V5 ---
    print("\nInstanciation V5...")
    from lectura_correcteur import CorrecteurV5
    v5 = CorrecteurV5(lexique)

    t0 = time.time()
    r5 = evaluer(v5, "V5")
    t5 = time.time() - t0
    afficher("V5 — V1 + P2G comme etiqueteur POS/MORPHO", r5, t5)

    # --- Tableau comparatif ---
    all_versions = [("V1", r1), ("V2", r2), ("V3", r3), ("V4", r4), ("V5", r5)]

    print(f"\n\n{'='*88}")
    print(f"  COMPARATIF")
    print(f"{'='*88}")
    print(f"  {'Categorie':20s}  {'V1':>8s}  {'V2':>8s}  {'V3':>8s}  {'V4':>8s}  {'V5':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for cat in CORPUS:
        n = len(CORPUS[cat])
        s1 = f"{r1[cat].corrects}/{n}"
        s2 = f"{r2[cat].corrects}/{n}"
        s3 = f"{r3[cat].corrects}/{n}"
        s4 = f"{r4[cat].corrects}/{n}"
        s5 = f"{r5[cat].corrects}/{n}"
        print(f"  {cat:20s}  {s1:>8s}  {s2:>8s}  {s3:>8s}  {s4:>8s}  {s5:>8s}")

    # Totals
    t1_ok = sum(r1[c].corrects for c in CORPUS)
    t2_ok = sum(r2[c].corrects for c in CORPUS)
    t3_ok = sum(r3[c].corrects for c in CORPUS)
    t4_ok = sum(r4[c].corrects for c in CORPUS)
    t5_ok = sum(r5[c].corrects for c in CORPUS)
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'TOTAL':20s}  {t1_ok:>3d}/{_TOTAL:<3d}  {t2_ok:>3d}/{_TOTAL:<3d}  {t3_ok:>3d}/{_TOTAL:<3d}  {t4_ok:>3d}/{_TOTAL:<3d}  {t5_ok:>3d}/{_TOTAL:<3d}")

    # F1 / FP
    for lbl, rx in all_versions:
        fp = sum(rx[c].fp for c in CORPUS)
        fn = sum(rx[c].fn for c in CORPUS)
        wr = sum(rx[c].wrong for c in CORPUS)
        guards = sum(rx[c]._gardes for c in CORPUS)
        tp = sum(rx[c].corrects for c in CORPUS) - guards
        total_corr_expected = tp + fn
        p = tp / (tp + fp + wr) if (tp + fp + wr) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"  {lbl} — F1={f1:.2f}  P={p:.2f}  R={r:.2f}  FP={fp}  FN={fn}  WR={wr}")

    print(f"{'='*88}")


if __name__ == "__main__":
    main()
