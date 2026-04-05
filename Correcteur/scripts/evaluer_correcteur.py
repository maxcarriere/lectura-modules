#!/usr/bin/env python3
"""Évaluation comparative : lectura-correcteur vs Grammalecte.

Usage :
    python scripts/evaluer_correcteur.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

LEXIQUE_PATH = Path(
    "/data/work/projets/lectura/workspace/lectura-main/correcteur/donnees/lexique.db"
)

# ---------------------------------------------------------------------------
# Imports projet
# ---------------------------------------------------------------------------
from corpus_evaluation import CORPUS, CasTest, categories  # noqa: E402
from lectura_correcteur import Correcteur  # noqa: E402


# ---------------------------------------------------------------------------
# Grammalecte
# ---------------------------------------------------------------------------
def _corriger_grammalecte(phrase: str) -> str:
    """Applique Grammalecte et reconstruit la phrase corrigée."""
    try:
        from pygrammalecte import grammalecte_text
        from pygrammalecte.pygrammalecte import GrammalecteGrammarMessage
    except ImportError:
        return phrase

    messages = list(grammalecte_text(phrase))
    grammar_msgs = [
        m for m in messages
        if isinstance(m, GrammalecteGrammarMessage) and m.suggestions
    ]
    grammar_msgs.sort(key=lambda m: m.start, reverse=True)

    result = phrase
    for msg in grammar_msgs:
        result = result[:msg.start] + msg.suggestions[0] + result[msg.end:]
    return result


# ---------------------------------------------------------------------------
# Structures de résultat
# ---------------------------------------------------------------------------
@dataclass
class ResultatPhrase:
    cas: CasTest
    corrigee_lectura: str = ""
    corrigee_grammalecte: str = ""
    exact_lectura: bool = False
    exact_grammalecte: bool = False


@dataclass
class MetriquesCategorie:
    categorie: str
    total: int = 0
    exact_lectura: int = 0
    exact_grammalecte: int = 0


@dataclass
class MetriquesGlobales:
    lec_vrais_positifs: int = 0
    lec_faux_positifs: int = 0
    lec_faux_negatifs: int = 0
    lec_vrais_negatifs: int = 0
    gram_vrais_positifs: int = 0
    gram_faux_positifs: int = 0
    gram_faux_negatifs: int = 0
    gram_vrais_negatifs: int = 0


# ---------------------------------------------------------------------------
# Comparaison
# ---------------------------------------------------------------------------
def _normaliser(phrase: str) -> str:
    """Normalise : minuscules, strip ponctuation finale, espaces."""
    phrase = phrase.strip()
    # Retirer la ponctuation finale (.!?…)
    while phrase and phrase[-1] in ".!?…":
        phrase = phrase[:-1]
    return " ".join(phrase.lower().split())


def _a_modifie(originale: str, corrigee: str) -> bool:
    return _normaliser(originale) != _normaliser(corrigee)


def _est_exacte(corrigee: str, attendue: str) -> bool:
    return _normaliser(corrigee) == _normaliser(attendue)


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------
def evaluer(verbose: bool = False) -> None:
    # --- Chargement du lexique ---
    print("Chargement du lexique SQLite...")
    from lectura_lexique import Lexique
    lexique = Lexique(str(LEXIQUE_PATH))
    print(f"  Lexique : {LEXIQUE_PATH.name}")

    # --- Chargement Lectura-correcteur ---
    print("Chargement de lectura-correcteur...")
    t0 = time.perf_counter()
    correcteur = Correcteur(lexique)
    t_load = time.perf_counter() - t0
    print(f"  Correcteur chargé en {t_load:.1f}s\n")

    # --- Grammalecte disponible ? ---
    grammalecte_ok = True
    try:
        from pygrammalecte import grammalecte_text  # noqa: F401
        list(grammalecte_text("test"))
        print("Grammalecte : disponible\n")
    except Exception as exc:
        grammalecte_ok = False
        print(f"Grammalecte : indisponible ({exc})\n")

    # --- Évaluation phrase par phrase ---
    resultats: list[ResultatPhrase] = []
    n = len(CORPUS)

    for i, cas in enumerate(CORPUS, 1):
        if verbose:
            print(f"  [{i:2d}/{n}] {cas.id:<14s} ", end="", flush=True)

        # Lectura
        res_lec = correcteur.corriger(cas.erronee)
        corr_lec = res_lec.phrase_corrigee

        # Grammalecte
        corr_gram = _corriger_grammalecte(cas.erronee) if grammalecte_ok else cas.erronee

        r = ResultatPhrase(
            cas=cas,
            corrigee_lectura=corr_lec,
            corrigee_grammalecte=corr_gram,
            exact_lectura=_est_exacte(corr_lec, cas.attendue),
            exact_grammalecte=_est_exacte(corr_gram, cas.attendue),
        )
        resultats.append(r)

        if verbose:
            sl = "✓" if r.exact_lectura else "✗"
            sg = "✓" if r.exact_grammalecte else "✗"
            print(f"L={sl}  G={sg}")

    # --- Métriques par catégorie ---
    cats = categories()
    metriques_cat: dict[str, MetriquesCategorie] = {}
    for cat in cats:
        mc = MetriquesCategorie(categorie=cat)
        for r in resultats:
            if cat in r.cas.categories:
                mc.total += 1
                if r.exact_lectura:
                    mc.exact_lectura += 1
                if r.exact_grammalecte:
                    mc.exact_grammalecte += 1
        metriques_cat[cat] = mc

    # --- Métriques globales ---
    glob = MetriquesGlobales()
    for r in resultats:
        est_ok = "OK" in r.cas.categories

        if est_ok:
            if _a_modifie(r.cas.erronee, r.corrigee_lectura) and not r.exact_lectura:
                glob.lec_faux_positifs += 1
            else:
                glob.lec_vrais_negatifs += 1

            if _a_modifie(r.cas.erronee, r.corrigee_grammalecte) and not r.exact_grammalecte:
                glob.gram_faux_positifs += 1
            else:
                glob.gram_vrais_negatifs += 1
        else:
            if r.exact_lectura:
                glob.lec_vrais_positifs += 1
            else:
                glob.lec_faux_negatifs += 1

            if r.exact_grammalecte:
                glob.gram_vrais_positifs += 1
            else:
                glob.gram_faux_negatifs += 1

    # --- Rapport ---
    _afficher_rapport(metriques_cat, cats, glob, resultats, grammalecte_ok)


# ---------------------------------------------------------------------------
# Rapport console
# ---------------------------------------------------------------------------
def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  —  "
    return f"{100 * n / total:5.1f}%"


def _f1(precision: float, rappel: float) -> float:
    if precision + rappel == 0:
        return 0.0
    return 2 * precision * rappel / (precision + rappel)


def _afficher_rapport(
    metriques_cat: dict[str, MetriquesCategorie],
    cats: list[str],
    glob: MetriquesGlobales,
    resultats: list[ResultatPhrase],
    grammalecte_ok: bool,
) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Évaluation comparative : lectura-correcteur vs Grammalecte")
    print(sep)

    # --- Par catégorie ---
    print(f"\n{'Catégorie':<16} | {'Nb':>3} | {'Lectura':^16} | {'Grammalecte':^16}")
    print("-" * 62)
    for cat in cats:
        mc = metriques_cat[cat]
        if mc.total == 0:
            continue
        l_str = f"{mc.exact_lectura}/{mc.total} ({_pct(mc.exact_lectura, mc.total)})"
        g_str = f"{mc.exact_grammalecte}/{mc.total} ({_pct(mc.exact_grammalecte, mc.total)})"
        print(f"{cat:<16} | {mc.total:>3} | {l_str:>16} | {g_str:>16}")

    # --- Faux positifs ---
    ok_resultats = [r for r in resultats if "OK" in r.cas.categories]
    n_ok = len(ok_resultats)
    print(f"\n--- Faux positifs (phrases correctes : {n_ok}) ---")
    print(f"  Lectura    : {glob.lec_faux_positifs}/{n_ok} ({_pct(glob.lec_faux_positifs, n_ok)})")
    if grammalecte_ok:
        print(f"  Grammalecte: {glob.gram_faux_positifs}/{n_ok} ({_pct(glob.gram_faux_positifs, n_ok)})")

    # --- Scores globaux ---
    erreurs = [r for r in resultats if "OK" not in r.cas.categories]
    n_err = len(erreurs)

    lec_prec = (
        glob.lec_vrais_positifs / (glob.lec_vrais_positifs + glob.lec_faux_positifs)
        if (glob.lec_vrais_positifs + glob.lec_faux_positifs) > 0 else 0.0
    )
    lec_rappel = glob.lec_vrais_positifs / n_err if n_err > 0 else 0.0
    lec_f1 = _f1(lec_prec, lec_rappel)

    gram_prec = (
        glob.gram_vrais_positifs / (glob.gram_vrais_positifs + glob.gram_faux_positifs)
        if (glob.gram_vrais_positifs + glob.gram_faux_positifs) > 0 else 0.0
    )
    gram_rappel = glob.gram_vrais_positifs / n_err if n_err > 0 else 0.0
    gram_f1 = _f1(gram_prec, gram_rappel)

    print(f"\n--- Scores globaux (sur {n_err} phrases erronées) ---")
    print(f"{'':16} | {'Précision':>10} | {'Rappel':>10} | {'F1':>10}")
    print("-" * 55)
    print(f"{'Lectura':<16} | {lec_prec:>10.3f} | {lec_rappel:>10.3f} | {lec_f1:>10.3f}")
    if grammalecte_ok:
        print(f"{'Grammalecte':<16} | {gram_prec:>10.3f} | {gram_rappel:>10.3f} | {gram_f1:>10.3f}")

    # --- Détails des erreurs ---
    print(f"\n--- Détail des phrases non corrigées exactement ---")
    for r in resultats:
        if "OK" in r.cas.categories:
            continue
        if r.exact_lectura and r.exact_grammalecte:
            continue

        impl_tag = "" if r.cas.implementee else " [non implémenté]"
        print(f"\n  [{r.cas.id}]{impl_tag} \"{r.cas.erronee}\"")
        print(f"    Attendu     : \"{r.cas.attendue}\"")

        sl = "✓" if r.exact_lectura else "✗"
        print(f"    Lectura     : \"{r.corrigee_lectura}\" {sl}")
        if grammalecte_ok:
            sg = "✓" if r.exact_grammalecte else "✗"
            print(f"    Grammalecte : \"{r.corrigee_grammalecte}\" {sg}")

    # --- Faux positifs détaillés ---
    fp_lec = [r for r in resultats if "OK" in r.cas.categories and not r.exact_lectura]
    fp_gram = [
        r for r in resultats
        if "OK" in r.cas.categories and not r.exact_grammalecte and grammalecte_ok
    ]
    if fp_lec or fp_gram:
        print(f"\n--- Détail des faux positifs ---")
        for r in fp_lec:
            print(f"  [{r.cas.id}] Lectura     : \"{r.cas.erronee}\" → \"{r.corrigee_lectura}\"")
        for r in fp_gram:
            print(f"  [{r.cas.id}] Grammalecte : \"{r.cas.erronee}\" → \"{r.corrigee_grammalecte}\"")

    print()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation comparative lectura-correcteur vs Grammalecte",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Affiche le résultat phrase par phrase",
    )
    args = parser.parse_args()
    evaluer(verbose=args.verbose)


if __name__ == "__main__":
    main()
