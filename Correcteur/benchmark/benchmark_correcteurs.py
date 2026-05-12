"""Benchmark comparatif des correcteurs français.

Compare :
  - lectura-correcteur (mode dégradé : lookup lexique)
  - lectura-correcteur (mode G2P Unifié V2)
  - LanguageTool (via language_tool_python)
  - Grammalecte (via pygrammalecte)

Corpus : 38 phrases avec fautes classées par catégorie.
Métriques :
  - Exact match (phrase entière)
  - Précision, Rappel, F1, F0.5 au niveau mot
  - Taux de faux positifs
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus de test
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    categorie: str
    original: str
    attendu: str
    description: str = ""


CORPUS: list[TestCase] = [
    # --- Accord sujet-verbe ---
    TestCase("accord_sv", "Les enfant mange des pomme.", "Les enfants mangent des pommes.",
             "pluriel nom + verbe + COD"),
    TestCase("accord_sv", "Les chat dort sur le canapé.", "Les chats dorment sur le canapé.",
             "pluriel nom + verbe irrégulier"),
    TestCase("accord_sv", "Les oiseau chante dans les arbre.", "Les oiseaux chantent dans les arbres.",
             "pluriel double + verbe"),
    TestCase("accord_sv", "Mon frere et ma soeur joue dehors.", "Mon frère et ma sœur jouent dehors.",
             "sujet coordonné + accent"),
    TestCase("accord_sv", "Nous avont fini le travail.", "Nous avons fini le travail.",
             "conjugaison P4"),

    # --- Participe passé / infinitif ---
    TestCase("pp_inf", "Il a manger tout les gâteaux.", "Il a mangé tous les gâteaux.",
             "PP avec avoir + tout/tous"),
    TestCase("pp_inf", "Tu a oublier ton cartable.", "Tu as oublié ton cartable.",
             "PP + conjugaison tu as"),
    TestCase("pp_inf", "Elle est parti sans prévenir.", "Elle est partie sans prévenir.",
             "PP accord avec être + féminin"),
    TestCase("pp_inf", "Ils ont manger à midi.", "Ils ont mangé à midi.",
             "PP avec avoir pluriel"),
    TestCase("pp_inf", "Je suis aller au marché.", "Je suis allé au marché.",
             "PP avec être"),

    # --- Homophones grammaticaux ---
    TestCase("homophone", "Il et parti hier.", "Il est parti hier.",
             "et/est"),
    TestCase("homophone", "Je vais a la plage.", "Je vais à la plage.",
             "a/à"),
    TestCase("homophone", "Il a bu son café.", "Il a bu son café.",
             "son correct — pas de fausse correction"),
    TestCase("homophone", "Ses enfants son gentils.", "Ses enfants sont gentils.",
             "son/sont"),
    TestCase("homophone", "Ou vas-tu ce soir ?", "Où vas-tu ce soir ?",
             "ou/où"),

    # --- Accents ---
    TestCase("accent", "L'ecole est fermee.", "L'école est fermée.",
             "accent é"),
    TestCase("accent", "Il a ete tres content.", "Il a été très content.",
             "accents multiples"),
    TestCase("accent", "La foret est magnifique.", "La forêt est magnifique.",
             "accent ê"),
    TestCase("accent", "Le probleme est resolu.", "Le problème est résolu.",
             "accent è + é"),

    # --- Orthographe (distance d'édition) ---
    TestCase("ortho", "Le farmacien a preparé le medicament.", "Le pharmacien a préparé le médicament.",
             "ph/f + accents"),
    TestCase("ortho", "Je cherche une adresse sur internett.", "Je cherche une adresse sur internet.",
             "doublement consonne"),
    TestCase("ortho", "Il fait beau aujourdhui.", "Il fait beau aujourd'hui.",
             "apostrophe manquante"),
    TestCase("ortho", "Les vacanse sont bientot finies.", "Les vacances sont bientôt finies.",
             "faute + accent"),

    # --- Accord adjectif ---
    TestCase("accord_adj", "Les voitures bleu sont garé.", "Les voitures bleues sont garées.",
             "accord adj + PP pluriel féminin"),
    TestCase("accord_adj", "Les petit filles jouent.", "Les petites filles jouent.",
             "accord adj antéposé féminin pluriel"),
    TestCase("accord_adj", "Une belle journee ensoleillé.", "Une belle journée ensoleillée.",
             "accord adj postposé + accent"),

    # --- Conjugaison ---
    TestCase("conjugaison", "Je mangera demain.", "Je mangerai demain.",
             "futur P1"),
    TestCase("conjugaison", "Ils finira bientôt.", "Ils finiront bientôt.",
             "futur P6"),
    TestCase("conjugaison", "Tu peut venir.", "Tu peux venir.",
             "présent irrégulier P2"),
    TestCase("conjugaison", "Il faut que je part.", "Il faut que je parte.",
             "subjonctif"),

    # --- Phrases complexes ---
    TestCase("complexe", "Les resultat de l'examen on ete publier hier.",
             "Les résultats de l'examen ont été publiés hier.",
             "accord + accent + PP + on/ont"),
    TestCase("complexe", "Si j'aurais su, je serai pas venu.",
             "Si j'avais su, je ne serais pas venu.",
             "conditionnel + négation"),
    TestCase("complexe", "Il ni a personne dans la salle.",
             "Il n'y a personne dans la salle.",
             "ni/n'y"),
    TestCase("complexe", "Tout les professeur sont la.",
             "Tous les professeurs sont là.",
             "tout/tous + pluriel + là/la"),

    # --- Pas de faute (faux positifs) ---
    TestCase("correct", "Le chat dort sur le canapé.", "Le chat dort sur le canapé.",
             "phrase correcte simple"),
    TestCase("correct", "Les enfants mangent des pommes.", "Les enfants mangent des pommes.",
             "phrase correcte pluriel"),
    TestCase("correct", "Il a mangé son repas hier soir.", "Il a mangé son repas hier soir.",
             "phrase correcte PP"),
    TestCase("correct", "Où sont les clés de la voiture ?", "Où sont les clés de la voiture ?",
             "phrase correcte homophones"),
]


# ---------------------------------------------------------------------------
# Correcteurs
# ---------------------------------------------------------------------------

def _normaliser(texte: str) -> str:
    """Normalise pour comparaison (strip, espaces multiples)."""
    return " ".join(texte.strip().split())


# ---------------------------------------------------------------------------
# Métriques mot-à-mot (TP / FP / FN / TN)
# ---------------------------------------------------------------------------

def _build_word_map(src: list[str], dst: list[str]) -> dict[int, str | None]:
    """Mappe chaque position src vers le mot correspondant dans dst.

    Utilise SequenceMatcher pour aligner les deux listes.
    Retourne {src_pos: dst_word} (None si le mot est supprimé).
    """
    sm = SequenceMatcher(None, src, dst)
    word_map: dict[int, str | None] = {}
    n_insertions = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                word_map[i1 + k] = src[i1 + k]
        elif op == "replace":
            n_src = i2 - i1
            n_dst = j2 - j1
            for k in range(n_src):
                word_map[i1 + k] = dst[j1 + k] if k < n_dst else None
            # Extra dst words = insertions (counted separately)
            if n_dst > n_src:
                n_insertions += n_dst - n_src
        elif op == "delete":
            for k in range(i2 - i1):
                word_map[i1 + k] = None
        elif op == "insert":
            n_insertions += j2 - j1
    return word_map


def _compter_insertions(src: list[str], dst: list[str]) -> list[str]:
    """Retourne les mots insérés dans dst qui n'ont pas de source dans src."""
    sm = SequenceMatcher(None, src, dst)
    insertions: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert":
            insertions.extend(dst[j1:j2])
        elif op == "replace":
            n_extra = (j2 - j1) - (i2 - i1)
            if n_extra > 0:
                insertions.extend(dst[j2 - n_extra : j2])
    return insertions


def calculer_metriques_mots(
    original: str, attendu: str, obtenu: str,
) -> tuple[int, int, int, int]:
    """Calcule (TP, FP, FN, TN) au niveau mot pour une phrase.

    - TP : mot fautif corrigé correctement
    - FP : mot correct modifié à tort
    - FN : mot fautif non corrigé (ou mal corrigé)
    - TN : mot correct laissé intact
    """
    orig = _normaliser(original).split()
    gold = _normaliser(attendu).split()
    syst = _normaliser(obtenu).split()

    gold_map = _build_word_map(orig, gold)
    sys_map = _build_word_map(orig, syst)

    tp = fp = fn = tn = 0
    for i, mot_orig in enumerate(orig):
        mot_gold = gold_map.get(i, mot_orig)
        mot_sys = sys_map.get(i, mot_orig)

        needs_change = (mot_gold != mot_orig)
        was_changed = (mot_sys != mot_orig)
        correct_change = (mot_sys == mot_gold)

        if needs_change and was_changed and correct_change:
            tp += 1
        elif needs_change:
            fn += 1
        elif not needs_change and was_changed:
            fp += 1
        else:
            tn += 1

    # Insertions dans gold non faites par le système = FN supplémentaires
    gold_ins = _compter_insertions(orig, gold)
    sys_ins = _compter_insertions(orig, syst)
    for mot in gold_ins:
        if mot in sys_ins:
            tp += 1
            sys_ins.remove(mot)
        else:
            fn += 1
    # Insertions du système non attendues = FP
    fp += len(sys_ins)

    return tp, fp, fn, tn


def _f_score(precision: float, recall: float, beta: float) -> float:
    """Calcule le F-score pour un beta donné."""
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


@dataclass
class Resultat:
    nom_correcteur: str
    categorie: str
    original: str
    attendu: str
    obtenu: str
    correct: bool       # obtenu == attendu (normalisé)
    modifie: bool       # obtenu != original
    temps_ms: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


def corriger_lectura(phrase: str, correcteur) -> str:
    r = correcteur.corriger(phrase)
    return r.phrase_corrigee


def corriger_languagetool(phrase: str, tool) -> str:
    return tool.correct(phrase)


def corriger_grammalecte(phrase: str, gc) -> str:
    """Applique les suggestions de Grammalecte séquentiellement."""
    aErrs = gc.getParagraphErrors(phrase)
    if not aErrs:
        return phrase

    # Trier par position décroissante pour remplacer de droite à gauche
    errs_sorted = sorted(aErrs, key=lambda e: e.get("nStart", 0), reverse=True)
    result = phrase
    for err in errs_sorted:
        suggestions = err.get("aSuggestions", [])
        if not suggestions:
            continue
        start = err.get("nStart", 0)
        end = err.get("nEnd", 0)
        result = result[:start] + suggestions[0] + result[end:]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    resultats: list[Resultat] = []

    # --- Lectura mode dégradé ---
    print("=" * 60)
    print("Chargement lectura-correcteur (lookup lexique)...")
    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur

    db_path = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura.db"
    lex = Lexique(db_path)

    t0 = time.perf_counter()
    c_degrade = Correcteur(lex)
    print(f"  Init: {(time.perf_counter()-t0)*1000:.0f} ms")

    # warmup
    c_degrade.corriger("test")

    for tc in CORPUS:
        t0 = time.perf_counter()
        obtenu = corriger_lectura(tc.original, c_degrade)
        dt = (time.perf_counter() - t0) * 1000
        obtenu_n = _normaliser(obtenu)
        attendu_n = _normaliser(tc.attendu)
        tp, fp, fn, tn = calculer_metriques_mots(tc.original, tc.attendu, obtenu)
        resultats.append(Resultat(
            "lectura_degrade", tc.categorie, tc.original, tc.attendu,
            obtenu, obtenu_n == attendu_n, obtenu_n != _normaliser(tc.original), dt,
            tp, fp, fn, tn,
        ))

    # --- Lectura G2P V2 ---
    print("Chargement lectura-correcteur (G2P Unifié V2)...")
    try:
        from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
        adapter = creer_adapter_g2p_unifie()
        if adapter is None:
            raise ImportError("adapter None")
        c_g2p = Correcteur(lex, tagger=adapter, g2p=adapter)
        c_g2p.corriger("test")

        for tc in CORPUS:
            t0 = time.perf_counter()
            obtenu = corriger_lectura(tc.original, c_g2p)
            dt = (time.perf_counter() - t0) * 1000
            obtenu_n = _normaliser(obtenu)
            attendu_n = _normaliser(tc.attendu)
            tp, fp, fn, tn = calculer_metriques_mots(tc.original, tc.attendu, obtenu)
            resultats.append(Resultat(
                "lectura_g2p_v2", tc.categorie, tc.original, tc.attendu,
                obtenu, obtenu_n == attendu_n, obtenu_n != _normaliser(tc.original), dt,
                tp, fp, fn, tn,
            ))
    except Exception as e:
        print(f"  SKIP: {e}")

    # --- LanguageTool ---
    print("Chargement LanguageTool (fr)...")
    try:
        import language_tool_python
        lt = language_tool_python.LanguageTool("fr")

        # warmup
        lt.correct("test")

        for tc in CORPUS:
            t0 = time.perf_counter()
            obtenu = corriger_languagetool(tc.original, lt)
            dt = (time.perf_counter() - t0) * 1000
            obtenu_n = _normaliser(obtenu)
            attendu_n = _normaliser(tc.attendu)
            tp, fp, fn, tn = calculer_metriques_mots(tc.original, tc.attendu, obtenu)
            resultats.append(Resultat(
                "languagetool", tc.categorie, tc.original, tc.attendu,
                obtenu, obtenu_n == attendu_n, obtenu_n != _normaliser(tc.original), dt,
                tp, fp, fn, tn,
            ))
        lt.close()
    except Exception as e:
        print(f"  SKIP LanguageTool: {e}")

    # --- Grammalecte ---
    print("Chargement Grammalecte...")
    try:
        import grammalecte
        gc = grammalecte.GrammarChecker("fr")
        oGCE = gc.getGCEngine()

        for tc in CORPUS:
            t0 = time.perf_counter()
            obtenu = corriger_grammalecte(tc.original, oGCE)
            dt = (time.perf_counter() - t0) * 1000
            obtenu_n = _normaliser(obtenu)
            attendu_n = _normaliser(tc.attendu)
            tp, fp, fn, tn = calculer_metriques_mots(tc.original, tc.attendu, obtenu)
            resultats.append(Resultat(
                "grammalecte", tc.categorie, tc.original, tc.attendu,
                obtenu, obtenu_n == attendu_n, obtenu_n != _normaliser(tc.original), dt,
                tp, fp, fn, tn,
            ))
    except Exception as e:
        print(f"  SKIP Grammalecte: {e}")

    # --- Résultats ---
    print("\n" + "=" * 60)
    print("RESULTATS")
    print("=" * 60)

    # Par correcteur
    correcteurs = sorted(set(r.nom_correcteur for r in resultats))
    categories = sorted(set(tc.categorie for tc in CORPUS))

    for nom in correcteurs:
        rs = [r for r in resultats if r.nom_correcteur == nom]
        n_correct = sum(1 for r in rs if r.correct)
        n_total = len(rs)
        temps_moy = sum(r.temps_ms for r in rs) / n_total if n_total else 0
        print(f"\n--- {nom} ---")
        print(f"  Exact match: {n_correct}/{n_total} ({n_correct/n_total*100:.0f}%)")
        print(f"  Temps moyen: {temps_moy:.1f} ms/phrase")

        # Par catégorie
        for cat in categories:
            rs_cat = [r for r in rs if r.categorie == cat]
            if not rs_cat:
                continue
            n_ok = sum(1 for r in rs_cat if r.correct)
            print(f"  {cat:15s}: {n_ok}/{len(rs_cat)}")

    # --- Métriques F-score ---
    print("\n" + "=" * 60)
    print("METRIQUES MOT-A-MOT (Precision / Rappel / F1 / F0.5)")
    print("=" * 60)

    for nom in correcteurs:
        rs = [r for r in resultats if r.nom_correcteur == nom]
        total_tp = sum(r.tp for r in rs)
        total_fp = sum(r.fp for r in rs)
        total_fn = sum(r.fn for r in rs)
        total_tn = sum(r.tn for r in rs)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = _f_score(precision, recall, 1.0)
        f05 = _f_score(precision, recall, 0.5)

        # Taux de faux positifs : mots corrects modifiés / total mots corrects
        mots_corrects = total_tn + total_fp
        taux_fp = total_fp / mots_corrects if mots_corrects else 0.0

        print(f"\n--- {nom} ---")
        print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}  TN={total_tn}")
        print(f"  Precision : {precision:.3f}  ({total_tp}/{total_tp + total_fp})")
        print(f"  Rappel    : {recall:.3f}  ({total_tp}/{total_tp + total_fn})")
        print(f"  F1        : {f1:.3f}")
        print(f"  F0.5      : {f05:.3f}  (pondère la précision)")
        print(f"  Taux FP   : {taux_fp:.3f}  ({total_fp}/{mots_corrects} mots corrects)")

        # Par catégorie
        for cat in categories:
            rs_cat = [r for r in rs if r.categorie == cat]
            if not rs_cat:
                continue
            cat_tp = sum(r.tp for r in rs_cat)
            cat_fp = sum(r.fp for r in rs_cat)
            cat_fn = sum(r.fn for r in rs_cat)
            cat_p = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) else 0.0
            cat_r = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) else 0.0
            cat_f1 = _f_score(cat_p, cat_r, 1.0)
            print(f"  {cat:15s}: P={cat_p:.2f} R={cat_r:.2f} F1={cat_f1:.2f}  (TP={cat_tp} FP={cat_fp} FN={cat_fn})")

    # Détails des erreurs
    print("\n" + "=" * 60)
    print("DETAILS DES ECARTS")
    print("=" * 60)

    for nom in correcteurs:
        erreurs = [r for r in resultats if r.nom_correcteur == nom and not r.correct]
        if not erreurs:
            print(f"\n--- {nom}: aucune erreur ---")
            continue
        print(f"\n--- {nom}: {len(erreurs)} écarts ---")
        for r in erreurs:
            print(f"  [{r.categorie}] {r.original!r}")
            print(f"    attendu: {r.attendu!r}")
            print(f"    obtenu:  {r.obtenu!r}")
            print(f"    (TP={r.tp} FP={r.fp} FN={r.fn})")

    # Export JSON
    out_path = Path(__file__).parent / "benchmark_resultats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in resultats], f, ensure_ascii=False, indent=2)
    print(f"\nResultats exportes: {out_path}")


if __name__ == "__main__":
    main()
