#!/usr/bin/env python3
"""Benchmark POS/MORPHO du P2G a partir de phrases phonetiques.

Deux scenarios :
  1. Phrases correctes → G2P → IPA → P2G sans ortho_words → eval POS/MORPHO
  2. Phrases erronees (post-passe1) → G2P → IPA → P2G sans ortho_words → eval

La reference POS/MORPHO est obtenue par le LexiqueTagger sur les phrases
correctes (ou manuellement annotee pour les cas ambigus).

Usage:
    python scripts/bench_p2g_morpho.py
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

logging.disable(logging.WARNING)

# ---- Corpus ----
# Chaque entree : (phrase_correcte, phrase_erronee_ou_None, annotations_morpho)
# annotations_morpho : dict[int, dict] par position de mot
#   cles: pos, Number, Gender, Person, VerbForm, Tense, Mood
# On annote manuellement les mots-cles pour avoir une reference fiable.

CORPUS: list[dict] = [
    # --- Accords nombre (det + nom + verbe) ---
    {
        "correcte": "les enfants mangent des pommes",
        "erronee": "les enfant mange des pomme",
        "categorie": "accord_nombre",
        "ref": {
            0: {"pos": "ART:def", "Number": "Plur"},
            1: {"pos": "NOM", "Number": "Plur"},
            2: {"pos": "VER", "Number": "Plur", "Person": "3", "VerbForm": "Fin", "Tense": "Pres"},
            3: {"pos": "ART:ind", "Number": "Plur"},
            4: {"pos": "NOM", "Number": "Plur"},
        },
    },
    {
        "correcte": "les chats dorment",
        "erronee": "les chat dort",
        "categorie": "accord_nombre",
        "ref": {
            0: {"pos": "ART:def", "Number": "Plur"},
            1: {"pos": "NOM", "Number": "Plur"},
            2: {"pos": "VER", "Number": "Plur", "Person": "3", "VerbForm": "Fin"},
        },
    },
    {
        "correcte": "ces petites filles jouent",
        "erronee": "ces petit fille joue",
        "categorie": "accord_nombre",
        "ref": {
            0: {"pos": "ART:dem", "Number": "Plur"},
            1: {"pos": "ADJ", "Number": "Plur", "Gender": "Fem"},
            2: {"pos": "NOM", "Number": "Plur", "Gender": "Fem"},
            3: {"pos": "VER", "Number": "Plur", "Person": "3"},
        },
    },
    {
        "correcte": "les maisons sont grandes",
        "erronee": "les maison sont grande",
        "categorie": "accord_nombre",
        "ref": {
            0: {"pos": "ART:def", "Number": "Plur"},
            1: {"pos": "NOM", "Number": "Plur", "Gender": "Fem"},
            2: {"pos": "VER", "Number": "Plur", "Person": "3"},
            3: {"pos": "ADJ", "Number": "Plur", "Gender": "Fem"},
        },
    },
    {
        "correcte": "mes amis arrivent demain",
        "erronee": "mes ami arrivent demain",
        "categorie": "accord_nombre",
        "ref": {
            0: {"pos": "ART:pos", "Number": "Plur"},
            1: {"pos": "NOM", "Number": "Plur"},
            2: {"pos": "VER", "Number": "Plur", "Person": "3"},
        },
    },
    {
        "correcte": "les oiseaux chantent",
        "erronee": "les oiseau chantent",
        "categorie": "accord_nombre",
        "ref": {
            0: {"pos": "ART:def", "Number": "Plur"},
            1: {"pos": "NOM", "Number": "Plur"},
            2: {"pos": "VER", "Number": "Plur", "Person": "3"},
        },
    },
    {
        "correcte": "les livres sont sur la table",
        "erronee": "les livre sont sur la table",
        "categorie": "accord_nombre",
        "ref": {
            0: {"pos": "ART:def", "Number": "Plur"},
            1: {"pos": "NOM", "Number": "Plur"},
            2: {"pos": "VER", "Number": "Plur", "Person": "3"},
            5: {"pos": "NOM", "Number": "Sing"},
        },
    },
    {
        "correcte": "les voitures rouges roulent vite",
        "erronee": "les voiture rouge roule vite",
        "categorie": "accord_nombre",
        "ref": {
            0: {"pos": "ART:def", "Number": "Plur"},
            1: {"pos": "NOM", "Number": "Plur", "Gender": "Fem"},
            2: {"pos": "ADJ", "Number": "Plur"},
            3: {"pos": "VER", "Number": "Plur", "Person": "3"},
        },
    },

    # --- Participes passes ---
    {
        "correcte": "il a mangé",
        "erronee": "il a manger",
        "categorie": "participe",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3", "Number": "Sing"},
            1: {"pos": "AUX", "Person": "3", "Number": "Sing"},
            2: {"pos": "VER", "VerbForm": "Part", "Tense": "Past"},
        },
    },
    {
        "correcte": "elle a acheté un livre",
        "erronee": "elle a acheter un livre",
        "categorie": "participe",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3"},
            1: {"pos": "AUX", "Person": "3"},
            2: {"pos": "VER", "VerbForm": "Part", "Tense": "Past"},
            4: {"pos": "NOM", "Number": "Sing"},
        },
    },
    {
        "correcte": "il a pris le train",
        "erronee": "il a prit le train",
        "categorie": "participe",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3"},
            1: {"pos": "AUX", "Person": "3"},
            2: {"pos": "VER", "VerbForm": "Part", "Tense": "Past"},
            4: {"pos": "NOM", "Number": "Sing"},
        },
    },
    {
        "correcte": "nous avons fini le travail",
        "erronee": "nous avons finir le travail",
        "categorie": "participe",
        "ref": {
            0: {"pos": "PRO:per", "Person": "1", "Number": "Plur"},
            1: {"pos": "AUX", "Person": "1", "Number": "Plur"},
            2: {"pos": "VER", "VerbForm": "Part", "Tense": "Past"},
        },
    },
    {
        "correcte": "ils ont préparé le repas",
        "erronee": "ils ont preparer le repas",
        "categorie": "participe",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3", "Number": "Plur"},
            1: {"pos": "AUX", "Person": "3", "Number": "Plur"},
            2: {"pos": "VER", "VerbForm": "Part", "Tense": "Past"},
        },
    },
    {
        "correcte": "il a chanté une chanson",
        "erronee": "il a chanter une chanson",
        "categorie": "participe",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3"},
            1: {"pos": "AUX", "Person": "3"},
            2: {"pos": "VER", "VerbForm": "Part", "Tense": "Past"},
        },
    },
    {
        "correcte": "elle a joué du piano",
        "erronee": "elle a jouer du piano",
        "categorie": "participe",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3"},
            1: {"pos": "AUX", "Person": "3"},
            2: {"pos": "VER", "VerbForm": "Part", "Tense": "Past"},
        },
    },

    # --- Homophones grammaticaux ---
    {
        "correcte": "il est venu",
        "erronee": "il et venu",
        "categorie": "homophones",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3"},
            1: {"pos": "AUX", "VerbForm": "Fin", "Person": "3"},
            2: {"pos": "VER", "VerbForm": "Part"},
        },
    },
    {
        "correcte": "ils ont dit",
        "erronee": "ils on dit",
        "categorie": "homophones",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3", "Number": "Plur"},
            1: {"pos": "AUX", "Person": "3", "Number": "Plur"},
            2: {"pos": "VER", "VerbForm": "Part"},
        },
    },
    {
        "correcte": "on a vu son film",
        "erronee": "on a vu sont film",
        "categorie": "homophones",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3"},
            1: {"pos": "AUX", "Person": "3"},
            2: {"pos": "VER", "VerbForm": "Part"},
            3: {"pos": "ART:pos", "Number": "Sing"},
            4: {"pos": "NOM", "Number": "Sing"},
        },
    },
    {
        "correcte": "grâce à la vie",
        "erronee": "grâce a la vie",
        "categorie": "homophones",
        "ref": {
            1: {"pos": "PRE"},
        },
    },
    {
        "correcte": "face à la mer",
        "erronee": "face a la mer",
        "categorie": "homophones",
        "ref": {
            1: {"pos": "PRE"},
        },
    },
    {
        "correcte": "elle est grande et forte",
        "erronee": None,
        "categorie": "homophones",
        "ref": {
            1: {"pos": "AUX", "Person": "3"},
            3: {"pos": "CON"},
        },
    },
    {
        "correcte": "il a une voiture",
        "erronee": None,
        "categorie": "homophones",
        "ref": {
            1: {"pos": "AUX", "Person": "3"},
        },
    },
    {
        "correcte": "il va à la plage",
        "erronee": None,
        "categorie": "homophones",
        "ref": {
            2: {"pos": "PRE"},
        },
    },

    # --- Conjugaison (personne/nombre du verbe) ---
    {
        "correcte": "tu manges bien",
        "erronee": "tu manger bien",
        "categorie": "conjugaison",
        "ref": {
            0: {"pos": "PRO:per", "Person": "2", "Number": "Sing"},
            1: {"pos": "VER", "Person": "2", "Number": "Sing", "VerbForm": "Fin"},
        },
    },
    {
        "correcte": "il part demain",
        "erronee": "il partir demain",
        "categorie": "conjugaison",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3", "Number": "Sing"},
            1: {"pos": "VER", "Person": "3", "Number": "Sing", "VerbForm": "Fin"},
        },
    },
    {
        "correcte": "nous partons demain",
        "erronee": "nous partir demain",
        "categorie": "conjugaison",
        "ref": {
            0: {"pos": "PRO:per", "Person": "1", "Number": "Plur"},
            1: {"pos": "VER", "Person": "1", "Number": "Plur", "VerbForm": "Fin"},
        },
    },
    {
        "correcte": "vous finissez le travail",
        "erronee": "vous finir le travail",
        "categorie": "conjugaison",
        "ref": {
            0: {"pos": "PRO:per", "Person": "2", "Number": "Plur"},
            1: {"pos": "VER", "Person": "2", "Number": "Plur", "VerbForm": "Fin"},
        },
    },
    {
        "correcte": "ils prennent le train",
        "erronee": "ils prendre le train",
        "categorie": "conjugaison",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3", "Number": "Plur"},
            1: {"pos": "VER", "Person": "3", "Number": "Plur", "VerbForm": "Fin"},
        },
    },
    {
        "correcte": "je mange une pomme",
        "erronee": None,
        "categorie": "conjugaison",
        "ref": {
            0: {"pos": "PRO:per", "Person": "1", "Number": "Sing"},
            1: {"pos": "VER", "Person": "1", "Number": "Sing", "VerbForm": "Fin"},
        },
    },
    {
        "correcte": "elle chante bien",
        "erronee": None,
        "categorie": "conjugaison",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3"},
            1: {"pos": "VER", "Person": "3", "Number": "Sing", "VerbForm": "Fin"},
        },
    },

    # --- Infinitif (ne pas confondre avec PP) ---
    {
        "correcte": "il va manger",
        "erronee": None,
        "categorie": "infinitif_vs_pp",
        "ref": {
            2: {"pos": "VER", "VerbForm": "Inf"},
        },
    },
    {
        "correcte": "je veux partir",
        "erronee": None,
        "categorie": "infinitif_vs_pp",
        "ref": {
            2: {"pos": "VER", "VerbForm": "Inf"},
        },
    },
    {
        "correcte": "pour manger dehors",
        "erronee": None,
        "categorie": "infinitif_vs_pp",
        "ref": {
            1: {"pos": "VER", "VerbForm": "Inf"},
        },
    },
    {
        "correcte": "il peut partir",
        "erronee": None,
        "categorie": "infinitif_vs_pp",
        "ref": {
            2: {"pos": "VER", "VerbForm": "Inf"},
        },
    },
    {
        "correcte": "elle doit finir",
        "erronee": None,
        "categorie": "infinitif_vs_pp",
        "ref": {
            2: {"pos": "VER", "VerbForm": "Inf"},
        },
    },
    {
        "correcte": "sans pouvoir dormir",
        "erronee": None,
        "categorie": "infinitif_vs_pp",
        "ref": {
            1: {"pos": "VER", "VerbForm": "Inf"},
            2: {"pos": "VER", "VerbForm": "Inf"},
        },
    },

    # --- Genre (masculin/feminin) ---
    {
        "correcte": "la grande maison",
        "erronee": None,
        "categorie": "genre",
        "ref": {
            0: {"pos": "ART:def", "Gender": "Fem"},
            1: {"pos": "ADJ", "Gender": "Fem"},
            2: {"pos": "NOM", "Gender": "Fem"},
        },
    },
    {
        "correcte": "le petit garçon",
        "erronee": None,
        "categorie": "genre",
        "ref": {
            0: {"pos": "ART:def", "Gender": "Masc"},
            1: {"pos": "ADJ", "Gender": "Masc"},
            2: {"pos": "NOM", "Gender": "Masc"},
        },
    },
    {
        "correcte": "une belle fleur",
        "erronee": None,
        "categorie": "genre",
        "ref": {
            0: {"pos": "ART:ind", "Gender": "Fem"},
            1: {"pos": "ADJ", "Gender": "Fem"},
            2: {"pos": "NOM", "Gender": "Fem"},
        },
    },
    {
        "correcte": "un beau jardin",
        "erronee": None,
        "categorie": "genre",
        "ref": {
            0: {"pos": "ART:ind", "Gender": "Masc"},
            1: {"pos": "ADJ", "Gender": "Masc"},
            2: {"pos": "NOM", "Gender": "Masc"},
        },
    },

    # --- Temps verbal ---
    {
        "correcte": "il mangeait du pain",
        "erronee": None,
        "categorie": "temps",
        "ref": {
            1: {"pos": "VER", "Tense": "Imp", "VerbForm": "Fin"},
        },
    },
    {
        "correcte": "nous mangeons du pain",
        "erronee": None,
        "categorie": "temps",
        "ref": {
            1: {"pos": "VER", "Tense": "Pres", "Person": "1", "Number": "Plur"},
        },
    },
    {
        "correcte": "ils mangeront demain",
        "erronee": None,
        "categorie": "temps",
        "ref": {
            1: {"pos": "VER", "Tense": "Fut", "Person": "3", "Number": "Plur"},
        },
    },

    # --- Phrases correctes (guard FP) ---
    {
        "correcte": "le chat mange",
        "erronee": None,
        "categorie": "guard_fp",
        "ref": {
            1: {"pos": "NOM", "Number": "Sing"},
            2: {"pos": "VER", "Number": "Sing", "Person": "3"},
        },
    },
    {
        "correcte": "je suis content",
        "erronee": None,
        "categorie": "guard_fp",
        "ref": {
            0: {"pos": "PRO:per", "Person": "1"},
            1: {"pos": "AUX", "Person": "1"},
            2: {"pos": "ADJ", "Number": "Sing", "Gender": "Masc"},
        },
    },
    {
        "correcte": "elle est contente",
        "erronee": None,
        "categorie": "guard_fp",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3"},
            1: {"pos": "AUX", "Person": "3"},
            2: {"pos": "ADJ", "Number": "Sing", "Gender": "Fem"},
        },
    },
    {
        "correcte": "nous avons bien travaillé",
        "erronee": None,
        "categorie": "guard_fp",
        "ref": {
            0: {"pos": "PRO:per", "Person": "1", "Number": "Plur"},
            1: {"pos": "AUX", "Person": "1", "Number": "Plur"},
            3: {"pos": "VER", "VerbForm": "Part", "Tense": "Past"},
        },
    },
    {
        "correcte": "tu as raison",
        "erronee": None,
        "categorie": "guard_fp",
        "ref": {
            0: {"pos": "PRO:per", "Person": "2", "Number": "Sing"},
            1: {"pos": "AUX", "Person": "2", "Number": "Sing"},
        },
    },
    {
        "correcte": "ils sont partis avec son vélo",
        "erronee": None,
        "categorie": "guard_fp",
        "ref": {
            0: {"pos": "PRO:per", "Person": "3", "Number": "Plur"},
            1: {"pos": "AUX", "Person": "3", "Number": "Plur"},
            2: {"pos": "VER", "VerbForm": "Part", "Number": "Plur"},
            4: {"pos": "ART:pos", "Number": "Sing"},
        },
    },
]


# ---- Evaluation ----

FEATURES = ["pos", "Number", "Gender", "Person", "VerbForm", "Tense", "Mood"]

# POS equivalences (P2G tags vs reference tags)
_POS_EQUIV = {
    "ART:def": {"ART:def", "DET"},
    "ART:ind": {"ART:ind", "DET"},
    "ART:dem": {"ART:dem", "DET"},
    "ART:pos": {"ART:pos", "DET"},
    "PRO:per": {"PRO:per", "PRO", "PRON"},
    "AUX": {"AUX", "VER"},  # AUX and VER are often confused
    "VER": {"VER", "AUX"},
    "CON": {"CON", "CCONJ", "KON"},
    "PRE": {"PRE", "PRP", "ADP"},
}

# Tense equivalences
_TENSE_EQUIV = {
    "Past": {"Past", "Pas"},
    "Pres": {"Pres", "Pre"},
    "Imp": {"Imp", "Impf"},
    "Fut": {"Fut"},
}


def _match_feature(feat: str, predicted: str, expected: str) -> bool:
    """Compare une feature predite vs attendue avec equivalences."""
    if predicted == expected:
        return True
    if feat == "pos":
        equiv = _POS_EQUIV.get(expected, {expected})
        return predicted in equiv
    if feat == "Tense":
        equiv = _TENSE_EQUIV.get(expected, {expected})
        return predicted in equiv
    return False


@dataclass
class FeatureStats:
    total: int = 0
    correct: int = 0
    errors: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def evaluate_p2g_morpho(
    g2p_tagger,
    p2g_adapter,
    scenario: str,
    passe1_fn=None,
    lexique=None,
) -> dict[str, dict[str, FeatureStats]]:
    """Evalue le POS/MORPHO du P2G sur le corpus.

    Args:
        scenario: "correcte" ou "erronee"
    """
    results_by_cat: dict[str, dict[str, FeatureStats]] = {}

    for entry in CORPUS:
        cat = entry["categorie"]
        ref = entry["ref"]

        if scenario == "correcte":
            phrase = entry["correcte"]
        else:
            phrase = entry.get("erronee") or entry["correcte"]

        mots = phrase.split()

        # Si scenario errone + passe1, appliquer la passe1
        if scenario == "erronee" and passe1_fn is not None and lexique is not None:
            from lectura_correcteur._types import MotV2
            mots_v2 = [
                MotV2(position=i, original=w, forme=w.lower())
                for i, w in enumerate(mots)
            ]
            g2p_for_p1 = g2p_tagger if hasattr(g2p_tagger, "prononcer") else None
            passe1_fn(mots_v2, lexique, g2p_for_p1)
            mots = [m.forme for m in mots_v2]

        # G2P
        tags = g2p_tagger.tag_words_rich(mots)
        ipa = [t.get("g2p", "") or "" for t in tags]
        # Fallback prononcer
        for i, ip in enumerate(ipa):
            if not ip and hasattr(g2p_tagger, "prononcer"):
                ipa[i] = g2p_tagger.prononcer(mots[i]) or mots[i]

        # P2G sans ortho_words
        p2g_result = p2g_adapter.transcrire_complet(ipa, ortho_words=None, k=5)

        # Extraire predictions
        p2g_pos = p2g_result.get("pos", [])
        p2g_morpho = p2g_result.get("morpho", {})
        p2g_ortho = p2g_result.get("ortho", [])

        if cat not in results_by_cat:
            results_by_cat[cat] = {f: FeatureStats() for f in FEATURES}

        # Evaluer chaque position annotee
        for pos_idx, expected in ref.items():
            if pos_idx >= len(mots):
                continue

            for feat in FEATURES:
                if feat not in expected:
                    continue

                expected_val = expected[feat]

                if feat == "pos":
                    predicted_val = p2g_pos[pos_idx] if pos_idx < len(p2g_pos) else ""
                else:
                    feat_list = p2g_morpho.get(feat, [])
                    predicted_val = feat_list[pos_idx] if pos_idx < len(feat_list) else "_"

                stats = results_by_cat[cat][feat]
                stats.total += 1

                if _match_feature(feat, predicted_val, expected_val):
                    stats.correct += 1
                else:
                    # Mot correct pour reference
                    mot_correct = entry["correcte"].split()[pos_idx] if pos_idx < len(entry["correcte"].split()) else "?"
                    mot_input = mots[pos_idx] if pos_idx < len(mots) else "?"
                    p2g_ortho_val = p2g_ortho[pos_idx] if pos_idx < len(p2g_ortho) else "?"
                    stats.errors.append({
                        "phrase": phrase,
                        "mot_correct": mot_correct,
                        "mot_input": mot_input,
                        "ipa": ipa[pos_idx] if pos_idx < len(ipa) else "?",
                        "p2g_ortho": p2g_ortho_val,
                        "feat": feat,
                        "expected": expected_val,
                        "predicted": predicted_val,
                    })

    return results_by_cat


def afficher_resultats(label: str, results: dict[str, dict[str, FeatureStats]]):
    """Affiche les resultats d'evaluation."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    # Par feature (agrege)
    global_stats: dict[str, FeatureStats] = {f: FeatureStats() for f in FEATURES}
    for cat, feat_stats in results.items():
        for feat, stats in feat_stats.items():
            global_stats[feat].total += stats.total
            global_stats[feat].correct += stats.correct
            global_stats[feat].errors.extend(stats.errors)

    print(f"\n  {'Feature':12s}  {'Correct':>8s}  {'Total':>6s}  {'Accuracy':>8s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*8}")
    for feat in FEATURES:
        s = global_stats[feat]
        if s.total > 0:
            print(f"  {feat:12s}  {s.correct:>8d}  {s.total:>6d}  {s.accuracy:>8.1%}")

    # Par categorie
    print(f"\n  Par categorie:")
    cats = sorted(results.keys())
    for cat in cats:
        feat_stats = results[cat]
        total = sum(s.total for s in feat_stats.values())
        correct = sum(s.correct for s in feat_stats.values())
        acc = correct / total if total > 0 else 0.0
        print(f"    {cat:20s}  {correct:>3d}/{total:<3d}  {acc:.1%}")

    # Erreurs detaillees
    all_errors = []
    for feat in FEATURES:
        all_errors.extend(global_stats[feat].errors)

    if all_errors:
        print(f"\n  Erreurs ({len(all_errors)}):")
        # Grouper par feature
        by_feat: dict[str, list] = {}
        for e in all_errors:
            by_feat.setdefault(e["feat"], []).append(e)

        for feat in FEATURES:
            errs = by_feat.get(feat, [])
            if not errs:
                continue
            print(f"\n    [{feat}] ({len(errs)} erreurs):")
            for e in errs[:10]:  # max 10 par feature
                print(f"      {e['mot_input']:15s} (IPA: {e['ipa']:10s}) "
                      f"P2G ortho: {e['p2g_ortho']:12s} "
                      f"predit={e['predicted']:8s} attendu={e['expected']:8s} "
                      f"| {e['phrase']}")
            if len(errs) > 10:
                print(f"      ... et {len(errs) - 10} autres")


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur._adapter_p2g import creer_adapter_p2g
    from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
    from lectura_correcteur._tagger_hybride import TaggerHybride
    from lectura_correcteur._passe1_orthographe import passe1_orthographe
    from lectura_correcteur._utils import LexiqueNormalise

    LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"

    print(f"Corpus: {len(CORPUS)} phrases")
    print(f"Chargement du lexique: {LEXIQUE_DB}")

    lexique = Lexique(LEXIQUE_DB)
    lex_norm = LexiqueNormalise(lexique)

    g2p_adapter = creer_adapter_g2p_unifie()
    g2p_tagger = TaggerHybride(g2p_adapter, lex_norm, lm_homophones=None)
    p2g_adapter = creer_adapter_p2g()

    if p2g_adapter is None:
        print("ERREUR: P2G indisponible")
        return

    # Scenario 1 : phrases correctes
    r1 = evaluate_p2g_morpho(g2p_tagger, p2g_adapter, "correcte")
    afficher_resultats("Scenario 1 — Phrases correctes → G2P → P2G", r1)

    # Scenario 2 : phrases erronees (post-passe1)
    r2 = evaluate_p2g_morpho(
        g2p_tagger, p2g_adapter, "erronee",
        passe1_fn=passe1_orthographe,
        lexique=lex_norm,
    )
    afficher_resultats("Scenario 2 — Phrases erronees (post-passe1) → G2P → P2G", r2)

    # Comparatif
    print(f"\n\n{'='*80}")
    print(f"  COMPARATIF")
    print(f"{'='*80}")
    print(f"  {'Feature':12s}  {'Correctes':>10s}  {'Erronees':>10s}  {'Delta':>8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")

    for feat in FEATURES:
        t1 = sum(r1[c][feat].total for c in r1)
        c1 = sum(r1[c][feat].correct for c in r1)
        t2 = sum(r2[c][feat].total for c in r2)
        c2 = sum(r2[c][feat].correct for c in r2)
        if t1 > 0 or t2 > 0:
            a1 = c1 / t1 if t1 > 0 else 0.0
            a2 = c2 / t2 if t2 > 0 else 0.0
            delta = a2 - a1
            print(f"  {feat:12s}  {a1:>10.1%}  {a2:>10.1%}  {delta:>+8.1%}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
