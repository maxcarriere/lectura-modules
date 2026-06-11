#!/usr/bin/env python3
"""Evaluation du correcteur sur un corpus de grammaire.

Evalue specifiquement les corrections grammaticales (conjugaison, accords,
homophones) sur un fichier TSV de paires (phrase_erronee, phrase_corrigee).

Format TSV attendu :
    type_erreur\tphrase_erronee\tphrase_corrigee

Types d'erreurs supportes :
    conjugaison, accord, homophone, participe

Usage :
    python scripts/evaluer_corpus_grammaire.py --corpus data/grammaire_test.tsv
    python scripts/evaluer_corpus_grammaire.py --built-in
    python scripts/evaluer_corpus_grammaire.py --built-in --g2p
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"


# ---------------------------------------------------------------------------
# Cas de test integres (grammaire)
# ---------------------------------------------------------------------------

_CAS_CONJUGAISON = [
    ("je revoir ce film", "je revois ce film"),
    ("tu manger bien", "tu manges bien"),
    ("il partir demain", "il part demain"),
    ("nous partir demain", "nous partons demain"),
    ("vous finir le travail", "vous finissez le travail"),
    ("ils prendre le train", "ils prennent le train"),
    # Guards : ne pas corriger
    ("je veux revoir ce film", "je veux revoir ce film"),
    ("il va manger", "il va manger"),
    ("pour revoir ce film", "pour revoir ce film"),
    ("il peut partir", "il peut partir"),
    ("elle doit finir", "elle doit finir"),
    # Conjugaison existante (pronom + VER)
    ("je mange bien", "je mange bien"),
    ("tu manges bien", "tu manges bien"),
    ("il mange bien", "il mange bien"),
    # Sujet nominal + verbe
    ("le chat mange", "le chat mange"),
    ("les chats mangent", "les chats mangent"),
]

_CAS_HOMOPHONES = [
    ("il a chaque fois", "il a chaque fois"),
    ("a chaque fois", "à chaque fois"),
    # Homophones grammaticaux (regles)
    ("il et venu", "il est venu"),
    ("ils on dit", "ils ont dit"),
    # Homophones LM trigramme
    ("grâce a la vie", "grâce à la vie"),
    ("il a été tres content", "il a été très content"),
    ("un peut plus tard", "un peu plus tard"),
    ("de son coté", "de son côté"),
    ("face a la mer", "face à la mer"),
    ("trois sans mètres", "trois cent mètres"),
    # ("du sang froid", "du sang-froid"),  # compose avec tiret, pas un homophone
    ("elle a vu sa fille", "elle a vu sa fille"),  # guard: ne pas changer
    ("on a vu sont film", "on a vu son film"),
    ("il est aller la bas", "il est allé là-bas"),
    # Guards : ne pas toucher quand correct
    ("il a raison", "il a raison"),
    ("de son côté", "de son côté"),
    ("un peu plus", "un peu plus"),
    ("sans doute", "sans doute"),
    ("elle est là", "elle est là"),
    # Guard a/à + PP : ne pas changer "a" devant un PP
    ("le téléphone a sonné", "le téléphone a sonné"),
    # Filtre lemme : ne pas changer des variantes flexionnelles
    ("ils sont partis avec son vélo", "ils sont partis avec son vélo"),
    # Scoring conjoint : corriger homophone + accent simultanément
    ("ils on prepare le repas", "ils ont préparé le repas"),
    # Regle participe elargie : AUX + present → PP
    ("il a prépare le repas", "il a préparé le repas"),
    ("il a sonne a la porte", "il a sonné à la porte"),
]

_CAS_ACCORDS = [
    ("les chat", "les chats"),
    ("le chats", "le chat"),
    ("une bon idee", "une bonne idee"),
]


@dataclass
class ResultatEval:
    """Resultats d'evaluation par categorie."""
    total: int = 0
    corrects: int = 0
    incorrects: int = 0
    non_corriges: int = 0  # attendait une correction mais rien fait
    faux_positifs: int = 0  # pas de correction attendue mais correction faite
    details: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        tp = self.corrects
        fp = self.incorrects + self.faux_positifs
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = self.corrects
        fn = self.non_corriges
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _normaliser(texte: str) -> str:
    return " ".join(texte.strip().lower().split())


_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)


def _extraire_mots(texte: str) -> list[str]:
    """Extrait les mots (sans ponctuation) d'un texte, en minuscules."""
    return [m.group().lower() for m in _MOT_RE.finditer(texte)]


def _trouver_mot_cible(erronee: str, attendue: str) -> tuple[str, str, int] | None:
    """Trouve le premier mot qui differe entre erronee et attendue.

    Utilise une extraction par mots (sans ponctuation) pour etre robuste
    aux variations de tokenisation WiCoPaCo (espaces autour de la ponctuation).

    Retourne (mot_erreur, mot_attendu, index_dans_mots) ou None.
    """
    mots_err = _extraire_mots(erronee)
    mots_att = _extraire_mots(attendue)
    if len(mots_err) != len(mots_att):
        return None
    for i, (a, b) in enumerate(zip(mots_err, mots_att)):
        if a != b:
            return (a, b, i)
    return None


def _tronquer_contexte(
    erronee: str, attendue: str, fenetre: int = 12,
) -> tuple[str, str] | None:
    """Tronque les phrases autour du mot cible.

    Extrait un contexte de ±fenetre mots autour du premier mot qui differe.
    Retourne (erronee_tronquee, attendue_tronquee) ou None si pas de diff.
    """
    # Utiliser les espaces WiCoPaCo (tokens separes par espaces)
    tokens_err = erronee.split()
    tokens_att = attendue.split()
    if len(tokens_err) != len(tokens_att):
        return None

    idx = None
    for i, (a, b) in enumerate(zip(tokens_err, tokens_att)):
        if a != b:
            idx = i
            break
    if idx is None:
        return None

    start = max(0, idx - fenetre)
    end = min(len(tokens_err), idx + fenetre + 1)
    return (
        " ".join(tokens_err[start:end]),
        " ".join(tokens_att[start:end]),
    )


def evaluer_paires(
    paires: list[tuple[str, str]],
    correcteur,
    label: str = "",
    *,
    mode_mot: bool = False,
) -> ResultatEval:
    """Evalue une liste de paires (erronee, attendue) sur le correcteur.

    Args:
        mode_mot: Si True, evaluation au niveau du mot cible uniquement
            (ignore les autres changements du correcteur). Utile pour les
            corpus longs type WiCoPaCo.
    """
    result = ResultatEval()
    for erronee, attendue in paires:
        result.total += 1
        res = correcteur.corriger(erronee)
        obtenu = res.phrase_corrigee

        if not mode_mot:
            # --- Mode phrase (comparaison globale) ---
            obtenu_norm = _normaliser(obtenu)
            attendu_norm = _normaliser(attendue)
            erronee_norm = _normaliser(erronee)

            correction_attendue = (attendu_norm != erronee_norm)
            correction_faite = (obtenu_norm != erronee_norm)

            if obtenu_norm == attendu_norm:
                result.corrects += 1
            elif correction_attendue and not correction_faite:
                result.non_corriges += 1
                result.details.append({
                    "type": "FN",
                    "erronee": erronee,
                    "attendue": attendue,
                    "obtenu": obtenu,
                })
            elif not correction_attendue and correction_faite:
                result.faux_positifs += 1
                result.details.append({
                    "type": "FP",
                    "erronee": erronee,
                    "attendue": attendue,
                    "obtenu": obtenu,
                })
            else:
                result.incorrects += 1
                result.details.append({
                    "type": "WRONG",
                    "erronee": erronee,
                    "attendue": attendue,
                    "obtenu": obtenu,
                })
        else:
            # --- Mode mot (comparaison du mot cible uniquement) ---
            # Tronquer les phrases autour du mot cible pour un contexte
            # realiste (le correcteur est prevu pour des phrases courtes)
            tronque = _tronquer_contexte(erronee, attendue)
            if tronque is None:
                result.incorrects += 1
                continue
            err_ctx, att_ctx = tronque

            cible = _trouver_mot_cible(err_ctx, att_ctx)
            if cible is None:
                result.incorrects += 1
                continue

            mot_err, mot_att, idx = cible

            # Corriger le contexte tronque
            res_ctx = correcteur.corriger(err_ctx)
            obtenu_ctx = res_ctx.phrase_corrigee
            mots_obtenu = _extraire_mots(obtenu_ctx)

            # Chercher le mot a la position idx dans la sortie
            # Tolerance : chercher dans une fenetre [idx-3, idx+3]
            mot_trouve = None
            for offset in (0, -1, 1, -2, 2, -3, 3):
                j = idx + offset
                if 0 <= j < len(mots_obtenu):
                    if mots_obtenu[j] == mot_att or mots_obtenu[j] == mot_err:
                        mot_trouve = mots_obtenu[j]
                        break

            correction_attendue = (mot_err != mot_att)

            if mot_trouve == mot_att:
                result.corrects += 1
            elif mot_trouve == mot_err and correction_attendue:
                result.non_corriges += 1
                result.details.append({
                    "type": "FN",
                    "mot": f"{mot_err} → {mot_att}",
                    "erronee": err_ctx[:80],
                    "attendue": att_ctx[:80],
                    "obtenu": obtenu_ctx[:80],
                })
            elif not correction_attendue:
                result.faux_positifs += 1
                result.details.append({
                    "type": "FP",
                    "mot": f"{mot_err} → {mot_att}",
                    "erronee": err_ctx[:80],
                    "attendue": att_ctx[:80],
                    "obtenu": obtenu_ctx[:80],
                })
            else:
                result.incorrects += 1
                result.details.append({
                    "type": "WRONG",
                    "mot": f"{mot_err} → {mot_att} (obtenu: {mot_trouve})",
                    "erronee": err_ctx[:80],
                    "attendue": att_ctx[:80],
                    "obtenu": obtenu_ctx[:80],
                })

    return result


def charger_corpus_tsv(path: str) -> dict[str, list[tuple[str, str]]]:
    """Charge un corpus TSV : type_erreur\\tphrase_erronee\\tphrase_corrigee."""
    categories: dict[str, list[tuple[str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            if row[0].startswith("#") or row[0] == "type_erreur":
                continue
            cat, erronee, corrigee = row[0].strip(), row[1].strip(), row[2].strip()
            categories.setdefault(cat, []).append((erronee, corrigee))
    return categories


def main():
    parser = argparse.ArgumentParser(description="Evaluation grammaire")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Chemin vers fichier TSV de paires")
    parser.add_argument("--built-in", action="store_true",
                        help="Utiliser les cas de test integres")
    parser.add_argument("--g2p", action="store_true",
                        help="Activer le G2P unifie (ONNX)")
    parser.add_argument("--double-tagging", action="store_true",
                        help="Activer le double tagging blind+lex")
    parser.add_argument("--word-level", action="store_true",
                        help="Evaluation au niveau du mot cible (pour corpus longs)")
    parser.add_argument("--max", type=int, default=0,
                        help="Limiter le nombre de paires par categorie (0=illimite)")
    args = parser.parse_args()

    if not args.corpus and not args.built_in:
        args.built_in = True

    # Charger le lexique
    sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
    from lectura_lexique import Lexique
    print(f"Chargement du lexique: {LEXIQUE_DB}")
    lexique = Lexique(LEXIQUE_DB)

    # Creer le correcteur
    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig

    config = CorrecteurConfig(
        activer_double_tagging=args.double_tagging,
    )

    tagger = None
    if args.g2p:
        from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
        tagger = creer_adapter_g2p_unifie()
        if tagger:
            print("G2P Unifie V2 charge")
        else:
            print("G2P Unifie V2 indisponible, fallback LexiqueTagger")

    correcteur = Correcteur(lexique, config=config, tagger=tagger)

    # Evaluer
    categories: dict[str, list[tuple[str, str]]] = {}

    if args.built_in:
        categories["conjugaison"] = _CAS_CONJUGAISON
        categories["homophones"] = _CAS_HOMOPHONES
        categories["accords"] = _CAS_ACCORDS

    if args.corpus:
        cats_tsv = charger_corpus_tsv(args.corpus)
        for cat, paires in cats_tsv.items():
            categories.setdefault(cat, []).extend(paires)

    # Limiter si demande
    if args.max > 0:
        for cat in categories:
            categories[cat] = categories[cat][:args.max]

    # Mode mot automatique pour corpus externe (phrases longues)
    mode_mot = args.word_level or (args.corpus is not None and not args.built_in)

    print(f"\n{'='*60}")
    mode_str = "mot" if mode_mot else "phrase"
    print(f"Evaluation grammaire — {sum(len(v) for v in categories.values())} cas (mode {mode_str})")
    print(f"{'='*60}\n")

    t0 = time.time()
    total_result = ResultatEval()

    for cat, paires in sorted(categories.items()):
        result = evaluer_paires(paires, correcteur, label=cat, mode_mot=mode_mot)
        print(f"[{cat}] {result.corrects}/{result.total} corrects "
              f"(P={result.precision:.2f} R={result.recall:.2f} F1={result.f1:.2f})")

        if result.details:
            for d in result.details[:10]:
                if "mot" in d:
                    print(f"  {d['type']}: {d['mot']}")
                else:
                    print(f"  {d['type']}: '{d['erronee']}' → attendu '{d['attendue']}' "
                          f"obtenu '{d['obtenu']}'")

        total_result.total += result.total
        total_result.corrects += result.corrects
        total_result.incorrects += result.incorrects
        total_result.non_corriges += result.non_corriges
        total_result.faux_positifs += result.faux_positifs
        total_result.details.extend(result.details)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_result.corrects}/{total_result.total} corrects")
    print(f"  Precision: {total_result.precision:.3f}")
    print(f"  Recall:    {total_result.recall:.3f}")
    print(f"  F1:        {total_result.f1:.3f}")
    print(f"  FP:        {total_result.faux_positifs}")
    print(f"  FN:        {total_result.non_corriges}")
    print(f"  Wrong:     {total_result.incorrects}")
    print(f"  Temps:     {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
