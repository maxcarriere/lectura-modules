#!/usr/bin/env python3
"""Benchmark large multi-outils sur corpus independants.

Corpora :
1. corpus_10000.jsonl — 10k paires correct/fautif, erreurs synthetiques
2. grammaire_wicopaco.tsv — 6k erreurs reelles Wikipedia
3. negatif_wicopaco.tsv — 1000 phrases propres (FP)

Metriques :
- Recall par type d'erreur (mot fautif -> mot correct)
- Target recall WiCoPaCo (le correcteur a-t-il corrige l'edit annote ?)
- Edit-level P/R/F via alignement difflib
- Faux positifs sur texte propre

Usage:
    python benchmark_large.py [--n-synth 2000] [--n-wiki 1000] [--n-fp 500]
    python benchmark_large.py --verbose
    python benchmark_large.py --all   # tout le corpus
    python benchmark_large.py --outils v6,grammalecte,langtool
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

WORKSPACE = ROOT.parent.parent
CORPUS_SYNTH = WORKSPACE / "Corpus" / "Correcteur" / "corpus_10000.jsonl"
CORPUS_WIKI = WORKSPACE / "Corpus" / "Correcteur" / "grammaire_wicopaco.tsv"
CORPUS_FP = WORKSPACE / "Corpus" / "Correcteur" / "negatif_wicopaco.tsv"
LEXIQUE_PATH = Path(
    "/data/work/projets/lectura/workspace/Lexique/bases/lexique_lectura.db"
)

SEED = 42


# ========================================================================
# Utilitaires
# ========================================================================

def normaliser(phrase: str) -> str:
    """Normalisation pour comparaison."""
    phrase = phrase.strip()
    phrase = " ".join(phrase.split())
    return phrase.lower()


def tokeniser_simple(phrase: str) -> list[str]:
    """Tokenisation simple pour comparaison (split sur espaces apres normalisation)."""
    return normaliser(phrase).split()


def diff_mots(src: list[str], tgt: list[str]) -> list[tuple[str, str]]:
    """Calcule les substitutions mot-a-mot entre src et tgt via SequenceMatcher.

    Retourne une liste de (mot_src, mot_tgt) pour chaque substitution.
    Les insertions sont representees comme ("", mot_tgt).
    Les suppressions comme (mot_src, "").
    """
    edits: list[tuple[str, str]] = []
    sm = SequenceMatcher(None, src, tgt)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            # Aligner les mots remplaces un a un
            src_chunk = src[i1:i2]
            tgt_chunk = tgt[j1:j2]
            for k in range(max(len(src_chunk), len(tgt_chunk))):
                s = src_chunk[k] if k < len(src_chunk) else ""
                t = tgt_chunk[k] if k < len(tgt_chunk) else ""
                edits.append((s, t))
        elif tag == "insert":
            for j in range(j1, j2):
                edits.append(("", tgt[j]))
        elif tag == "delete":
            for i in range(i1, i2):
                edits.append((src[i], ""))
    return edits


# ========================================================================
# Chargement des corpus
# ========================================================================

@dataclass
class CasSynth:
    fautif: str
    correct: str
    erreurs: list[dict]
    n_erreurs: int


def charger_corpus_synth(n: int, seed: int = SEED) -> list[CasSynth]:
    """Charge n phrases du corpus synthetique."""
    if not CORPUS_SYNTH.exists():
        print(f"  [WARN] Corpus synthetique non trouve : {CORPUS_SYNTH}")
        return []
    all_cases: list[CasSynth] = []
    with open(CORPUS_SYNTH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            all_cases.append(CasSynth(
                fautif=d["fautif"],
                correct=d["correct"],
                erreurs=d.get("erreurs", []),
                n_erreurs=d.get("n_erreurs", 0),
            ))
    rng = random.Random(seed)
    if len(all_cases) > n:
        all_cases = rng.sample(all_cases, n)
    return all_cases


@dataclass
class CasWiki:
    type_erreur: str
    erronee: str
    corrigee: str


def charger_corpus_wiki(n: int, seed: int = SEED) -> list[CasWiki]:
    """Charge n phrases de WiCoPaCo grammaire."""
    if not CORPUS_WIKI.exists():
        print(f"  [WARN] Corpus WiCoPaCo non trouve : {CORPUS_WIKI}")
        return []
    all_cases: list[CasWiki] = []
    with open(CORPUS_WIKI, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                # Filtrer les phrases trop longues (paragraphes)
                if len(parts[1]) > 500:
                    continue
                all_cases.append(CasWiki(
                    type_erreur=parts[0],
                    erronee=parts[1],
                    corrigee=parts[2],
                ))
    rng = random.Random(seed)
    if len(all_cases) > n:
        all_cases = rng.sample(all_cases, n)
    return all_cases


def charger_fp(n: int, seed: int = SEED) -> list[str]:
    """Charge n phrases propres pour test FP."""
    if not CORPUS_FP.exists():
        print(f"  [WARN] Corpus FP non trouve : {CORPUS_FP}")
        return []
    phrases: list[str] = []
    with open(CORPUS_FP, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] == "negatif":
                texte = parts[1].strip()
                if texte and len(texte) <= 500:
                    phrases.append(texte)
    rng = random.Random(seed)
    if len(phrases) > n:
        phrases = rng.sample(phrases, n)
    return phrases


# ========================================================================
# Evaluation
# ========================================================================

def evaluer_synth(correcteur, corpus: list[CasSynth], verbose: bool) -> dict:
    """Evalue le corpus synthetique avec recall par type + edits P/R/F."""
    by_type: dict[str, list[bool]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    echecs: list[tuple[str, str, str, str]] = []  # (type, perturbe, original, obtenu)

    for cas in corpus:
        result = correcteur.corriger(cas.fautif)
        sortie = result.phrase_corrigee

        # Edits gold et systeme via difflib
        fautif_mots = tokeniser_simple(cas.fautif)
        correct_mots = tokeniser_simple(cas.correct)
        sortie_mots = tokeniser_simple(sortie)

        gold_edits = set(diff_mots(fautif_mots, correct_mots))
        sys_edits = set(diff_mots(fautif_mots, sortie_mots))

        tp = len(gold_edits & sys_edits)
        fp = len(sys_edits - gold_edits)
        fn = len(gold_edits - sys_edits)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Recall par type d'erreur (mot specifique corrige ?)
        for err in cas.erreurs:
            t = err["type"]
            if t not in by_type:
                by_type[t] = []
            original = err["original"].lower()
            perturbe = err["perturbe"].lower()
            # Check : le mot original est-il dans la sortie ?
            if original in sortie_mots:
                by_type[t].append(True)
            else:
                by_type[t].append(False)
                if verbose and len(echecs) < 5:
                    # Chercher ce qui a ete produit
                    obtenu = "?"
                    pos = err["position"]
                    if pos < len(sortie_mots):
                        obtenu = sortie_mots[pos]
                    echecs.append((t, perturbe, original, obtenu))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0

    return {
        "by_type": by_type,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": precision, "recall": recall, "f1": f1, "f05": f05,
        "echecs": echecs,
    }


def evaluer_wiki(correcteur, corpus: list[CasWiki], verbose: bool) -> dict:
    """Evalue WiCoPaCo avec target recall et edit-level metrics.

    Target recall = le correcteur a-t-il produit les memes substitutions
    que l'editeur Wikipedia (meme partiellement) ?
    """
    by_type: dict[str, list[bool]] = {}
    total_target_ok = 0
    total_target = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    # Compteur de phrases ou le correcteur a fait au moins un changement utile
    phrases_ameliorees = 0
    echecs: list[tuple[str, str, str, str]] = []

    for cas in corpus:
        result = correcteur.corriger(cas.erronee)
        sortie = result.phrase_corrigee

        erronee_mots = tokeniser_simple(cas.erronee)
        corrigee_mots = tokeniser_simple(cas.corrigee)
        sortie_mots = tokeniser_simple(sortie)

        # Gold edits : erronee -> corrigee (ce que Wikipedia a corrige)
        gold_edits = set(diff_mots(erronee_mots, corrigee_mots))
        # Sys edits : erronee -> sortie (ce que le correcteur a change)
        sys_edits = set(diff_mots(erronee_mots, sortie_mots))

        # Target recall : combien d'edits gold le correcteur a-t-il reproduits ?
        tp = len(gold_edits & sys_edits)
        # FP : edits du correcteur qui ne sont PAS dans le gold
        # Note : certains de ces "FP" sont en fait des corrections valides
        # d'autres erreurs dans la phrase (non annotees par WiCoPaCo)
        fp = len(sys_edits - gold_edits)
        fn = len(gold_edits - sys_edits)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_target += len(gold_edits)
        total_target_ok += tp

        if tp > 0:
            phrases_ameliorees += 1

        # Par type : au moins un gold edit reproduit ?
        t = cas.type_erreur
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(tp > 0)

        if verbose and tp == 0 and len(echecs) < 3:
            # Montrer les edits gold non reproduits
            for (old, new) in list(gold_edits)[:1]:
                obtenu = "inchange"
                # Chercher ce que le correcteur a produit pour ce mot
                for (so, sn) in sys_edits:
                    if so == old:
                        obtenu = sn
                        break
                echecs.append((t, old, new, obtenu))

    target_recall = total_target_ok / total_target if total_target > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0

    return {
        "by_type": by_type,
        "target_recall": target_recall,
        "phrases_ameliorees": phrases_ameliorees,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": precision, "recall": recall, "f1": f1, "f05": f05,
        "echecs": echecs,
    }


def evaluer_fp(correcteur, corpus: list[str], verbose: bool) -> dict:
    """Evalue le taux de faux positifs sur texte propre.

    Note : le corpus 'negatif' WiCoPaCo peut contenir des phrases avec de
    vraies erreurs non annotees. Les modifications sont comptees puis
    classifiees en 'probablement TP' (accent, resegmentation) vs 'probablement FP'.
    """
    modifications: list[tuple[str, str, list[tuple[str, str]]]] = []

    for phrase in corpus:
        result = correcteur.corriger(phrase)
        sortie = result.phrase_corrigee
        if normaliser(sortie) != normaliser(phrase):
            phrase_mots = tokeniser_simple(phrase)
            sortie_mots = tokeniser_simple(sortie)
            edits = diff_mots(phrase_mots, sortie_mots)
            modifications.append((phrase[:150], sortie[:150], edits))

    # Classifier les modifications
    n_accent = 0
    n_reseg = 0
    n_accord = 0
    n_autre = 0
    for _, _, edits in modifications:
        for old, new in edits:
            if not old or not new:
                n_reseg += 1
            elif _est_correction_accent(old, new):
                n_accent += 1
            elif _est_correction_accord(old, new):
                n_accord += 1
            else:
                n_autre += 1

    return {
        "total": len(corpus),
        "modifiees": len(modifications),
        "details": modifications[:20] if verbose else modifications[:5],
        "n_accent": n_accent,
        "n_reseg": n_reseg,
        "n_accord": n_accord,
        "n_autre": n_autre,
    }


def _est_correction_accent(old: str, new: str) -> bool:
    """Verifie si la difference entre old et new est juste un accent."""
    import unicodedata
    def sans_accents(s: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
    return sans_accents(old) == sans_accents(new)


def _est_correction_accord(old: str, new: str) -> bool:
    """Verifie si la difference est un accord (ajout/retrait de s/e/es/ent)."""
    if old.endswith("s") and new == old[:-1]:
        return True
    if new.endswith("s") and old == new[:-1]:
        return True
    if old.endswith("e") and new == old[:-1]:
        return True
    if new.endswith("e") and old == new[:-1]:
        return True
    if old.endswith("es") and new == old[:-2]:
        return True
    if new.endswith("es") and old == new[:-2]:
        return True
    return False


# ========================================================================
# Adaptateurs multi-outils
# ========================================================================

from abc import ABC, abstractmethod


class Outil(ABC):
    """Interface abstraite pour un correcteur."""
    nom: str

    @abstractmethod
    def initialiser(self) -> None: ...

    @abstractmethod
    def corriger(self, phrase: str) -> str: ...


class OutilV6(Outil):
    """Lectura V6 avec config complete (toutes les regles actives)."""
    nom = "V6"

    def initialiser(self) -> None:
        from lectura_correcteur import CorrecteurV6, CorrecteurV6Config
        from lectura_lexique import Lexique
        lexique = Lexique(str(LEXIQUE_PATH))
        config = CorrecteurV6Config()
        self._correcteur = CorrecteurV6(lexique, config=config)

    def corriger(self, phrase: str) -> str:
        return self._correcteur.corriger(phrase).phrase_corrigee


class OutilV6Safe(Outil):
    """Lectura V6 profil safe (regles P2G desactivees)."""
    nom = "V6-safe"

    def initialiser(self) -> None:
        from lectura_correcteur import CorrecteurV6, CorrecteurV6Config
        from lectura_lexique import Lexique
        lexique = Lexique(str(LEXIQUE_PATH))
        config = CorrecteurV6Config(
            activer_p2g_global=False,
            activer_verbe_p2g=False,
            activer_accent_p2g=False,
            activer_accord_attribut=False,
            activer_pp_etre=False,
        )
        self._correcteur = CorrecteurV6(lexique, config=config)

    def corriger(self, phrase: str) -> str:
        return self._correcteur.corriger(phrase).phrase_corrigee


class OutilGrammalecte(Outil):
    """Grammalecte (mode complet)."""
    nom = "Grammalecte"

    def initialiser(self) -> None:
        import grammalecte
        self._gc = grammalecte.GrammarChecker("fr")

    def corriger(self, phrase: str) -> str:
        result = self._gc.getParagraphErrors(phrase)
        all_errors = list(result[0]) + list(result[1])
        all_fixes: list[tuple[int, int, str]] = []
        for err in all_errors:
            suggs = err.get("aSuggestions", [])
            if suggs:
                all_fixes.append((err["nStart"], err["nEnd"], suggs[0]))
        all_fixes.sort(key=lambda x: x[0], reverse=True)
        result_str = phrase
        used: list[tuple[int, int]] = []
        for start, end, repl in all_fixes:
            overlap = any(s < end and start < e for s, e in used)
            if not overlap:
                result_str = result_str[:start] + repl + result_str[end:]
                used.append((start, end))
        return result_str


class OutilLangTool(Outil):
    """LanguageTool pour le francais."""
    nom = "LangTool"

    def initialiser(self) -> None:
        import language_tool_python
        self._tool = language_tool_python.LanguageTool("fr")

    def corriger(self, phrase: str) -> str:
        return self._tool.correct(phrase)


OUTILS_DISPONIBLES: dict[str, type[Outil]] = {
    "v6": OutilV6,
    "v6-safe": OutilV6Safe,
    "grammalecte": OutilGrammalecte,
    "langtool": OutilLangTool,
}


# ========================================================================
# Affichage multi-outils
# ========================================================================

def _fmt_pct(v: float) -> str:
    return f"{v:>6.1f}%"


def afficher_resultats_multi(
    outils: list[Outil],
    resultats_synth: dict[str, dict],
    resultats_wiki: dict[str, dict],
    resultats_fp: dict[str, dict],
    corpus_synth: list,
    corpus_wiki: list,
    corpus_fp: list,
    verbose: bool,
    duree: float,
) -> None:
    """Affiche les resultats pour tous les outils en tableaux comparatifs."""

    noms = [o.nom for o in outils]
    col_w = max(12, *(len(n) for n in noms)) + 1

    # ==================================================================
    # A. RECALL — Corpus synthetique
    # ==================================================================
    print("=" * 76)
    print("  A. RECALL — Corpus synthetique (erreurs generees)")
    print(f"  {len(corpus_synth)} phrases")
    print("=" * 76)

    # Collecter tous les types
    all_types: set[str] = set()
    for rs in resultats_synth.values():
        all_types.update(rs["by_type"].keys())

    # Tableau recall par type
    print()
    header = f"  {'Type':<12s}"
    for nom in noms:
        header += f" | {nom:>{col_w}s}"
    print(header)
    sep = f"  {'-'*12}"
    for _ in noms:
        sep += f"-+-{'-'*col_w}"
    print(sep)

    for t in sorted(all_types):
        row = f"  {t:<12s}"
        for nom in noms:
            rs = resultats_synth[nom]
            results = rs["by_type"].get(t, [])
            ok = sum(results)
            n = len(results)
            pct = 100.0 * ok / n if n else 0
            row += f" | {pct:>{col_w-1}.1f}%"
        print(row)

    print(sep)
    row = f"  {'TOTAL':<12s}"
    for nom in noms:
        rs = resultats_synth[nom]
        total_ok = sum(sum(v) for v in rs["by_type"].values())
        total_n = sum(len(v) for v in rs["by_type"].values())
        pct = 100.0 * total_ok / total_n if total_n else 0
        row += f" | {pct:>{col_w-1}.1f}%"
    print(row)

    # Edits P/R/F
    print()
    print(f"  {'Metrique':<12s}", end="")
    for nom in noms:
        print(f" | {nom:>{col_w}s}", end="")
    print()
    print(sep)
    for metric in ["precision", "recall", "f1", "f05"]:
        label = {"precision": "Precision", "recall": "Recall", "f1": "F1", "f05": "F0.5"}[metric]
        row = f"  {label:<12s}"
        for nom in noms:
            v = resultats_synth[nom][metric]
            row += f" | {v:>{col_w}.3f}"
        print(row)

    # ==================================================================
    # B. RECALL — WiCoPaCo grammaire
    # ==================================================================
    print()
    print("=" * 76)
    print("  B. RECALL — WiCoPaCo grammaire (erreurs reelles Wikipedia)")
    print(f"  {len(corpus_wiki)} phrases")
    print("=" * 76)

    all_types_w: set[str] = set()
    for rw in resultats_wiki.values():
        all_types_w.update(rw["by_type"].keys())

    print()
    header = f"  {'Type':<12s}"
    for nom in noms:
        header += f" | {nom:>{col_w}s}"
    print(header)
    print(sep)
    for t in sorted(all_types_w):
        row = f"  {t:<12s}"
        for nom in noms:
            rw = resultats_wiki[nom]
            results = rw["by_type"].get(t, [])
            ok = sum(results)
            n = len(results)
            pct = 100.0 * ok / n if n else 0
            row += f" | {pct:>{col_w-1}.1f}%"
        print(row)
    print(sep)
    row = f"  {'TOTAL':<12s}"
    for nom in noms:
        rw = resultats_wiki[nom]
        total_ok = sum(sum(v) for v in rw["by_type"].values())
        total_n = sum(len(v) for v in rw["by_type"].values())
        pct = 100.0 * total_ok / total_n if total_n else 0
        row += f" | {pct:>{col_w-1}.1f}%"
    print(row)

    print()
    print(f"  {'Metrique':<12s}", end="")
    for nom in noms:
        print(f" | {nom:>{col_w}s}", end="")
    print()
    print(sep)
    for metric in ["target_recall", "precision", "recall", "f1", "f05"]:
        label = {"target_recall": "Tgt Recall", "precision": "Precision",
                 "recall": "Recall", "f1": "F1", "f05": "F0.5"}[metric]
        row = f"  {label:<12s}"
        for nom in noms:
            v = resultats_wiki[nom][metric]
            row += f" | {v:>{col_w}.3f}"
        print(row)

    # ==================================================================
    # C. FAUX POSITIFS — Texte propre
    # ==================================================================
    print()
    print("=" * 76)
    print("  C. FAUX POSITIFS — Texte propre (WiCoPaCo negatif)")
    print(f"  {len(corpus_fp)} phrases")
    print("=" * 76)
    print()

    header = f"  {'Categorie':<16s}"
    for nom in noms:
        header += f" | {nom:>{col_w}s}"
    print(header)
    sep2 = f"  {'-'*16}"
    for _ in noms:
        sep2 += f"-+-{'-'*col_w}"
    print(sep2)
    for label, key in [
        ("Modifiees", "modifiees"),
        ("Accent", "n_accent"),
        ("Resegmentation", "n_reseg"),
        ("Accord", "n_accord"),
        ("Autre (FP)", "n_autre"),
    ]:
        row = f"  {label:<16s}"
        for nom in noms:
            rf = resultats_fp[nom]
            v = rf[key]
            if key == "modifiees":
                pct = 100.0 * v / rf["total"] if rf["total"] else 0
                row += f" | {f'{v} ({pct:.1f}%)':>{col_w}s}"
            else:
                row += f" | {v:>{col_w}d}"
        print(row)

    if verbose:
        for nom in noms:
            rf = resultats_fp[nom]
            if rf["details"]:
                print()
                print(f"  Detail [{nom}] :")
                for i, (orig, corr, edits) in enumerate(rf["details"][:5], 1):
                    edits_str = ", ".join(f"{o}->{n}" for o, n in edits[:3])
                    print(f"    [{i:2d}] {edits_str}")
                    print(f"         {orig[:100]}")

    # ==================================================================
    # RESUME
    # ==================================================================
    print()
    print("=" * 76)
    print("  RESUME COMPARATIF")
    print("=" * 76)
    print()

    header = f"  {'Metrique':<25s}"
    for nom in noms:
        header += f" | {nom:>{col_w}s}"
    print(header)
    sep3 = f"  {'-'*25}"
    for _ in noms:
        sep3 += f"-+-{'-'*col_w}"
    print(sep3)

    # Synth recall
    row = f"  {'Synth recall erreur':<25s}"
    for nom in noms:
        rs = resultats_synth[nom]
        total_ok = sum(sum(v) for v in rs["by_type"].values())
        total_n = sum(len(v) for v in rs["by_type"].values())
        pct = 100.0 * total_ok / total_n if total_n else 0
        row += f" | {pct:>{col_w-1}.1f}%"
    print(row)

    # Synth P/R/F
    for metric, label in [("precision", "Synth Precision"), ("recall", "Synth Recall edit"),
                           ("f1", "Synth F1"), ("f05", "Synth F0.5")]:
        row = f"  {label:<25s}"
        for nom in noms:
            row += f" | {resultats_synth[nom][metric]:>{col_w}.3f}"
        print(row)

    print(sep3)

    # Wiki target recall
    row = f"  {'Wiki target recall':<25s}"
    for nom in noms:
        row += f" | {resultats_wiki[nom]['target_recall']:>{col_w}.1%}"
    print(row)

    for metric, label in [("precision", "Wiki Precision"), ("recall", "Wiki Recall edit"),
                           ("f1", "Wiki F1"), ("f05", "Wiki F0.5")]:
        row = f"  {label:<25s}"
        for nom in noms:
            row += f" | {resultats_wiki[nom][metric]:>{col_w}.3f}"
        print(row)

    print(sep3)

    # FP
    row = f"  {'FP modifiees':<25s}"
    for nom in noms:
        rf = resultats_fp[nom]
        pct = 100.0 * rf["modifiees"] / rf["total"] if rf["total"] else 0
        row += f" | {pct:>{col_w-1}.1f}%"
    print(row)

    row = f"  {'FP vrais estimes':<25s}"
    for nom in noms:
        row += f" | {resultats_fp[nom]['n_autre']:>{col_w}d}"
    print(row)

    print()
    print(f"  Duree totale : {duree:.1f}s")


# ========================================================================
# Main
# ========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark large multi-outils — Recall + FP sur corpus independants",
    )
    parser.add_argument("--n-synth", type=int, default=2000,
                        help="Nombre de phrases synthetiques (defaut: 2000)")
    parser.add_argument("--n-wiki", type=int, default=1000,
                        help="Nombre de phrases WiCoPaCo (defaut: 1000)")
    parser.add_argument("--n-fp", type=int, default=500,
                        help="Nombre de phrases FP (defaut: 500)")
    parser.add_argument("--all", action="store_true",
                        help="Utiliser tout le corpus (10k synth, 6k wiki, 1k fp)")
    parser.add_argument("--outils", type=str, default="v6",
                        help="Outils a comparer, separes par des virgules "
                             f"(dispo: {', '.join(OUTILS_DISPONIBLES)})")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.n_synth = 999999
        args.n_wiki = 999999
        args.n_fp = 999999

    # --- Selection des outils ---
    noms_outils = [n.strip().lower() for n in args.outils.split(",")]
    outils: list[Outil] = []
    for nom in noms_outils:
        cls = OUTILS_DISPONIBLES.get(nom)
        if cls is None:
            print(f"Outil inconnu : {nom} (dispo: {', '.join(OUTILS_DISPONIBLES)})")
            sys.exit(1)
        outils.append(cls())

    # --- Chargement ---
    for outil in outils:
        print(f"Chargement {outil.nom}...")
        outil.initialiser()
        print(f"  {outil.nom} pret.")

    print()
    print("Chargement des corpus...")
    corpus_synth = charger_corpus_synth(args.n_synth)
    corpus_wiki = charger_corpus_wiki(args.n_wiki)
    corpus_fp = charger_fp(args.n_fp)
    print(f"  Synthetique : {len(corpus_synth)} phrases")
    print(f"  WiCoPaCo    : {len(corpus_wiki)} phrases")
    print(f"  FP (propre) : {len(corpus_fp)} phrases")
    print()

    t0 = time.time()

    # --- Evaluation par outil ---
    resultats_synth: dict[str, dict] = {}
    resultats_wiki: dict[str, dict] = {}
    resultats_fp: dict[str, dict] = {}

    for outil in outils:
        print(f"Evaluation {outil.nom}...")

        # Wrapper pour uniformiser l'interface
        class _Wrapper:
            def __init__(self, o: Outil):
                self._o = o
            def corriger(self, phrase: str):
                class _R:
                    def __init__(self, p: str):
                        self.phrase_corrigee = p
                return _R(self._o.corriger(phrase))

        wrapper = _Wrapper(outil)
        t_outil = time.time()
        resultats_synth[outil.nom] = evaluer_synth(wrapper, corpus_synth, args.verbose)
        resultats_wiki[outil.nom] = evaluer_wiki(wrapper, corpus_wiki, args.verbose)
        resultats_fp[outil.nom] = evaluer_fp(wrapper, corpus_fp, args.verbose)
        dt = time.time() - t_outil
        print(f"  {outil.nom} termine en {dt:.1f}s")

    duree = time.time() - t0

    # --- Affichage ---
    print()
    afficher_resultats_multi(
        outils, resultats_synth, resultats_wiki, resultats_fp,
        corpus_synth, corpus_wiki, corpus_fp,
        args.verbose, duree,
    )


if __name__ == "__main__":
    main()
