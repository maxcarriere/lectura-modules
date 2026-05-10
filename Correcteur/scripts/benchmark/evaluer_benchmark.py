#!/usr/bin/env python3
"""Benchmark GEC debiaise — Evaluation multi-outils.

Compare lectura-correcteur, Grammalecte (mode complet), LanguageTool
et un baseline (ne rien faire) sur un corpus de 120 phrases annotees.

Metriques standard GEC : edit-level Precision/Rappel/F0.5/F1.

Usage :
    python scripts/benchmark/evaluer_benchmark.py
    python scripts/benchmark/evaluer_benchmark.py --verbose
    python scripts/benchmark/evaluer_benchmark.py --skip-slow
    python scripts/benchmark/evaluer_benchmark.py --outils lectura,grammalecte
    python scripts/benchmark/evaluer_benchmark.py --markdown rapport.md
"""

from __future__ import annotations

import argparse
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))

LEXIQUE_PATH = Path(
    "/data/work/projets/lectura/workspace/Lexique/lexique_lectura.db"
)

from corpus_benchmark import CATEGORIES, CORPUS, CasBenchmark  # noqa: E402
from corpus_validation import CORPUS_VALIDATION  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Edits & metriques GEC
# ═══════════════════════════════════════════════════════════════════════════

class Edit(NamedTuple):
    """Une operation d'edition entre source et cible."""
    tag: str      # replace, insert, delete
    src: str      # tokens source (jointure espace)
    tgt: str      # tokens cible  (jointure espace)
    i1: int
    i2: int


def tokeniser_gec(phrase: str) -> list[str]:
    """Tokenise pour la comparaison GEC.

    - Strip ponctuation finale (.!?…)
    - Lowercase
    - Split les apostrophes : "l'ecole" → ["l'", "ecole"]
    - NE PAS supprimer les accents (a ≠ à)
    """
    phrase = phrase.strip()
    while phrase and phrase[-1] in ".!?…»":
        phrase = phrase[:-1]
    phrase = phrase.strip()
    phrase = phrase.lower()

    tokens: list[str] = []
    for word in phrase.split():
        if "'" in word and not word.startswith("'") and not word.endswith("'"):
            idx = word.index("'")
            tokens.append(word[: idx + 1])
            rest = word[idx + 1 :]
            if rest:
                tokens.append(rest)
        elif "'" in word and not word.startswith("'") and not word.endswith("'"):
            idx = word.index("'")
            tokens.append(word[: idx + 1])
            rest = word[idx + 1 :]
            if rest:
                tokens.append(rest)
        else:
            # Retirer ponctuation residuelle en debut/fin
            clean = word.strip(",:;«»\"()")
            if clean:
                tokens.append(clean)
    return tokens


def extraire_edits(source_tokens: list[str], target_tokens: list[str]) -> set[Edit]:
    """Extrait les edits (replace/insert/delete) entre source et cible."""
    sm = SequenceMatcher(None, source_tokens, target_tokens)
    edits: set[Edit] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        src = " ".join(source_tokens[i1:i2])
        tgt = " ".join(target_tokens[j1:j2])
        edits.add(Edit(tag=tag, src=src, tgt=tgt, i1=i1, i2=i2))
    return edits


@dataclass
class CompteursGEC:
    """Compteurs TP/FP/FN pour le calcul des metriques GEC."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 1.0
        return self.tp / (self.tp + self.fp)

    @property
    def rappel(self) -> float:
        if self.tp + self.fn == 0:
            return 1.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.rappel
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def f05(self) -> float:
        p, r = self.precision, self.rappel
        if p + r == 0:
            return 0.0
        return (1 + 0.25) * p * r / (0.25 * p + r)

    def __iadd__(self, other: CompteursGEC) -> CompteursGEC:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self


def evaluer_phrase(
    source: str,
    hypothese: str,
    references: list[str],
) -> CompteursGEC:
    """Evalue une phrase avec multi-reference (meilleur F1)."""
    src_tok = tokeniser_gec(source)
    hyp_tok = tokeniser_gec(hypothese)
    sys_edits = extraire_edits(src_tok, hyp_tok)

    meilleur: CompteursGEC | None = None
    meilleur_f1 = -1.0

    for ref in references:
        ref_tok = tokeniser_gec(ref)
        gold_edits = extraire_edits(src_tok, ref_tok)

        tp = len(gold_edits & sys_edits)
        fp = len(sys_edits - gold_edits)
        fn = len(gold_edits - sys_edits)
        c = CompteursGEC(tp=tp, fp=fp, fn=fn)

        if c.f1 > meilleur_f1:
            meilleur_f1 = c.f1
            meilleur = c

    return meilleur if meilleur is not None else CompteursGEC()


# ═══════════════════════════════════════════════════════════════════════════
#  Adaptateurs correcteurs
# ═══════════════════════════════════════════════════════════════════════════

class Adaptateur(ABC):
    """Interface commune pour tous les correcteurs."""

    nom: str

    @abstractmethod
    def corriger(self, phrase: str) -> str: ...

    def initialiser(self) -> None:
        """Hook d'initialisation (optionnel)."""


class AdaptateurLectura(Adaptateur):
    nom = "Lectura"

    def __init__(self) -> None:
        self._correcteur = None

    def initialiser(self) -> None:
        from lectura_correcteur import Correcteur
        from lectura_lexique import Lexique

        lexique = Lexique(str(LEXIQUE_PATH))

        # G2P optionnel pour suggestions phonetiques d<=1
        g2p = None
        try:
            from lectura_nlp.inference_numpy import NumpyInferenceEngine
            import pathlib
            g2p_dir = pathlib.Path(__file__).resolve().parents[3] / "G2P"
            weights = g2p_dir / "modeles_numpy" / "unifie_weights.json"
            vocab = g2p_dir / "src" / "lectura_nlp" / "modeles" / "unifie_vocab.json"
            if weights.exists() and vocab.exists():
                _engine = NumpyInferenceEngine(
                    weights_path=str(weights), vocab_path=str(vocab),
                )
                class _G2PAdapter:
                    def prononcer(self, mot: str) -> str | None:
                        try:
                            result = _engine.analyser([mot])
                            phones = result.get("g2p", [])
                            return phones[0] if phones else None
                        except Exception:
                            return None
                g2p = _G2PAdapter()
                print("  G2P phonetique d<=1: active")
        except Exception as e:
            print(f"  G2P phonetique: indisponible ({e})")

        self._correcteur = Correcteur(lexique, g2p=g2p)

    def corriger(self, phrase: str) -> str:
        assert self._correcteur is not None
        return self._correcteur.corriger(phrase).phrase_corrigee


class AdaptateurLecturaScoring(Adaptateur):
    """Lectura avec scoring unifie active."""

    nom = "Lec+Score"

    def __init__(self, seuil: float = 0.15) -> None:
        self._correcteur = None
        self._seuil = seuil

    def initialiser(self) -> None:
        from lectura_correcteur import Correcteur
        from lectura_correcteur._config import CorrecteurConfig
        from lectura_lexique import Lexique

        lexique = Lexique(str(LEXIQUE_PATH))

        g2p = None
        try:
            from lectura_nlp.inference_numpy import NumpyInferenceEngine
            import pathlib
            g2p_dir = pathlib.Path(__file__).resolve().parents[3] / "G2P"
            weights = g2p_dir / "modeles_numpy" / "unifie_weights.json"
            vocab = g2p_dir / "src" / "lectura_nlp" / "modeles" / "unifie_vocab.json"
            if weights.exists() and vocab.exists():
                _engine = NumpyInferenceEngine(
                    weights_path=str(weights), vocab_path=str(vocab),
                )
                class _G2PAdapter:
                    def prononcer(self, mot: str) -> str | None:
                        try:
                            result = _engine.analyser([mot])
                            phones = result.get("g2p", [])
                            return phones[0] if phones else None
                        except Exception:
                            return None
                g2p = _G2PAdapter()
                print("  G2P phonetique d<=1: active")
        except Exception as e:
            print(f"  G2P phonetique: indisponible ({e})")

        config = CorrecteurConfig(
            activer_scoring=True,
            seuil_remplacement=self._seuil,
        )
        self._correcteur = Correcteur(lexique, config=config, g2p=g2p)

    def corriger(self, phrase: str) -> str:
        assert self._correcteur is not None
        return self._correcteur.corriger(phrase).phrase_corrigee


class AdaptateurLecturaScoringAzerty(Adaptateur):
    """Lectura avec scoring unifie + AZERTY actives."""

    nom = "Lec+Az"

    def __init__(self, seuil: float = 0.15) -> None:
        self._correcteur = None
        self._seuil = seuil

    def initialiser(self) -> None:
        from lectura_correcteur import Correcteur
        from lectura_correcteur._config import CorrecteurConfig
        from lectura_lexique import Lexique

        lexique = Lexique(str(LEXIQUE_PATH))

        g2p = None
        try:
            from lectura_nlp.inference_numpy import NumpyInferenceEngine
            import pathlib
            g2p_dir = pathlib.Path(__file__).resolve().parents[3] / "G2P"
            weights = g2p_dir / "modeles_numpy" / "unifie_weights.json"
            vocab = g2p_dir / "src" / "lectura_nlp" / "modeles" / "unifie_vocab.json"
            if weights.exists() and vocab.exists():
                _engine = NumpyInferenceEngine(
                    weights_path=str(weights), vocab_path=str(vocab),
                )
                class _G2PAdapter:
                    def prononcer(self, mot: str) -> str | None:
                        try:
                            result = _engine.analyser([mot])
                            phones = result.get("g2p", [])
                            return phones[0] if phones else None
                        except Exception:
                            return None
                g2p = _G2PAdapter()
                print("  G2P phonetique d<=1: active")
        except Exception as e:
            print(f"  G2P phonetique: indisponible ({e})")

        config = CorrecteurConfig(
            activer_scoring=True,
            activer_azerty=True,
            seuil_remplacement=self._seuil,
        )
        self._correcteur = Correcteur(lexique, config=config, g2p=g2p)

    def corriger(self, phrase: str) -> str:
        assert self._correcteur is not None
        return self._correcteur.corriger(phrase).phrase_corrigee


class AdaptateurGrammalecte(Adaptateur):
    """Grammalecte en mode complet (grammaire + orthographe)."""

    nom = "Grammalecte"

    def __init__(self) -> None:
        self._gc = None

    def initialiser(self) -> None:
        import grammalecte

        self._gc = grammalecte.GrammarChecker("fr")

    def corriger(self, phrase: str) -> str:
        assert self._gc is not None
        aGrammErrs, aSpellErrs = self._gc.getParagraphErrors(
            phrase, bSpellSugg=True,
        )

        # Fusionner grammar + spelling errors
        all_fixes: list[tuple[int, int, str]] = []

        # Grammar errors
        for err in aGrammErrs:
            suggs = err.get("aSuggestions", [])
            if suggs:
                all_fixes.append((err["nStart"], err["nEnd"], suggs[0]))

        # Spelling errors
        for err in aSpellErrs:
            suggs = err.get("aSuggestions", [])
            if suggs:
                all_fixes.append((err["nStart"], err["nEnd"], suggs[0]))

        # Appliquer en reverse order (pour ne pas decaler les offsets)
        all_fixes.sort(key=lambda x: x[0], reverse=True)

        # Deduplication : si deux corrections couvrent la meme zone, garder la premiere
        result = phrase
        used_ranges: list[tuple[int, int]] = []
        for start, end, replacement in all_fixes:
            overlap = any(
                s < end and start < e for s, e in used_ranges
            )
            if not overlap:
                result = result[:start] + replacement + result[end:]
                used_ranges.append((start, end))

        return result


class AdaptateurLanguageTool(Adaptateur):
    nom = "LangTool"

    def __init__(self) -> None:
        self._tool = None

    def initialiser(self) -> None:
        import language_tool_python

        self._tool = language_tool_python.LanguageTool("fr")

    def corriger(self, phrase: str) -> str:
        assert self._tool is not None
        return self._tool.correct(phrase)


class AdaptateurBaseline(Adaptateur):
    """Ne rien faire (baseline)."""

    nom = "Baseline"

    def corriger(self, phrase: str) -> str:
        return phrase


# ═══════════════════════════════════════════════════════════════════════════
#  Resultats
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResultatOutil:
    """Resultats d'un outil sur tout le corpus."""
    nom: str
    temps_total: float = 0.0
    compteurs_global: CompteursGEC = field(default_factory=CompteursGEC)
    compteurs_par_cat: dict[str, CompteursGEC] = field(default_factory=dict)
    faux_positifs_ok: int = 0  # nb de phrases OK modifiees
    n_ok: int = 0


@dataclass
class ResultatPhrase:
    """Resultat par phrase et par outil."""
    cas: CasBenchmark
    corrections: dict[str, str] = field(default_factory=dict)  # nom_outil -> phrase_corrigee
    compteurs: dict[str, CompteursGEC] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _normaliser_simple(phrase: str) -> str:
    """Normalisation legere pour comparer si un outil a modifie la phrase."""
    phrase = phrase.strip()
    while phrase and phrase[-1] in ".!?…»":
        phrase = phrase[:-1]
    return " ".join(phrase.lower().split())


def evaluer_corpus(
    outils: list[Adaptateur],
    verbose: bool = False,
    corpus: list[CasBenchmark] | None = None,
) -> tuple[list[ResultatOutil], list[ResultatPhrase]]:
    """Evalue tous les outils sur le corpus."""
    if corpus is None:
        corpus = CORPUS

    # Initialisation
    for outil in outils:
        print(f"  Chargement {outil.nom}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            outil.initialiser()
            dt = time.perf_counter() - t0
            print(f"OK ({dt:.1f}s)")
        except Exception as exc:
            print(f"ERREUR ({exc})")
            outils.remove(outil)

    print()

    # Resultats
    res_outils: dict[str, ResultatOutil] = {
        o.nom: ResultatOutil(nom=o.nom) for o in outils
    }
    res_phrases: list[ResultatPhrase] = []
    n = len(corpus)

    for i, cas in enumerate(corpus, 1):
        rp = ResultatPhrase(cas=cas)

        if verbose:
            print(f"  [{i:3d}/{n}] {cas.id:<10s} ", end="", flush=True)

        for outil in outils:
            ro = res_outils[outil.nom]

            # Corriger
            t0 = time.perf_counter()
            try:
                corrigee = outil.corriger(cas.erronee)
            except Exception:
                corrigee = cas.erronee
            dt = time.perf_counter() - t0
            ro.temps_total += dt

            rp.corrections[outil.nom] = corrigee

            # Metriques GEC
            compteurs = evaluer_phrase(cas.erronee, corrigee, cas.attendue)
            rp.compteurs[outil.nom] = compteurs

            # Accumulation globale
            ro.compteurs_global += compteurs

            # Par categorie
            if cas.categorie not in ro.compteurs_par_cat:
                ro.compteurs_par_cat[cas.categorie] = CompteursGEC()
            ro.compteurs_par_cat[cas.categorie] += compteurs

            # Faux positifs sur phrases OK
            if cas.categorie == "OK":
                ro.n_ok += 1
                src_norm = _normaliser_simple(cas.erronee)
                hyp_norm = _normaliser_simple(corrigee)
                if src_norm != hyp_norm:
                    ro.faux_positifs_ok += 1

        if verbose:
            tags = []
            for outil in outils:
                c = rp.compteurs[outil.nom]
                sym = "=" if c.tp == 0 and c.fp == 0 and c.fn == 0 else (
                    "+" if c.fn == 0 and c.fp == 0 else "~"
                )
                tags.append(f"{outil.nom[0]}={sym}")
            print("  ".join(tags))

        res_phrases.append(rp)

    return list(res_outils.values()), res_phrases


# ═══════════════════════════════════════════════════════════════════════════
#  Rapport console
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(val: float) -> str:
    return f"{val:.3f}"


def afficher_rapport(
    res_outils: list[ResultatOutil],
    res_phrases: list[ResultatPhrase],
    verbose: bool = False,
) -> str:
    """Genere et affiche le rapport. Retourne le texte complet."""
    n_total = len(res_phrases)
    n_ok = sum(1 for rp in res_phrases if rp.cas.categorie == "OK")
    n_err = n_total - n_ok
    lines: list[str] = []

    def p(text: str = "") -> None:
        lines.append(text)
        print(text)

    sep = "=" * 76
    p(sep)
    p(f"  BENCHMARK GEC DEBIAISE — lectura-correcteur")
    p(f"  Corpus: {n_total} phrases ({n_err} erronees, {n_ok} correctes)")
    p(sep)

    # --- Scores globaux ---
    p()
    p("--- SCORES GLOBAUX (micro-averaged) ---")
    header = f"  {'':16s} | {'Precision':>10s} | {'Rappel':>10s} | {'F0.5':>10s} | {'F1':>10s}"
    p(header)
    p("  " + "-" * 64)
    for ro in res_outils:
        c = ro.compteurs_global
        p(f"  {ro.nom:<16s} | {_fmt(c.precision):>10s} | {_fmt(c.rappel):>10s}"
          f" | {_fmt(c.f05):>10s} | {_fmt(c.f1):>10s}")

    # --- Faux positifs ---
    p()
    p(f"--- FAUX POSITIFS (sur {n_ok} phrases correctes) ---")
    for ro in res_outils:
        n = ro.faux_positifs_ok
        total = n_ok
        pct = 100 * n / total if total > 0 else 0
        p(f"  {ro.nom:<16s} :  {n}/{total} ({pct:.1f}%)")

    # --- Par categorie ---
    p()
    cat_header = f"  {'Categorie':<12s} | {'Nb':>3s}"
    for ro in res_outils:
        cat_header += f" | {ro.nom[:10]:>10s}"
    p("--- PAR CATEGORIE (F1) ---")
    p(cat_header)
    p("  " + "-" * (20 + 13 * len(res_outils)))

    for cat in CATEGORIES:
        nb = sum(1 for rp in res_phrases if rp.cas.categorie == cat)
        if nb == 0:
            continue
        row = f"  {cat:<12s} | {nb:>3d}"
        for ro in res_outils:
            c = ro.compteurs_par_cat.get(cat, CompteursGEC())
            row += f" | {_fmt(c.f1):>10s}"
        p(row)

    # --- Temps d'execution ---
    p()
    p("--- TEMPS D'EXECUTION ---")
    for ro in res_outils:
        ms_phrase = 1000 * ro.temps_total / n_total if n_total > 0 else 0
        p(f"  {ro.nom:<16s} :  {ro.temps_total:.1f}s ({ms_phrase:.0f}ms/phrase)")

    # --- Detail verbose ---
    if verbose:
        p()
        p("--- DETAIL PHRASE PAR PHRASE ---")
        for rp in res_phrases:
            cas = rp.cas
            # Afficher si au moins un outil a des erreurs (FP ou FN)
            any_issue = any(
                rp.compteurs[ro.nom].fp > 0 or rp.compteurs[ro.nom].fn > 0
                for ro in res_outils
            )
            if not any_issue:
                continue

            p(f"\n  [{cas.id}] ({cas.categorie}/{cas.sous_categorie}, {cas.niveau})")
            p(f"    Erronee  : {cas.erronee}")
            p(f"    Attendue : {cas.attendue[0]}")
            if len(cas.attendue) > 1:
                for alt in cas.attendue[1:]:
                    p(f"             | {alt}")

            for ro in res_outils:
                c = rp.compteurs[ro.nom]
                corrigee = rp.corrections[ro.nom]
                sym = "+" if c.fn == 0 and c.fp == 0 else "~" if c.tp > 0 else "-"
                detail = f"TP={c.tp} FP={c.fp} FN={c.fn}"
                p(f"    {ro.nom:<12s} : {corrigee}  [{sym}] ({detail})")

    p()
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Export markdown
# ═══════════════════════════════════════════════════════════════════════════

def exporter_markdown(
    res_outils: list[ResultatOutil],
    res_phrases: list[ResultatPhrase],
    path: str,
) -> None:
    """Exporte le rapport en markdown."""
    n_total = len(res_phrases)
    n_ok = sum(1 for rp in res_phrases if rp.cas.categorie == "OK")
    n_err = n_total - n_ok

    md: list[str] = []
    md.append("# Benchmark GEC debiaise — Rapport")
    md.append("")
    md.append(f"Corpus : {n_total} phrases ({n_err} erronees, {n_ok} correctes)")
    md.append("")

    # Scores globaux
    md.append("## Scores globaux (micro-averaged)")
    md.append("")
    header = "| Outil | Precision | Rappel | F0.5 | F1 |"
    md.append(header)
    md.append("|-------|-----------|--------|------|-----|")
    for ro in res_outils:
        c = ro.compteurs_global
        md.append(f"| {ro.nom} | {_fmt(c.precision)} | {_fmt(c.rappel)}"
                  f" | {_fmt(c.f05)} | {_fmt(c.f1)} |")
    md.append("")

    # Faux positifs
    md.append("## Faux positifs")
    md.append("")
    md.append("| Outil | Modifiees | Total OK | % |")
    md.append("|-------|-----------|----------|---|")
    for ro in res_outils:
        pct = 100 * ro.faux_positifs_ok / n_ok if n_ok > 0 else 0
        md.append(f"| {ro.nom} | {ro.faux_positifs_ok} | {n_ok} | {pct:.1f}% |")
    md.append("")

    # Par categorie
    md.append("## Par categorie (F1)")
    md.append("")
    cat_header = "| Categorie | Nb |"
    for ro in res_outils:
        cat_header += f" {ro.nom} |"
    md.append(cat_header)
    cat_sep = "|-----------|---:|"
    for _ in res_outils:
        cat_sep += "------:|"
    md.append(cat_sep)
    for cat in CATEGORIES:
        nb = sum(1 for rp in res_phrases if rp.cas.categorie == cat)
        if nb == 0:
            continue
        row = f"| {cat} | {nb} |"
        for ro in res_outils:
            c = ro.compteurs_par_cat.get(cat, CompteursGEC())
            row += f" {_fmt(c.f1)} |"
        md.append(row)
    md.append("")

    # Temps
    md.append("## Temps d'execution")
    md.append("")
    md.append("| Outil | Total | Par phrase |")
    md.append("|-------|-------|------------|")
    for ro in res_outils:
        ms = 1000 * ro.temps_total / n_total if n_total > 0 else 0
        md.append(f"| {ro.nom} | {ro.temps_total:.1f}s | {ms:.0f}ms |")

    Path(path).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nRapport markdown exporté : {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Point d'entree
# ═══════════════════════════════════════════════════════════════════════════

OUTILS_DISPONIBLES = {
    "lectura": AdaptateurLectura,
    "lectura+scoring": AdaptateurLecturaScoring,
    "lectura+azerty": AdaptateurLecturaScoringAzerty,
    "grammalecte": AdaptateurGrammalecte,
    "langtool": AdaptateurLanguageTool,
    "baseline": AdaptateurBaseline,
}

OUTILS_LENTS = {"langtool"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GEC debiaise — Evaluation multi-outils",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Detail phrase par phrase",
    )
    parser.add_argument(
        "--skip-slow", action="store_true",
        help="Exclut LanguageTool (lent ~4min)",
    )
    parser.add_argument(
        "--outils", type=str, default=None,
        help="Outils a evaluer, separes par des virgules (ex: lectura,grammalecte)",
    )
    parser.add_argument(
        "--markdown", type=str, default=None,
        help="Chemin de sortie pour le rapport markdown",
    )
    parser.add_argument(
        "--corpus", type=str, default="dev",
        choices=["dev", "validation", "all"],
        help="Corpus a utiliser : dev (120 phrases), validation (60), all (180)",
    )
    args = parser.parse_args()

    # Selection du corpus
    if args.corpus == "dev":
        corpus_actif = CORPUS
    elif args.corpus == "validation":
        corpus_actif = CORPUS_VALIDATION
    else:
        corpus_actif = CORPUS + CORPUS_VALIDATION

    # Selection des outils
    if args.outils:
        noms = [n.strip().lower() for n in args.outils.split(",")]
    else:
        noms = list(OUTILS_DISPONIBLES.keys())

    if args.skip_slow:
        noms = [n for n in noms if n not in OUTILS_LENTS]

    outils: list[Adaptateur] = []
    for nom in noms:
        cls = OUTILS_DISPONIBLES.get(nom)
        if cls is None:
            print(f"Outil inconnu : {nom} (disponibles : {', '.join(OUTILS_DISPONIBLES)})")
            sys.exit(1)
        outils.append(cls())

    print(f"\nOutils : {', '.join(o.nom for o in outils)}")
    print(f"Corpus : {args.corpus} ({len(corpus_actif)} phrases)\n")

    res_outils, res_phrases = evaluer_corpus(
        outils, verbose=args.verbose, corpus=corpus_actif,
    )
    afficher_rapport(res_outils, res_phrases, verbose=args.verbose)

    if args.markdown:
        exporter_markdown(res_outils, res_phrases, args.markdown)


if __name__ == "__main__":
    main()
