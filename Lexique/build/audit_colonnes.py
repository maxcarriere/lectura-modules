#!/usr/bin/env python3
"""Audit de coherence des colonnes phone / syllabes / orthocode.

Verifie la coherence des 3 colonnes cles de la table ``formes`` dans
le lexique Lectura (v5/v6).

Usage :
    python audit_colonnes.py
    python audit_colonnes.py --db ../lexique_lectura_v6.db
    python audit_colonnes.py --db ../lexique_lectura_v6.db --export anomalies.tsv
    python audit_colonnes.py --verbose
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ── Inventaire phonemique IPA autorise ──────────────────────────────────────

VOYELLES = set("aeiouyøœɑɔəɛ")
CONSONNES = set("bdfɡklmnpstvzʁʃʒɲŋ")
SEMI_VOYELLES = set("jwɥ")
COMBINING_TILDE = "\u0303"  # nasalisation

PHONEMES_IPA = VOYELLES | CONSONNES | SEMI_VOYELLES
PHONE_ALPHABET = PHONEMES_IPA | {COMBINING_TILDE, " "}

# Tokens valides entre crochets dans syllabes
BRACKET_TOKENS = {"z", "t", "n", "ʁ", "k", "p", "l", "-", "'"}

# Alphabet autorise pour syllabes (hors brackets)
SYLLABES_ALPHABET = PHONEMES_IPA | {COMBINING_TILDE, ".", " "}

# Alphabet autorise pour orthocode
_ORTHO_ACCENTS = set("àâäéèêëïîôùûüÿçæœÀÂÄÉÈÊËÏÎÔÙÛÜŸÇÆŒ")
_ORTHO_BASE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_ORTHO_SPECIAL = set(".°²'-() ")
ORTHOCODE_ALPHABET = _ORTHO_BASE | _ORTHO_ACCENTS | _ORTHO_SPECIAL

# Caracteres interdits connus dans phone
PHONE_FORBIDDEN = {
    ".": "dot (sigle epelle)",
    "‿": "undertie",
    "g": "ASCII g (au lieu de ɡ)",
    "[": "crochet ouvrant",
    "]": "crochet fermant",
    "°": "degree sign",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

_BRACKET_RE = re.compile(r"\[[^\]]*\]")


def _strip_brackets(s: str) -> str:
    """Supprime les tokens entre crochets d'une chaine de syllabes."""
    return _BRACKET_RE.sub("", s)


def _strip_syll_to_phone(syllabes: str) -> str:
    """Extrait le contenu phonemique brut de la colonne syllabes.

    Supprime les dots (separateurs de syllabes) et les tokens entre crochets.
    """
    s = _strip_brackets(syllabes)
    return s.replace(".", "")


def _count_syllabes_expected(syllabes: str) -> int:
    """Calcule le nombre de syllabes attendu depuis la colonne syllabes.

    Pour chaque composant (separe par espace), nb = dots + 1.
    Les tokens entre crochets ne comptent pas comme separateurs.
    """
    total = 0
    stripped = _strip_brackets(syllabes)
    for composant in stripped.split(" "):
        if not composant:
            continue
        total += composant.count(".") + 1
    return total


def _classify_phone_chars(phone: str) -> list[tuple[str, str]]:
    """Identifie les caracteres invalides dans la colonne phone.

    Returns:
        Liste de (char, description) pour chaque caractere invalide.
    """
    issues: list[tuple[str, str]] = []
    for ch in phone:
        if ch in PHONE_ALPHABET:
            continue
        if ch in PHONE_FORBIDDEN:
            issues.append((ch, PHONE_FORBIDDEN[ch]))
        elif ch.isascii() and ch.isalpha():
            issues.append((ch, f"ASCII latin '{ch}'"))
        elif unicodedata.category(ch).startswith("L"):
            issues.append((ch, f"lettre non-IPA '{ch}' (U+{ord(ch):04X})"))
        else:
            name = unicodedata.name(ch, f"U+{ord(ch):04X}")
            issues.append((ch, f"caractere interdit '{ch}' ({name})"))
    return issues


def _classify_syllabes_chars(syllabes: str) -> list[tuple[str, str]]:
    """Identifie les caracteres invalides dans la colonne syllabes (hors brackets)."""
    stripped = _strip_brackets(syllabes)
    issues: list[tuple[str, str]] = []
    for ch in stripped:
        if ch in SYLLABES_ALPHABET:
            continue
        if ch == "‿":
            issues.append((ch, "undertie"))
        elif ch.isascii() and ch.isalpha():
            issues.append((ch, f"ASCII latin '{ch}'"))
        elif unicodedata.category(ch).startswith("L"):
            issues.append((ch, f"lettre non-IPA '{ch}' (U+{ord(ch):04X})"))
        else:
            name = unicodedata.name(ch, f"U+{ord(ch):04X}")
            issues.append((ch, f"caractere interdit '{ch}' ({name})"))
    return issues


def _classify_orthocode_chars(orthocode: str) -> list[tuple[str, str]]:
    """Identifie les caracteres invalides dans la colonne orthocode."""
    issues: list[tuple[str, str]] = []
    for ch in orthocode:
        if ch in ORTHOCODE_ALPHABET:
            continue
        name = unicodedata.name(ch, f"U+{ord(ch):04X}")
        issues.append((ch, f"caractere interdit '{ch}' ({name})"))
    return issues


# ── Structure de collecte ────────────────────────────────────────────────────

class AnomalyCollector:
    """Collecte les anomalies par categorie avec comptage et echantillons."""

    def __init__(self, max_samples: int = 10):
        self.max_samples = max_samples
        self.counts: Counter[str] = Counter()
        self.by_source: dict[str, Counter[str]] = defaultdict(Counter)
        self.samples: dict[str, list[tuple]] = defaultdict(list)

    def add(self, category: str, source: str, sample: tuple) -> None:
        self.counts[category] += 1
        self.by_source[category][source] += 1
        if len(self.samples[category]) < self.max_samples:
            self.samples[category].append(sample)

    def add_all(self, category: str, source: str, sample: tuple) -> None:
        """Comme add() mais conserve tous les echantillons (mode verbose/export)."""
        self.counts[category] += 1
        self.by_source[category][source] += 1
        self.samples[category].append(sample)


# ── Audit principal ──────────────────────────────────────────────────────────

def run_audit(db_path: str, verbose: bool = False,
              export_path: str | None = None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM formes")
    total = cur.fetchone()[0]
    print(f"Base : {db_path}")
    print(f"Total formes : {total:,}\n")

    max_samples = 999_999_999 if (verbose or export_path) else 10
    phone_col = AnomalyCollector(max_samples)
    syll_col = AnomalyCollector(max_samples)
    ortho_col = AnomalyCollector(max_samples)

    # Compteurs globaux
    phone_ok = 0
    syll_ok = 0
    ortho_ok = 0
    phone_empty = 0

    # Inventaire de tous les caracteres uniques dans phone
    phone_char_counter: Counter[str] = Counter()

    cur.execute(
        "SELECT id, ortho, phone, syllabes, nb_syllabes, orthocode, source "
        "FROM formes"
    )

    for row in cur:
        rid = row["id"]
        ortho = row["ortho"] or ""
        phone = row["phone"] or ""
        syllabes = row["syllabes"] or ""
        nb_syll = row["nb_syllabes"]
        orthocode = row["orthocode"] or ""
        source = row["source"] or ""

        sample_base = (rid, ortho, phone, syllabes, nb_syll, orthocode, source)

        # ── 1. Audit phone ──────────────────────────────────────────────

        if not phone:
            phone_empty += 1
        else:
            phone_char_counter.update(phone)
            issues = _classify_phone_chars(phone)
            if issues:
                # Regrouper par type d'anomalie
                seen_types: set[str] = set()
                for _ch, desc in issues:
                    anom_type = f"phone:{desc.split('(')[0].strip()}"
                    if anom_type not in seen_types:
                        seen_types.add(anom_type)
                        if verbose or export_path:
                            phone_col.add_all(anom_type, source, sample_base)
                        else:
                            phone_col.add(anom_type, source, sample_base)
            else:
                phone_ok += 1

        # ── 2. Audit syllabes ───────────────────────────────────────────

        if syllabes:
            syll_issues = _classify_syllabes_chars(syllabes)
            syll_has_issue = False

            if syll_issues:
                syll_has_issue = True
                seen_types_s: set[str] = set()
                for _ch, desc in syll_issues:
                    anom_type = f"syllabes:char:{desc.split('(')[0].strip()}"
                    if anom_type not in seen_types_s:
                        seen_types_s.add(anom_type)
                        if verbose or export_path:
                            syll_col.add_all(anom_type, source, sample_base)
                        else:
                            syll_col.add(anom_type, source, sample_base)

            # Coherence phone vs syllabes (contenu phonemique)
            if phone and not _classify_phone_chars(phone):
                syll_phone_content = _strip_syll_to_phone(syllabes)
                if syll_phone_content != phone:
                    syll_has_issue = True
                    anom = "syllabes:contenu_phone_mismatch"
                    if verbose or export_path:
                        syll_col.add_all(anom, source, sample_base)
                    else:
                        syll_col.add(anom, source, sample_base)

            # Coherence nb_syllabes vs dots
            if nb_syll is not None:
                expected = _count_syllabes_expected(syllabes)
                if expected != nb_syll:
                    syll_has_issue = True
                    anom = "syllabes:nb_syllabes_mismatch"
                    if verbose or export_path:
                        syll_col.add_all(anom, source, sample_base)
                    else:
                        syll_col.add(anom, source, sample_base)

            if not syll_has_issue:
                syll_ok += 1

        # ── 3. Audit orthocode ──────────────────────────────────────────

        if phone and not orthocode:
            anom = "orthocode:vide_avec_phone"
            if verbose or export_path:
                ortho_col.add_all(anom, source, sample_base)
            else:
                ortho_col.add(anom, source, sample_base)
        elif orthocode:
            oc_issues = _classify_orthocode_chars(orthocode)
            if oc_issues:
                seen_types_o: set[str] = set()
                for _ch, desc in oc_issues:
                    anom_type = f"orthocode:char:{desc.split('(')[0].strip()}"
                    if anom_type not in seen_types_o:
                        seen_types_o.add(anom_type)
                        if verbose or export_path:
                            ortho_col.add_all(anom_type, source, sample_base)
                        else:
                            ortho_col.add(anom_type, source, sample_base)
            else:
                ortho_ok += 1

    conn.close()

    # ── Rapport ─────────────────────────────────────────────────────────

    print("=" * 72)
    print("INVENTAIRE CARACTERES PHONE")
    print("=" * 72)
    for ch, cnt in phone_char_counter.most_common():
        status = "OK" if ch in PHONE_ALPHABET else "INTERDIT"
        name = unicodedata.name(ch, f"U+{ord(ch):04X}")
        print(f"  '{ch}'  U+{ord(ch):04X}  {name:40s}  {cnt:>10,}  {status}")

    _print_section("PHONE", phone_col, phone_ok, phone_empty, total)
    _print_section("SYLLABES", syll_col, syll_ok, 0, total)
    _print_section("ORTHOCODE", ortho_col, ortho_ok, 0, total)

    # ── Export TSV ──────────────────────────────────────────────────────

    if export_path:
        _export_tsv(export_path, phone_col, syll_col, ortho_col)


def _print_section(
    title: str,
    collector: AnomalyCollector,
    ok_count: int,
    empty_count: int,
    total: int,
) -> None:
    anomaly_total = sum(collector.counts.values())
    print()
    print("=" * 72)
    print(f"AUDIT {title}")
    print("=" * 72)
    print(f"  OK       : {ok_count:>10,}")
    if empty_count:
        print(f"  Vides    : {empty_count:>10,}")
    print(f"  Anomalies: {anomaly_total:>10,}")
    print()

    if not collector.counts:
        print("  Aucune anomalie detectee.")
        return

    for anom_type, count in collector.counts.most_common():
        print(f"  [{anom_type}] : {count:,}")
        # Detail par source
        for src, src_cnt in collector.by_source[anom_type].most_common():
            print(f"      source={src or '(vide)'} : {src_cnt:,}")
        # Echantillons
        samples = collector.samples[anom_type]
        n_show = min(10, len(samples))
        if n_show:
            print(f"    Echantillons ({n_show}/{count}) :")
            for s in samples[:n_show]:
                rid, ortho, phone, syllabes, nb_syll, orthocode, source = s
                print(
                    f"      id={rid}  ortho={ortho!r}  phone={phone!r}  "
                    f"syllabes={syllabes!r}  nb_syll={nb_syll}  "
                    f"orthocode={orthocode!r}  src={source}"
                )
        print()


def _export_tsv(
    path: str,
    phone_col: AnomalyCollector,
    syll_col: AnomalyCollector,
    ortho_col: AnomalyCollector,
) -> None:
    """Exporte toutes les anomalies en TSV."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        header = "categorie\tid\tortho\tphone\tsyllabes\tnb_syllabes\torthocode\tsource\n"
        f.write(header)
        for collector in (phone_col, syll_col, ortho_col):
            for anom_type, samples in collector.samples.items():
                for s in samples:
                    rid, ortho, phone, syllabes, nb_syll, orthocode, source = s
                    line = (
                        f"{anom_type}\t{rid}\t{ortho}\t{phone}\t{syllabes}\t"
                        f"{nb_syll}\t{orthocode}\t{source}\n"
                    )
                    f.write(line)
                    count += 1
    print(f"\nExport : {count:,} anomalies ecrites dans {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit de coherence phone / syllabes / orthocode"
    )
    # Chercher la BDD dans les emplacements courants
    _script_dir = Path(__file__).resolve().parent
    _candidates = [
        _script_dir.parent / "lexique_lectura_v6.db",           # Modules/Lexique/
        _script_dir.parents[2] / "Lexique" / "lexique_lectura_v6.db",  # workspace/Lexique/
    ]
    _default_db = str(next((p for p in _candidates if p.exists()), _candidates[0]))

    parser.add_argument(
        "--db",
        default=_default_db,
        help="Chemin vers la BDD (defaut: auto-detection)",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Exporter les anomalies en TSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher tous les echantillons (pas juste 10)",
    )
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"Erreur : BDD introuvable : {db}", file=sys.stderr)
        sys.exit(1)

    run_audit(str(db), verbose=args.verbose, export_path=args.export)


if __name__ == "__main__":
    main()
