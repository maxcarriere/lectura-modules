#!/usr/bin/env python3
"""Correction des anomalies phone / syllabes / orthocode dans le lexique.

Corrige les anomalies identifiees par ``audit_colonnes.py`` :
- Passe 1 : phone (ASCII g, dots, undertie, accents latins, divers)
- Passe 2 : syllabes (resyllabification, recalcul nb_syllabes)
- Passe 3 : orthocode (apostrophe typographique, orthocodes vides)

Usage :
    python corriger_colonnes.py
    python corriger_colonnes.py --db ../lexique_lectura_v6.db
    python corriger_colonnes.py --dry-run
    python corriger_colonnes.py --skip-orthocode-regen
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sqlite3
import sys
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent

# ── Auto-detection de la BDD ─────────────────────────────────────────

_DB_CANDIDATES = [
    _SCRIPT_DIR.parent / "lexique_lectura_v6.db",
    _SCRIPT_DIR.parents[2] / "Lexique" / "lexique_lectura_v6.db",
]

# ── Fonctions IPA importees de patch_ipa_dico.py ─────────────────────

_VOYELLES: set[str] = {
    "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə",
}
_CONSONNES: set[str] = {
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
    "ʃ", "ʒ", "m", "n", "ɲ", "ŋ", "l", "ʁ",
}
_SEMI_VOYELLES: set[str] = {"j", "w", "ɥ"}
_VALID_BASES = _VOYELLES | _CONSONNES | _SEMI_VOYELLES

_VOWEL_PHONEMES = {
    "a", "e", "i", "o", "u", "y", "ø", "œ", "ɔ", "ə", "ɛ", "ɑ",
    "ɑ̃", "ɔ̃", "ɛ̃", "œ̃",
}

_LEGAL_ONSETS_2: set[tuple[str, str]] = {
    ("p", "l"), ("b", "l"), ("k", "l"), ("ɡ", "l"), ("f", "l"),
    ("p", "ʁ"), ("b", "ʁ"), ("t", "ʁ"), ("d", "ʁ"), ("k", "ʁ"),
    ("ɡ", "ʁ"), ("f", "ʁ"), ("v", "ʁ"),
    ("p", "j"), ("b", "j"), ("t", "j"), ("d", "j"), ("k", "j"),
    ("ɡ", "j"), ("f", "j"), ("v", "j"), ("m", "j"), ("n", "j"),
    ("p", "ɥ"), ("b", "ɥ"), ("t", "ɥ"), ("d", "ɥ"), ("k", "ɥ"),
    ("ɡ", "ɥ"), ("f", "ɥ"), ("v", "ɥ"), ("n", "ɥ"), ("l", "ɥ"),
    ("p", "w"), ("b", "w"), ("t", "w"), ("d", "w"), ("k", "w"),
    ("ɡ", "w"), ("f", "w"), ("v", "w"), ("m", "w"), ("n", "w"),
    ("s", "w"), ("ʃ", "w"), ("ʒ", "w"), ("l", "w"),
    ("s", "l"), ("s", "n"), ("s", "m"), ("s", "k"), ("s", "p"),
    ("s", "t"), ("ʃ", "l"), ("ʃ", "n"), ("ʃ", "m"),
}

_LEGAL_ONSETS_3: set[tuple[str, str, str]] = {
    ("s", "t", "ʁ"), ("s", "k", "ʁ"), ("s", "p", "l"),
    ("s", "p", "ʁ"), ("s", "k", "l"),
}


def iter_phonemes(ipa: str) -> list[str]:
    """Itere sur les phonemes d'une chaine IPA, en regroupant les combining marks."""
    if not ipa:
        return []
    phonemes: list[str] = []
    current = ""
    for ch in ipa:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            current += ch
        else:
            if current:
                phonemes.append(current)
            current = ch
    if current:
        phonemes.append(current)
    return phonemes


def _is_legal_onset(consonants: list[str]) -> bool:
    n = len(consonants)
    if n <= 1:
        return True
    if n == 2:
        return (consonants[0], consonants[1]) in _LEGAL_ONSETS_2
    if n == 3:
        return (consonants[0], consonants[1], consonants[2]) in _LEGAL_ONSETS_3
    return False


def syllabify_ipa(phone: str) -> tuple[int, str]:
    """Syllabifie une transcription IPA (Maximal Onset Principle).

    Retourne (nb_syllabes, syllabes_separees_par_points).
    """
    if not phone:
        return 0, ""
    phonemes = iter_phonemes(phone)
    if not phonemes:
        return 0, ""

    nuclei: list[int] = [i for i, ph in enumerate(phonemes) if ph in _VOWEL_PHONEMES]

    if not nuclei:
        return 1, phone
    if len(nuclei) == 1:
        return 1, phone

    syllables: list[list[str]] = []
    prev_end = 0

    for ni in range(len(nuclei) - 1):
        v1_idx = nuclei[ni]
        v2_idx = nuclei[ni + 1]
        cons_start = v1_idx + 1
        inter_consonants = phonemes[cons_start:v2_idx]

        if not inter_consonants:
            syllables.append(phonemes[prev_end:v1_idx + 1])
            prev_end = v2_idx
        else:
            split_at = cons_start
            for k in range(len(inter_consonants) + 1):
                if _is_legal_onset(list(inter_consonants[k:])):
                    split_at = cons_start + k
                    break
            syllables.append(phonemes[prev_end:split_at])
            prev_end = split_at

    syllables.append(phonemes[prev_end:])
    return len(syllables), ".".join("".join(s) for s in syllables)


# ── reverse_phone_ipa (de _utils.py) ─────────────────────────────────

_COMBINANTS = set("\u0303\u02D0\u02C8\u02CC\u0325\u0324\u0330\u032A\u033A\u033B\u033C\u030A")


def _tokenize_ipa(ipa: str) -> list[str]:
    """Decoupe une chaine IPA en segments phonetiques."""
    segments: list[str] = []
    i = 0
    while i < len(ipa):
        seg = ipa[i]
        i += 1
        while i < len(ipa) and ipa[i] in _COMBINANTS:
            seg += ipa[i]
            i += 1
        segments.append(seg)
    return segments


def reverse_phone_ipa(phone: str) -> str:
    """Inverse une transcription IPA au niveau des phonemes."""
    if not phone:
        return ""
    # Traiter les composants separes par espace independamment
    parts = phone.split(" ")
    reversed_parts: list[str] = []
    for part in parts:
        segments = _tokenize_ipa(part)
        segments.reverse()
        reversed_parts.append("".join(segments))
    reversed_parts.reverse()
    return " ".join(reversed_parts)


# ── _count_syllabes_expected (de audit_colonnes.py) ──────────────────

_BRACKET_RE = re.compile(r"\[[^\]]*\]")


def _strip_brackets(s: str) -> str:
    """Supprime les tokens entre crochets d'une chaine de syllabes."""
    return _BRACKET_RE.sub("", s)


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


# ── Mapping accents latins → IPA ─────────────────────────────────────

_ACCENT_MAP: dict[str, str] = {
    "é": "e",
    "è": "ɛ",
    "â": "a",
    "ê": "ɛ",
    "î": "i",
    "ô": "o",
    "ç": "s",
}

# Prefixe h aspire des entrees kaikki
_H_ASPIRE_RE = re.compile(r"\^\(\(h aspiré\)\)")

# Caracteres parasites a supprimer dans phone
_PARASITES_PHONE = set(":'\\\u2019-")


# ══════════════════════════════════════════════════════════════════════
# Passe 1 : Corrections phone
# ══════════════════════════════════════════════════════════════════════

def _collapse_spaces(s: str) -> str:
    """Collapse les espaces multiples et strip."""
    return re.sub(r" {2,}", " ", s).strip()


def _correct_phone_python(phone: str) -> str:
    """Applique les corrections Python sur une valeur phone.

    Corrections :
    - Accents latins (mapping)
    - Prefixe ^((h aspire))
    - Caracteres parasites (: ' \\ - ')
    """
    # Prefixe h aspire
    phone = _H_ASPIRE_RE.sub("", phone)

    # Accents latins
    result = []
    for ch in phone:
        if ch in _ACCENT_MAP:
            result.append(_ACCENT_MAP[ch])
        elif ch in _PARASITES_PHONE:
            continue  # supprimer
        else:
            result.append(ch)
    phone = "".join(result)

    # Nettoyer les espaces
    phone = _collapse_spaces(phone)
    return phone


def _correct_syllabes_python(syllabes: str) -> str:
    """Applique les corrections Python sur une valeur syllabes.

    Corrections :
    - Accents latins (mapping)
    - Caracteres parasites (: ' \\ - ')
    Note : les dots dans syllabes sont des separateurs valides, on ne les touche pas.
    """
    result = []
    for ch in syllabes:
        if ch in _ACCENT_MAP:
            result.append(_ACCENT_MAP[ch])
        elif ch in _PARASITES_PHONE:
            continue
        else:
            result.append(ch)
    return "".join(result)


def _needs_python_fix(phone: str) -> bool:
    """Determine si une valeur phone a besoin de corrections Python."""
    if _H_ASPIRE_RE.search(phone):
        return True
    for ch in phone:
        if ch in _ACCENT_MAP or ch in _PARASITES_PHONE:
            return True
    return False


def passe1_phone(conn: sqlite3.Connection, dry_run: bool) -> dict[int, str]:
    """Passe 1 : corrections sur la colonne phone.

    Retourne un dict id → nouveau phone pour les entrees modifiees.
    """
    log.info("=== Passe 1 : Corrections phone ===")
    stats: dict[str, int] = {
        "ascii_g": 0,
        "dots": 0,
        "undertie": 0,
        "accents_latins": 0,
        "h_aspire_prefix": 0,
        "parasites": 0,
    }

    # 1a. Corrections SQL directes (ASCII g → IPA ɡ)
    if not dry_run:
        # ASCII g (U+0067) → IPA ɡ (U+0261) dans phone
        cur = conn.execute(
            "SELECT COUNT(*) FROM formes WHERE phone LIKE '%g%' COLLATE BINARY"
        )
        n_ascii_g = cur.fetchone()[0]
        stats["ascii_g"] = n_ascii_g
        if n_ascii_g > 0:
            conn.execute(
                "UPDATE formes SET phone = REPLACE(phone, 'g', 'ɡ') "
                "WHERE phone LIKE '%g%' COLLATE BINARY"
            )
            # Meme chose pour syllabes
            conn.execute(
                "UPDATE formes SET syllabes = REPLACE(syllabes, 'g', 'ɡ') "
                "WHERE syllabes LIKE '%g%' COLLATE BINARY"
            )
            log.info("  ASCII g → IPA ɡ : %d formes (phone + syllabes)", n_ascii_g)
    else:
        cur = conn.execute(
            "SELECT COUNT(*) FROM formes WHERE phone LIKE '%g%' COLLATE BINARY"
        )
        stats["ascii_g"] = cur.fetchone()[0]
        log.info("  [dry-run] ASCII g → IPA ɡ : %d formes", stats["ascii_g"])

    # 1b. Dots → espaces dans phone
    if not dry_run:
        cur = conn.execute(
            "SELECT COUNT(*) FROM formes WHERE phone LIKE '%.%'"
        )
        n_dots = cur.fetchone()[0]
        stats["dots"] = n_dots
        if n_dots > 0:
            conn.execute(
                "UPDATE formes SET phone = REPLACE(phone, '.', ' ') "
                "WHERE phone LIKE '%.%'"
            )
            log.info("  Dots → espaces : %d formes", n_dots)
    else:
        cur = conn.execute(
            "SELECT COUNT(*) FROM formes WHERE phone LIKE '%.%'"
        )
        stats["dots"] = cur.fetchone()[0]
        log.info("  [dry-run] Dots → espaces : %d formes", stats["dots"])

    # 1c. Undertie ‿ → supprimer dans phone et syllabes
    if not dry_run:
        cur = conn.execute(
            "SELECT COUNT(*) FROM formes WHERE phone LIKE '%‿%'"
        )
        n_undertie = cur.fetchone()[0]
        stats["undertie"] = n_undertie
        if n_undertie > 0:
            conn.execute(
                "UPDATE formes SET phone = REPLACE(phone, '‿', '') "
                "WHERE phone LIKE '%‿%'"
            )
            conn.execute(
                "UPDATE formes SET syllabes = REPLACE(syllabes, '‿', '') "
                "WHERE syllabes LIKE '%‿%'"
            )
            log.info("  Undertie supprime : %d formes (phone + syllabes)", n_undertie)
    else:
        cur = conn.execute(
            "SELECT COUNT(*) FROM formes WHERE phone LIKE '%‿%'"
        )
        stats["undertie"] = cur.fetchone()[0]
        log.info("  [dry-run] Undertie : %d formes", stats["undertie"])

    # 1d. Collapse des espaces multiples (apres dots→espaces)
    if not dry_run:
        cur = conn.execute(
            "SELECT COUNT(*) FROM formes WHERE phone LIKE '%  %'"
        )
        n_multi_spaces = cur.fetchone()[0]
        if n_multi_spaces > 0:
            # SQLite ne supporte pas le REGEXP natif, on fait en Python
            cur = conn.execute(
                "SELECT id, phone FROM formes WHERE phone LIKE '%  %'"
            )
            batch: list[tuple[str, int]] = []
            for row in cur:
                fid, phone = row
                cleaned = _collapse_spaces(phone)
                if cleaned != phone:
                    batch.append((cleaned, fid))
            if batch:
                conn.executemany(
                    "UPDATE formes SET phone = ? WHERE id = ?", batch
                )
            log.info("  Espaces multiples collapse : %d formes", len(batch))

    # 1e. Corrections Python (accents latins, prefixe h aspire, parasites)
    log.info("  Scan Python pour accents latins / prefixe h aspire / parasites...")
    cur = conn.execute("SELECT id, phone, syllabes FROM formes WHERE phone IS NOT NULL AND phone != ''")
    python_updates: list[tuple[str, str, int]] = []  # (new_phone, new_syllabes, id)
    n_python = 0

    for row in cur:
        fid, phone, syllabes = row
        if not _needs_python_fix(phone):
            continue
        n_python += 1

        # Compter les types
        if _H_ASPIRE_RE.search(phone):
            stats["h_aspire_prefix"] += 1
        for ch in phone:
            if ch in _ACCENT_MAP:
                stats["accents_latins"] += 1
                break
        for ch in phone:
            if ch in _PARASITES_PHONE:
                stats["parasites"] += 1
                break

        new_phone = _correct_phone_python(phone)
        new_syllabes = _correct_syllabes_python(syllabes) if syllabes else syllabes
        python_updates.append((new_phone, new_syllabes or "", fid))

    log.info("  Corrections Python : %d formes", n_python)
    for k, v in stats.items():
        if k not in ("ascii_g", "dots", "undertie") and v > 0:
            log.info("    %s : %d", k, v)

    if python_updates and not dry_run:
        conn.executemany(
            "UPDATE formes SET phone = ?, syllabes = ? WHERE id = ?",
            python_updates,
        )

    # 1f. Recalcul phone_reversed pour toutes les lignes modifiees
    log.info("  Recalcul phone_reversed...")
    # On recalcule pour TOUT ce qui a pu etre modifie (plus simple et sur)
    # On repere les lignes dont phone contient maintenant un ɡ (ex-g),
    # ou qui avaient un dot, ou sont dans python_updates.
    # Plus simple : recalculer tout phone_reversed (c'est rapide).
    if not dry_run:
        cur = conn.execute(
            "SELECT id, phone FROM formes WHERE phone IS NOT NULL AND phone != ''"
        )
        rev_batch: list[tuple[str, int]] = []
        n_rev = 0
        for row in cur:
            fid, phone = row
            new_rev = reverse_phone_ipa(phone)
            rev_batch.append((new_rev, fid))
            n_rev += 1
            if len(rev_batch) >= 10_000:
                conn.executemany(
                    "UPDATE formes SET phone_reversed = ? WHERE id = ?",
                    rev_batch,
                )
                rev_batch.clear()
        if rev_batch:
            conn.executemany(
                "UPDATE formes SET phone_reversed = ? WHERE id = ?",
                rev_batch,
            )
        log.info("  phone_reversed recalcule pour %d formes", n_rev)

    conn.commit()

    # Construire le set d'IDs modifies pour la passe 2
    modified_ids: dict[int, str] = {}
    if not dry_run:
        # Re-lire les phones corriges pour les IDs qui ont ete modifies
        # On ne peut pas facilement tracker tous les IDs modifies par SQL,
        # donc on collecte les IDs des corrections Python + on note
        # qu'il y a eu des corrections SQL globales.
        for new_phone, _, fid in python_updates:
            modified_ids[fid] = new_phone

    total_corrections = (
        stats["ascii_g"] + stats["dots"] + stats["undertie"]
        + n_python
    )
    log.info("  Passe 1 terminee : %d corrections totales", total_corrections)
    for k, v in sorted(stats.items()):
        log.info("    %-20s : %d", k, v)

    return modified_ids


# ══════════════════════════════════════════════════════════════════════
# Passe 2 : Corrections syllabes
# ══════════════════════════════════════════════════════════════════════

def passe2_syllabes(conn: sqlite3.Connection, dry_run: bool) -> None:
    """Passe 2 : resyllabification et recalcul nb_syllabes."""
    log.info("=== Passe 2 : Corrections syllabes ===")

    # 2a. Resyllabifier toutes les entrees qui ont un phone valide
    # mais dont syllabes ne correspond pas (ou a ete modifie en passe 1).
    # Strategie simple : resyllabifier tout phone mono-composant et
    # multi-composant pour coherence.
    log.info("  Resyllabification depuis phone corrige...")

    cur = conn.execute(
        "SELECT id, phone, syllabes, nb_syllabes FROM formes "
        "WHERE phone IS NOT NULL AND phone != ''"
    )

    resyll_batch: list[tuple[str, int, int]] = []  # (syllabes, nb_syll, id)
    n_resyll = 0

    for row in cur:
        fid, phone, old_syllabes, old_nb = row

        # Syllabifier chaque composant (separes par espace) separement
        components = phone.split(" ")
        syll_parts: list[str] = []
        total_syll = 0
        for comp in components:
            if not comp:
                continue
            nb, syll = syllabify_ipa(comp)
            syll_parts.append(syll)
            total_syll += nb

        new_syllabes = " ".join(syll_parts)
        new_nb = total_syll

        # Ne mettre a jour que si different (pour eviter les ecritures inutiles)
        if new_syllabes != (old_syllabes or "") or new_nb != old_nb:
            resyll_batch.append((new_syllabes, new_nb, fid))
            n_resyll += 1

            if len(resyll_batch) >= 10_000 and not dry_run:
                conn.executemany(
                    "UPDATE formes SET syllabes = ?, nb_syllabes = ? WHERE id = ?",
                    resyll_batch,
                )
                resyll_batch.clear()

    if resyll_batch and not dry_run:
        conn.executemany(
            "UPDATE formes SET syllabes = ?, nb_syllabes = ? WHERE id = ?",
            resyll_batch,
        )

    log.info("  Resyllabifie : %d formes", n_resyll)

    # 2b. Recalculer nb_syllabes pour les entrees avec brackets dans syllabes
    # (les brackets ne sont pas geres par syllabify_ipa, donc on utilise
    # la logique de _count_syllabes_expected)
    log.info("  Recalcul nb_syllabes pour entrees avec brackets...")
    cur = conn.execute(
        "SELECT id, syllabes, nb_syllabes FROM formes "
        "WHERE syllabes LIKE '%[%'"
    )
    bracket_batch: list[tuple[int, int]] = []  # (nb_syll, id)
    n_bracket = 0

    for row in cur:
        fid, syllabes, old_nb = row
        expected = _count_syllabes_expected(syllabes)
        if expected != old_nb:
            bracket_batch.append((expected, fid))
            n_bracket += 1

    if bracket_batch and not dry_run:
        conn.executemany(
            "UPDATE formes SET nb_syllabes = ? WHERE id = ?",
            bracket_batch,
        )

    log.info("  nb_syllabes recalcule (brackets) : %d formes", n_bracket)

    conn.commit()
    log.info("  Passe 2 terminee")


# ══════════════════════════════════════════════════════════════════════
# Passe 3 : Corrections orthocode
# ══════════════════════════════════════════════════════════════════════

def passe3_orthocode(
    conn: sqlite3.Connection,
    dry_run: bool,
    skip_regen: bool,
) -> None:
    """Passe 3 : corrections orthocode."""
    log.info("=== Passe 3 : Corrections orthocode ===")

    # 3a. Apostrophe typographique → ASCII
    cur = conn.execute(
        "SELECT COUNT(*) FROM formes WHERE orthocode LIKE '%\u2019%'"
    )
    n_apo = cur.fetchone()[0]
    log.info("  Apostrophe typographique \\u2019 → \\u0027 : %d formes", n_apo)

    if n_apo > 0 and not dry_run:
        conn.execute(
            "UPDATE formes SET orthocode = REPLACE(orthocode, '\u2019', '''') "
            "WHERE orthocode LIKE '%\u2019%'"
        )

    # 3b. Orthocodes vides avec phone → regenerer
    cur = conn.execute(
        "SELECT COUNT(*) FROM formes "
        "WHERE phone IS NOT NULL AND phone != '' "
        "AND (orthocode IS NULL OR orthocode = '')"
    )
    n_vides = cur.fetchone()[0]
    log.info("  Orthocodes vides avec phone : %d formes", n_vides)

    if n_vides > 0 and not skip_regen and not dry_run:
        try:
            from lectura_aligneur import LecturaSyllabeur

            class NullPhonemizer:
                def phonemize(self, word: str) -> str:
                    return ""

            syllabeur = LecturaSyllabeur(phonemizer=NullPhonemizer())
            log.info("  Syllabeur charge, regeneration des orthocodes...")

            cur = conn.execute(
                "SELECT id, ortho, phone FROM formes "
                "WHERE phone IS NOT NULL AND phone != '' "
                "AND (orthocode IS NULL OR orthocode = '')"
            )
            regen_batch: list[tuple[str, str, int, int]] = []
            n_regen = 0
            n_err = 0

            # Import du compactage
            _DOUBLE_CARRE_RE = re.compile(r"([a-zA-ZÀ-ÿ])\1²")

            for row in cur:
                fid, ortho, phone = row
                try:
                    result = syllabeur.analyze(ortho, phone=phone)
                    if result and result.syllabes:
                        orthocode = ".".join(s.ortho for s in result.syllabes)
                        orthocode = _DOUBLE_CARRE_RE.sub(r"\1²", orthocode)
                        syllabes = ".".join(s.phone for s in result.syllabes)
                        nb = len(result.syllabes)
                        regen_batch.append((orthocode, syllabes, nb, fid))
                        n_regen += 1
                    else:
                        n_err += 1
                except Exception:
                    n_err += 1

                if len(regen_batch) >= 5_000:
                    conn.executemany(
                        "UPDATE formes SET orthocode = ?, syllabes = ?, nb_syllabes = ? "
                        "WHERE id = ?",
                        regen_batch,
                    )
                    regen_batch.clear()

            if regen_batch:
                conn.executemany(
                    "UPDATE formes SET orthocode = ?, syllabes = ?, nb_syllabes = ? "
                    "WHERE id = ?",
                    regen_batch,
                )

            log.info("  Orthocodes regeneres : %d ok, %d erreurs", n_regen, n_err)

        except ImportError:
            log.warning(
                "  lectura_aligneur non disponible — "
                "regeneration des orthocodes vides ignoree. "
                "Relancer enrichir_orthocodes.py manuellement."
            )
    elif skip_regen and n_vides > 0:
        log.info("  --skip-orthocode-regen : regeneration ignoree")

    conn.commit()
    log.info("  Passe 3 terminee")


# ══════════════════════════════════════════════════════════════════════
# Rapport final
# ══════════════════════════════════════════════════════════════════════

def rapport_final(conn: sqlite3.Connection) -> None:
    """Affiche un rapport de l'etat de la BDD apres corrections."""
    log.info("=== Rapport final ===")

    cur = conn.execute("SELECT COUNT(*) FROM formes")
    total = cur.fetchone()[0]
    log.info("  Total formes : %d", total)

    for col in ("phone", "syllabes", "orthocode"):
        cur = conn.execute(
            f"SELECT COUNT(*) FROM formes WHERE {col} IS NOT NULL AND {col} != ''"  # noqa: S608
        )
        count = cur.fetchone()[0]
        log.info("  %-15s non vides : %d", col, count)

    # Verif rapide : combien d'ASCII g restent dans phone ?
    cur = conn.execute(
        "SELECT COUNT(*) FROM formes WHERE phone LIKE '%g%' COLLATE BINARY"
    )
    n_g = cur.fetchone()[0]
    if n_g > 0:
        log.warning("  ATTENTION : %d formes contiennent encore un ASCII g dans phone", n_g)
    else:
        log.info("  ASCII g dans phone : 0 (OK)")

    # Dots dans phone
    cur = conn.execute("SELECT COUNT(*) FROM formes WHERE phone LIKE '%.%'")
    n_dots = cur.fetchone()[0]
    if n_dots > 0:
        log.warning("  ATTENTION : %d formes contiennent encore un dot dans phone", n_dots)
    else:
        log.info("  Dots dans phone : 0 (OK)")

    # Undertie
    cur = conn.execute("SELECT COUNT(*) FROM formes WHERE phone LIKE '%‿%'")
    n_undertie = cur.fetchone()[0]
    if n_undertie > 0:
        log.warning("  ATTENTION : %d formes contiennent encore un undertie dans phone", n_undertie)
    else:
        log.info("  Undertie dans phone : 0 (OK)")

    # Apostrophe typographique dans orthocode
    cur = conn.execute(
        "SELECT COUNT(*) FROM formes WHERE orthocode LIKE '%\u2019%'"
    )
    n_apo = cur.fetchone()[0]
    if n_apo > 0:
        log.warning(
            "  ATTENTION : %d formes contiennent encore une apostrophe typographique dans orthocode",
            n_apo,
        )
    else:
        log.info("  Apostrophe typographique dans orthocode : 0 (OK)")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correction des anomalies phone / syllabes / orthocode",
    )
    _default_db = str(
        next((p for p in _DB_CANDIDATES if p.exists()), _DB_CANDIDATES[0])
    )
    parser.add_argument(
        "--db", default=_default_db,
        help="Chemin vers la BDD (defaut: auto-detection)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Afficher les corrections sans les appliquer",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Ne pas creer de copie .bak",
    )
    parser.add_argument(
        "--skip-orthocode-regen", action="store_true",
        help="Ne pas tenter de regenerer les orthocodes vides",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Logs debug",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = Path(args.db)
    if not db_path.exists():
        log.error("BDD introuvable : %s", db_path)
        sys.exit(1)

    log.info("BDD : %s", db_path)

    # Backup
    if not args.no_backup and not args.dry_run:
        bak = db_path.with_suffix(".db.bak")
        shutil.copy2(db_path, bak)
        log.info("Backup : %s", bak)

    # Connexion
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-64000")

    try:
        # Passe 1 : phone
        passe1_phone(conn, dry_run=args.dry_run)

        # Passe 2 : syllabes
        passe2_syllabes(conn, dry_run=args.dry_run)

        # Passe 3 : orthocode
        passe3_orthocode(
            conn,
            dry_run=args.dry_run,
            skip_regen=args.skip_orthocode_regen,
        )

        # Rapport
        if not args.dry_run:
            rapport_final(conn)

    finally:
        conn.close()

    log.info("=== Termine ===")

    # Relancer l'audit si pas dry-run
    if not args.dry_run:
        audit_script = _SCRIPT_DIR / "audit_colonnes.py"
        if audit_script.exists():
            log.info("Relance de l'audit pour verification...")
            import subprocess
            subprocess.run(
                [sys.executable, str(audit_script), "--db", str(db_path)],
                check=False,
            )


if __name__ == "__main__":
    main()
