#!/usr/bin/env python3
"""Phonetise les noms propres sans IPA via le modele ONNX V2.

Usage :
    python phonetiser_noms_propres.py [--db chemin.db] [--dry-run]

Pipeline :
    1. Lit les noms propres sans phone dans la BDD
    2. Filtre les NP non-phonetisables (chiffres purs, codes, etc.)
    3. Passe chaque lemme par ONNX V2 (mot par mot)
    4. Met a jour noms_propres.phone dans la BDD
    5. Ajoute les NP dans la table formes (si absent)
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_LEXIQUE_DIR = _SCRIPT_DIR.parent
_MODULES_DIR = _LEXIQUE_DIR.parent

_PRIVATE_DIR = _MODULES_DIR / "lectura-modules-private" / "Phonemiseur"
_ONNX_MODEL = _PRIVATE_DIR / "modeles" / "unifie_v2_int8.onnx"
_ONNX_VOCAB = _PRIVATE_DIR / "modeles" / "unifie_v2_vocab.json"

_G2P_DIR = _MODULES_DIR / "Phonemiseur" / "src" / "lectura_phonemiseur"
_CORRECTIONS_PATH = _G2P_DIR / "modeles" / "g2p_corrections_unifie.json"
_HOMOGRAPHES_PATH = _G2P_DIR / "modeles" / "homographes.json"

_DEFAULT_DB = _LEXIQUE_DIR / "lexique_lectura.db"

# ── Filtres ──────────────────────────────────────────────────────────────

# Regex pour detecter les NP non-phonetisables
_RE_DIGITS_ONLY = re.compile(r"^[\d\s\-\./:,]+$")
_RE_URL = re.compile(r"https?://|www\.")
_RE_CODE = re.compile(r"^[A-Z]{2,5}-\d")

# Caracteres min/max pour un lemme phonetisable
_MIN_ALPHA = 2


def _est_phonetisable(lemme: str) -> bool:
    """Verifie si un lemme peut raisonnablement etre phonetise."""
    if not lemme or len(lemme) < 2:
        return False
    if _RE_DIGITS_ONLY.match(lemme):
        return False
    if _RE_URL.search(lemme):
        return False
    if _RE_CODE.match(lemme):
        return False
    # Doit contenir au moins 2 lettres
    n_alpha = sum(1 for c in lemme if c.isalpha())
    return n_alpha >= _MIN_ALPHA


def _tokeniser_lemme(lemme: str) -> list[str]:
    """Decoupe un lemme NP en tokens pour le G2P.

    Ex: "clermont-ferrand" → ["clermont", "ferrand"]
        "10 downing street" → ["downing", "street"]  (chiffres filtres)
        "d'artagnan" → ["d'artagnan"]  (apostrophe gardee, le G2P gere)
    """
    # Split sur espaces et tirets (mais garder apostrophes)
    parts = re.split(r"[\s\-]+", lemme)
    tokens = []
    for p in parts:
        p = p.strip("'\"()[]{}.,;:!?")
        if not p:
            continue
        # Filtrer les tokens purement numeriques
        if p.isdigit():
            continue
        if p.isalpha() or "'" in p or "\u2019" in p:
            tokens.append(p)
    return tokens


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phonetise les noms propres sans IPA via ONNX V2."
    )
    parser.add_argument(
        "--db", type=Path, default=_DEFAULT_DB,
        help=f"Base SQLite (defaut: {_DEFAULT_DB.name})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche les stats sans modifier la BDD.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limite de NP a traiter (0 = tous).",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERREUR : BDD introuvable : {args.db}", file=sys.stderr)
        sys.exit(1)

    # Charger engine ONNX V2
    print("Chargement du modele ONNX V2...", file=sys.stderr)
    sys.path.insert(0, str(_PRIVATE_DIR))
    from inference_onnx_v2 import OnnxInferenceEngineV2
    from lectura_phonemiseur.posttraitement import (
        charger_corrections, charger_homographes, corriger_g2p,
    )

    engine = OnnxInferenceEngineV2(
        onnx_path=_ONNX_MODEL,
        vocab_path=_ONNX_VOCAB,
    )

    if _CORRECTIONS_PATH.exists():
        charger_corrections(_CORRECTIONS_PATH)
    if _HOMOGRAPHES_PATH.exists():
        charger_homographes(_HOMOGRAPHES_PATH)

    # Lire les NP sans phone
    conn = sqlite3.connect(str(args.db))
    cur = conn.cursor()

    query = "SELECT id, lemme FROM noms_propres WHERE phone IS NULL OR phone = ''"
    if args.limit:
        query += f" LIMIT {args.limit}"

    rows = cur.execute(query).fetchall()
    print(f"{len(rows):,} noms propres a phonetiser", file=sys.stderr)

    # Collecter les lemmes deja dans formes (pour eviter doublons)
    existing_formes = set()
    for r in cur.execute(
        "SELECT DISTINCT lower(ortho) FROM formes WHERE cgram = 'NOM PROPRE'"
    ).fetchall():
        existing_formes.add(r[0])
    print(f"{len(existing_formes):,} NP deja dans formes", file=sys.stderr)

    # Phonetiser
    t0 = time.time()
    n_ok = 0
    n_skip = 0
    n_err = 0

    updates_np: list[tuple[str, int]] = []      # (phone, id) pour UPDATE noms_propres
    inserts_formes: list[tuple[str, str, str]] = []  # (ortho, phone, cgram) pour INSERT formes

    for i, (np_id, lemme) in enumerate(rows):
        if not _est_phonetisable(lemme):
            n_skip += 1
            continue

        tokens = _tokeniser_lemme(lemme)
        if not tokens:
            n_skip += 1
            continue

        try:
            analyse = engine.analyser(tokens)
        except Exception:
            n_err += 1
            continue

        ipa_list = analyse["g2p"]
        pos_list = analyse.get("pos", [])

        phones = []
        for j, tok in enumerate(tokens):
            ipa = ipa_list[j] if j < len(ipa_list) else ""
            pos = pos_list[j] if j < len(pos_list) else None
            ipa = corriger_g2p(tok, ipa, pos)
            if ipa:
                phones.append(ipa)

        if not phones:
            n_err += 1
            continue

        phone_str = "".join(phones)
        n_ok += 1

        if not args.dry_run:
            updates_np.append((phone_str, np_id))

            # Ajouter dans formes si absent
            lemme_lower = lemme.lower()
            if lemme_lower not in existing_formes:
                inserts_formes.append((lemme_lower, phone_str, "NOM PROPRE"))
                existing_formes.add(lemme_lower)

        # Flush par batch
        if len(updates_np) >= 5000:
            cur.executemany(
                "UPDATE noms_propres SET phone = ? WHERE id = ?", updates_np
            )
            if inserts_formes:
                cur.executemany(
                    "INSERT INTO formes (ortho, lemme, cgram, phone) VALUES (?, ?, ?, ?)",
                    [(o, o, c, p) for o, p, c in inserts_formes],
                )
            conn.commit()
            updates_np.clear()
            inserts_formes.clear()

        if (i + 1) % 10_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  {i + 1:>8,}/{len(rows):,}  "
                f"ok={n_ok:,}  skip={n_skip:,}  err={n_err:,}  "
                f"{rate:.0f}/s",
                file=sys.stderr,
            )

    # Flush final
    if updates_np:
        cur.executemany(
            "UPDATE noms_propres SET phone = ? WHERE id = ?", updates_np
        )
    if inserts_formes:
        cur.executemany(
            "INSERT INTO formes (ortho, lemme, cgram, phone) VALUES (?, ?, ?, ?)",
            [(o, o, c, p) for o, p, c in inserts_formes],
        )
    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    print(
        f"\nTermine en {elapsed:.0f}s :\n"
        f"  Phonetises : {n_ok:,}\n"
        f"  Ignores    : {n_skip:,} (non-phonetisables)\n"
        f"  Erreurs    : {n_err:,}\n"
        f"  {'(dry-run, BDD non modifiee)' if args.dry_run else 'BDD mise a jour'}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
