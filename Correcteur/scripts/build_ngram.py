#!/usr/bin/env python3
"""Construction de modeles n-gram depuis un corpus Wikipedia FR.

Usage :
    python scripts/build_ngram.py \\
        --corpus /data/work/projets/lectura/data/wikipedia/frwiki_phrases.txt \\
        --lexique /data/work/projets/lectura/workspace/Lexique/lexique_lectura.csv \\
        --output ngram_3.db --order 3

    python scripts/build_ngram.py \\
        --corpus frwiki_phrases.txt \\
        --output ngram_4.db --order 4

Le corpus est un fichier texte avec une phrase par ligne (lowercase,
ponctuation supprimee sauf apostrophes).

Etapes :
    1. Compter les n-grams (streaming, memoire constante via batches)
    2. Calculer les probabilites avec lissage Kneser-Ney modifie
    3. Ecrire dans une base SQLite
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

# Tokens speciaux
BOS = "<s>"
EOS = "</s>"

# Parametres Kneser-Ney modifie
D1 = 0.75  # discount pour count=1
D2 = 1.00  # discount pour count=2
D3 = 1.50  # discount pour count>=3


def lire_vocabulaire_lexique(lexique_path: str) -> set[str] | None:
    """Lit le vocabulaire depuis un lexique CSV ou SQLite.

    Si le fichier est CSV : colonne 'ortho' ou premiere colonne.
    Retourne None si le fichier n'existe pas.
    """
    p = Path(lexique_path)
    if not p.exists():
        return None

    vocab: set[str] = set()

    if p.suffix == ".csv":
        import csv
        with open(p, encoding="utf-8", newline="") as f:
            # Detecter le delimiteur (virgule ou tabulation)
            sample = f.read(4096)
            f.seek(0)
            delim = "\t" if "\t" in sample else ","
            reader = csv.DictReader(f, delimiter=delim)
            if reader.fieldnames and "ortho" in reader.fieldnames:
                for row in reader:
                    ortho = row.get("ortho", "").strip().lower()
                    if ortho:
                        vocab.add(ortho)
            else:
                f.seek(0)
                for line in f:
                    parts = line.strip().split(delim)
                    if parts:
                        vocab.add(parts[0].lower())
    elif p.suffix == ".db":
        conn = sqlite3.connect(p)
        try:
            for row in conn.execute("SELECT DISTINCT ortho FROM lexique"):
                vocab.add(row[0].lower())
        except sqlite3.OperationalError:
            pass
        conn.close()

    return vocab if vocab else None


def compter_ngrams(
    corpus_path: str,
    order: int,
    vocab: set[str] | None = None,
    max_phrases: int = 0,
    min_count: int = 2,
) -> dict[int, Counter]:
    """Compte les n-grams du corpus en streaming.

    Args:
        corpus_path: Chemin vers le fichier de phrases (1/ligne).
        order: Ordre max (3 ou 4).
        vocab: Vocabulaire ferme (si None, tous les mots sont gardes).
        max_phrases: Limite de phrases (0 = illimite).
        min_count: Seuil minimum de frequence (les n-grams < min_count sont
            elagages apres comptage pour limiter la memoire).

    Returns:
        Dict {n: Counter(tuple -> count)} pour n=1..order.
    """
    counts: dict[int, Counter] = {n: Counter() for n in range(1, order + 1)}
    n_phrases = 0
    t0 = time.time()

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if not phrase:
                continue

            mots = phrase.lower().split()

            # Filtrer vocabulaire
            if vocab is not None:
                mots = [m if m in vocab else "<unk>" for m in mots]

            # Filtrer phrases trop courtes/longues
            if len(mots) < 3 or len(mots) > 50:
                continue

            # Trop de <unk> -> phrase pas utile
            if vocab is not None:
                n_unk = sum(1 for m in mots if m == "<unk>")
                if n_unk > len(mots) * 0.3:
                    continue

            tokens = [BOS] + mots + [EOS]

            for n in range(1, order + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i + n])
                    counts[n][ngram] += 1

            n_phrases += 1
            if n_phrases % 1_000_000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  {n_phrases / 1e6:.0f}M phrases, "
                    f"{elapsed:.0f}s, "
                    f"1-grams: {len(counts[1]):,}",
                    file=sys.stderr,
                )

            if max_phrases and n_phrases >= max_phrases:
                break

    print(
        f"Total : {n_phrases:,} phrases en {time.time() - t0:.0f}s",
        file=sys.stderr,
    )
    for n in range(1, order + 1):
        print(f"  {n}-grams uniques : {len(counts[n]):,}", file=sys.stderr)

    # Elaguer les n-grams rares (n>=2) pour limiter la memoire
    if min_count > 1:
        for n in range(2, order + 1):
            before = len(counts[n])
            counts[n] = Counter(
                {k: v for k, v in counts[n].items() if v >= min_count}
            )
            after = len(counts[n])
            print(
                f"  Elagage {n}-grams: {before:,} -> {after:,} "
                f"(supprime {before - after:,} hapax)",
                file=sys.stderr,
            )

    return counts


def discount(count: int) -> float:
    """Discount Kneser-Ney modifie selon le comptage."""
    if count == 1:
        return D1
    if count == 2:
        return D2
    return D3


def calculer_probabilites(
    counts: dict[int, Counter],
    order: int,
) -> dict[int, dict[tuple, tuple[float, float]]]:
    """Calcule les log-probabilites et poids de backoff.

    Args:
        counts: Dict {n: Counter(ngram -> count)}.
        order: Ordre max.

    Returns:
        Dict {n: {ngram: (log10_prob, log10_backoff)}}.
    """
    probs: dict[int, dict[tuple, tuple[float, float]]] = {}

    # --- Unigrammes ---
    total_unigrams = sum(counts[1].values())
    probs[1] = {}
    for ngram, count in counts[1].items():
        p = count / total_unigrams
        probs[1][ngram] = (math.log10(p), 0.0)

    # --- Bigrammes et plus ---
    for n in range(2, order + 1):
        probs[n] = {}

        # Grouper les n-grams par prefixe (contexte)
        prefix_counts: dict[tuple, int] = Counter()
        prefix_types: dict[tuple, int] = Counter()
        for ngram, count in counts[n].items():
            prefix = ngram[:-1]
            prefix_counts[prefix] += count
            prefix_types[prefix] += 1

        for ngram, count in counts[n].items():
            prefix = ngram[:-1]
            total = prefix_counts[prefix]

            # Probabilite avec discount
            d = discount(count)
            p = max(count - d, 0.0) / total

            # Masse de probabilite redistribuee
            n_types = prefix_types[prefix]
            lambda_prefix = (
                D1 * sum(1 for ng, c in counts[n].items()
                         if ng[:-1] == prefix and c == 1)
                + D2 * sum(1 for ng, c in counts[n].items()
                           if ng[:-1] == prefix and c == 2)
                + D3 * sum(1 for ng, c in counts[n].items()
                           if ng[:-1] == prefix and c >= 3)
            ) / total

            log_p = math.log10(p + 1e-20) if p > 0 else -10.0
            log_bo = math.log10(lambda_prefix + 1e-20) if lambda_prefix > 0 else 0.0

            probs[n][ngram] = (log_p, log_bo)

    return probs


def calculer_probabilites_fast(
    counts: dict[int, Counter],
    order: int,
) -> dict[int, dict[tuple, tuple[float, float]]]:
    """Version optimisee du calcul de probabilites.

    Evite le scan O(N) par prefixe en pre-calculant les statistiques.
    """
    probs: dict[int, dict[tuple, tuple[float, float]]] = {}

    # --- Unigrammes ---
    total_unigrams = sum(counts[1].values())
    probs[1] = {}
    for ngram, count in counts[1].items():
        p = count / total_unigrams
        probs[1][ngram] = (math.log10(p), 0.0)

    # --- Bigrammes et plus ---
    for n in range(2, order + 1):
        probs[n] = {}

        # Pre-calculer les stats par prefixe en un seul passage
        prefix_total: dict[tuple, int] = Counter()
        prefix_d1: dict[tuple, int] = Counter()
        prefix_d2: dict[tuple, int] = Counter()
        prefix_d3: dict[tuple, int] = Counter()

        for ngram, count in counts[n].items():
            prefix = ngram[:-1]
            prefix_total[prefix] += count
            if count == 1:
                prefix_d1[prefix] += 1
            elif count == 2:
                prefix_d2[prefix] += 1
            else:
                prefix_d3[prefix] += 1

        for ngram, count in counts[n].items():
            prefix = ngram[:-1]
            total = prefix_total[prefix]

            d = discount(count)
            p = max(count - d, 0.0) / total

            lambda_prefix = (
                D1 * prefix_d1.get(prefix, 0)
                + D2 * prefix_d2.get(prefix, 0)
                + D3 * prefix_d3.get(prefix, 0)
            ) / total

            log_p = math.log10(p + 1e-20) if p > 0 else -10.0
            log_bo = math.log10(lambda_prefix + 1e-20) if lambda_prefix > 0 else 0.0

            probs[n][ngram] = (log_p, log_bo)

    return probs


def ecrire_sqlite(
    probs: dict[int, dict[tuple, tuple[float, float]]],
    counts: dict[int, Counter],
    order: int,
    output_path: str,
) -> None:
    """Ecrit les n-grams dans une base SQLite.

    Args:
        probs: Dict {n: {ngram: (log_prob, log_backoff)}}.
        counts: Dict {n: Counter(ngram -> count)}.
        order: Ordre max.
        output_path: Chemin du fichier SQLite de sortie.
    """
    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=OFF")

    # Unigrammes
    conn.execute("""
        CREATE TABLE IF NOT EXISTS unigrams (
            word TEXT PRIMARY KEY,
            count INTEGER,
            prob REAL
        )
    """)

    # Bigrammes
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bigrams (
            w1 TEXT, w2 TEXT,
            count INTEGER,
            prob REAL,
            backoff REAL,
            PRIMARY KEY (w1, w2)
        )
    """)

    # Trigrammes
    if order >= 3:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trigrams (
                w1 TEXT, w2 TEXT, w3 TEXT,
                count INTEGER,
                prob REAL,
                backoff REAL,
                PRIMARY KEY (w1, w2, w3)
            )
        """)

    # 4-grammes
    if order >= 4:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fourgrams (
                w1 TEXT, w2 TEXT, w3 TEXT, w4 TEXT,
                count INTEGER,
                prob REAL,
                backoff REAL,
                PRIMARY KEY (w1, w2, w3, w4)
            )
        """)

    # Inserer unigrammes
    print("Ecriture unigrammes...", file=sys.stderr)
    batch = []
    for ngram, (log_p, _) in probs[1].items():
        w = ngram[0]
        c = counts[1][ngram]
        batch.append((w, c, log_p))
    conn.executemany(
        "INSERT OR REPLACE INTO unigrams (word, count, prob) VALUES (?, ?, ?)",
        batch,
    )

    # Inserer bigrammes
    if 2 in probs:
        print("Ecriture bigrammes...", file=sys.stderr)
        batch = []
        for ngram, (log_p, log_bo) in probs[2].items():
            c = counts[2][ngram]
            batch.append((ngram[0], ngram[1], c, log_p, log_bo))
        conn.executemany(
            "INSERT OR REPLACE INTO bigrams (w1, w2, count, prob, backoff)"
            " VALUES (?, ?, ?, ?, ?)",
            batch,
        )

    # Inserer trigrammes
    if order >= 3 and 3 in probs:
        print("Ecriture trigrammes...", file=sys.stderr)
        batch = []
        for ngram, (log_p, log_bo) in probs[3].items():
            c = counts[3][ngram]
            batch.append((ngram[0], ngram[1], ngram[2], c, log_p, log_bo))
        conn.executemany(
            "INSERT OR REPLACE INTO trigrams (w1, w2, w3, count, prob, backoff)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )

    # Inserer 4-grammes
    if order >= 4 and 4 in probs:
        print("Ecriture 4-grammes...", file=sys.stderr)
        batch = []
        for ngram, (log_p, log_bo) in probs[4].items():
            c = counts[4][ngram]
            batch.append((
                ngram[0], ngram[1], ngram[2], ngram[3],
                c, log_p, log_bo,
            ))
        conn.executemany(
            "INSERT OR REPLACE INTO fourgrams"
            " (w1, w2, w3, w4, count, prob, backoff)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch,
        )

    conn.commit()

    # Index pour les lookups de backoff
    print("Creation des index...", file=sys.stderr)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bi_w1 ON bigrams(w1)")
    if order >= 3:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tri_w1w2 ON trigrams(w1, w2)"
        )
    if order >= 4:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_four_w1w2w3"
            " ON fourgrams(w1, w2, w3)"
        )
    conn.commit()
    conn.close()

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Base ecrite : {output_path} ({size_mb:.1f} Mo)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construit un modele n-gram depuis un corpus de phrases."
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Fichier de phrases (1 par ligne, lowercase).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Chemin du fichier SQLite de sortie (ex: ngram_3.db).",
    )
    parser.add_argument(
        "--order", type=int, default=3, choices=[2, 3, 4],
        help="Ordre max du modele (defaut: 3).",
    )
    parser.add_argument(
        "--lexique", default="",
        help="Chemin lexique CSV/SQLite pour vocabulaire ferme (optionnel).",
    )
    parser.add_argument(
        "--max-phrases", type=int, default=0,
        help="Limite de phrases (0 = illimite).",
    )
    parser.add_argument(
        "--min-count", type=int, default=2,
        help="Seuil min de frequence pour n-grams n>=2 (defaut: 2).",
    )
    args = parser.parse_args()

    vocab = None
    if args.lexique:
        print(f"Chargement vocabulaire : {args.lexique}", file=sys.stderr)
        vocab = lire_vocabulaire_lexique(args.lexique)
        if vocab:
            print(f"  {len(vocab):,} formes", file=sys.stderr)
        else:
            print("  ERREUR : vocabulaire vide", file=sys.stderr)

    print(f"Comptage {args.order}-grams depuis {args.corpus}...", file=sys.stderr)
    counts = compter_ngrams(
        args.corpus, args.order, vocab, args.max_phrases, args.min_count,
    )

    print("Calcul des probabilites (Kneser-Ney modifie)...", file=sys.stderr)
    probs = calculer_probabilites_fast(counts, args.order)

    print(f"Ecriture SQLite : {args.output}", file=sys.stderr)
    ecrire_sqlite(probs, counts, args.order, args.output)

    print("Termine.", file=sys.stderr)


if __name__ == "__main__":
    main()
