#!/usr/bin/env python3
"""Construit une base n-gram POS (+ POS+Morpho) depuis le Kit-G2P-P2G.

Lit les phrases annotees du corpus Universal Dependencies French-GSD
(via le Kit-G2P-P2G), extrait les sequences de tags POS et de tuples
(POS, Number, Gender, Person), compte les n-grammes (1 a 4), et
sauvegarde dans une base SQLite.

Les log-probabilites sont calculees avec lissage Kneser-Ney modifie
(avec backoff) au lieu du lissage Laplace naif.

Usage :
    python scripts/construire_pos_ngram.py
    python scripts/construire_pos_ngram.py --corpus /chemin/vers/sentences_train.json
    python scripts/construire_pos_ngram.py --all-splits  # train+dev+test
    python scripts/construire_pos_ngram.py --all-splits --lexique path/to/lexique.db
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin par defaut vers le corpus Kit-G2P-P2G
KIT_DIR = "/home/moi/Documents/work/projets/lectura/workspace/Corpus/Kit-G2P-P2G/corpus/phrases"

OUTPUT_PATH = os.path.join(
    _PROJECT_ROOT, "src", "lectura_correcteur", "data", "pos_ngram.db",
)

# Symboles speciaux
BOS = "<BOS>"
EOS = "<EOS>"

# Kneser-Ney modifie — discounts
D1 = 0.5   # discount pour count=1
D2 = 0.75  # discount pour count=2
D3 = 1.0   # discount pour count>=3


def charger_phrases(paths: list[str]) -> list[list[dict]]:
    """Charge les phrases depuis un ou plusieurs fichiers JSON."""
    phrases = []
    for path in paths:
        print(f"  Lecture de {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sent in data:
            tokens = sent.get("tokens", [])
            if tokens:
                phrases.append(tokens)
    return phrases


def extraire_pos_seq(tokens: list[dict]) -> list[str]:
    """Extrait la sequence POS d'une phrase."""
    return [t["pos_tag"] for t in tokens if t.get("pos_tag")]


# ---------------------------------------------------------------------------
# Enrichissement genre NOM (Phase 2)
# ---------------------------------------------------------------------------

def _charger_lexique(lexique_path: str):
    """Charge le lexique pour l'enrichissement du genre NOM.

    Retourne un objet avec .info() ou None si indisponible.
    """
    if not lexique_path or not os.path.exists(lexique_path):
        return None

    # Essayer d'importer le lexique lite du correcteur
    try:
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
        from lectura_correcteur._lexique_lite import LexiqueLite
        return LexiqueLite(lexique_path)
    except ImportError:
        pass

    # Fallback : lecture directe SQLite
    try:
        conn = sqlite3.connect(lexique_path)
        conn.row_factory = sqlite3.Row

        class _LexiqueFallback:
            def __init__(self, c):
                self._conn = c
            def info(self, mot):
                cur = self._conn.execute(
                    "SELECT * FROM lexique WHERE ortho = ? COLLATE NOCASE",
                    (mot,),
                )
                return [dict(row) for row in cur.fetchall()]

        return _LexiqueFallback(conn)
    except Exception:
        return None


# Cache pour eviter les lookups repetes
_gender_cache: dict[str, str] = {}


def _infer_nom_gender(
    token: dict,
    tokens: list[dict],
    idx: int,
    lexique,
) -> str:
    """Infere le genre d'un NOM via une cascade a 3 niveaux.

    1. Lexique : si toutes les entrees NOM du mot ont le meme genre
    2. Contexte : genre de l'ART/DET precedent dans les annotations UD
    3. Defaut : "Masc"

    Returns:
        "Masc", "Fem", ou "_" si pas de genre applicable.
    """
    form = token.get("form", "").lower()
    if not form:
        return "_"

    # --- Tier 1 : lexique ---
    if lexique is not None:
        cache_key = form
        if cache_key in _gender_cache:
            cached = _gender_cache[cache_key]
            if cached != "_AMBIG_":
                return cached
            # Ambigu → passer au tier 2
        else:
            try:
                infos = lexique.info(form)
            except Exception:
                infos = []

            genres = set()
            for entry in infos:
                cgram = entry.get("cgram", "")
                if not cgram:
                    continue
                # Seulement les entrees NOM
                if cgram.split(":")[0] != "NOM":
                    continue
                g = entry.get("genre", "")
                if g in ("m", "Masc"):
                    genres.add("Masc")
                elif g in ("f", "Fem"):
                    genres.add("Fem")

            if len(genres) == 1:
                result = genres.pop()
                _gender_cache[cache_key] = result
                return result
            elif len(genres) > 1:
                # Epicene — marquer comme ambigu, passer au tier 2
                _gender_cache[cache_key] = "_AMBIG_"
            else:
                # Pas dans le lexique ou pas de genre — passer au tier 2
                _gender_cache[cache_key] = "_AMBIG_"

    # --- Tier 2 : contexte (ART/DET precedent avec Gender annote) ---
    for j in range(idx - 1, max(idx - 4, -1), -1):
        prev = tokens[j]
        prev_pos = prev.get("pos_tag", "")
        prev_base = prev_pos.split(":")[0]
        if prev_base in ("ART", "DET"):
            prev_gender = prev.get("morpho", {}).get("Gender", "")
            if prev_gender in ("Masc", "Fem"):
                return prev_gender
            break
        elif prev_base not in ("ADJ", "ADV"):
            break

    # --- Tier 3 : defaut masculin ---
    return "Masc"


def extraire_pos_morpho_seq(
    tokens: list[dict],
    lexique=None,
) -> list[str]:
    """Extrait la sequence POS+Number+Gender+Person sous forme de cle composee.

    Exemple : "NOM|Plur|Masc|_", "VER|Sing|_|3", "ART:def|Plur|_|_"
    Person est crucial pour la conjugaison (accord sujet-verbe en personne).

    Si lexique est fourni, le genre des NOM est enrichi via _infer_nom_gender.
    """
    result = []
    for i, t in enumerate(tokens):
        pos = t.get("pos_tag", "")
        if not pos:
            continue
        morpho = t.get("morpho", {})
        number = morpho.get("Number", "_")
        gender = morpho.get("Gender", "_")
        person = morpho.get("Person", "_")

        # Enrichir le genre des NOM si absent et lexique disponible
        pos_base = pos.split(":")[0]
        if pos_base == "NOM" and gender == "_" and lexique is not None:
            gender = _infer_nom_gender(t, tokens, i, lexique)

        result.append(f"{pos}|{number}|{gender}|{person}")
    return result


# ---------------------------------------------------------------------------
# Comptage n-grammes (jusqu'a 4-grammes)
# ---------------------------------------------------------------------------

def compter_ngrams(
    phrases_seqs: list[list[str]],
    order: int = 4,
) -> dict[int, Counter]:
    """Compte les n-grammes de 1 a order avec BOS/EOS.

    Utilise (order-1) BOS en padding.
    """
    counts: dict[int, Counter] = {n: Counter() for n in range(1, order + 1)}

    for seq in phrases_seqs:
        if not seq:
            continue
        # Padding : (order-1) BOS + seq + EOS
        n_bos = order - 1
        padded = [BOS] * n_bos + seq + [EOS]

        for i in range(n_bos, len(padded)):
            for n in range(1, min(order, i + 1) + 1):
                ngram = tuple(padded[i - n + 1: i + 1])
                counts[n][ngram] += 1

        # BOS counts
        counts[1][(BOS,)] += n_bos

    return counts


# ---------------------------------------------------------------------------
# Kneser-Ney modifie
# ---------------------------------------------------------------------------

def _discount(count: int) -> float:
    """Discount Kneser-Ney modifie selon le comptage."""
    if count == 1:
        return D1
    if count == 2:
        return D2
    return D3


def calculer_probabilites_kn(
    counts: dict[int, Counter],
    order: int,
) -> dict[int, dict[tuple, tuple[float, float]]]:
    """Calcule les log-probabilites et poids de backoff (Kneser-Ney modifie).

    Returns:
        Dict {n: {ngram: (log_prob, log_backoff)}} pour n=1..order.
        log_prob et log_backoff sont en log naturel (ln).
    """
    probs: dict[int, dict[tuple, tuple[float, float]]] = {}

    # --- Unigrammes ---
    total_unigrams = sum(counts[1].values())
    probs[1] = {}
    for ngram, count in counts[1].items():
        p = count / total_unigrams
        probs[1][ngram] = (math.log(p), 0.0)

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

            d = _discount(count)
            p = max(count - d, 0.0) / total

            lambda_prefix = (
                D1 * prefix_d1.get(prefix, 0)
                + D2 * prefix_d2.get(prefix, 0)
                + D3 * prefix_d3.get(prefix, 0)
            ) / total

            log_p = math.log(p + 1e-20) if p > 0 else -20.0
            log_bo = math.log(lambda_prefix + 1e-20) if lambda_prefix > 0 else 0.0

            probs[n][ngram] = (log_p, log_bo)

    return probs


# ---------------------------------------------------------------------------
# Ecriture SQLite
# ---------------------------------------------------------------------------

def creer_db(
    db_path: str,
    pos_probs: dict[int, dict[tuple, tuple[float, float]]],
    pm_probs: dict[int, dict[tuple, tuple[float, float]]],
    pos_counts: dict[int, Counter],
    pm_counts: dict[int, Counter],
    order: int,
    stats: dict,
) -> None:
    """Cree la base SQLite avec les n-grammes POS et PM."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=OFF")
    cur = conn.cursor()

    # Table de metadonnees
    cur.execute("""
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    for k, v in stats.items():
        cur.execute("INSERT INTO meta VALUES (?, ?)", (k, str(v)))

    # --- Tables POS ---
    _creer_tables_ngram(cur, conn, "pos", pos_probs, pos_counts, order)

    # --- Tables POS+Morpho ---
    _creer_tables_ngram(cur, conn, "pm", pm_probs, pm_counts, order)

    conn.commit()

    # Index pour les lookups de backoff
    for prefix in ("pos", "pm"):
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{prefix}_bi_w1 ON {prefix}_bigrams(w1)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{prefix}_tri_w1w2 ON {prefix}_trigrams(w1, w2)"
        )
        if order >= 4:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{prefix}_four_w1w2w3"
                f" ON {prefix}_fourgrams(w1, w2, w3)"
            )
    conn.commit()
    conn.close()


def _creer_tables_ngram(
    cur,
    conn,
    prefix: str,
    probs: dict[int, dict[tuple, tuple[float, float]]],
    counts: dict[int, Counter],
    order: int,
) -> None:
    """Cree et peuple les tables n-grammes pour un prefixe (pos ou pm)."""

    # Unigrammes
    cur.execute(f"""
        CREATE TABLE {prefix}_unigrams (
            w1 TEXT PRIMARY KEY,
            logp REAL,
            backoff REAL
        )
    """)
    batch = []
    for ngram, (logp, logbo) in probs[1].items():
        batch.append((ngram[0], logp, logbo))
    conn.executemany(
        f"INSERT INTO {prefix}_unigrams VALUES (?, ?, ?)", batch,
    )

    # Bigrammes
    cur.execute(f"""
        CREATE TABLE {prefix}_bigrams (
            w1 TEXT,
            w2 TEXT,
            logp REAL,
            backoff REAL,
            PRIMARY KEY (w1, w2)
        )
    """)
    if 2 in probs:
        batch = []
        for ngram, (logp, logbo) in probs[2].items():
            batch.append((ngram[0], ngram[1], logp, logbo))
        conn.executemany(
            f"INSERT INTO {prefix}_bigrams VALUES (?, ?, ?, ?)", batch,
        )

    # Trigrammes
    cur.execute(f"""
        CREATE TABLE {prefix}_trigrams (
            w1 TEXT,
            w2 TEXT,
            w3 TEXT,
            logp REAL,
            backoff REAL,
            PRIMARY KEY (w1, w2, w3)
        )
    """)
    if 3 in probs:
        batch = []
        for ngram, (logp, logbo) in probs[3].items():
            batch.append((ngram[0], ngram[1], ngram[2], logp, logbo))
        conn.executemany(
            f"INSERT INTO {prefix}_trigrams VALUES (?, ?, ?, ?, ?)", batch,
        )

    # 4-grammes
    if order >= 4:
        cur.execute(f"""
            CREATE TABLE {prefix}_fourgrams (
                w1 TEXT,
                w2 TEXT,
                w3 TEXT,
                w4 TEXT,
                logp REAL,
                backoff REAL,
                PRIMARY KEY (w1, w2, w3, w4)
            )
        """)
        if 4 in probs:
            batch = []
            for ngram, (logp, logbo) in probs[4].items():
                batch.append((
                    ngram[0], ngram[1], ngram[2], ngram[3], logp, logbo,
                ))
            conn.executemany(
                f"INSERT INTO {prefix}_fourgrams VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construire la base n-gram POS depuis le Kit-G2P-P2G",
    )
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="Chemin vers un fichier sentences_*.json specifique",
    )
    parser.add_argument(
        "--all-splits", action="store_true",
        help="Utiliser train + dev + test (par defaut : train seulement)",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH,
        help=f"Chemin de sortie (defaut: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--lexique", type=str, default="",
        help="Chemin vers la base lexique SQLite pour enrichir le genre des NOM",
    )
    parser.add_argument(
        "--order", type=int, default=4, choices=[2, 3, 4],
        help="Ordre max du modele n-gram (defaut: 4)",
    )
    args = parser.parse_args()

    # Determiner les fichiers d'entree
    if args.corpus:
        paths = [args.corpus]
    elif args.all_splits:
        paths = [
            os.path.join(KIT_DIR, f"sentences_{split}.json")
            for split in ("train", "dev", "test")
        ]
    else:
        paths = [os.path.join(KIT_DIR, "sentences_train.json")]

    for p in paths:
        if not os.path.exists(p):
            print(f"ERREUR: fichier introuvable: {p}", file=sys.stderr)
            sys.exit(1)

    # Charger le lexique pour enrichissement genre NOM
    lexique = None
    if args.lexique:
        print(f"Chargement lexique : {args.lexique}")
        lexique = _charger_lexique(args.lexique)
        if lexique is not None:
            print("  Lexique charge (enrichissement genre NOM actif)")
        else:
            print("  ATTENTION : lexique introuvable, genre NOM non enrichi")

    # 1. Charger les phrases
    print("Chargement des phrases...")
    phrases = charger_phrases(paths)
    print(f"  {len(phrases)} phrases chargees")

    total_tokens = sum(len(p) for p in phrases)
    print(f"  {total_tokens} tokens")

    # 2. Extraire les sequences POS
    print("\nExtraction des sequences POS...")
    pos_seqs = [extraire_pos_seq(p) for p in phrases]
    pos_seqs = [s for s in pos_seqs if s]

    # 3. Extraire les sequences POS+Morpho (avec enrichissement genre NOM)
    print("Extraction des sequences POS+Morpho...")
    if lexique is not None:
        # Reset du cache pour stats
        _gender_cache.clear()

    pm_seqs = [extraire_pos_morpho_seq(p, lexique=lexique) for p in phrases]
    pm_seqs = [s for s in pm_seqs if s]

    if lexique is not None:
        # Stats enrichissement
        n_enriched = sum(1 for v in _gender_cache.values() if v not in ("_AMBIG_",))
        n_ambig = sum(1 for v in _gender_cache.values() if v == "_AMBIG_")
        print(f"  Genre NOM enrichi : {n_enriched} formes uniques (genre unique)")
        print(f"  Genre NOM ambigu  : {n_ambig} formes epicenes (fallback contexte/Masc)")

    order = args.order

    # 4. Compter les n-grammes POS
    print(f"\nComptage n-grammes POS (ordre {order})...")
    pos_counts = compter_ngrams(pos_seqs, order=order)
    pos_vocab = set(k[0] for k in pos_counts[1].keys())
    print(f"  Vocabulaire POS: {len(pos_vocab)} labels")
    for n in range(1, order + 1):
        print(f"  {n}-grammes: {len(pos_counts[n])}")

    # 5. Compter les n-grammes POS+Morpho
    print(f"\nComptage n-grammes POS+Morpho (ordre {order})...")
    pm_counts = compter_ngrams(pm_seqs, order=order)
    pm_vocab = set(k[0] for k in pm_counts[1].keys())
    print(f"  Vocabulaire POS+Morpho: {len(pm_vocab)} labels")
    for n in range(1, order + 1):
        n_singles = sum(1 for c in pm_counts[n].values() if c == 1)
        pct = 100 * n_singles / len(pm_counts[n]) if pm_counts[n] else 0
        print(f"  {n}-grammes: {len(pm_counts[n])} ({pct:.1f}% singletons)")

    # 6. Calculer les log-probabilites (Kneser-Ney modifie)
    print(f"\nCalcul des log-probabilites (Kneser-Ney modifie, D1={D1}, D2={D2}, D3={D3})...")
    pos_probs = calculer_probabilites_kn(pos_counts, order)
    pm_probs = calculer_probabilites_kn(pm_counts, order)

    # 7. Sauvegarder en SQLite
    stats = {
        "n_phrases": len(phrases),
        "n_tokens": total_tokens,
        "smoothing": "kneser_ney_modified",
        "D1": D1,
        "D2": D2,
        "D3": D3,
        "order": order,
        "n_pos_labels": len(pos_vocab),
        "n_pm_labels": len(pm_vocab),
        "n_pos_trigrams": len(pos_counts.get(3, {})),
        "n_pm_trigrams": len(pm_counts.get(3, {})),
        "n_pos_fourgrams": len(pos_counts.get(4, {})),
        "n_pm_fourgrams": len(pm_counts.get(4, {})),
        "genre_enrichment": "lexique" if lexique else "none",
        "source": "Kit-G2P-P2G (UD French-GSD)",
    }

    print(f"\nSauvegarde dans {args.output}...")
    creer_db(
        args.output,
        pos_probs, pm_probs,
        pos_counts, pm_counts,
        order, stats,
    )

    file_size = os.path.getsize(args.output)
    print(f"  Taille: {file_size / 1024:.1f} KB")

    # 8. Verification : top trigrammes POS
    print("\nTop 20 trigrammes POS (par frequence):")
    top_tri = pos_counts[3].most_common(20) if 3 in pos_counts else []
    for (w1, w2, w3), count in top_tri:
        logp = pos_probs[3].get((w1, w2, w3), (-99.0, 0.0))[0]
        print(f"  {count:6d}  {w1:12s} {w2:12s} {w3:12s}  logp={logp:.3f}")

    # 9. Verification : trigrammes pertinents pour les homophones
    print("\nTrigrammes pertinents pour desambiguation:")
    test_trigrams = [
        ("PRO:per", "AUX", "NOM"),
        ("PRO:per", "PRE", "NOM"),
        ("NOM", "AUX", "ADJ"),
        ("NOM", "CON", "ADJ"),
        ("PRO:per", "AUX", "VER"),
        ("PRO:per", "VER", "VER"),
    ]
    for tri in test_trigrams:
        count = pos_counts[3].get(tri, 0)
        logp = pos_probs[3].get(tri, (-99.0, 0.0))[0]
        print(f"  {tri[0]:12s} {tri[1]:12s} {tri[2]:12s}  count={count:5d}  logp={logp:.3f}")

    # 10. Verification : bigrammes PM NOM+genre → VER
    print("\nBigrammes PM NOM → VER (verification enrichissement genre):")
    test_pm_bi = [
        ("NOM|Plur|_|_", "VER|Plur|_|3"),
        ("NOM|Plur|Masc|_", "VER|Plur|_|3"),
        ("NOM|Plur|Fem|_", "VER|Plur|_|3"),
        ("NOM|Sing|Masc|_", "VER|Sing|_|3"),
        ("NOM|Sing|Fem|_", "VER|Sing|_|3"),
    ]
    for w1, w2 in test_pm_bi:
        count = pm_counts[2].get((w1, w2), 0)
        entry = pm_probs[2].get((w1, w2))
        logp = entry[0] if entry else -99.0
        print(f"  {w1:22s} {w2:22s}  count={count:5d}  logp={logp:.3f}")

    # POS+Morpho: AUX + VER
    print("\nTrigrammes POS+Morpho (VerbForm):")
    if 3 in pm_counts:
        aux_ver_tri = [
            ((w1, w2, w3), c)
            for (w1, w2, w3), c in pm_counts[3].items()
            if w1.startswith("AUX") and w2.startswith("VER")
        ]
        aux_ver_tri.sort(key=lambda x: -x[1])
        for (w1, w2, w3), count in aux_ver_tri[:15]:
            logp = pm_probs[3].get((w1, w2, w3), (-99.0, 0.0))[0]
            print(f"  {w1:20s} {w2:20s} {w3:20s}  count={count:5d}  logp={logp:.3f}")

    print("\nTermine.")


if __name__ == "__main__":
    main()
