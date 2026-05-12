#!/usr/bin/env python3
"""Generateur de corpus d'entrainement pour le correcteur Lectura.

Lit des phrases propres (Wikipedia FR), injecte des erreurs realistes
via les perturbateurs, et ecrit des paires (fautif, correct) en JSONL.

Usage:
    python scripts/corpus/generer.py                      # 10 000 paires (defaut)
    python scripts/corpus/generer.py -n 200000            # 200k paires
    python scripts/corpus/generer.py -n 1000 --preview    # apercu sans ecrire
    python scripts/corpus/generer.py --stats              # statistiques seulement
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# --- Chemins par defaut ---
WIKI_PATH = Path("/data/work/projets/lectura/data/wikipedia/frwiki_phrases.txt")
LEXIQUE_PATH = Path("/data/work/projets/lectura/workspace/Lexique/lexique_lectura.db")
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"

# --- Import perturbateurs (relatif au script) ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.corpus.perturbateurs import (
    TokenEnrichi,
    Erreur,
    enrichir_tokens,
    appliquer_perturbations,
    DISTRIBUTION_ERREURS,
)


# ---------------------------------------------------------------------------
# Chargement et filtrage des phrases source
# ---------------------------------------------------------------------------

# Mots fonctionnels courants (pour verifier que la phrase est du francais)
MOTS_FR = frozenset(
    "le la les un une de des du en et est a au aux il elle on je tu nous vous "
    "ils elles ce qui que ne pas dans par pour avec son sa ses sur".split()
)

# Pattern pour les artefacts wiki
RE_WIKI = re.compile(r"[''{}|<>\[\]#*=]")


def est_phrase_valide(phrase: str, min_mots: int = 5, max_mots: int = 30) -> bool:
    """Filtre les phrases exploitables pour le corpus."""
    # Artefacts wiki
    if RE_WIKI.search(phrase):
        return False
    # Longueur
    mots = phrase.split()
    if len(mots) < min_mots or len(mots) > max_mots:
        return False
    # Au moins 2 mots FR fonctionnels (proxy pour "c'est du francais")
    n_fr = sum(1 for m in mots if m.lower() in MOTS_FR)
    if n_fr < 2:
        return False
    # Pas de chiffres ni URLs
    if re.search(r"\d{3,}|http|www\.", phrase):
        return False
    return True


def charger_phrases(
    path: Path,
    n_max: int,
    *,
    seed: int = 42,
    max_scan: int = 0,
) -> list[str]:
    """Charge N phrases valides depuis le fichier Wikipedia.

    Utilise un reservoir sampling pour eviter de tout charger en memoire.
    Si max_scan > 0, arrete le scan apres max_scan phrases valides
    (utile pour les tests rapides).
    """
    random.seed(seed)
    reservoir: list[str] = []
    n_total = 0

    print(f"Chargement depuis {path.name}...")
    with open(path, encoding="utf-8") as f:
        for ligne in f:
            phrase = ligne.strip()
            if not phrase:
                continue
            if not est_phrase_valide(phrase):
                continue

            n_total += 1

            # Reservoir sampling
            if len(reservoir) < n_max:
                reservoir.append(phrase)
            else:
                j = random.randint(0, n_total - 1)
                if j < n_max:
                    reservoir[j] = phrase

            # Arret anticipe pour les tests
            if max_scan and n_total >= max_scan:
                break

            # Progression
            if n_total % 500_000 == 0:
                print(f"  {n_total:>10,} phrases valides scanees...", flush=True)

    print(f"  {n_total:,} phrases valides trouvees, {len(reservoir):,} selectionnees")
    random.shuffle(reservoir)
    return reservoir


# ---------------------------------------------------------------------------
# Tokenisation simple (sans dependance externe)
# ---------------------------------------------------------------------------

RE_TOKEN = re.compile(
    r"(?:[a-zàâäéèêëïîôùûüÿçœæ](?:['-][a-zàâäéèêëïîôùûüÿçœæ])?)+|[^\s]",
    re.IGNORECASE,
)


def tokeniser_simple(phrase: str) -> list[str]:
    """Tokenisation basique respectant les apostrophes et traits d'union."""
    return RE_TOKEN.findall(phrase)


# ---------------------------------------------------------------------------
# Generation du corpus
# ---------------------------------------------------------------------------

def generer_paire(
    phrase: str,
    lexique,
    *,
    n_erreurs_min: int = 1,
    n_erreurs_max: int = 2,
) -> dict | None:
    """Genere une paire (fautif, correct) a partir d'une phrase propre.

    Returns:
        Dict avec cles: correct, fautif, erreurs, n_erreurs
        ou None si aucune erreur n'a pu etre injectee.
    """
    tokens_str = tokeniser_simple(phrase)
    if len(tokens_str) < 3:
        return None

    tokens_enrichis = enrichir_tokens(tokens_str, lexique)

    # Nombre d'erreurs aleatoire
    n_err = random.randint(n_erreurs_min, n_erreurs_max)

    formes_perturbees, erreurs = appliquer_perturbations(
        tokens_enrichis, lexique, n_erreurs=n_err,
    )

    if not erreurs:
        return None

    phrase_fautive = " ".join(formes_perturbees)
    phrase_correcte = " ".join(t.forme for t in tokens_enrichis)

    return {
        "correct": phrase_correcte,
        "fautif": phrase_fautive,
        "n_erreurs": len(erreurs),
        "erreurs": [
            {
                "position": e.position,
                "original": e.original,
                "perturbe": e.perturbe,
                "type": e.type_erreur,
            }
            for e in erreurs
        ],
    }


def generer_corpus(
    phrases: list[str],
    lexique,
    *,
    n_cible: int = 10_000,
    n_erreurs_min: int = 1,
    n_erreurs_max: int = 2,
) -> list[dict]:
    """Genere N paires a partir de phrases propres."""
    corpus: list[dict] = []
    n_echecs = 0
    t0 = time.time()

    for i, phrase in enumerate(phrases):
        if len(corpus) >= n_cible:
            break

        paire = generer_paire(
            phrase, lexique,
            n_erreurs_min=n_erreurs_min,
            n_erreurs_max=n_erreurs_max,
        )
        if paire:
            corpus.append(paire)
        else:
            n_echecs += 1

        if (i + 1) % 2000 == 0:
            dt = time.time() - t0
            pct = len(corpus) / n_cible * 100
            print(
                f"  {len(corpus):>8,}/{n_cible:,} paires ({pct:.0f}%) "
                f"| {i+1:,} phrases traitees "
                f"| {dt:.1f}s",
                flush=True,
            )

    dt = time.time() - t0
    taux = len(corpus) / (len(corpus) + n_echecs) * 100 if corpus else 0
    print(f"\nGeneration terminee : {len(corpus):,} paires en {dt:.1f}s")
    print(f"Taux de succes : {taux:.1f}% ({n_echecs:,} echecs)")
    return corpus


def afficher_stats(corpus: list[dict]) -> None:
    """Affiche les statistiques du corpus genere."""
    if not corpus:
        print("Corpus vide.")
        return

    # Distribution des types d'erreurs
    compteurs: dict[str, int] = {}
    n_erreurs_total = 0
    for paire in corpus:
        for err in paire["erreurs"]:
            t = err["type"]
            compteurs[t] = compteurs.get(t, 0) + 1
            n_erreurs_total += 1

    print(f"\n{'='*60}")
    print(f"  STATISTIQUES CORPUS — {len(corpus):,} paires")
    print(f"{'='*60}")
    print(f"  Erreurs totales : {n_erreurs_total:,}")
    print(f"  Moyenne par phrase : {n_erreurs_total/len(corpus):.2f}")
    print()
    print(f"  {'Type':<10} {'Count':>8}  {'%':>6}  Distribution cible")
    print(f"  {'-'*50}")
    for type_err in sorted(compteurs, key=lambda t: -compteurs[t]):
        count = compteurs[type_err]
        pct = count / n_erreurs_total * 100
        cible = DISTRIBUTION_ERREURS.get(type_err, 0) * 100
        bar = "█" * int(pct / 2)
        print(f"  {type_err:<10} {count:>8,}  {pct:>5.1f}%  {bar} (cible: {cible:.0f}%)")

    # Distribution du nombre d'erreurs par phrase
    print()
    distrib_n = {}
    for paire in corpus:
        n = paire["n_erreurs"]
        distrib_n[n] = distrib_n.get(n, 0) + 1
    for n in sorted(distrib_n):
        pct = distrib_n[n] / len(corpus) * 100
        print(f"  {n} erreur(s) : {distrib_n[n]:>8,} phrases ({pct:.1f}%)")


def afficher_preview(corpus: list[dict], n: int = 20) -> None:
    """Affiche un apercu du corpus."""
    print(f"\n{'='*60}")
    print(f"  APERCU — {min(n, len(corpus))} premiers exemples")
    print(f"{'='*60}")
    for i, paire in enumerate(corpus[:n]):
        print(f"\n  [{i+1}]")
        print(f"  CORRECT : {paire['correct']}")
        print(f"  FAUTIF  : {paire['fautif']}")
        for err in paire["erreurs"]:
            print(f"    -> [{err['type']}] {err['original']!r} -> {err['perturbe']!r} (pos {err['position']})")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genere un corpus d'entrainement pour le correcteur Lectura.",
    )
    parser.add_argument(
        "-n", "--n-paires", type=int, default=10_000,
        help="Nombre de paires a generer (defaut: 10 000)",
    )
    parser.add_argument(
        "--erreurs-min", type=int, default=1,
        help="Nombre minimum d'erreurs par phrase (defaut: 1)",
    )
    parser.add_argument(
        "--erreurs-max", type=int, default=2,
        help="Nombre maximum d'erreurs par phrase (defaut: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Graine aleatoire (defaut: 42)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Afficher un apercu sans ecrire le fichier",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Afficher les statistiques seulement",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Chemin du fichier de sortie (defaut: data/corpus/corpus_NNN.jsonl)",
    )
    parser.add_argument(
        "--wiki", type=Path, default=WIKI_PATH,
        help="Chemin du fichier Wikipedia",
    )
    parser.add_argument(
        "--lexique", type=Path, default=LEXIQUE_PATH,
        help="Chemin du fichier lexique (.db)",
    )
    parser.add_argument(
        "--max-scan", type=int, default=0,
        help="Limiter le scan a N phrases valides (0 = tout, utile pour tests)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Charger le lexique
    print("Chargement du lexique...")
    from lectura_lexique import Lexique
    lexique = Lexique(args.lexique)
    print(f"  {len(lexique._formes):,} formes chargees")

    # On charge plus de phrases que demande (toutes ne donneront pas une paire)
    n_phrases = int(args.n_paires * 1.5)

    # Charger les phrases
    phrases = charger_phrases(
        args.wiki, n_phrases, seed=args.seed, max_scan=args.max_scan,
    )

    # Generer le corpus
    print(f"\nGeneration de {args.n_paires:,} paires...")
    corpus = generer_corpus(
        phrases, lexique,
        n_cible=args.n_paires,
        n_erreurs_min=args.erreurs_min,
        n_erreurs_max=args.erreurs_max,
    )

    # Stats
    afficher_stats(corpus)

    # Preview
    if args.preview:
        afficher_preview(corpus, n=30)
        return

    if args.stats:
        return

    # Ecrire le fichier
    output_path = args.output
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"corpus_{len(corpus)}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for paire in corpus:
            f.write(json.dumps(paire, ensure_ascii=False) + "\n")

    taille_mo = output_path.stat().st_size / (1024 * 1024)
    print(f"\nCorpus ecrit : {output_path}")
    print(f"  {len(corpus):,} paires | {taille_mo:.1f} Mo")


if __name__ == "__main__":
    main()
