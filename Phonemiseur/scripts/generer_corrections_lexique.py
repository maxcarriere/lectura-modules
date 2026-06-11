#!/usr/bin/env python3
"""Compare le G2P neural avec le lexique de reference et genere :

1. ``g2p_corrections_lexique.json`` — corrections pour les mots frequents
   (freq > 0), couvrant ~99% de la couverture de la langue francaise.
2. ``g2p_corrections_lexique_etendu.json`` — toutes les corrections
   (y compris mots rares a freq=0), pour l'annotation de corpus.
3. Mise a jour de ``homographes.json`` — ajout des entrees NOM / NOM PROPRE
   pour les mots dont la prononciation differe selon la categorie
   (ex. jean=dʒin vs Jean=ʒɑ̃).

Filtres appliques :
- Sigles (multext Y*) exclus (geres par le Tokeniseur/Formules)
- Homographes du pipeline (homographes.json) exclus (geres contextuellement)
- Mots contenant des chiffres exclus (geres par le Tokeniseur/Formules)

Mode --sep (V4) :
- Les corrections sont stockees avec separateurs (ex: bo-pɛʁ pour beau-pere)
- Un mot est en erreur si le phone OU la position/type des separateurs differe
- Reference construite depuis la colonne ``syllabes`` de la base v6
  (marqueurs [-] et ['] pour tiret et apostrophe)

Usage :
    python generer_corrections_lexique.py --raw
    python generer_corrections_lexique.py --raw --sep
    python generer_corrections_lexique.py --raw --sep --model-path model.onnx --vocab-path vocab.json
    python generer_corrections_lexique.py --raw --limit 10000
    python generer_corrections_lexique.py --raw --batch-size 1  # sans contexte
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "3"

# ── Chemins ──────────────────────────────────────────────────────────────────

_LEXIQUE_DB = Path(
    "/data/work/projets/lectura/workspace/Modules/Correcteur/"
    "src/lectura_correcteur/data/lexique_correcteur.db"
)
_LEXIQUE_V6_DB = Path(
    "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v6.db"
)
_OUTPUT_DIR = Path(
    "/data/work/projets/lectura/workspace/Modules/Phonemiseur/"
    "src/lectura_phonemiseur/data"
)
_OUTPUT_BASE = _OUTPUT_DIR / "g2p_corrections_lexique.json"
_OUTPUT_ETENDU = _OUTPUT_DIR / "g2p_corrections_lexique_etendu.json"
_HOMOGRAPHES_PATH = Path(
    "/data/work/projets/lectura/workspace/Modules/Phonemiseur/"
    "src/lectura_phonemiseur/data/homographes.json"
)

_RE_CONTIENT_CHIFFRE = re.compile(r"\d")


# ── Separateurs (mode --sep) ─────────────────────────────────────────────────

def syllabes_to_phonesep(syllabes: str) -> str | None:
    """Convertit la colonne ``syllabes`` (v6) en phone avec separateurs.

    Marqueurs reconnus dans les crochets :
    - ``[-]`` → ``-``  (tiret dans le mot compose)
    - ``[']`` → ``'``  (apostrophe dans le mot compose)
    - ``[x]`` suivi de ``[-]`` ou ``[']`` → consonne de liaison, gardee
    - ``[x]`` en fin de mot → consonne latente, ignoree
    - ``.``   → supprime (frontiere syllabique)

    Retourne ``None`` si aucun separateur n'est present.
    """
    if not syllabes or "[" not in syllabes:
        return None

    # Parser tous les tokens
    tokens: list[tuple[str, str]] = []  # (type, content)
    i = 0
    n = len(syllabes)
    while i < n:
        ch = syllabes[i]
        if ch == "[":
            j = syllabes.index("]", i + 1)
            content = syllabes[i + 1:j]
            if content == "-":
                tokens.append(("sep", "-"))
            elif content == "'":
                tokens.append(("sep", "'"))
            else:
                tokens.append(("bracket", content))
            i = j + 1
        elif ch == ".":
            i += 1
        else:
            tokens.append(("char", ch))
            i += 1

    if not any(t == "sep" for t, _ in tokens):
        return None

    # Construire le resultat : les [x] suivis d'un sep sont des liaisons
    # (garder), les [x] en fin ou suivis d'autre chose sont des consonnes
    # latentes (ignorer)
    result: list[str] = []
    for idx, (typ, content) in enumerate(tokens):
        if typ == "char" or typ == "sep":
            result.append(content)
        elif typ == "bracket":
            # Garder seulement si suivi d'un separateur [-] ou [']
            next_is_sep = (
                idx + 1 < len(tokens) and tokens[idx + 1][0] == "sep"
            )
            if next_is_sep:
                result.append(content)
            # Sinon : consonne latente → ignorer
    return "".join(result)


def charger_phonesep_v6(
    db_path: Path,
) -> dict[str, set[str]]:
    """Charge les references phonesep depuis la base v6.

    Lit la colonne ``syllabes`` pour les formes ayant des marqueurs
    ``[-]`` ou ``[']``, et convertit en phone avec separateurs.

    Retourne mot_lower -> set de phonesep valides.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        SELECT lower(ortho), syllabes FROM formes
        WHERE phone IS NOT NULL AND phone != ''
        AND (syllabes LIKE '%[-]%' OR syllabes LIKE '%[''%')
    """)
    phonesep_refs: dict[str, set[str]] = {}
    n_parsed = 0
    n_err = 0
    for ortho, syllabes in cur.fetchall():
        ps = syllabes_to_phonesep(syllabes)
        if ps:
            phonesep_refs.setdefault(ortho, set()).add(ps)
            n_parsed += 1
        else:
            n_err += 1
    conn.close()
    print(f"  Phonesep v6 : {len(phonesep_refs):,} mots, {n_parsed:,} formes")
    if n_err:
        print(f"  ({n_err} formes non parsees)")
    return phonesep_refs


# ── Lexique ──────────────────────────────────────────────────────────────────

def charger_lexique(
    db_path: Path,
    limit: int = 0,
    exclude_sigles: bool = True,
    exclude_chiffres: bool = True,
    exclude_homographes: set[str] | None = None,
) -> tuple[dict[str, set[str]], dict[str, float]]:
    """Charge le lexique : mot -> ensemble de prononciations valides.

    Retourne (lexique, frequences) ou :
    - lexique: mot -> set de prononciations
    - frequences: mot -> frequence max
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    where = "WHERE phone IS NOT NULL AND phone != ''"
    if exclude_sigles:
        where += " AND (multext IS NULL OR multext NOT LIKE 'Y%')"

    cur.execute(f"SELECT lower(ortho), phone, freq FROM lexique {where}")

    lexique: dict[str, set[str]] = {}
    frequences: dict[str, float] = {}
    n_sigles = 0
    n_chiffres = 0
    n_homographes = 0

    for ortho, phone, freq in cur.fetchall():
        lexique.setdefault(ortho, set()).add(phone)
        if freq and (ortho not in frequences or freq > frequences[ortho]):
            frequences[ortho] = freq
    conn.close()

    # Filtrer les mots contenant des chiffres
    if exclude_chiffres:
        to_remove = {m for m in lexique if _RE_CONTIENT_CHIFFRE.search(m)}
        n_chiffres = len(to_remove)
        for m in to_remove:
            del lexique[m]
            frequences.pop(m, None)

    # Filtrer les homographes du pipeline (geres contextuellement)
    if exclude_homographes:
        to_remove = {m for m in lexique if m in exclude_homographes}
        n_homographes = len(to_remove)
        for m in to_remove:
            del lexique[m]
            frequences.pop(m, None)

    if n_chiffres or n_homographes:
        print(f"  Filtres: {n_chiffres} chiffres, {n_homographes} homographes pipeline")

    if limit:
        keys = sorted(lexique.keys())[:limit]
        lexique = {k: lexique[k] for k in keys}
        frequences = {k: frequences[k] for k in keys if k in frequences}

    return lexique, frequences


def extraire_homographes_np(
    db_path: Path,
) -> dict[str, dict[str, str]]:
    """Extrait les mots ayant des prononciations differentes selon NOM vs NOM PROPRE.

    Retourne un dict mot_lower -> {"NOM": ipa_nom, "NOM PROPRE": ipa_np}.
    """
    conn = sqlite3.connect(str(db_path))

    # Charger les phones NOM et NOM PROPRE en memoire
    np_phones: dict[str, set[str]] = defaultdict(set)
    nom_phones: dict[str, set[str]] = defaultdict(set)

    for ortho, phone, cgram in conn.execute("""
        SELECT LOWER(ortho), phone, cgram FROM lexique
        WHERE cgram IN ('NOM', 'NOM PROPRE')
        AND phone IS NOT NULL AND phone != ''
    """):
        if cgram == "NOM PROPRE":
            np_phones[ortho].add(phone)
        else:
            nom_phones[ortho].add(phone)
    conn.close()

    # Mots avec les deux categories et des phones differents
    result: dict[str, dict[str, str]] = {}
    both = set(np_phones) & set(nom_phones)
    for mot in sorted(both):
        if np_phones[mot] != nom_phones[mot]:
            # Prendre la premiere prononciation (triee) pour chaque categorie
            result[mot] = {
                "NOM": sorted(nom_phones[mot])[0],
                "NOM PROPRE": sorted(np_phones[mot])[0],
            }
    return result


def fusionner_homographes(
    homographes_path: Path, nouvelles_entrees: dict[str, dict[str, str]]
) -> int:
    """Fusionne les nouvelles entrees NOM/NOM PROPRE dans homographes.json.

    Les entrees existantes sont preservees (pas ecrasees).
    Retourne le nombre d'entrees ajoutees.
    """
    if homographes_path.exists():
        with open(homographes_path, encoding="utf-8") as f:
            homographes = json.load(f)
    else:
        homographes = {}

    n_added = 0
    for mot, entry in nouvelles_entrees.items():
        if mot in homographes:
            # Ajouter NOM PROPRE si absent
            if "NOM PROPRE" not in homographes[mot]:
                homographes[mot]["NOM PROPRE"] = entry["NOM PROPRE"]
                n_added += 1
        else:
            homographes[mot] = entry
            n_added += 1

    # Sauvegarder trie
    homographes = dict(sorted(homographes.items()))
    with open(homographes_path, "w", encoding="utf-8") as f:
        json.dump(homographes, f, ensure_ascii=False, indent=2)

    return n_added


def main():
    parser = argparse.ArgumentParser(
        description="Genere des corrections G2P depuis le lexique"
    )
    parser.add_argument("--db", type=Path, default=_LEXIQUE_DB)
    parser.add_argument("--db-v6", type=Path, default=_LEXIQUE_V6_DB,
                        help="Base lexique v6 (pour --sep)")
    parser.add_argument("--output-base", type=Path, default=_OUTPUT_BASE,
                        help="Fichier de corrections de base (freq > 0)")
    parser.add_argument("--output-etendu", type=Path, default=_OUTPUT_ETENDU,
                        help="Fichier de corrections etendu (tout)")
    parser.add_argument("--homographes", type=Path, default=_HOMOGRAPHES_PATH)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limiter le nombre de mots (0 = tout)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Taille de batch pour inference (1 = sans contexte)")
    parser.add_argument("--skip-np", action="store_true",
                        help="Ne pas generer les homographes NOM PROPRE")
    parser.add_argument("--raw", action="store_true",
                        help="Utiliser le moteur ONNX brut (sans corrections)")
    parser.add_argument("--sep", action="store_true",
                        help="Mode separateurs V4 : corrections avec - et '")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Chemin vers le modele ONNX (defaut: auto-detect)")
    parser.add_argument("--vocab-path", type=Path, default=None,
                        help="Chemin vers le vocab JSON (defaut: auto-detect)")
    parser.add_argument("--exclude-sigles", action="store_true", default=True,
                        help="Exclure les sigles (multext Y*) du lexique (defaut: oui)")
    parser.add_argument("--no-exclude-sigles", action="store_false", dest="exclude_sigles",
                        help="Inclure les sigles dans le lexique")
    args = parser.parse_args()

    # ── 1) Homographes NOM / NOM PROPRE ──
    if not args.skip_np:
        print("Extraction homographes NOM / NOM PROPRE...")
        np_entries = extraire_homographes_np(args.db)
        print(f"  {len(np_entries):,} mots avec prononciation NOM != NOM PROPRE")
        n_added = fusionner_homographes(args.homographes, np_entries)
        print(f"  {n_added:,} entrees ajoutees dans {args.homographes.name}")

    # ── 2) Charger les homographes du pipeline (a exclure) ──
    homographes_pipeline: set[str] = set()
    if args.homographes.exists():
        with open(args.homographes, encoding="utf-8") as f:
            homographes_pipeline = set(json.load(f).keys())
        print(f"\nHomographes pipeline: {len(homographes_pipeline):,} (seront exclus)")

    # ── 3) Charger le lexique ──
    print(f"\nChargement lexique : {args.db}")
    if args.exclude_sigles:
        print("  (sigles multext Y* exclus)")
    t0 = time.time()
    lexique, frequences = charger_lexique(
        args.db, args.limit,
        exclude_sigles=args.exclude_sigles,
        exclude_chiffres=True,
        exclude_homographes=homographes_pipeline,
    )
    mots = sorted(lexique.keys())
    n_avec_freq = sum(1 for m in mots if frequences.get(m, 0) > 0)
    print(f"  {len(mots):,} mots ({n_avec_freq:,} avec freq>0) ({time.time()-t0:.1f}s)")

    # ── 3b) Charger les references phonesep (mode --sep) ──
    phonesep_refs: dict[str, set[str]] = {}
    if args.sep:
        print(f"\nChargement phonesep depuis {args.db_v6.name}...")
        phonesep_refs = charger_phonesep_v6(args.db_v6)

    # ── 4) Charger le moteur ──
    if args.raw:
        print("\nChargement moteur G2P brut (sans corrections)...")
        from lectura_phonemiseur.inference_onnx_v2 import OnnxInferenceEngineV2
        from lectura_phonemiseur import _resoudre_modeles_dir, _resoudre_lexique

        if args.model_path and args.vocab_path:
            onnx_path = args.model_path
            vocab_path = args.vocab_path
            models_dir = args.model_path.parent
        else:
            models_dir = _resoudre_modeles_dir()
            if models_dir is None:
                raise RuntimeError("Aucun modele ONNX trouve (unifie_v4_int8.onnx)")
            onnx_path = models_dir / "unifie_v4_int8.onnx"
            vocab_path = models_dir / "unifie_v4_vocab.json"

        lexicon_path = _resoudre_lexique(None, models_dir)
        engine = OnnxInferenceEngineV2(
            onnx_path, vocab_path, lexicon_path=lexicon_path,
        )
        print(f"  Modele: {onnx_path}")
    else:
        print("\nChargement moteur G2P (avec corrections)...")
        from lectura_phonemiseur import creer_engine
        engine = creer_engine()
    engine.analyser(["test"])  # warmup
    print("  Pret.")

    erreurs: dict[str, str] = {}
    n_ok = 0
    n_err = 0
    n_skip = 0
    n_sep_ok = 0
    n_sep_err = 0

    # ── 5) Inference en batch ──
    print(f"\n--- Inference en batch de {args.batch_size} ---")
    if args.batch_size == 1:
        print("  (mode sans contexte)")
    if args.sep:
        print("  (mode separateurs V4)")
    t0 = time.time()

    for batch_start in range(0, len(mots), args.batch_size):
        batch = mots[batch_start:batch_start + args.batch_size]

        if batch_start > 0 and batch_start % max(args.batch_size * 50, 10000) == 0:
            elapsed = time.time() - t0
            speed = batch_start / elapsed
            eta = (len(mots) - batch_start) / speed
            sep_info = f" | sep_err={n_sep_err:,}" if args.sep else ""
            print(
                f"  {batch_start:>9,}/{len(mots):,} | "
                f"{n_err:,} erreurs{sep_info} | {speed:.0f} mots/s | "
                f"ETA {eta/60:.1f}min",
                flush=True,
            )

        try:
            if args.sep:
                r = engine.analyser(
                    batch, sep_hyphen=True, sep_apos=True,
                )
            else:
                r = engine.analyser(batch)
        except Exception:
            n_skip += len(batch)
            continue

        g2p_list = r.get("g2p", [])
        for mot, g2p in zip(batch, g2p_list):
            if args.sep and mot in phonesep_refs:
                # Mode sep : comparer avec la reference phonesep
                refs_sep = phonesep_refs[mot]
                if g2p in refs_sep:
                    n_ok += 1
                    n_sep_ok += 1
                else:
                    n_err += 1
                    n_sep_err += 1
                    ref_best = sorted(refs_sep)[0]
                    erreurs[mot] = ref_best
            else:
                # Mode normal : comparer avec le phone du lexique
                # En mode --sep, nettoyer les separateurs du g2p
                # (le modele peut inserer - ou ' mais la ref lexique n'en a pas)
                g2p_cmp = g2p.replace("-", "").replace("'", "") if args.sep else g2p
                refs = lexique.get(mot, set())
                if g2p_cmp in refs:
                    n_ok += 1
                else:
                    n_err += 1
                    ref_best = sorted(refs)[0] if refs else g2p_cmp
                    erreurs[mot] = ref_best

    elapsed_total = time.time() - t0

    # ── 6) Resultats ──
    total = n_ok + n_err + n_skip
    print(f"\n{'='*60}")
    print(f"RESULTATS")
    print(f"{'='*60}")
    print(f"  Total mots    : {total:,}")
    print(f"  Corrects      : {n_ok:,} ({n_ok/max(1,total):.1%})")
    print(f"  Erreurs       : {n_err:,} ({n_err/max(1,total):.1%})")
    if args.sep:
        n_sep_total = n_sep_ok + n_sep_err
        print(f"  dont sep      : {n_sep_err:,}/{n_sep_total:,} erreurs separateur")
    print(f"  Ignores       : {n_skip:,}")
    print(f"  Temps total   : {elapsed_total:.0f}s ({total/max(1,elapsed_total):.0f} mots/s)")

    # ── 7) Sauvegarder les deux fichiers ──
    all_corrections = dict(sorted(erreurs.items()))

    # Fichier etendu : toutes les corrections
    args.output_etendu.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_etendu, "w", encoding="utf-8") as f:
        json.dump(all_corrections, f, ensure_ascii=False, indent=1)
    n_etendu = len(all_corrections)
    sz_etendu = args.output_etendu.stat().st_size / 1024
    print(f"\n  Etendu  : {args.output_etendu.name}")
    print(f"           {n_etendu:,} entrees ({sz_etendu:.0f} Ko)")

    # Fichier base : seulement les mots avec freq > 0
    base_corrections = {
        m: ipa for m, ipa in all_corrections.items()
        if frequences.get(m, 0) > 0
    }
    with open(args.output_base, "w", encoding="utf-8") as f:
        json.dump(base_corrections, f, ensure_ascii=False, indent=1)
    n_base = len(base_corrections)
    sz_base = args.output_base.stat().st_size / 1024
    print(f"  Base    : {args.output_base.name}")
    print(f"           {n_base:,} entrees ({sz_base:.0f} Ko)")

    # ── 8) Echantillon d'erreurs ──
    print(f"\n--- Echantillon d'erreurs (top frequence) ---")
    top_freq = sorted(erreurs.items(), key=lambda x: -frequences.get(x[0], 0))[:30]
    for mot, ref in top_freq:
        freq = frequences.get(mot, 0)
        try:
            if args.sep:
                r = engine.analyser([mot], sep_hyphen=True, sep_apos=True)
            else:
                r = engine.analyser([mot])
            g2p = r["g2p"][0] if r.get("g2p") else "?"
        except Exception:
            g2p = "?"
        sep_marker = " [SEP]" if mot in phonesep_refs else ""
        print(f"  {mot:30s}  g2p={g2p:20s}  ref={ref:20s}  freq={freq:.1f}{sep_marker}")

    # ── 9) Echantillon erreurs separateurs (mode --sep) ──
    if args.sep and n_sep_err:
        print(f"\n--- Erreurs separateurs (top frequence) ---")
        sep_erreurs = {m: r for m, r in erreurs.items() if m in phonesep_refs}
        top_sep = sorted(sep_erreurs.items(), key=lambda x: -frequences.get(x[0], 0))[:30]
        for mot, ref in top_sep:
            freq = frequences.get(mot, 0)
            try:
                r = engine.analyser([mot], sep_hyphen=True, sep_apos=True)
                g2p = r["g2p"][0] if r.get("g2p") else "?"
            except Exception:
                g2p = "?"
            # Determiner le type d'erreur : phone ou separateur
            g2p_clean = g2p.replace("-", "").replace("'", "")
            ref_clean = ref.replace("-", "").replace("'", "")
            err_type = "SEP" if g2p_clean == ref_clean else "PHONE"
            print(f"  {mot:30s}  g2p={g2p:20s}  ref={ref:20s}  freq={freq:.1f}  [{err_type}]")


if __name__ == "__main__":
    main()
