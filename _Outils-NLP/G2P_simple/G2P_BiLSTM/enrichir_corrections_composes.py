#!/usr/bin/env python3
"""Enrichir les corrections G2P BiLSTM avec les parties de mots composés/apostrophe.

Le dico.csv contient des mots composés (avec tiret) et des mots avec apostrophe.
Quand le G2P traite du texte réel, ces mots sont souvent découpés par le tokenizer
en parties (ex: "aujourd'hui" → "aujourd" + "hui"). Certaines de ces parties
n'existent pas comme mots isolés dans le dico et n'ont donc pas de corrections.

Ce script :
1. Extrait les "parties orphelines" des mots composés/apostrophe
2. Détermine leur phonémisation gold par déduction (soustraction)
3. Teste le modèle BiLSTM
4. Ajoute les corrections nécessaires dans le fichier JSON

Usage :
    python enrichir_corrections_composes.py --dico /chemin/vers/dico.csv --dry-run
    python enrichir_corrections_composes.py --dico /chemin/vers/dico.csv --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# On importe depuis le lectura_g2p.py local (G2P_BiLSTM/)
sys.path.insert(0, str(Path(__file__).parent))
from lectura_g2p import (  # noqa: E402
    LecturaG2P,
    postprocess,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Séparateurs pour la découpe des mots composés
# ═══════════════════════════════════════════════════════════════════════════════

_SEPARATORS = re.compile(r"[-''']")

# Élisions à ignorer (parties d'un seul caractère ou très courtes sans sens)
_ELISIONS = {"c", "d", "l", "s", "n", "j", "m", "t", "qu"}


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 1 : Charger le dico
# ═══════════════════════════════════════════════════════════════════════════════

def load_dico(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Charge le dico.csv.

    Returns:
        (mots_simples, mots_composes) :
        - mots_simples: {ortho: set[phones]} pour les mots SANS séparateur
        - mots_composes: {ortho: set[phones]} pour les mots AVEC séparateur
    """
    simples: dict[str, set[str]] = {}
    composes: dict[str, set[str]] = {}

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            phone = row.get("phone", "").strip()

            if not ortho or not phone:
                continue
            # Ignorer les entrées multi-mots (avec espace)
            if " " in ortho:
                continue

            if _SEPARATORS.search(ortho):
                composes.setdefault(ortho, set()).add(phone)
            else:
                simples.setdefault(ortho, set()).add(phone)

    return simples, composes


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 2 : Trouver les parties orphelines
# ═══════════════════════════════════════════════════════════════════════════════

def split_compound(word: str) -> list[str]:
    """Découpe un mot composé en parties (séparateurs : -, ', ')."""
    return [p for p in _SEPARATORS.split(word) if p]


def find_orphan_parts(
    simples: dict[str, set[str]],
    composes: dict[str, set[str]],
) -> dict[str, list[tuple[str, str]]]:
    """Trouve les parties orphelines des mots composés.

    Returns:
        {partie: [(mot_source, phone_source), ...]}
        Chaque partie orpheline est associée à ses mots sources et phones gold.
    """
    orphans: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for compound, phones in composes.items():
        parts = split_compound(compound)
        for part in parts:
            # Filtrer élisions et parties trop courtes
            if len(part) <= 1:
                continue
            if part in _ELISIONS:
                continue
            # Filtrer les parties déjà présentes comme mot isolé
            if part in simples:
                continue
            # Ajouter chaque phone source
            for phone in phones:
                orphans[part].append((compound, phone))

    return dict(orphans)


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 3 : Déterminer la phonémisation gold (par déduction uniquement)
# ═══════════════════════════════════════════════════════════════════════════════

def determine_gold(
    part: str,
    sources: list[tuple[str, str]],
    simples: dict[str, set[str]],
) -> str | None:
    """Détermine la phonémisation gold d'une partie orpheline.

    Heuristique : pour chaque mot source, essayer de déduire le phone de la
    partie en retirant le phone des parties connues. Vote majoritaire parmi
    les déductions.

    Returns:
        Le phone gold, ou None si indéterminable.
    """
    candidates: list[str] = []

    for compound, compound_phone in sources:
        phone = _deduce_phone_from_compound(part, compound, compound_phone, simples)
        if phone:
            candidates.append(phone)

    if candidates:
        return _majority_vote(candidates)

    return None


def _deduce_phone_from_compound(
    target_part: str,
    compound: str,
    compound_phone: str,
    simples: dict[str, set[str]],
) -> str | None:
    """Essaie de déduire le phone d'une partie à partir du mot composé.

    Pour "X-Y" avec phone "ph_XY" :
    - Si X est connu → phone(Y) = suffixe de ph_XY après phone(X)
    - Si Y est connu → phone(X) = préfixe de ph_XY avant phone(Y)
    """
    parts = split_compound(compound)
    if len(parts) < 2:
        return None

    target_idx = None
    for i, p in enumerate(parts):
        if p == target_part:
            target_idx = i
            break
    if target_idx is None:
        return None

    # Cas simple : 2 parties
    if len(parts) == 2:
        other_idx = 1 - target_idx
        other_part = parts[other_idx]

        # L'autre partie est-elle connue ?
        other_phones = _get_known_phones(other_part, simples)
        if not other_phones:
            return None

        for other_phone in other_phones:
            if target_idx == 1:
                # target est en 2e position → retirer le préfixe
                if compound_phone.startswith(other_phone):
                    result = compound_phone[len(other_phone):]
                    if result:
                        return result
            else:
                # target est en 1e position → retirer le suffixe
                if compound_phone.endswith(other_phone):
                    result = compound_phone[:-len(other_phone)]
                    if result:
                        return result

        return None

    # Cas 3+ parties : essayer de retirer les parties connues autour
    remaining_phone = compound_phone

    # Retirer les parties avant la cible (de gauche à droite)
    for i in range(target_idx):
        p = parts[i]
        if len(p) <= 1 or p in _ELISIONS:
            if remaining_phone:
                remaining_phone = remaining_phone[1:]
            continue
        known = _get_known_phones(p, simples)
        if not known:
            return None
        matched = False
        for kp in known:
            if remaining_phone.startswith(kp):
                remaining_phone = remaining_phone[len(kp):]
                matched = True
                break
        if not matched:
            return None

    # Retirer les parties après la cible (de droite à gauche)
    for i in range(len(parts) - 1, target_idx, -1):
        p = parts[i]
        if len(p) <= 1 or p in _ELISIONS:
            if remaining_phone:
                remaining_phone = remaining_phone[:-1]
            continue
        known = _get_known_phones(p, simples)
        if not known:
            return None
        matched = False
        for kp in known:
            if remaining_phone.endswith(kp):
                remaining_phone = remaining_phone[:-len(kp)]
                matched = True
                break
        if not matched:
            return None

    return remaining_phone if remaining_phone else None


def _get_known_phones(part: str, simples: dict[str, set[str]]) -> list[str]:
    """Retourne les phones connues pour une partie, ou []."""
    if part in simples:
        return sorted(simples[part])
    return []


def _majority_vote(candidates: list[str]) -> str | None:
    """Retourne le candidat le plus fréquent s'il a une majorité."""
    if not candidates:
        return None
    counts: dict[str, int] = defaultdict(int)
    for c in candidates:
        counts[c] += 1
    best = max(counts, key=lambda k: counts[k])
    if counts[best] >= 2 or len(set(candidates)) == 1:
        return best
    if len(candidates) == 1:
        return candidates[0]
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 4 : Charger le modèle BiLSTM
# ═══════════════════════════════════════════════════════════════════════════════

def load_model() -> LecturaG2P:
    """Charge le modèle BiLSTM (sans corrections)."""
    here = Path(__file__).parent
    model_path = here / "modele" / "g2p_model_bilstm_int8.onnx"
    vocab_path = here / "modele" / "g2p_vocab.json"

    if not model_path.exists():
        print(f"ERREUR : modèle non trouvé : {model_path}", file=sys.stderr)
        sys.exit(1)

    return LecturaG2P(model_path, vocab_path=vocab_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 5 : Tester le modèle et construire les corrections
# ═══════════════════════════════════════════════════════════════════════════════

def build_corrections(
    orphans_gold: dict[str, str],
    model: LecturaG2P,
) -> dict[str, str]:
    """Teste le modèle BiLSTM sur les parties orphelines et construit les corrections.

    Returns:
        {word: gold_phone} pour les mots où le modèle se trompe.
    """
    corrections: dict[str, str] = {}

    for word, gold in orphans_gold.items():
        try:
            predicted = model._model.predict(word)
            if predicted:
                predicted = postprocess(word, predicted)
        except Exception:
            predicted = ""

        if predicted != gold:
            corrections[word] = gold

    return corrections


# ═══════════════════════════════════════════════════════════════════════════════
# Étape 6 : Mettre à jour le fichier de corrections JSON
# ═══════════════════════════════════════════════════════════════════════════════

def update_correction_file(
    corrections: dict[str, str],
    dry_run: bool = True,
) -> int:
    """Met à jour le fichier de corrections BiLSTM.

    Returns:
        Nombre de corrections ajoutées.
    """
    here = Path(__file__).parent
    fpath = here / "modele" / "g2p_corrections_bilstm.json"

    if not corrections:
        return 0

    if not fpath.exists():
        print(f"  WARN: Fichier non trouvé : {fpath}", file=sys.stderr)
        return 0

    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    g2p = data.get("g2p", {})
    count_before = len(g2p)

    for word, phone in corrections.items():
        if word not in g2p:
            g2p[word] = phone

    added = len(g2p) - count_before

    if added > 0:
        data["g2p"] = dict(sorted(g2p.items()))

        if dry_run:
            print(f"  [DRY-RUN] {fpath.name}: +{added} corrections")
        else:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"  {fpath.name}: +{added} corrections écrites")

    return len(corrections)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrichir les corrections G2P BiLSTM avec les parties de mots composés.",
    )
    parser.add_argument(
        "--dico",
        type=Path,
        default=Path("/data/work/projets/lectura/workspace/lectura-main/src/lecteur_syllabique/data/dico.csv"),
        help="Chemin vers dico.csv",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Voir les changements sans écrire")
    group.add_argument("--apply", action="store_true", help="Écrire les corrections")

    args = parser.parse_args()

    if not args.dico.exists():
        print(f"ERREUR : dico non trouvé : {args.dico}", file=sys.stderr)
        sys.exit(1)

    # --- Étape 1 : Charger le dico ---
    print("═══ Étape 1 : Chargement du dico ═══")
    simples, composes = load_dico(args.dico)
    print(f"  Mots simples (uniques) : {len(simples):,}")
    print(f"  Mots composés/apostrophe : {len(composes):,}")

    # --- Étape 2 : Trouver les parties orphelines ---
    print("\n═══ Étape 2 : Recherche des parties orphelines ═══")
    orphans = find_orphan_parts(simples, composes)
    print(f"  Parties orphelines trouvées : {len(orphans)}")

    if not orphans:
        print("  Aucune partie orpheline trouvée. Rien à faire.")
        return

    examples = sorted(orphans.keys())[:10]
    print(f"  Exemples : {', '.join(examples)}")

    # --- Étape 3 : Charger le modèle ---
    print("\n═══ Étape 3 : Chargement du modèle BiLSTM ═══")
    model = load_model()
    print(f"  Modèle chargé : {model.backend}")

    # --- Étape 4 : Déterminer les gold ---
    print("\n═══ Étape 4 : Détermination des phonémisations gold ═══")
    orphans_gold: dict[str, str] = {}
    undetermined: list[str] = []

    for part in sorted(orphans.keys()):
        gold = determine_gold(part, orphans[part], simples)
        if gold:
            orphans_gold[part] = gold
        else:
            undetermined.append(part)

    print(f"  Gold déterminés : {len(orphans_gold)}")
    print(f"  Indéterminés    : {len(undetermined)}")

    if undetermined and len(undetermined) <= 20:
        print(f"  Parties indéterminées : {', '.join(undetermined)}")

    for part in sorted(orphans_gold.keys())[:10]:
        sources = orphans[part]
        compound_ex = sources[0][0] if sources else "?"
        print(f"    {part:20} → /{orphans_gold[part]}/  (source: {compound_ex})")

    # --- Étape 5 : Tester le modèle ---
    print("\n═══ Étape 5 : Test du modèle BiLSTM et construction des corrections ═══")
    corrections = build_corrections(orphans_gold, model)
    print(f"  BiLSTM : {len(corrections)} corrections nécessaires")

    for check_word in ["lorsqu", "aujourd", "quelqu", "puisqu", "quoiqu", "jusq"]:
        if check_word in orphans_gold:
            gold = orphans_gold[check_word]
            try:
                pred = model._model.predict(check_word)
                if pred:
                    pred = postprocess(check_word, pred)
            except Exception:
                pred = "?"
            status = "OK" if pred == gold else "ERREUR"
            needs_corr = check_word in corrections
            print(f"  Vérification '{check_word}' → gold: /{gold}/ — BiLSTM: /{pred}/ {status}{' → correction ajoutée' if needs_corr else ''}")

    # --- Étape 6 : Mettre à jour le fichier ---
    print(f"\n═══ Étape 6 : {'[DRY-RUN] ' if args.dry_run else ''}Mise à jour du fichier ═══")
    n_added = update_correction_file(corrections, dry_run=args.dry_run)

    # --- Résumé ---
    print("\n═══ Résumé ═══")
    print(f"  Mots composés analysés    : {len(composes):,}")
    print(f"  Parties orphelines        : {len(orphans)}")
    print(f"  Gold déterminés           : {len(orphans_gold)}")
    print(f"  Corrections BiLSTM        : {n_added}")

    if args.dry_run:
        print("\n  → Relancer avec --apply pour écrire les corrections.")


if __name__ == "__main__":
    main()
