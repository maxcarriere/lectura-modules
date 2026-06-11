#!/usr/bin/env python3
"""Test critique : le G2P comprend-il la structure malgre les fautes ?

Pour chaque paire WiCoPaCo (erronee, corrigee), on tague les deux
versions et on compare les POS position par position.

Questions :
1. Le G2P donne-t-il le meme POS sur le mot errone et le mot corrige ?
   (ex: "les chat" → chat=NOM dans les deux cas ?)
2. Le contexte errone perturbe-t-il le tagging des mots voisins corrects ?
3. Le LexiqueTagger est-il plus stable (meme POS erreur/corrige) ?
4. Apres correction ortho, le POS s'ameliore-t-il ?

On mesure aussi : si on donne d'abord les ancres non-ambigues au G2P
via le tagger hybride, est-ce que ca stabilise les tags des mots ambigus ?
"""

import csv
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
WICOPACO_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/grammaire_wicopaco.tsv"

_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)


def extraire_mots_aligns(phrase_err, phrase_cor):
    """Extrait les mots alignes des deux phrases (meme nombre de tokens)."""
    toks_e = phrase_err.split()
    toks_c = phrase_cor.split()
    if len(toks_e) != len(toks_c):
        return None
    # Extraire les mots (ignorer ponctuation isolee)
    pairs = []
    for te, tc in zip(toks_e, toks_c):
        is_word_e = bool(re.match(r"[\w]", te))
        is_word_c = bool(re.match(r"[\w]", tc))
        if is_word_e and is_word_c:
            pairs.append((te, tc))
    return pairs


def tronquer(phrase, max_mots=30):
    tokens = phrase.split()
    if len(tokens) > max_mots:
        tokens = tokens[:max_mots]
    return " ".join(tokens)


def tag_phrase(tagger, phrase, rich=False):
    """Tague une phrase, retourne list[(mot, pos, nombre, confiance)]."""
    if hasattr(tagger, "tokenize"):
        tokens = tagger.tokenize(phrase)
        words = [t for t, is_w in tokens if is_w]
    else:
        words = [m.group() for m in _MOT_RE.finditer(phrase)]
    if not words:
        return []
    if rich and hasattr(tagger, "tag_words_rich"):
        tags = tagger.tag_words_rich(words)
    else:
        tags = tagger.tag_words(words)
    result = []
    for i, w in enumerate(words):
        if i < len(tags):
            t = tags[i]
            result.append((w, t.get("pos", ""), t.get("nombre", ""), t.get("confiance_pos", 0.5)))
        else:
            result.append((w, "", "", 0.5))
    return result


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur._tagger_lexique import LexiqueTagger
    from lectura_correcteur._utils import LexiqueNormalise

    print("Chargement du lexique...")
    lexique = Lexique(LEXIQUE_DB)
    lex_norm = LexiqueNormalise(lexique)

    # Charger paires accord (erronee, corrigee)
    paires = []
    with open(WICOPACO_TSV, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3 or row[0].startswith("#") or row[0] == "type_erreur":
                continue
            if row[0].strip() != "accord":
                continue
            paires.append((row[1].strip(), row[2].strip()))
            if len(paires) >= 1000:
                break

    print(f"Paires accord chargees : {len(paires)}")

    # Preparer les taggers
    lex_tagger = LexiqueTagger(lex_norm)

    g2p_tagger = None
    hyb_tagger = None
    try:
        from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
        g2p_tagger = creer_adapter_g2p_unifie()
        if g2p_tagger:
            from lectura_correcteur._tagger_hybride import TaggerHybride
            from lectura_correcteur._lm_homophones import LMHomophones
            from pathlib import Path
            lm_path = Path(__file__).resolve().parent.parent / "src" / "lectura_correcteur" / "data" / "homophones_trigrams.db"
            lm_homo = LMHomophones(str(lm_path), lexique=lex_norm) if lm_path.exists() else None
            hyb_tagger = TaggerHybride(g2p_tagger, lex_norm, lm_homophones=lm_homo)
    except Exception as e:
        print(f"  G2P/Hybride indisponible: {e}")

    taggers = [("LexiqueTagger", lex_tagger, False)]
    if g2p_tagger:
        taggers.append(("G2P_V2", g2p_tagger, True))
    if hyb_tagger:
        taggers.append(("Hybride", hyb_tagger, True))

    for tagger_name, tagger, rich in taggers:
        print(f"\n{'='*70}")
        print(f"  {tagger_name}")
        print(f"{'='*70}")

        t0 = time.time()

        # Metriques
        # 1. Stabilite POS : meme POS sur mot errone et corrige ?
        stable_pos = 0
        unstable_pos = 0
        unstable_details = Counter()  # (pos_err, pos_cor) -> count

        # 2. Stabilite POS des voisins (mots inchanges)
        voisin_stable = 0
        voisin_unstable = 0
        voisin_details = Counter()

        # 3. POS correct sur le mot cible errone
        # (le tagger identifie-t-il la bonne categorie malgre la faute ?)
        target_pos_from_lex = 0  # pos(mot_errone) == pos_lexique(mot_corrige)
        target_pos_total = 0

        # 4. NOMBRE correct sur le mot cible
        nombre_correct_err = 0
        nombre_correct_cor = 0
        nombre_total = 0

        # 5. Structure globale : POS sequence similaire ?
        seq_match_ratio_sum = 0
        seq_count = 0

        skipped = 0

        for phrase_err, phrase_cor in paires:
            phrase_err = tronquer(phrase_err)
            phrase_cor = tronquer(phrase_cor)

            tags_err = tag_phrase(tagger, phrase_err, rich)
            tags_cor = tag_phrase(tagger, phrase_cor, rich)

            if len(tags_err) != len(tags_cor):
                skipped += 1
                continue

            n = len(tags_err)
            if n == 0:
                continue

            # Trouver le mot cible (celui qui differe)
            target_idx = None
            for j in range(n):
                if tags_err[j][0].lower() != tags_cor[j][0].lower():
                    target_idx = j
                    break

            if target_idx is None:
                # Pas de difference detectee (peut arriver si tokenisation differe)
                skipped += 1
                continue

            # 1. Stabilite POS du mot cible
            pos_e = tags_err[target_idx][1]
            pos_c = tags_cor[target_idx][1]
            if pos_e == pos_c:
                stable_pos += 1
            else:
                unstable_pos += 1
                unstable_details[(pos_e, pos_c)] += 1

            # 2. Stabilite POS des voisins
            for j in range(n):
                if j == target_idx:
                    continue
                if tags_err[j][0].lower() != tags_cor[j][0].lower():
                    continue  # mot aussi change
                if tags_err[j][1] == tags_cor[j][1]:
                    voisin_stable += 1
                else:
                    voisin_unstable += 1
                    voisin_details[(tags_err[j][0].lower(), tags_err[j][1], tags_cor[j][1])] += 1

            # 3. POS correct sur mot errone vs POS lexique du mot corrige
            mot_cor = tags_cor[target_idx][0].lower()
            infos_cor = lex_norm.info(mot_cor)
            if infos_cor:
                best = max(infos_cor, key=lambda e: float(e.get("freq") or 0))
                lex_pos = best.get("cgram", "")
                if lex_pos:
                    target_pos_total += 1
                    base_e = pos_e.split(":")[0] if pos_e else ""
                    base_l = lex_pos.split(":")[0]
                    if base_e == base_l:
                        target_pos_from_lex += 1

            # 4. NOMBRE sur mot cible
            # Le nombre attendu est celui du mot CORRIGE dans le lexique
            nombre_err = tags_err[target_idx][2]
            nombre_cor_tag = tags_cor[target_idx][2]
            if infos_cor:
                best_nombre = max(infos_cor, key=lambda e: float(e.get("freq") or 0))
                nombre_lex = best_nombre.get("nombre", "")
                if nombre_lex:
                    # Normaliser
                    if nombre_lex in ("Sing", "s"):
                        nombre_lex_n = "s"
                    elif nombre_lex in ("Plur", "p"):
                        nombre_lex_n = "p"
                    else:
                        nombre_lex_n = nombre_lex
                    nombre_total += 1
                    if nombre_err == nombre_lex_n:
                        nombre_correct_err += 1
                    if nombre_cor_tag == nombre_lex_n:
                        nombre_correct_cor += 1

            # 5. Sequence POS
            matches = sum(1 for j in range(n) if tags_err[j][1] == tags_cor[j][1])
            seq_match_ratio_sum += matches / n
            seq_count += 1

        elapsed = time.time() - t0

        # Affichage
        total_target = stable_pos + unstable_pos
        print(f"\n  Temps: {elapsed:.1f}s  (skipped: {skipped})")

        pct_stable = 100 * stable_pos / total_target if total_target > 0 else 0
        print(f"\n  1. STABILITE POS mot cible (errone vs corrige)")
        print(f"     Stable: {stable_pos}/{total_target} ({pct_stable:.1f}%)")
        print(f"     Instable: {unstable_pos}/{total_target} ({100-pct_stable:.1f}%)")
        if unstable_details:
            print(f"     Top changements POS :")
            for (pe, pc), cnt in unstable_details.most_common(10):
                print(f"       {pe:12s} -> {pc:12s} : {cnt}")

        total_v = voisin_stable + voisin_unstable
        pct_v = 100 * voisin_stable / total_v if total_v > 0 else 0
        print(f"\n  2. STABILITE POS voisins (mots inchanges)")
        print(f"     Stable: {voisin_stable}/{total_v} ({pct_v:.1f}%)")
        if voisin_details:
            print(f"     Top instabilites :")
            for (mot, pe, pc), cnt in voisin_details.most_common(10):
                print(f"       {mot:12s} {pe:12s} -> {pc:12s} : {cnt}")

        pct_t = 100 * target_pos_from_lex / target_pos_total if target_pos_total > 0 else 0
        print(f"\n  3. POS CORRECT sur mot errone (vs lexique du mot corrige)")
        print(f"     Correct: {target_pos_from_lex}/{target_pos_total} ({pct_t:.1f}%)")

        if nombre_total > 0:
            pct_ne = 100 * nombre_correct_err / nombre_total
            pct_nc = 100 * nombre_correct_cor / nombre_total
            print(f"\n  4. NOMBRE correct sur mot cible")
            print(f"     Sur phrase erronee:  {nombre_correct_err}/{nombre_total} ({pct_ne:.1f}%)")
            print(f"     Sur phrase corrigee: {nombre_correct_cor}/{nombre_total} ({pct_nc:.1f}%)")

        if seq_count > 0:
            avg_seq = seq_match_ratio_sum / seq_count
            print(f"\n  5. SEQUENCE POS : {100*avg_seq:.1f}% des positions ont le meme POS")
            print(f"     (errone vs corrige, moyenne par phrase)")


if __name__ == "__main__":
    main()
