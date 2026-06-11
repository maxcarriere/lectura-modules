#!/usr/bin/env python3
"""Diagnostic POS/MORPHO — compare LexiqueTagger vs G2P Unifie V2.

Evalue la qualite de l'etiquetage POS/MORPHO sur les cas en erreur du
benchmark. Identifie les mots-ancres mal etiquetes, les incoherences
n-gram, et quantifie le potentiel d'amelioration.

Usage :
    python scripts/diagnostic_pos_morpho.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura.db"

# Mots-ancres : POS non-ambigu (ou presque)
ANCRES_POS = {
    # Pronoms personnels sujets
    "je": "PRO:per", "j'": "PRO:per", "tu": "PRO:per",
    "il": "PRO:per", "elle": "PRO:per", "on": "PRO:per",
    "nous": "PRO:per", "vous": "PRO:per",
    "ils": "PRO:per", "elles": "PRO:per",
    # Articles
    "le": "ART:def", "la": "ART:def", "les": "ART:def", "l'": "ART:def",
    "un": "ART:ind", "une": "ART:ind", "des": "ART:ind",
    "du": "ART", "au": "PRE", "aux": "ART:def",
    # Determinants possessifs
    "mon": "ADJ:pos", "ton": "ADJ:pos", "son": "ADJ:pos",
    "ma": "ADJ:pos", "ta": "ADJ:pos", "sa": "ADJ:pos",
    "mes": "ADJ:pos", "tes": "ADJ:pos", "ses": "ADJ:pos",
    "notre": "ADJ:pos", "votre": "ADJ:pos",
    "nos": "ADJ:pos", "vos": "ADJ:pos", "leur": "ADJ:pos",
    # Prepositions
    "de": "PRE", "dans": "PRE", "sur": "PRE",
    "pour": "PRE", "par": "PRE", "avec": "PRE",
    "chez": "PRE", "sans": "PRE", "entre": "PRE",
    # Conjonctions
    "et": "CON", "ou": "CON", "mais": "CON",
    "que": "CON", "qui": "PRO:rel",
    # Auxiliaires (formes courantes)
    "est": "AUX", "sont": "AUX", "suis": "AUX",
    "a": "AUX", "ont": "AUX", "ai": "AUX",
    "avons": "AUX", "avez": "AUX",
    "sommes": "AUX", "etes": "AUX", "êtes": "AUX",
    # Negation
    "ne": "ADV", "n'": "ADV", "pas": "ADV",
}

# Contraintes de succession impossibles apres un pronom personnel sujet
IMPOSSIBLE_APRES_PRO_SUJET = {"NOM", "ADJ", "ART:def", "ART:ind", "ART", "ADJ:pos"}

# Morpho attendue pour les pronoms
ANCRES_MORPHO = {
    "je": {"personne": "1", "nombre": "s"},
    "tu": {"personne": "2", "nombre": "s"},
    "il": {"personne": "3", "nombre": "s", "genre": "m"},
    "elle": {"personne": "3", "nombre": "s", "genre": "f"},
    "on": {"personne": "3", "nombre": "s"},
    "nous": {"personne": "1", "nombre": "p"},
    "vous": {"personne": "2", "nombre": "p"},
    "ils": {"personne": "3", "nombre": "p", "genre": "m"},
    "elles": {"personne": "3", "nombre": "p", "genre": "f"},
}


def tokenize_simple(text):
    """Tokenisation simple pour le diagnostic."""
    import re
    tokens = re.findall(
        r"(?:[dlnmtscjqDLNMTSCJQ]|[Qq]u|[Ll]orsqu|[Pp]uisqu)['\u2019]"
        r"|[\w]+(?:-[\w]+)*"
        r"|[^\s\w]",
        text,
    )
    return tokens


def is_punct(tok):
    return all(not c.isalnum() for c in tok)


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur._tagger_lexique import LexiqueTagger
    from lectura_correcteur._utils import LexiqueNormalise
    from lectura_correcteur._pos_ngram import PosNgram

    lexique_raw = Lexique(LEXIQUE_DB)
    lexique = LexiqueNormalise(lexique_raw)
    lex_tagger = LexiqueTagger(lexique)

    # Charger G2P si disponible
    try:
        from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
        g2p_tagger = creer_adapter_g2p_unifie()
    except Exception:
        g2p_tagger = None

    # Charger POS n-gram
    try:
        from pathlib import Path
        db_path = Path(__file__).resolve().parent.parent / "src" / "lectura_correcteur" / "data" / "pos_ngram.db"
        pos_ngram = PosNgram(str(db_path))
    except Exception:
        pos_ngram = None

    # Charger les corpus
    from scripts.benchmark.run_benchmark import charger_corpus_evaluation, charger_corpus_benchmark
    corpus = charger_corpus_evaluation() + charger_corpus_benchmark()

    print("=" * 78)
    print("  DIAGNOSTIC POS / MORPHO")
    print("=" * 78)
    print(f"  Corpus : {len(corpus)} cas")
    print(f"  LexiqueTagger : oui")
    print(f"  G2P Unifie V2 : {'oui' if g2p_tagger else 'non'}")
    print(f"  POS n-gram    : {'oui' if pos_ngram else 'non'}")
    print()

    # =====================================================================
    # 1. Qualite POS sur les phrases du benchmark
    # =====================================================================
    print("=" * 78)
    print("  1. COMPARAISON POS : LexiqueTagger vs G2P sur les phrases attendues")
    print("=" * 78)
    print()

    # On etiquette les phrases ATTENDUES (pas erronees) pour avoir une ref
    lex_pos_stats = Counter()
    g2p_pos_stats = Counter()
    ancre_erreurs_lex = []
    ancre_erreurs_g2p = []
    ngram_violations = []
    total_tokens = 0
    total_ancres = 0
    lex_ancre_ok = 0
    g2p_ancre_ok = 0
    lex_g2p_agree = 0
    lex_g2p_disagree = 0
    disagree_details = Counter()  # (lex_pos, g2p_pos) -> count

    for cas in corpus:
        phrase_attendue = cas.attendu[0]
        tokens = tokenize_simple(phrase_attendue)
        word_tokens = [t for t in tokens if not is_punct(t)]

        if not word_tokens:
            continue

        # LexiqueTagger
        lex_tags = lex_tagger.tag_words(word_tokens)

        # G2P
        g2p_tags = None
        if g2p_tagger:
            g2p_tags = g2p_tagger.tag_words_rich(word_tokens)

        total_tokens += len(word_tokens)

        for j, w in enumerate(word_tokens):
            low = w.lower()
            lex_pos = lex_tags[j].get("pos", "") if j < len(lex_tags) else ""
            g2p_pos = g2p_tags[j].get("pos", "") if g2p_tags and j < len(g2p_tags) else ""

            lex_pos_stats[lex_pos] += 1
            if g2p_pos:
                g2p_pos_stats[g2p_pos] += 1

            # Comparaison
            if g2p_pos and lex_pos:
                if lex_pos == g2p_pos:
                    lex_g2p_agree += 1
                else:
                    lex_g2p_disagree += 1
                    disagree_details[(low, lex_pos, g2p_pos)] += 1

            # Verification mots-ancres
            if low in ANCRES_POS:
                total_ancres += 1
                expected = ANCRES_POS[low]
                exp_base = expected.split(":")[0]

                lex_base = lex_pos.split(":")[0]
                if lex_base == exp_base or lex_pos == expected:
                    lex_ancre_ok += 1
                else:
                    ancre_erreurs_lex.append((phrase_attendue, w, expected, lex_pos))

                if g2p_pos:
                    g2p_base = g2p_pos.split(":")[0]
                    if g2p_base == exp_base or g2p_pos == expected:
                        g2p_ancre_ok += 1
                    else:
                        ancre_erreurs_g2p.append((phrase_attendue, w, expected, g2p_pos))

            # Contrainte n-gram : apres PRO:per sujet, pas de NOM/ADJ direct
            if (
                j > 0
                and word_tokens[j - 1].lower() in ANCRES_MORPHO
                and lex_pos.split(":")[0] in IMPOSSIBLE_APRES_PRO_SUJET
            ):
                ngram_violations.append((
                    phrase_attendue,
                    word_tokens[j - 1], w, lex_pos,
                ))

    print(f"  Tokens analyses : {total_tokens}")
    print(f"  Mots-ancres     : {total_ancres}")
    print()

    print(f"  Ancres correctes LexiqueTagger : {lex_ancre_ok}/{total_ancres} ({100*lex_ancre_ok/max(total_ancres,1):.1f}%)")
    if g2p_tagger:
        print(f"  Ancres correctes G2P           : {g2p_ancre_ok}/{total_ancres} ({100*g2p_ancre_ok/max(total_ancres,1):.1f}%)")
    print()

    if g2p_tagger:
        total_both = lex_g2p_agree + lex_g2p_disagree
        print(f"  Accord Lex/G2P  : {lex_g2p_agree}/{total_both} ({100*lex_g2p_agree/max(total_both,1):.1f}%)")
        print(f"  Desaccords      : {lex_g2p_disagree}")
        print()

    # =====================================================================
    # 2. Erreurs ancres LexiqueTagger
    # =====================================================================
    print("-" * 78)
    print(f"  Erreurs ancres LexiqueTagger ({len(ancre_erreurs_lex)}) :")
    for phrase, mot, attendu, obtenu in ancre_erreurs_lex[:20]:
        print(f"    '{mot}' : attendu={attendu}, obtenu={obtenu}")
        print(f"      dans : {phrase}")
    if len(ancre_erreurs_lex) > 20:
        print(f"    ... et {len(ancre_erreurs_lex) - 20} autres")
    print()

    if ancre_erreurs_g2p:
        print("-" * 78)
        print(f"  Erreurs ancres G2P ({len(ancre_erreurs_g2p)}) :")
        for phrase, mot, attendu, obtenu in ancre_erreurs_g2p[:20]:
            print(f"    '{mot}' : attendu={attendu}, obtenu={obtenu}")
            print(f"      dans : {phrase}")
        if len(ancre_erreurs_g2p) > 20:
            print(f"    ... et {len(ancre_erreurs_g2p) - 20} autres")
        print()

    # =====================================================================
    # 3. Violations de contraintes n-gram
    # =====================================================================
    print("-" * 78)
    print(f"  Violations PRO_sujet + NOM/ADJ/ART (LexiqueTagger) : {len(ngram_violations)}")
    for phrase, pro, mot, pos in ngram_violations[:15]:
        print(f"    '{pro}' + '{mot}'={pos}")
        print(f"      dans : {phrase}")
    print()

    # =====================================================================
    # 4. Top desaccords Lex vs G2P
    # =====================================================================
    if disagree_details:
        print("-" * 78)
        print(f"  Top 20 desaccords Lex vs G2P (sur mots-types) :")
        for (mot, lex_p, g2p_p), cnt in disagree_details.most_common(20):
            print(f"    '{mot}' : Lex={lex_p:12s}  G2P={g2p_p:12s}  (x{cnt})")
        print()

    # =====================================================================
    # 5. Distribution POS
    # =====================================================================
    print("-" * 78)
    print("  Distribution POS LexiqueTagger :")
    for pos, cnt in lex_pos_stats.most_common():
        print(f"    {pos:15s} : {cnt:5d}")
    print()

    if g2p_pos_stats:
        print("  Distribution POS G2P :")
        for pos, cnt in g2p_pos_stats.most_common():
            print(f"    {pos:15s} : {cnt:5d}")
        print()

    # =====================================================================
    # 6. Diagnostic sur les cas en ERREUR du benchmark
    # =====================================================================
    print("=" * 78)
    print("  2. POS DES CAS EN ERREUR (phrase erronee vs phrase attendue)")
    print("=" * 78)
    print()

    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig
    config = CorrecteurConfig(activer_scoring=True)
    correcteur = Correcteur(lexique_raw, config=config)

    n_err = 0
    pos_mismatch_in_errors = []

    for cas in corpus:
        result = correcteur.corriger(cas.entree)
        match = any(
            result.phrase_corrigee.lower().strip().rstrip(".")
            == att.lower().strip().rstrip(".")
            for att in cas.attendu
        )
        if match:
            continue
        n_err += 1

        # Etiqueter la phrase erronee et la phrase attendue
        tok_err = tokenize_simple(cas.entree)
        tok_att = tokenize_simple(cas.attendu[0])
        words_err = [t for t in tok_err if not is_punct(t)]
        words_att = [t for t in tok_att if not is_punct(t)]

        lex_err = lex_tagger.tag_words(words_err)
        lex_att = lex_tagger.tag_words(words_att)

        g2p_err = g2p_tagger.tag_words_rich(words_err) if g2p_tagger else None
        g2p_att = g2p_tagger.tag_words_rich(words_att) if g2p_tagger else None

        if n_err <= 30:
            print(f"  [{cas.categorie}] {cas.entree}")
            print(f"  ATT: {cas.attendu[0]}")
            print(f"  OUT: {result.phrase_corrigee}")
            print(f"  {'mot':15s} {'Lex(err)':12s} {'G2P(err)':12s} {'Lex(att)':12s} {'G2P(att)':12s} {'morpho_g2p':15s}")
            max_len = max(len(words_err), len(words_att))
            for j in range(max_len):
                w_e = words_err[j] if j < len(words_err) else "---"
                w_a = words_att[j] if j < len(words_att) else "---"
                lp_e = lex_err[j].get("pos", "") if j < len(lex_err) else ""
                gp_e = g2p_err[j].get("pos", "") if g2p_err and j < len(g2p_err) else ""
                lp_a = lex_att[j].get("pos", "") if j < len(lex_att) else ""
                gp_a = g2p_att[j].get("pos", "") if g2p_att and j < len(g2p_att) else ""

                # Morpho G2P (err)
                morpho_str = ""
                if g2p_err and j < len(g2p_err):
                    d = g2p_err[j]
                    parts = []
                    for k in ("nombre", "genre", "personne"):
                        v = d.get(k, "")
                        if v:
                            parts.append(f"{k[0]}={v}")
                    morpho_str = " ".join(parts)

                flag = ""
                if w_e != w_a:
                    flag = " <<<"
                elif lp_e != lp_a:
                    flag = " [POS diff]"

                label = w_e if w_e == w_a else f"{w_e}/{w_a}"
                print(f"  {label:15s} {lp_e:12s} {gp_e:12s} {lp_a:12s} {gp_a:12s} {morpho_str:15s}{flag}")

            print()

    print(f"  Total cas en erreur : {n_err}")
    print()

    # =====================================================================
    # 7. Impact potentiel des mots-ancres fixes
    # =====================================================================
    print("=" * 78)
    print("  3. IMPACT POTENTIEL : sequencage par ponctuation")
    print("=" * 78)
    print()

    # Mesurer la longueur des segments entre ponctuations
    seg_lengths = []
    for cas in corpus:
        tokens = tokenize_simple(cas.attendu[0])
        seg_len = 0
        for t in tokens:
            if is_punct(t):
                if seg_len > 0:
                    seg_lengths.append(seg_len)
                seg_len = 0
            else:
                seg_len += 1
        if seg_len > 0:
            seg_lengths.append(seg_len)

    if seg_lengths:
        avg = sum(seg_lengths) / len(seg_lengths)
        print(f"  Segments (entre ponctuations) : {len(seg_lengths)}")
        print(f"  Longueur moyenne : {avg:.1f} mots")
        print(f"  Longueur max     : {max(seg_lengths)} mots")
        print(f"  Longueur mediane : {sorted(seg_lengths)[len(seg_lengths)//2]} mots")
        distrib = Counter()
        for l in seg_lengths:
            if l <= 3:
                distrib["1-3"] += 1
            elif l <= 6:
                distrib["4-6"] += 1
            elif l <= 10:
                distrib["7-10"] += 1
            else:
                distrib["11+"] += 1
        for bucket in ["1-3", "4-6", "7-10", "11+"]:
            print(f"    {bucket:5s} : {distrib[bucket]:4d} ({100*distrib[bucket]/len(seg_lengths):.0f}%)")
    print()

    # =====================================================================
    # 8. Confiance G2P sur les mots-ancres vs mots de contenu
    # =====================================================================
    if g2p_tagger:
        print("=" * 78)
        print("  4. CONFIANCE G2P : ancres vs contenu")
        print("=" * 78)
        print()

        conf_ancres = []
        conf_contenu = []
        for cas in corpus:
            tokens = tokenize_simple(cas.attendu[0])
            words = [t for t in tokens if not is_punct(t)]
            if not words:
                continue
            tags = g2p_tagger.tag_words_rich(words)
            for j, w in enumerate(words):
                if j >= len(tags):
                    break
                c = tags[j].get("confiance_pos", 1.0)
                if w.lower() in ANCRES_POS:
                    conf_ancres.append(c)
                else:
                    conf_contenu.append(c)

        if conf_ancres:
            print(f"  Confiance moyenne ancres  : {sum(conf_ancres)/len(conf_ancres):.3f} ({len(conf_ancres)} mots)")
        if conf_contenu:
            print(f"  Confiance moyenne contenu : {sum(conf_contenu)/len(conf_contenu):.3f} ({len(conf_contenu)} mots)")
        print()

    if pos_ngram:
        pos_ngram.close()


if __name__ == "__main__":
    main()
