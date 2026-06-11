#!/usr/bin/env python3
"""Evaluation de la qualite POS/MORPHO sur phrases erronees.

Compare 3 taggers sur des phrases WiCoPaCo (avec erreurs) :
- LexiqueTagger (baseline)
- G2P Unifie V2
- TaggerHybride (G2P + overrides + boost)

Mesure la qualite POS en comparant avec des ancres fiables :
1. Mots non-ambigus (1 seul POS dans le lexique)
2. Mots-outils (POS connu a priori)
3. Determinants/pronoms (POS evident par forme)

Mesure aussi la qualite du trait NOMBRE (s/p) car c'est le
trait critique pour les regles d'accord.
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

# Ancres POS : mots dont le POS est certain
_ANCRES_POS = {
    # Determinants
    "le": "ART", "la": "ART", "les": "ART", "un": "ART:ind", "une": "ART:ind",
    "des": "ART", "du": "ART", "au": "PRE", "aux": "ART:def",
    "ce": "PRO:dem", "cette": "ADJ:dem", "ces": "ADJ:dem",
    "mon": "ADJ:pos", "ma": "ADJ:pos", "ton": "ADJ:pos", "ta": "ADJ:pos",
    "son": "ADJ:pos", "sa": "ADJ:pos", "mes": "ADJ:pos", "tes": "ADJ:pos",
    "ses": "ADJ:pos", "nos": "ADJ:pos", "vos": "ADJ:pos", "leurs": "ADJ:pos",
    # Prepositions
    "de": "PRE", "dans": "PRE", "par": "PRE", "pour": "PRE",
    "avec": "PRE", "sur": "PRE", "sous": "PRE", "entre": "PRE",
    "chez": "PRE", "vers": "PRE", "depuis": "PRE", "sans": "PRE",
    # Conjonctions
    "et": "CON", "ou": "CON", "mais": "CON", "donc": "CON",
    "car": "CON", "ni": "CON", "que": "CON",
    # Pronoms
    "je": "PRO:per", "tu": "PRO:per", "il": "PRO:per", "elle": "PRO:per",
    "nous": "PRO:per", "vous": "PRO:per", "ils": "PRO:per", "elles": "PRO:per",
    "on": "PRO:per", "se": "PRO:per",
    # Auxiliaires
    "est": "AUX", "sont": "AUX", "était": "AUX", "étaient": "AUX",
    "sera": "AUX", "seront": "AUX", "a": "AUX", "ont": "AUX",
    "avait": "AUX", "avaient": "AUX",
    # Adverbes tres frequents
    "ne": "ADV", "pas": "ADV", "plus": "ADV", "très": "ADV",
    "bien": "ADV", "aussi": "ADV", "encore": "ADV",
}

# Nombre attendu pour certains mots (ancres nombre)
_ANCRES_NOMBRE = {
    "le": "s", "la": "s", "un": "s", "une": "s", "ce": "s", "cette": "s",
    "mon": "s", "ma": "s", "ton": "s", "ta": "s", "son": "s", "sa": "s",
    "les": "p", "des": "p", "ces": "p", "mes": "p", "tes": "p",
    "ses": "p", "nos": "p", "vos": "p", "leurs": "p",
    "il": "s", "elle": "s", "ils": "p", "elles": "p",
    "est": "s", "sont": "p", "était": "s", "étaient": "p",
    "a": "s", "ont": "p", "avait": "s", "avaient": "p",
}


def extraire_mots(texte):
    return [m.group() for m in _MOT_RE.finditer(texte)]


def tronquer_phrase(phrase, max_mots=30):
    """Tronque les phrases trop longues pour le tagging."""
    tokens = phrase.split()
    if len(tokens) > max_mots:
        tokens = tokens[:max_mots]
    return " ".join(tokens)


def pos_compatible(tag_predit, tag_attendu):
    """Verifie compatibilite POS (prefixe)."""
    if not tag_predit or not tag_attendu:
        return False
    # Match exact
    if tag_predit == tag_attendu:
        return True
    # Match prefixe (ART:ind match ART, PRO:per match PRO)
    base_p = tag_predit.split(":")[0]
    base_a = tag_attendu.split(":")[0]
    return base_p == base_a


def evaluer_tagger(tagger, phrases, lexique, label=""):
    """Evalue un tagger sur un echantillon de phrases erronees."""
    # Stats POS
    pos_total = 0
    pos_correct = 0
    pos_errors = Counter()  # (attendu, predit) -> count

    # Stats POS par type de mot
    pos_par_type = defaultdict(lambda: {"total": 0, "ok": 0})

    # Stats nombre
    nombre_total = 0
    nombre_correct = 0
    nombre_errors = Counter()

    # Stats non-ambigus (1 seul POS dans lexique)
    unambig_total = 0
    unambig_correct = 0

    # Stats confiance
    confiances = []

    tokenizer = tagger.tokenize if hasattr(tagger, "tokenize") else None
    has_rich = hasattr(tagger, "tag_words_rich")

    for phrase in phrases:
        phrase = tronquer_phrase(phrase)
        if tokenizer:
            tokens_raw = tokenizer(phrase)
            words = [t for t, is_w in tokens_raw if is_w]
        else:
            words = extraire_mots(phrase)

        if not words:
            continue

        if has_rich:
            tags = tagger.tag_words_rich(words)
        else:
            tags = tagger.tag_words(words)

        for j, word in enumerate(words):
            if j >= len(tags):
                break

            low = word.lower()
            tag = tags[j]
            pos_predit = tag.get("pos", "")
            nombre_predit = tag.get("nombre", "")
            confiance = tag.get("confiance_pos", 0.5)

            # 1. Evaluation POS sur ancres
            if low in _ANCRES_POS:
                pos_attendu = _ANCRES_POS[low]
                pos_total += 1
                cat = pos_attendu.split(":")[0]
                pos_par_type[cat]["total"] += 1
                if pos_compatible(pos_predit, pos_attendu):
                    pos_correct += 1
                    pos_par_type[cat]["ok"] += 1
                else:
                    pos_errors[(pos_attendu, pos_predit)] += 1

            # 2. Evaluation POS sur non-ambigus
            infos = lexique.info(low)
            if infos:
                cgrams = {e.get("cgram") for e in infos if e.get("cgram")}
                if len(cgrams) == 1:
                    unambig_total += 1
                    expected_pos = next(iter(cgrams))
                    if pos_compatible(pos_predit, expected_pos):
                        unambig_correct += 1

            # 3. Evaluation NOMBRE sur ancres
            if low in _ANCRES_NOMBRE:
                nombre_attendu = _ANCRES_NOMBRE[low]
                nombre_total += 1
                if nombre_predit == nombre_attendu:
                    nombre_correct += 1
                else:
                    nombre_errors[(low, nombre_attendu, nombre_predit)] += 1

            # 4. Confiance
            if has_rich:
                confiances.append(confiance)

    return {
        "pos_total": pos_total,
        "pos_correct": pos_correct,
        "pos_errors": pos_errors,
        "pos_par_type": dict(pos_par_type),
        "nombre_total": nombre_total,
        "nombre_correct": nombre_correct,
        "nombre_errors": nombre_errors,
        "unambig_total": unambig_total,
        "unambig_correct": unambig_correct,
        "confiances": confiances,
    }


def print_results(label, r):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # POS ancres
    pct = 100 * r["pos_correct"] / r["pos_total"] if r["pos_total"] > 0 else 0
    print(f"\n  POS (ancres) : {r['pos_correct']}/{r['pos_total']} ({pct:.1f}%)")

    print(f"    Par type :")
    for cat in sorted(r["pos_par_type"].keys()):
        s = r["pos_par_type"][cat]
        p = 100 * s["ok"] / s["total"] if s["total"] > 0 else 0
        print(f"      {cat:10s} : {s['ok']:4d}/{s['total']:4d} ({p:.1f}%)")

    print(f"    Top erreurs POS :")
    for (att, pred), cnt in r["pos_errors"].most_common(15):
        print(f"      {att:12s} -> {pred:12s} : {cnt}")

    # POS non-ambigus
    pct_ua = 100 * r["unambig_correct"] / r["unambig_total"] if r["unambig_total"] > 0 else 0
    print(f"\n  POS (non-ambigus lexique) : {r['unambig_correct']}/{r['unambig_total']} ({pct_ua:.1f}%)")

    # NOMBRE
    pct_n = 100 * r["nombre_correct"] / r["nombre_total"] if r["nombre_total"] > 0 else 0
    print(f"\n  NOMBRE (ancres) : {r['nombre_correct']}/{r['nombre_total']} ({pct_n:.1f}%)")
    print(f"    Top erreurs NOMBRE :")
    for (mot, att, pred), cnt in r["nombre_errors"].most_common(15):
        print(f"      {mot:12s} attendu={att} predit={pred:5s} : {cnt}")

    # Confiance
    if r["confiances"]:
        confs = r["confiances"]
        avg = sum(confs) / len(confs)
        high = sum(1 for c in confs if c >= 0.9) / len(confs)
        low = sum(1 for c in confs if c < 0.5) / len(confs)
        print(f"\n  Confiance POS : moy={avg:.3f}  >=0.9={100*high:.1f}%  <0.5={100*low:.1f}%")


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur._tagger_lexique import LexiqueTagger
    from lectura_correcteur._utils import LexiqueNormalise

    print("Chargement du lexique...")
    lexique = Lexique(LEXIQUE_DB)
    lex_norm = LexiqueNormalise(lexique)

    # Charger un echantillon de phrases erronees (accord)
    phrases = []
    with open(WICOPACO_TSV, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3 or row[0].startswith("#") or row[0] == "type_erreur":
                continue
            phrases.append(row[1].strip())  # phrase erronee
            if len(phrases) >= 2000:
                break

    print(f"Phrases chargees : {len(phrases)}")

    # 1. LexiqueTagger
    print("\nEvaluation LexiqueTagger...")
    t0 = time.time()
    lex_tagger = LexiqueTagger(lex_norm)
    r_lex = evaluer_tagger(lex_tagger, phrases, lex_norm, "LexiqueTagger")
    print(f"  Temps: {time.time()-t0:.1f}s")
    print_results("LEXIQUE TAGGER", r_lex)

    # 2. G2P Unifie V2
    print("\nEvaluation G2P Unifie V2...")
    try:
        from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
        g2p_tagger = creer_adapter_g2p_unifie()
        if g2p_tagger:
            t0 = time.time()
            r_g2p = evaluer_tagger(g2p_tagger, phrases, lex_norm, "G2P Unifie V2")
            print(f"  Temps: {time.time()-t0:.1f}s")
            print_results("G2P UNIFIE V2", r_g2p)
        else:
            print("  G2P indisponible")
            r_g2p = None
    except Exception as e:
        print(f"  Erreur G2P: {e}")
        r_g2p = None

    # 3. TaggerHybride
    print("\nEvaluation TaggerHybride...")
    try:
        from lectura_correcteur._tagger_hybride import TaggerHybride
        from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
        g2p_adapter = creer_adapter_g2p_unifie()
        if g2p_adapter:
            # Charger LM homophones pour le check ambiguite
            from lectura_correcteur._lm_homophones import LMHomophones
            from pathlib import Path
            lm_path = Path(__file__).resolve().parent.parent / "src" / "lectura_correcteur" / "data" / "homophones_trigrams.db"
            lm_homo = LMHomophones(str(lm_path), lexique=lex_norm) if lm_path.exists() else None

            hyb_tagger = TaggerHybride(g2p_adapter, lex_norm, lm_homophones=lm_homo)
            t0 = time.time()
            r_hyb = evaluer_tagger(hyb_tagger, phrases, lex_norm, "TaggerHybride")
            print(f"  Temps: {time.time()-t0:.1f}s")
            print_results("TAGGER HYBRIDE (G2P + overrides + boost)", r_hyb)
        else:
            print("  G2P indisponible")
            r_hyb = None
    except Exception as e:
        print(f"  Erreur Hybride: {e}")
        import traceback
        traceback.print_exc()
        r_hyb = None

    # Comparaison
    print(f"\n{'='*60}")
    print("  COMPARAISON")
    print(f"{'='*60}")
    print(f"\n  {'Metrique':<30s} {'Lexique':>10s} {'G2P':>10s} {'Hybride':>10s}")
    print(f"  {'-'*60}")

    def fmt(r, key_ok, key_total):
        if r is None:
            return "N/A"
        t = r[key_total]
        o = r[key_ok]
        return f"{100*o/t:.1f}%" if t > 0 else "N/A"

    print(f"  {'POS ancres':<30s} {fmt(r_lex, 'pos_correct', 'pos_total'):>10s} {fmt(r_g2p, 'pos_correct', 'pos_total') if r_g2p else 'N/A':>10s} {fmt(r_hyb, 'pos_correct', 'pos_total') if r_hyb else 'N/A':>10s}")
    print(f"  {'POS non-ambigus':<30s} {fmt(r_lex, 'unambig_correct', 'unambig_total'):>10s} {fmt(r_g2p, 'unambig_correct', 'unambig_total') if r_g2p else 'N/A':>10s} {fmt(r_hyb, 'unambig_correct', 'unambig_total') if r_hyb else 'N/A':>10s}")
    print(f"  {'NOMBRE ancres':<30s} {fmt(r_lex, 'nombre_correct', 'nombre_total'):>10s} {fmt(r_g2p, 'nombre_correct', 'nombre_total') if r_g2p else 'N/A':>10s} {fmt(r_hyb, 'nombre_correct', 'nombre_total') if r_hyb else 'N/A':>10s}")

    if r_hyb and r_hyb["confiances"]:
        confs = r_hyb["confiances"]
        avg = sum(confs) / len(confs)
        print(f"\n  Confiance moyenne hybride: {avg:.3f}")
        buckets = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]
        for i in range(len(buckets)-1):
            lo, hi = buckets[i], buckets[i+1]
            cnt = sum(1 for c in confs if lo <= c < hi)
            print(f"    [{lo:.2f}, {hi:.2f}) : {cnt:5d} ({100*cnt/len(confs):.1f}%)")
        cnt_1 = sum(1 for c in confs if c >= 1.0)
        print(f"    [1.00, 1.00] : {cnt_1:5d} ({100*cnt_1/len(confs):.1f}%)")


if __name__ == "__main__":
    main()
