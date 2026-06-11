#!/usr/bin/env python3
"""Diagnostic FP : tracer l'orthographe pour 'des'."""
import os, sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Phonemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Graphemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Tokeniseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/G2P-Pipeline/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Formules/src")

LEXIQUE_DB = os.path.join(
    _PROJECT_ROOT, "src", "lectura_correcteur", "data", "lexique_correcteur.db",
)

from lectura_correcteur._lexique_lite import LexiqueLite
from lectura_correcteur import CorrecteurV5, CorrecteurV5Config

lexique = LexiqueLite(LEXIQUE_DB)

# Verifier "des" dans le lexique
print(f"'des' dans lexique: {lexique.existe('des')}")
print(f"'du' dans lexique: {lexique.existe('du')}")
infos_des = lexique.info('des')
infos_du = lexique.info('du')
print(f"Info 'des': {infos_des}")
print(f"Info 'du': {infos_du}")
print()

cfg = CorrecteurV5Config()
correcteur = CorrecteurV5(lexique, config=cfg)

# Tracer l'etape 5 (orthographe) directement
# Reproduire le pipeline V5 manuellement
phrase = "Elle envoyait des messages à Marc."

# 1. Tokeniser
raw_tokens = correcteur._tokenize(phrase)
tokens = [tok for tok, _is_word in raw_tokens]
_non_word_forms = frozenset(tok for tok, iw in raw_tokens if not iw)

print(f"Tokens: {tokens}")

from lectura_correcteur._utils import PUNCT_RE
is_punct = [bool(PUNCT_RE.match(t)) for t in tokens]
is_skip = [p or t in _non_word_forms for t, p in zip(tokens, is_punct)]
word_tokens = [t for t, s in zip(tokens, is_skip) if not s]

print(f"Word tokens: {word_tokens}")

# Morpho results (tagger lexique)
morpho_results = correcteur._v5_lex_tagger.tag_words(word_tokens)
for i, (w, m) in enumerate(zip(word_tokens, morpho_results)):
    print(f"  Lex tag [{i}] {w:15s} -> {m}")

# Ortho
analyses = correcteur._verificateur.verifier_phrase(word_tokens, morpho_results)
print(f"\nApres orthographe:")
for i, a in enumerate(analyses):
    changed = a.original != a.corrige
    marker = " **CHANGED**" if changed else ""
    print(f"  [{i}] orig={a.original!r:15s} corr={a.corrige!r:15s} pos={a.pos:10s} lex={a.dans_lexique}{marker}")
    if hasattr(a, 'suggestions') and a.suggestions:
        print(f"       suggestions: {a.suggestions}")

# Enrichir analyses avec POS/morpho (les lignes 256-277 du pipeline)
for j, analysis in enumerate(analyses):
    if j < len(morpho_results):
        mr = morpho_results[j]
        if not analysis.pos:
            analysis.pos = mr.get("pos", "")
        morpho_dict = {}
        for key in ("genre", "nombre", "temps", "mode", "personne"):
            val = mr.get(key)
            if val is not None:
                morpho_dict[key] = val
        if not analysis.morpho:
            analysis.morpho = morpho_dict

# Fallback POS via lexique (lines 271-277)
for j, analysis in enumerate(analyses):
    if analysis.pos and lexique is not None:
        infos = lexique.info(analysis.corrige)
        if infos:
            cgrams = {e["cgram"] for e in infos if e.get("cgram")}
            if analysis.pos not in cgrams and len(cgrams) == 1:
                old_pos = analysis.pos
                analysis.pos = next(iter(cgrams))
                print(f"  Fallback POS [{j}] {analysis.corrige}: {old_pos} -> {analysis.pos}")

print(f"\nApres enrichissement:")
for i, a in enumerate(analyses):
    print(f"  [{i}] corr={a.corrige!r:15s} pos={a.pos:10s}")
