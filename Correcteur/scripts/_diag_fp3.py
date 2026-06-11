#!/usr/bin/env python3
"""Diagnostic FP : tracer en amont du pipeline V5."""
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
cfg = CorrecteurV5Config()
correcteur = CorrecteurV5(lexique, config=cfg)

# Monkey-patch _pipeline_regles_v5 pour tracer
_orig_pipeline = correcteur._pipeline_regles_v5

def _traced_pipeline(analyses, word_tokens, morpho_results, corrections, **kwargs):
    print(f"  [PIPELINE_INPUT] word_tokens: {word_tokens}")
    print(f"  [PIPELINE_INPUT] analyses.corrige: {[a.corrige for a in analyses]}")
    print(f"  [PIPELINE_INPUT] analyses.original: {[a.original for a in analyses]}")
    print(f"  [PIPELINE_INPUT] analyses.pos: {[a.pos for a in analyses]}")

    pos_p2g = kwargs.get("pos_list_p2g", [])
    morpho_p2g = kwargs.get("morpho_p2g", {})
    print(f"  [PIPELINE_INPUT] pos_p2g: {pos_p2g}")
    print(f"  [PIPELINE_INPUT] morpho_p2g.nombre: {morpho_p2g.get('nombre', [])}")

    result, corrs = _orig_pipeline(analyses, word_tokens, morpho_results, corrections, **kwargs)
    print(f"  [PIPELINE_OUTPUT] after_rules: {result}")
    return result, corrs

correcteur._pipeline_regles_v5 = _traced_pipeline

# Aussi tracer la ligne 480 : decided_words
import lectura_correcteur.correcteur_v5 as v5_mod
_orig_corriger_v5 = correcteur._corriger_v5

def _traced_corriger(phrase):
    # On veut tracer ce qui se passe entre ortho et pipeline
    # Monkey-patch analyses
    return _orig_corriger_v5(phrase)

phrase = "Elle envoyait des messages à Marc."
print(f"PHRASE: {phrase}")
r = correcteur.corriger(phrase)
print(f"\nCORRIGE: {r.phrase_corrigee}")
for c in r.corrections:
    print(f"  [{c.regle}] {c.original!r} -> {c.corrige!r}")
