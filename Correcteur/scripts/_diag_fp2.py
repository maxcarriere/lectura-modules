#!/usr/bin/env python3
"""Diagnostic FP : tracer exactement ou 'des' devient 'du'."""
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

# Monkey-patch appliquer_grammaire pour tracer
import lectura_correcteur.grammaire as gram_mod
_orig_appliquer = gram_mod.appliquer_grammaire
_orig_accords = gram_mod.verifier_accords
_orig_homos = gram_mod.verifier_homophones

def _traced_homos(mots, *args, **kwargs):
    result, corrs = _orig_homos(mots, *args, **kwargs)
    for i, (a, b) in enumerate(zip(mots, result)):
        if a.lower() != b.lower():
            print(f"  [HOMOPHONES] pos {i}: {a!r} -> {b!r}")
    return result, corrs

def _traced_accords(mots, *args, **kwargs):
    result, corrs = _orig_accords(mots, *args, **kwargs)
    for i, (a, b) in enumerate(zip(mots, result)):
        if a.lower() != b.lower():
            print(f"  [ACCORDS] pos {i}: {a!r} -> {b!r}")
    return result, corrs

gram_mod.verifier_homophones = _traced_homos
gram_mod.verifier_accords = _traced_accords

# Aussi tracer la pipeline entiere
import lectura_correcteur.grammaire._homophones as homo_mod
_orig_homo_fn = homo_mod.verifier_homophones
def _traced_homo_fn(mots, *args, **kwargs):
    result, corrs = _orig_homo_fn(mots, *args, **kwargs)
    for i, (a, b) in enumerate(zip(mots, result)):
        if a.lower() != b.lower():
            print(f"  [HOMO_DETAIL] pos {i}: {a!r} -> {b!r}")
    return result, corrs
homo_mod.verifier_homophones = _traced_homo_fn
gram_mod.verifier_homophones = _traced_homo_fn

# Also trace inside verifier_accords to find the exact rule
import lectura_correcteur.grammaire._accord as acc_mod
_orig_va = acc_mod.verifier_accords

def _traced_va(mots, pos_tags, morpho, lexique, origs=None, **kwargs):
    # Trace with internal modification detection
    result = list(mots)
    # We need to intercept the result list modifications
    print(f"  [VA_INPUT] {result}")
    print(f"  [VA_POS]   {pos_tags}")
    nombres = morpho.get("nombre", [])
    print(f"  [VA_NOMBRE] {nombres}")
    r, c = _orig_va(mots, pos_tags, morpho, lexique, origs, **kwargs)
    print(f"  [VA_OUTPUT] {r}")
    for corr in c:
        print(f"  [VA_CORR] {corr.regle}: {corr.original!r} -> {corr.corrige!r} | {corr.explication}")
    return r, c

acc_mod.verifier_accords = _traced_va
gram_mod.verifier_accords = _traced_va

phrase = "Elle envoyait des messages à Marc."
print(f"PHRASE: {phrase}")
r = correcteur.corriger(phrase)
print(f"CORRIGE: {r.phrase_corrigee}")
for c in r.corrections:
    print(f"  [{c.regle}] {c.original!r} -> {c.corrige!r}")
