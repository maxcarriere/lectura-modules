"""Exemple d'intégration — Lectura Liaisons.

Montre comment intégrer le moteur de jonctions dans un pipeline NLP.
"""

from lectura_liaisons import (
    LecturaLiaisons,
    MotInfo,
    TokenMot,
    TokenSep,
    TokenPonct,
    GroupeJonction,
    JonctionOptions,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Pipeline avec Lectura Tokeniseur + POS Tagger + G2P
# ══════════════════════════════════════════════════════════════════════════════

# from lectura_tokeniseur import LecturaTokeniseur
# from lectura_pos import PosTagger
# from lectura_g2p import LecturaG2P
#
# tok = LecturaTokeniseur()
# pos = PosTagger("modele/pos_model_crf.json")
# g2p = LecturaG2P("modele/g2p_model_crf.json", ...)
# lia = LecturaLiaisons()
#
# text = "L'enfant est peut-être arrivé"
# result = tok.tokenise(text)
#
# # Convertir les tokens du tokeniseur en tokens Liaisons
# liaison_tokens = []
# for t in result.tokens:
#     if t.type.value == "mot":
#         phone = g2p.predict(t.text)
#         pos_tag = pos.tag_word(t.text)
#         liaison_tokens.append(TokenMot(t.text, phone, [pos_tag], t.span))
#     elif t.type.value == "separateur":
#         sep_type = "space"
#         if t.text == "'":
#             sep_type = "apostrophe"
#         elif t.text == "-":
#             sep_type = "hyphen"
#         liaison_tokens.append(TokenSep(t.text, sep_type, t.span))
#     elif t.type.value == "ponctuation":
#         liaison_tokens.append(TokenPonct(t.text, t.span))
#
# groups = lia.apply_jonctions(liaison_tokens)
# for g in groups:
#     print(f"  /{g.phone}/  ({g.jonction_type or 'simple'})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Simulation avec données manuelles
# ══════════════════════════════════════════════════════════════════════════════

print("=== Pipeline simulé ===\n")

lia = LecturaLiaisons()

# Phrase : "L'enfant est peut-être arrivé avec elle"
tokens = [
    TokenMot("L'", "l", ["ART:def"], (0, 2)),
    TokenSep("'", "apostrophe", (1, 2)),
    TokenMot("enfant", "ɑ̃fɑ̃", ["NOM"], (2, 8)),
    TokenSep(" ", "space", (8, 9)),
    TokenMot("est", "ɛ", ["AUX"], (9, 12)),
    TokenSep(" ", "space", (12, 13)),
    TokenMot("peut", "pø", ["VER"], (13, 17)),
    TokenSep("-", "hyphen", (17, 18)),
    TokenMot("être", "ɛtʁ", ["VER"], (18, 22)),
    TokenSep(" ", "space", (22, 23)),
    TokenMot("arrivé", "aʁive", ["VER"], (23, 29)),
    TokenSep(" ", "space", (29, 30)),
    TokenMot("avec", "avɛk", ["PRE"], (30, 34)),
    TokenSep(" ", "space", (34, 35)),
    TokenMot("elle", "ɛl", ["PRO:per"], (35, 39)),
]

# Toutes les jonctions activées, y compris enchaînements
opts = JonctionOptions(
    elisions=True,
    mots_composes=True,
    liaisons_gram=True,
    enchainements=True,
)

groups = lia.apply_jonctions(tokens, opts)

print("Phrase : L'enfant est peut-être arrivé avec elle\n")

for g in groups:
    parts = []
    for c in g.components:
        if isinstance(c, TokenMot):
            parts.append(c.ortho)
        elif isinstance(c, TokenSep):
            parts.append(c.text)
    label = "".join(parts)
    typ = g.jonction_type or "simple"
    print(f"  {label:20s}  /{g.phone:15s}/  {typ}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Options : désactiver certaines jonctions
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Liaisons seules (sans élisions ni composés) ===\n")

opts_liaisons_only = JonctionOptions(
    elisions=False,
    mots_composes=False,
    liaisons_gram=True,
    enchainements=False,
)

groups2 = lia.apply_jonctions(tokens, opts_liaisons_only)

for g in groups2:
    parts = []
    for c in g.components:
        if isinstance(c, TokenMot):
            parts.append(c.ortho)
        elif isinstance(c, TokenSep):
            parts.append(c.text)
    label = "".join(parts)
    typ = g.jonction_type or "simple"
    print(f"  {label:20s}  /{g.phone:15s}/  {typ}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. API par paires (niveau bas)
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== API par paires ===\n")

pairs = [
    (MotInfo("les", "le", ["ART:def"]), MotInfo("enfants", "ɑ̃fɑ̃", ["NOM"])),
    (MotInfo("un", "œ̃", ["ART:ind"]), MotInfo("ami", "ami", ["NOM"])),
    (MotInfo("neuf", "nœf", ["NUM"]), MotInfo("heures", "œʁ", ["NOM"])),
]

for w1, w2 in pairs:
    decision, merged = lia.analyze_pair(w1, w2)
    print(f"  {w1.ortho} + {w2.ortho} → /{merged}/  ({decision.kind}, {decision.typ})")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Liste h aspiré custom
# ══════════════════════════════════════════════════════════════════════════════

# Pour un domaine spécifique, on peut fournir sa propre liste :
# lia = LecturaLiaisons(h_aspire_path="mon_h_aspire.txt")
