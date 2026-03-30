"""Exemple basique — Lectura Liaisons.

Montre les cas d'usage les plus courants.
"""

from lectura_liaisons import LecturaLiaisons, MotInfo, TokenMot, TokenSep

lia = LecturaLiaisons()

# ── Classification d'une liaison ──
print("=== Classification ===\n")

decision = lia.classify(
    MotInfo("les", "le", ["ART:def"]),
    MotInfo("enfants", "ɑ̃fɑ̃", ["NOM"]),
)
print(f"Kind : {decision.kind}")           # grammaticale
print(f"Type : {decision.typ}")            # obligatoire
print(f"Latent : /{decision.latent_phoneme}/")  # z
print()

# ── Fusion phonétique ──
print("=== Fusion phonétique ===\n")

phone = lia.merge("le", "ɑ̃fɑ̃", decision)
print(f"les + enfants → /{phone}/")        # lezɑ̃fɑ̃
print()

# ── Raccourci analyze_pair ──
print("=== Analyze pair ===\n")

decision, merged = lia.analyze_pair(
    MotInfo("petit", "pəti", ["ADJ"]),
    MotInfo("oiseau", "wazo", ["NOM"]),
)
print(f"petit + oiseau → /{merged}/ ({decision.typ})")
print()

# ── Format lisible ──
print("=== Format lisible ===\n")

pairs = [
    (MotInfo("ils", "il", ["PRO:per"]), MotInfo("ont", "ɔ̃", ["AUX"])),
    (MotInfo("et", "e", ["CON"]), MotInfo("alors", "alɔʁ", ["ADV"])),
    (MotInfo("les", "le", ["ART:def"]), MotInfo("héros", "eʁo", ["NOM"])),
]
for w1, w2 in pairs:
    print(lia.format_decision(w1, w2))
    print()

# ── Vérification h aspiré ──
print("=== H aspiré ===\n")

for mot in ["haricot", "homme", "héros", "heure", "honte", "habitude"]:
    print(f"  {mot:12s} → h aspiré : {'oui' if lia.is_h_aspire(mot) else 'non'}")

print()

# ── Pipeline apply_jonctions ──
print("=== Pipeline apply_jonctions ===\n")

tokens = [
    TokenMot("Les", "le", ["ART:def"], (0, 3)),
    TokenSep(" ", "space", (3, 4)),
    TokenMot("enfants", "ɑ̃fɑ̃", ["NOM"], (4, 11)),
    TokenSep(" ", "space", (11, 12)),
    TokenMot("ont", "ɔ̃", ["AUX"], (12, 15)),
    TokenSep(" ", "space", (15, 16)),
    TokenMot("mangé", "mɑ̃ʒe", ["VER"], (16, 21)),
]

groups = lia.apply_jonctions(tokens)

for g in groups:
    parts = []
    for c in g.components:
        if isinstance(c, TokenMot):
            parts.append(c.ortho)
        elif isinstance(c, TokenSep):
            parts.append(c.text)
    label = "".join(parts)
    print(f"  {label:20s}  /{g.phone}/  ({g.jonction_type or 'simple'})")
