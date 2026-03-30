"""Exemple basique — Lectura Tokeniseur.

Montre les cas d'usage les plus courants.
"""

from lectura_tokeniseur import LecturaTokeniseur, Mot, Separateur, Nombre, Sigle

tok = LecturaTokeniseur()

# ── Normalisation seule ──
print("=== Normalisation ===\n")

exemples = [
    "L'enfant  mange...du  chocolat",
    '"Bonjour" dit-il.',
    "Il a 1 000 000 euros.",
    "C'est-à-dire ( oui ) !",
]
for text in exemples:
    print(f"  {text!r}")
    print(f"  → {tok.normalize(text)!r}")
    print()


# ── Tokenisation complète ──
print("=== Tokenisation ===\n")

result = tok.analyze("L'enfant mange-t-il du chocolat ?")
print(result.format_table())
print()

# ── Accès aux données structurées ──
print("=== Données structurées ===\n")

for t in result.tokens:
    if isinstance(t, Mot):
        print(f"  MOT: «{t.text}» ortho={t.ortho!r} span={t.span}")
    elif isinstance(t, Separateur):
        print(f"  SEP: «{t.text}» type={t.sep_type} span={t.span}")
    elif isinstance(t, Nombre):
        print(f"  NUM: «{t.text}» span={t.span}")
    elif isinstance(t, Sigle):
        print(f"  SIG: «{t.text}» lettres={[c.text for c in t.children]} span={t.span}")

# ── Extraction simple des mots ──
print(f"\n=== Mots extraits ===\n")
print(tok.extract_words("Le SNCF transporte 3 millions de voyageurs."))
