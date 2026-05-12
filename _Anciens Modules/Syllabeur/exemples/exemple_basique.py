"""Exemple basique — Lectura Syllabeur.

Montre les cas d'usage les plus courants.
"""

from lectura_syllabeur import LecturaSyllabeur

# ── Création du syllabeur (eSpeak par défaut) ──
syl = LecturaSyllabeur()

# ── Analyse d'un mot ──
result = syl.analyze("chocolat")
print(result.format_simple())
# → chocolat → /ʃo.ko.la/ (3 syll.)

print(result.format_detail())
# → chocolat → /ʃokola/
#   σ1: /ʃo/ «cho» [0:3] att=ʃ noy=o cod=-
#   σ2: /ko/ «co» [3:5] att=k noy=o cod=-
#   σ3: /la/ «lat» [5:8] att=l noy=a cod=-

# ── Accès aux données structurées ──
for s in result.syllabes:
    print(f"  {s.ortho:8s} /{s.phone:6s}/  span={s.span}")
    for p in s.attaque.phonemes:
        print(f"    attaque: /{p.ipa}/ ← «{p.grapheme}»")
    for p in s.noyau.phonemes:
        print(f"    noyau:   /{p.ipa}/ ← «{p.grapheme}»")
    for p in s.coda.phonemes:
        print(f"    coda:    /{p.ipa}/ ← «{p.grapheme}»")

# ── Analyse d'une phrase entière ──
print("\n--- Phrase ---")
for r in syl.analyze_text("Les enfants jouent dans la cour"):
    print(r.format_simple())

# ── IPA direct (sans phonémiseur) ──
print("\n--- IPA direct ---")
sylls = syl.syllabify_ipa("ɛkstʁaɔʁdinɛʁ")
print(f"ɛkstʁaɔʁdinɛʁ → {'.'.join(sylls)}")

# ── Phonétique manuelle (bypass eSpeak) ──
print("\n--- Phonétique manuelle ---")
r = syl.analyze("oignon", phone="ɔɲɔ̃")
print(r.format_detail())
