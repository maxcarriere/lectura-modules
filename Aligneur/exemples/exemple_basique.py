"""Exemple basique — Lectura Syllabeur Complet.

Montre les cas d'usage les plus courants, incluant les groupes de lecture.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_aligneur import (
    LecturaSyllabeur, MotAnalyse, OptionsGroupes,
    GroupeLecture, LectureFormule, EventFormule,
)


# -- Token factice pour les exemples --
@dataclass
class FakeToken:
    text: str
    span: tuple


# -- Creation du syllabeur (eSpeak par defaut) --
syl = LecturaSyllabeur()

# -- Analyse d'un mot (API retrocompatible) --
print("=== Analyse d'un mot ===\n")

result = syl.analyze("chocolat", phone="ʃɔkɔla")
print(result.format_simple())
print(result.format_detail())

# -- Acces aux donnees structurees --
print("\n=== Donnees structurees ===\n")

for s in result.syllabes:
    print(f"  {s.ortho:8s} /{s.phone:6s}/  span={s.span}")
    for p in s.attaque.phonemes:
        print(f"    attaque: /{p.ipa}/ <- <<{p.grapheme}>>")
    for p in s.noyau.phonemes:
        print(f"    noyau:   /{p.ipa}/ <- <<{p.grapheme}>>")
    for p in s.coda.phonemes:
        print(f"    coda:    /{p.ipa}/ <- <<{p.grapheme}>>")

# -- IPA direct (sans phonemiseur) --
print("\n=== IPA direct ===\n")
sylls = syl.syllabify_ipa("ɛkstʁaɔʁdinɛʁ")
print(f"ɛkstʁaɔʁdinɛʁ -> {'.'.join(sylls)}")

# -- Groupes de lecture avec liaison --
print("\n=== Groupes de lecture (liaison) ===\n")

mots = [
    MotAnalyse(token=FakeToken("les", (0, 3)), phone="lez", liaison="Lz"),
    MotAnalyse(token=FakeToken("enfants", (4, 11)), phone="ɑ̃fɑ̃"),
    MotAnalyse(token=FakeToken("jouent", (12, 18)), phone="ʒu"),
]
r = syl.analyser_complet(mots)
print(r.format_detail())

# -- Groupes de lecture avec enchainement --
print("\n=== Groupes de lecture (enchainement) ===\n")

mots = [
    MotAnalyse(token=FakeToken("avec", (0, 4)), phone="avɛk"),
    MotAnalyse(token=FakeToken("elle", (5, 9)), phone="ɛl"),
]
r = syl.analyser_complet(mots)
print(r.format_detail())

# -- Options : desactiver enchainement --
print("\n=== Sans enchainement ===\n")

opts = OptionsGroupes(gerer_enchainement=False)
r = syl.analyser_complet(mots, options=opts)
print(r.format_detail())

# -- Formule avec lecture pre-calculee --
print("\n=== Formule avec lecture ===\n")

groupe = GroupeLecture(
    mots=[MotAnalyse(token=FakeToken("42", (0, 2)), phone="kaʁɑ̃tdø")],
    phone_groupe="kaʁɑ̃tdø",
    span=(0, 2),
    est_formule=True,
    lecture=LectureFormule(
        display_fr="quarante-deux",
        events=[
            EventFormule(ortho="quarante", phone="kaʁɑ̃t", span_source=(0, 2)),
            EventFormule(ortho="deux", phone="dø", span_source=(0, 2)),
        ],
    ),
)
result_groupes = syl.syllabifier_groupes([groupe])
for s in result_groupes[0].syllabes:
    print(f"  <<{s.ortho}>> /{s.phone}/")

# -- E1 seul : construire des groupes --
print("\n=== E1 seul ===\n")

mots = [
    MotAnalyse(token=FakeToken("les", (0, 3)), phone="lez", liaison="Lz"),
    MotAnalyse(token=FakeToken("enfants", (4, 11)), phone="ɑ̃fɑ̃"),
    MotAnalyse(token=FakeToken("sont", (12, 16)), phone="sɔ̃"),
    MotAnalyse(token=FakeToken("arrives", (17, 24)), phone="aʁive"),
]
groupes = syl.construire_groupes(mots)
for g in groupes:
    print(f"  <<{g.text}>> /{g.phone_groupe}/ jonctions={g.jonctions}")
