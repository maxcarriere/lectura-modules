"""Exemple basique — Lectura Syllabeur Complet.

Montre les cas d'usage les plus courants, incluant les groupes de lecture
construits via le module G2P (lectura_phonemiseur.groupes_lecture).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_aligneur import (
    LecturaSyllabeur, OptionsGroupes,
    GroupeLecture, LectureFormule, EventFormule,
)
from lectura_phonemiseur.groupes_lecture import (
    construire_groupes_lecture,
    OptionsGroupes as G2POptionsGroupes,
)
from lectura_phonemiseur.pipeline_formules import MotAnalyseG2P, ResultatPhraseG2P


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

# -- Groupes de lecture avec liaison (via G2P) --
print("\n=== Groupes de lecture (liaison) ===\n")

result_g2p = ResultatPhraseG2P(mots=[
    MotAnalyseG2P(text="les", phone="lez", liaison="Lz"),
    MotAnalyseG2P(text="enfants", phone="ɑ̃fɑ̃"),
    MotAnalyseG2P(text="jouent", phone="ʒu"),
])
groupes = construire_groupes_lecture(result_g2p)
r = syl.analyser_complet(groupes=groupes)
print(r.format_detail())

# -- Groupes de lecture avec enchainement --
print("\n=== Groupes de lecture (enchainement) ===\n")

result_g2p = ResultatPhraseG2P(mots=[
    MotAnalyseG2P(text="avec", phone="avɛk"),
    MotAnalyseG2P(text="elle", phone="ɛl"),
])
groupes = construire_groupes_lecture(result_g2p)
r = syl.analyser_complet(groupes=groupes)
print(r.format_detail())

# -- Options : desactiver enchainement --
print("\n=== Sans enchainement ===\n")

opts = G2POptionsGroupes(gerer_enchainement=False)
groupes = construire_groupes_lecture(result_g2p, opts)
r = syl.analyser_complet(groupes=groupes)
print(r.format_detail())

# -- Formule avec lecture pre-calculee --
print("\n=== Formule avec lecture ===\n")

groupe = GroupeLecture(
    mots=[MotAnalyseG2P(text="42", phone="kaʁɑ̃tdø", est_formule=True)],
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
print("\n=== E1 seul (via G2P) ===\n")

result_g2p = ResultatPhraseG2P(mots=[
    MotAnalyseG2P(text="les", phone="lez", liaison="Lz"),
    MotAnalyseG2P(text="enfants", phone="ɑ̃fɑ̃"),
    MotAnalyseG2P(text="sont", phone="sɔ̃"),
    MotAnalyseG2P(text="arrives", phone="aʁive"),
])
groupes = construire_groupes_lecture(result_g2p)
for g in groupes:
    print(f"  <<{g.text}>> /{g.phone_groupe}/ jonctions={g.jonctions}")
