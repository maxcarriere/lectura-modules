"""Exemple 4 — Analyse syllabique et groupes de lecture.

pip install lectura-aligneur
"""

from lectura_aligneur import (
    LecturaSyllabeur,
    EspeakPhonemizer,
    OptionsGroupes,
)

# --- Verifier la disponibilite d'eSpeak-NG ---
if EspeakPhonemizer.is_available():
    phonemizer = EspeakPhonemizer()
    print("eSpeak-NG disponible")
else:
    print("eSpeak-NG non disponible — les exemples ci-dessous")
    print("necessitent eSpeak-NG pour la phonemisation.")
    print("Installation : sudo apt install espeak-ng")
    phonemizer = None

if phonemizer:
    syl = LecturaSyllabeur(phonemizer=phonemizer)

    # --- Analyse d'un mot ---
    mot = "papillon"
    resultat = syl.analyze(mot)
    print(f"\nMot : {mot}")
    print(f"Syllabes : {[s.texte for s in resultat.syllabes]}")
    print(f"Phonemes : {resultat.phonemes_ipa}")

    # --- Groupes de lecture ---
    mots = ["les", "enfants", "jouent"]
    phones = ["le", "ɑ̃fɑ̃", "ʒu"]

    options = OptionsGroupes(mode="E1")
    groupes = syl.analyser_complet(
        mots=[{"ortho": m, "phonemes": p} for m, p in zip(mots, phones)],
    )

    print(f"\nGroupes de lecture :")
    for g in groupes.groupes:
        print(f"  Groupe : {g.texte} → /{g.phonemes}/")
        for s in g.syllabes:
            print(f"    Syllabe : {s.texte}")
