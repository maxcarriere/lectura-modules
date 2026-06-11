"""Debug: comprendre pourquoi la regle accord_sujet_verbe ne fire pas
sur les cas avec pronom sujet."""
import sys
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Correcteur/src')
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Lexique/src')
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Phonemiseur/src')
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Graphemiseur/src')
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Tokeniseur/src')
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/G2P-Pipeline/src')
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/P2G-Pipeline/src')
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Formules/src')

from lectura_lexique import Lexique
from lectura_correcteur._utils import LexiqueNormalise, normaliser_morpho
from lectura_correcteur.correcteur_v6 import CorrecteurV6
from lectura_correcteur._config import CorrecteurV6Config

lex = Lexique('/data/work/projets/lectura/workspace/Lexique/bases/lexique_lectura.db')
lnorm = LexiqueNormalise(lex)
config = CorrecteurV6Config()
config.mode_analyse = True
correcteur = CorrecteurV6(lex, config=config)

_MODES_CONJ = frozenset({'ind', 'sub', 'con'})

test_cases = [
    ('il', 'travaillez', 'il travaillez au liberia'),
    ('il', 'exercent', 'il exercent sa profession'),
    ('elle', 'mourons', 'elle mourons le a bay shore'),
    ('on', 'parlons', 'on parlons aussi de commerces'),
    ('ils', 'joues', 'ils joues aussi avec the fratellis'),
    ('il', 'existent', 'il existent des generateurs'),
    ('il', 'decrochons', 'il decrochons le titre'),
]

_PRO = {
    'je': ('1', 's'), 'tu': ('2', 's'),
    'il': ('3', 's'), 'elle': ('3', 's'), 'on': ('3', 's'),
    'ils': ('3', 'p'), 'elles': ('3', 'p'),
}

print("=" * 80)
print("SIMULATION DE LA LOGIQUE _corriger_accord_sujet_verbe")
print("=" * 80)

for pronom, forme, phrase in test_cases:
    forme_low = forme.lower()
    print(f"\n--- {pronom} {forme} ---")
    print(f"Phrase: {phrase}")

    # Step 1: le mot existe dans le lexique ?
    existe = lnorm.existe(forme_low)
    print(f"  existe('{forme_low}'): {existe}")
    if not existe:
        print("  -> SKIP (OOV)")
        continue

    # Step 2: entrees verbales conjuguees
    infos = lnorm.info(forme_low)
    verb_entries = [
        e for e in infos
        if (e.get('cgram', '').startswith('VER') or e.get('cgram') == 'AUX')
        and e.get('mode', '') in _MODES_CONJ
    ]
    print(f"  verb_entries conj: {len(verb_entries)}")
    for e in verb_entries:
        print(f"    cgram={e.get('cgram')} mode={e.get('mode')} temps={e.get('temps')} pers={e.get('personne')} nb={e.get('nombre')} freq={e.get('freq',0)}")

    if not verb_entries:
        print("  -> SKIP (pas de VER conj)")
        continue

    # Step 3: G2P POS
    # On va executer le correcteur pour obtenir le G2P POS
    res = correcteur.corriger(phrase)
    # Trouver le MotV6 correspondant
    # En mode_analyse, les mots sont dans res.mots
    print(f"  Correction output: '{res.phrase_corrigee}'")

    # Step 4: meilleure entree
    verb_entries.sort(key=lambda e: float(e.get('freq', 0)), reverse=True)
    best = verb_entries[0]
    v_pers = best.get('personne', '')
    v_nombre = normaliser_morpho(best.get('nombre', '')) if best.get('nombre') else ''
    v_lemme = best.get('lemme', '')
    v_mode = best.get('mode', '')
    v_temps = best.get('temps', '')

    sujet_pers, sujet_nombre = _PRO[pronom]
    desaccord = not ((not v_pers or v_pers == sujet_pers) and (not v_nombre or v_nombre == sujet_nombre))

    print(f"  best: mode={v_mode} temps={v_temps} pers={v_pers} nb={v_nombre} lemme={v_lemme}")
    print(f"  sujet: {sujet_pers}{sujet_nombre} desaccord: {desaccord}")

    if desaccord:
        formes = lnorm.formes_de(v_lemme)
        candidat = None
        candidat_freq = -1.0
        for f in formes:
            f_cgram = f.get('cgram', '')
            if not (f_cgram.startswith('VER') or f_cgram == 'AUX'):
                continue
            f_mode = normaliser_morpho(f.get('mode', ''))
            f_temps = normaliser_morpho(f.get('temps', ''))
            if f_mode != v_mode or f_temps != v_temps:
                continue
            f_pers = f.get('personne', '')
            f_nombre = normaliser_morpho(f.get('nombre', ''))
            if f_pers == sujet_pers and f_nombre == sujet_nombre:
                f_freq = float(f.get('freq', 0))
                if f_freq > candidat_freq:
                    candidat = f.get('ortho', '')
                    candidat_freq = f_freq
        print(f"  -> candidat trouvee: {candidat}")
    else:
        print(f"  -> PAS de desaccord, la regle ne fire pas")

# Test additionnel: est-ce que l'etape 1 (ortho) corrige d'abord le verbe?
print("\n" + "=" * 80)
print("TEST: est-ce que l'etape 1 ortho change le verbe avant l'etape 3?")
print("=" * 80)

test_phrases_complet = [
    "il travaillez au liberia et en il est elu president du liberia college",
    "il exercent sa profession notamment a marseille",
    "elle mourons le a bay shore new york",
    "on parlons aussi de commerces associes",
    "ils joues aussi avec the fratellis",
    "il existent des generateurs en ligne",
    "il decrochons le titre de champion de belgique",
]

config2 = CorrecteurV6Config()
correcteur2 = CorrecteurV6(lex, config=config2)

for phrase in test_phrases_complet:
    res = correcteur2.corriger(phrase)
    if res.corrections:
        print(f"\n  Phrase: {phrase}")
        print(f"  Corrigee: {res.phrase_corrigee}")
        for c in res.corrections:
            print(f"    {c.original} -> {c.corrige} (regle: {c.regle})")
    else:
        print(f"\n  Phrase: {phrase}")
        print(f"  -> AUCUNE correction")
