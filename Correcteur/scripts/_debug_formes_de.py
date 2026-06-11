"""Debug: pourquoi formes_de ne retourne pas les formes 3s."""
import sys
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Correcteur/src')
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Lexique/src')

from lectura_lexique import Lexique
from lectura_correcteur._utils import LexiqueNormalise, normaliser_morpho

lex = Lexique('/data/work/projets/lectura/workspace/Lexique/bases/lexique_lectura.db')
lnorm = LexiqueNormalise(lex)

lemmes = ['travailler', 'exercer', 'parler', 'exister', 'jouer', 'mourir']

for lemme in lemmes:
    formes = lnorm.formes_de(lemme)
    print(f"\n=== formes_de('{lemme}') ===")
    # Filtrer present indicatif
    present_ind = [
        f for f in formes
        if f.get('cgram', '').startswith('VER') or f.get('cgram') == 'AUX'
        if normaliser_morpho(f.get('mode', '')) == 'ind'
        and normaliser_morpho(f.get('temps', '')) == 'pre'
    ]
    for f in present_ind:
        print(f"  ortho={f.get('ortho'):15s} pers={f.get('personne')} nb={normaliser_morpho(f.get('nombre',''))} mode={normaliser_morpho(f.get('mode',''))} temps={normaliser_morpho(f.get('temps',''))}")

    # Y a-t-il une forme 3s ?
    has_3s = any(
        f.get('personne') == '3' and normaliser_morpho(f.get('nombre', '')) == 's'
        for f in present_ind
    )
    print(f"  3s present: {has_3s}")

    # Et via info() ? (la forme 3s existe-t-elle ?)
    stem = lemme.rstrip('er').rstrip('ir').rstrip('re')
    for candidate in [lemme[:-2] + 'e', lemme[:-2] + 't', lemme[:-2] + 'd']:
        if lnorm.existe(candidate):
            infos = lnorm.info(candidate)
            verb_3s = [e for e in infos
                       if (e.get('cgram', '').startswith('VER') or e.get('cgram') == 'AUX')
                       and e.get('personne') == '3'
                       and normaliser_morpho(e.get('nombre', '')) == 's'
                       and normaliser_morpho(e.get('mode', '')) == 'ind']
            if verb_3s:
                print(f"  MAIS info('{candidate}') a la 3s: {verb_3s[0].get('ortho')}")
