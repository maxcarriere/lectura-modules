"""Table des homophones pour la desambiguation par predictions du modele.

Deux mecanismes :
- POS-level  : le POS predit determine la forme (ces/ses, ca/sa, son/sont, on/ont)
               → utilise dans step 1c de corriger_phrase_v3
- Morpho-level : la prediction morpho (Number) determine la forme (il/ils, au/aux, etc.)
               → utilise dans step 1d de corriger_phrase_v3

Pas de modele n-gramme externe : on exploite directement les tetes POS et morpho
du modele P2G unifie.
"""

from __future__ import annotations


# ── Desambiguation morpho-level ─────────────────────────────────────
#
# Homophones dont la distinction porte sur un trait morphologique
# (Number, Gender, Person) que le modele predit deja.
#
# Format : lower_form -> (feature_name, {feature_value: correct_form, ...})
#
# Le modele P2G predit Number, Gender, Person par mot. Si la prediction
# morpho contredit la forme actuelle, on la corrige.

_HOMOPHONES_MORPHO: dict[str, tuple[str, dict[str, str]]] = {
    # il / ils  (Number: Sing vs Plur)
    # Benchmark : +3 net (8 fixes, 5 casses)
    "il":       ("Number", {"Sing": "il",      "Plur": "ils"}),
    "ils":      ("Number", {"Sing": "il",      "Plur": "ils"}),

    # elle / elles
    # Benchmark : +1 net (1 fix, 0 casse)
    "elle":     ("Number", {"Sing": "elle",    "Plur": "elles"}),
    "elles":    ("Number", {"Sing": "elle",    "Plur": "elles"}),

    # au / aux  (PRE:det Sing vs Plur)
    # Benchmark : +1 net (1 fix, 0 casse)
    "au":       ("Number", {"Sing": "au",      "Plur": "aux"}),
    "aux":      ("Number", {"Sing": "au",      "Plur": "aux"}),

    # Paires retirees (net negatif sur dev set) :
    # leur/leurs  : -5 net — "leur" pronom datif invariable confondu avec Plur
    # etait/etaient : -1 net
    # ma/mes, ta/tes : 0 changement observe
}
