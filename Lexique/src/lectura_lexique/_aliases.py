"""Resolution d'alias pour les noms de colonnes.

Chaque source de lexique (Lexique3, GLAFF, custom CSV, etc.) utilise des
noms de colonnes differents. Ce module fournit un mapping canonique pour
unifier l'acces.
"""

from __future__ import annotations

# Mapping : nom canonique -> ensemble d'alias connus
ALIAS_MAP: dict[str, frozenset[str]] = {
    "ortho": frozenset({"graphie", "form", "word", "forme", "flexion"}),
    "lemme": frozenset({"lemma"}),
    "cgram": frozenset({"category", "pos", "cat", "catgram"}),
    "genre": frozenset({"gender", "gen"}),
    "nombre": frozenset({"number", "num"}),
    "mode": frozenset({"mood"}),
    "temps": frozenset({"tense"}),
    "personne": frozenset({"person", "pers"}),
    "phone": frozenset({"phon", "phon_ipa", "ipa", "prononciation"}),
    "freq": frozenset({
        "freq_opensubs", "freqfilms2", "freq_frwac_forme_pmw",
        "freq_frantext", "freq_lm10", "freq_frwac",
    }),
}

# Index inverse : alias -> nom canonique (construit une fois)
_ALIAS_INVERSE: dict[str, str] = {}
for _canon, _aliases in ALIAS_MAP.items():
    for _alias in _aliases:
        _ALIAS_INVERSE[_alias] = _canon

# Priorite pour les colonnes de frequence : si plusieurs sont presentes,
# on prend la premiere trouvee dans cet ordre
FREQ_PRIORITE: list[str] = [
    "freq_opensubs",
    "freqfilms2",
    "freq_frwac_forme_pmw",
    "freq_frwac",
    "freq_lm10",
    "freq_frantext",
]


def resoudre_colonnes(colonnes: list[str]) -> dict[str, str]:
    """Mappe les noms de colonnes source vers les noms canoniques.

    Args:
        colonnes: Liste des noms de colonnes tels qu'ils apparaissent
                  dans la source (CSV header, colonnes SQLite, etc.)

    Returns:
        Dictionnaire ``{nom_source: nom_canonique}`` pour les colonnes
        reconnues. Les colonnes inconnues sont conservees telles quelles.
    """
    mapping: dict[str, str] = {}
    for col in colonnes:
        col_lower = col.lower().strip()
        if col_lower in _ALIAS_INVERSE:
            mapping[col] = _ALIAS_INVERSE[col_lower]
        elif col_lower in ALIAS_MAP:
            # Le nom est deja canonique
            mapping[col] = col_lower
        else:
            # Inconnu -> garder tel quel
            mapping[col] = col_lower
    return mapping
