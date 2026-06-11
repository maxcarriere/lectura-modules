"""Fusion morpho P2G (format UD) vers format V1 (short-form) + lexique.

Ce module convertit les predictions P2G (POS/Morpho au format UD) en
format V1 (short-form : m/f, s/p, ind/sub/imp...) et comble les trous
avec le lexique quand le P2G ne predit pas un trait (ex: genre sans
determineur visible).
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Mapping UD -> V1 short-form
# ---------------------------------------------------------------------------

_GENDER_MAP: dict[str, str] = {"Masc": "m", "Fem": "f"}
_NUMBER_MAP: dict[str, str] = {"Sing": "s", "Plur": "p"}
_PERSON_MAP: dict[str, str] = {"1": "1", "2": "2", "3": "3"}

# VerbForm + Mood -> mode V1
_MODE_MAP: dict[tuple[str, str], str] = {
    ("Fin", "Ind"): "ind",
    ("Fin", "Sub"): "sub",
    ("Fin", "Imp"): "imp",
    ("Fin", "Cnd"): "con",
    ("Inf", "_"): "inf",
    ("Part", "_"): "par",
}

# VerbForm + Tense -> temps V1
_TEMPS_MAP: dict[tuple[str, str], str] = {
    ("Fin", "Pres"): "pre",
    ("Fin", "Imp"): "imp",
    ("Fin", "Past"): "pas",
    ("Fin", "Fut"): "fut",
    ("Part", "Past"): "pas",
    ("Part", "Pres"): "pre",
}


def convertir_p2g_vers_v1(
    p2g_result: dict[str, Any],
    n_mots: int,
) -> tuple[list[str], dict[str, list[str]]]:
    """Convertit le resultat P2G (format UD) en format V1 (short-form).

    Args:
        p2g_result: dict brut du moteur P2G (cles: pos, morpho, etc.)
        n_mots: nombre de mots attendus

    Returns:
        (pos_list, morpho_dict_lists) avec morpho_dict_lists =
        {"genre": [...], "nombre": [...], "temps": [...], "mode": [...], "personne": [...]}
    """
    pos_list_raw = p2g_result.get("pos", [])
    morpho = p2g_result.get("morpho", {})

    gender_raw = morpho.get("Gender", [])
    number_raw = morpho.get("Number", [])
    person_raw = morpho.get("Person", [])
    verbform_raw = morpho.get("VerbForm", [])
    mood_raw = morpho.get("Mood", [])
    tense_raw = morpho.get("Tense", [])

    pos_list: list[str] = []
    genre_list: list[str] = []
    nombre_list: list[str] = []
    personne_list: list[str] = []
    mode_list: list[str] = []
    temps_list: list[str] = []

    for i in range(n_mots):
        # POS
        pos_list.append(pos_list_raw[i] if i < len(pos_list_raw) else "")

        # Genre
        g = gender_raw[i] if i < len(gender_raw) else "_"
        genre_list.append(_GENDER_MAP.get(g, "_"))

        # Nombre
        n = number_raw[i] if i < len(number_raw) else "_"
        nombre_list.append(_NUMBER_MAP.get(n, "_"))

        # Personne
        p = person_raw[i] if i < len(person_raw) else "_"
        personne_list.append(_PERSON_MAP.get(p, "_"))

        # VerbForm + Mood -> mode
        vf = verbform_raw[i] if i < len(verbform_raw) else "_"
        mood = mood_raw[i] if i < len(mood_raw) else "_"
        mode_list.append(_MODE_MAP.get((vf, mood), "_"))

        # VerbForm + Tense -> temps
        tense = tense_raw[i] if i < len(tense_raw) else "_"
        temps_list.append(_TEMPS_MAP.get((vf, tense), "_"))

    morpho_dict_lists: dict[str, list[str]] = {
        "genre": genre_list,
        "nombre": nombre_list,
        "temps": temps_list,
        "mode": mode_list,
        "personne": personne_list,
    }

    return pos_list, morpho_dict_lists


def fusionner_avec_lexique(
    pos_list: list[str],
    morpho: dict[str, list[str]],
    formes: list[str],
    lexique: Any,
) -> tuple[list[str], dict[str, list[str]]]:
    """Comble les trous ("_") de la morpho P2G avec le lexique.

    Pour chaque feature a "_", cherche dans lexique.info(forme) la valeur
    correspondante. Particulierement utile pour Genre (84% de "_" dans P2G
    sont dus a l'absence de determineur visible).

    Args:
        pos_list: POS par mot (peut etre modifie si le lexique donne un POS
            plus precis pour un mot sans POS P2G)
        morpho: dict de listes {"genre": [...], "nombre": [...], ...}
        formes: formes (lowered) des mots
        lexique: objet avec .info(forme) -> list[dict]

    Returns:
        (pos_list, morpho) avec les trous combles
    """
    if not hasattr(lexique, "info"):
        return pos_list, morpho

    n = len(formes)

    # Mapping lexique feat keys -> morpho keys + valeur courte
    _FEAT_MAP: dict[str, tuple[str, dict[str, str]]] = {
        "genre": ("genre", {"m": "m", "f": "f"}),
        "nombre": ("nombre", {"s": "s", "p": "p"}),
        "personne": ("personne", {"1": "1", "2": "2", "3": "3"}),
        "temps": ("temps", {}),  # pas de mapping additionnel
        "mode": ("mode", {}),
    }

    for i in range(n):
        forme = formes[i].lower()
        infos = lexique.info(forme)
        if not infos:
            continue

        # Trouver l'entree la plus frequente du bon POS
        pos_i = pos_list[i]
        matched = [e for e in infos if e.get("cgram") == pos_i] if pos_i else []
        if not matched:
            # Fallback: entree la plus frequente tous POS confondus
            matched = infos

        best = max(matched, key=lambda e: float(e.get("freq") or 0))

        for feat_key, (morpho_key, val_map) in _FEAT_MAP.items():
            if morpho_key not in morpho:
                continue
            if i < len(morpho[morpho_key]) and morpho[morpho_key][i] == "_":
                val = best.get(feat_key)
                if val is not None and val != "_":
                    # Normaliser la valeur si un mapping existe
                    if val_map:
                        val = val_map.get(val, val)
                    morpho[morpho_key][i] = val

    return pos_list, morpho
