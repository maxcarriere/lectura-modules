"""Fonctions de morphologie : conjugaison, formes flechies, lemmatisation."""

from __future__ import annotations

from typing import Any


# Mapping personne -> cle de conjugaison
_PERSONNE_LABELS = {"1": "1", "2": "2", "3": "3"}
_NOMBRE_LABELS = {"s": "s", "p": "p"}

# Position dans le code multext pour les verbes :
# 0=cat(V), 1=type(m/a), 2=mode, 3=temps, 4=personne, 5=nombre, 6=genre
_MULTEXT_POS_PERSONNE = 4
_MULTEXT_POS_NOMBRE = 5


def _personne_nombre_from_entry(entry: dict[str, Any]) -> tuple[str, str]:
    """Extrait personne et nombre d'une entree, avec fallback multext."""
    personne = str(entry.get("personne", "") or "")
    nombre = str(entry.get("nombre", "") or "")

    # Fallback : extraire de multext si champs vides
    if (not personne or not nombre) and entry.get("multext"):
        multext = str(entry["multext"])
        if len(multext) > _MULTEXT_POS_PERSONNE and not personne:
            c = multext[_MULTEXT_POS_PERSONNE]
            if c in "123":
                personne = c
        if len(multext) > _MULTEXT_POS_NOMBRE and not nombre:
            c = multext[_MULTEXT_POS_NOMBRE]
            if c in "sp":
                nombre = c

    return personne, nombre


def conjuguer(entries: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, str]]]:
    """Table de conjugaison depuis les entrees d'un lemme verbal.

    Retour : {mode: {temps: {"1s": forme, "2s": forme, ...}}}
    Ne conserve que les entrees VER ou AUX.
    """
    table: dict[str, dict[str, dict[str, str]]] = {}

    for e in entries:
        cgram = str(e.get("cgram", "") or "")
        if not cgram.startswith(("VER", "AUX")):
            continue

        mode = str(e.get("mode", "") or "")
        temps = str(e.get("temps", "") or "")
        ortho = str(e.get("ortho", "") or "")

        if not mode or not ortho:
            continue

        personne, nombre = _personne_nombre_from_entry(e)

        # Construire la cle de personne
        if personne and nombre:
            cle = f"{personne}{nombre}"
        elif personne:
            cle = personne
        else:
            # Formes non personnelles (infinitif, participe)
            cle = ""

        if mode not in table:
            table[mode] = {}
        if temps not in table[mode]:
            table[mode][temps] = {}

        # Ne pas ecraser si deja present (premiere forme = plus frequente)
        if cle not in table[mode][temps]:
            table[mode][temps][cle] = ortho

    return table


def formes_de(entries: list[dict[str, Any]], cgram: str | None = None) -> list[dict[str, Any]]:
    """Toutes les formes flechies d'un lemme, avec filtre POS optionnel."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    for e in entries:
        if cgram is not None:
            e_cgram = str(e.get("cgram", "") or "")
            if not e_cgram.startswith(cgram):
                continue

        ortho = str(e.get("ortho", "") or "")
        if ortho and ortho.lower() not in seen:
            seen.add(ortho.lower())
            result.append(e)

    return result


def lemme_de(entries: list[dict[str, Any]]) -> str | None:
    """Lemme le plus frequent parmi les entrees d'un mot."""
    if not entries:
        return None

    # Compter les frequences par lemme
    freq_par_lemme: dict[str, float] = {}
    for e in entries:
        lemme = str(e.get("lemme", "") or "")
        if not lemme:
            continue
        freq = 0.0
        try:
            freq = float(e.get("freq", 0) or 0)
        except (ValueError, TypeError):
            pass
        freq_par_lemme[lemme] = freq_par_lemme.get(lemme, 0.0) + freq

    if not freq_par_lemme:
        return None

    return max(freq_par_lemme, key=lambda k: freq_par_lemme[k])
