"""Suggestions orthographiques par edit-distance 1 (et optionnellement 2)."""

from __future__ import annotations

ALPHABET_FR = "abcdefghijklmnopqrstuvwxyzàâäéèêëïîôùûüÿçœæ"

# Cap d'expansion pour distance 2 (eviter explosion combinatoire)
_MAX_D1_EXPAND = 500


def _edits_distance_1(mot: str) -> set[str]:
    """Genere tous les candidats a distance d'edition 1."""
    splits = [(mot[:i], mot[i:]) for i in range(len(mot) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALPHABET_FR]
    inserts = [L + c + R for L, R in splits for c in ALPHABET_FR]
    return set(deletes + transposes + replaces + inserts)


def suggerer(
    mot: str, lexique, max_n: int = 5, distance: int = 1,
) -> list[str]:
    """Retourne les suggestions triees par distance puis par frequence.

    Les candidats a distance 1 sont toujours classes avant ceux a distance 2,
    de sorte qu'un mot proche (d=1) ne soit pas supplante par un mot tres
    frequent mais plus eloigne (d=2).

    Args:
        mot: Mot a corriger.
        lexique: Lexique (existe, frequence).
        max_n: Nombre max de suggestions.
        distance: Distance d'edition max (1 ou 2).
    """
    low = mot.lower()
    d1 = _edits_distance_1(low)
    valides_d1: list[tuple[str, float]] = []
    for c in d1:
        if lexique.existe(c):
            freq = lexique.frequence(c) if hasattr(lexique, "frequence") else 0.0
            valides_d1.append((c, freq))

    valides_d2: list[tuple[str, float]] = []
    if distance >= 2 and len(valides_d1) < max_n:
        seen = {c for c, _ in valides_d1}
        count = 0
        for c in d1:
            if lexique.existe(c):
                continue
            count += 1
            if count > _MAX_D1_EXPAND:
                break
            for c2 in _edits_distance_1(c):
                if c2 not in seen and c2 != low and lexique.existe(c2):
                    freq = lexique.frequence(c2) if hasattr(lexique, "frequence") else 0.0
                    valides_d2.append((c2, freq))
                    seen.add(c2)

    # D=1 first (sorted by freq), then D=2 (sorted by freq)
    valides_d1.sort(key=lambda x: -x[1])
    valides_d2.sort(key=lambda x: -x[1])
    combined = valides_d1 + valides_d2
    return [c for c, _ in combined[:max_n]]
