"""Suggestions orthographiques par edit-distance 1 (et optionnellement 2)."""

from __future__ import annotations

from itertools import product as _product

ALPHABET_FR = "abcdefghijklmnopqrstuvwxyzàâäéèêëïîôùûüÿçœæ"

# Cap d'expansion pour distance 2 (eviter explosion combinatoire)
_MAX_D1_EXPAND = 500

# Ligatures digraphe <-> caractere unique
_LIGATURES = [("oe", "\u0153"), ("ae", "\u00e6")]  # oe->œ, ae->æ

# Table de desaccentuation pour detecter les variantes accent-only
_DESACCENTUER = str.maketrans(
    "àâäéèêëïîôùûüÿçœæ",
    "aaaeeeeiioouuycoa",
)

# Table d'accentuation : caractere de base -> variantes possibles
_ACCENT_MAP: dict[str, tuple[str, ...]] = {
    "a": ("à", "â"), "e": ("é", "è", "ê"), "i": ("î",), "o": ("ô",),
    "u": ("ù", "û"), "c": ("ç",),
    # Reverse : accent -> base
    "à": ("a",), "â": ("a",), "é": ("e",), "è": ("e",), "ê": ("e",),
    "î": ("i",), "ô": ("o",), "ù": ("u",), "û": ("u",), "ç": ("c",),
}


def _est_variante_accent(original: str, candidat: str) -> bool:
    """Verifie si le candidat ne differe que par les accents."""
    if len(original) != len(candidat):
        return False
    return original.translate(_DESACCENTUER) == candidat.translate(_DESACCENTUER)


def _variantes_accents(mot: str, lexique) -> list[tuple[str, float]]:
    """Genere les variantes d'accentuation d'un mot et filtre par le lexique.

    Pour les mots courts (<=4 positions accentuables) : toutes les combinaisons.
    Pour les mots plus longs : position par position puis paires.
    Retourne les candidats valides tries par frequence (priorite absolue).
    """
    low = mot.lower()
    # Identifier les positions accentuables
    positions: list[int] = []
    for idx, ch in enumerate(low):
        if ch in _ACCENT_MAP:
            positions.append(idx)

    if not positions:
        return []

    valides: list[tuple[str, float]] = []
    seen: set[str] = set()

    if len(positions) <= 4:
        # Toutes les combinaisons (max 3^4 = 81 candidats)
        options = []
        for p in positions:
            ch = low[p]
            options.append((ch,) + _ACCENT_MAP[ch])
        for combo in _product(*options):
            chars = list(low)
            for p, c in zip(positions, combo):
                chars[p] = c
            candidate = "".join(chars)
            if candidate != low and candidate not in seen and lexique.existe(candidate):
                freq = lexique.frequence(candidate) if hasattr(lexique, "frequence") else 0.0
                valides.append((candidate, freq))
                seen.add(candidate)
    else:
        # Position par position
        for p in positions:
            ch = low[p]
            for alt in _ACCENT_MAP[ch]:
                chars = list(low)
                chars[p] = alt
                candidate = "".join(chars)
                if candidate != low and candidate not in seen and lexique.existe(candidate):
                    freq = lexique.frequence(candidate) if hasattr(lexique, "frequence") else 0.0
                    valides.append((candidate, freq))
                    seen.add(candidate)
        # Paires
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pi, pj = positions[i], positions[j]
                chi, chj = low[pi], low[pj]
                for ai in _ACCENT_MAP[chi]:
                    for aj in _ACCENT_MAP[chj]:
                        chars = list(low)
                        chars[pi] = ai
                        chars[pj] = aj
                        candidate = "".join(chars)
                        if candidate != low and candidate not in seen and lexique.existe(candidate):
                            freq = lexique.frequence(candidate) if hasattr(lexique, "frequence") else 0.0
                            valides.append((candidate, freq))
                            seen.add(candidate)

    valides.sort(key=lambda x: -x[1])
    return valides


def _edits_distance_1(mot: str) -> set[str]:
    """Genere tous les candidats a distance d'edition 1."""
    splits = [(mot[:i], mot[i:]) for i in range(len(mot) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALPHABET_FR]
    inserts = [L + c + R for L, R in splits for c in ALPHABET_FR]

    # Ligatures : digraphe -> caractere unique et inverse
    ligature_edits: list[str] = []
    for digraph, ligature in _LIGATURES:
        # oe -> œ
        idx = 0
        while True:
            idx = mot.find(digraph, idx)
            if idx == -1:
                break
            ligature_edits.append(mot[:idx] + ligature + mot[idx + 2:])
            idx += 1
        # œ -> oe
        idx = 0
        while True:
            idx = mot.find(ligature, idx)
            if idx == -1:
                break
            ligature_edits.append(mot[:idx] + digraph + mot[idx + 1:])
            idx += 1

    return set(deletes + transposes + replaces + inserts + ligature_edits)


def _homophones_via_lexique(
    candidats: list[tuple[str, float]], lexique, seen: set[str], mot_low: str,
) -> list[tuple[str, float]]:
    """Trouve des homophones des candidats existants via le lexique.

    Pour chaque candidat deja trouve, recupere sa prononciation puis
    tous les mots partageant cette prononciation.
    """
    if not hasattr(lexique, "phone_de") or not hasattr(lexique, "homophones"):
        return []

    phones_seen: set[str] = set()
    result: list[tuple[str, float]] = []

    for cand, _ in candidats:
        phone = lexique.phone_de(cand)
        if not phone or phone in phones_seen:
            continue
        phones_seen.add(phone)
        for entry in lexique.homophones(phone):
            ortho = entry.get("ortho", "")
            if not ortho:
                continue
            ortho_low = ortho.lower()
            if ortho_low not in seen and ortho_low != mot_low:
                freq = entry.get("freq") or 0.0
                result.append((ortho_low, freq))
                seen.add(ortho_low)

    result.sort(key=lambda x: -x[1])
    return result


def suggerer(
    mot: str, lexique, max_n: int = 5, distance: int = 2,
    g2p=None,
) -> list[str]:
    """Pipeline de suggestions en 5 niveaux de priorite.

    1. Distance d'edition <= 1 (variantes d'accent prioritaires)
    2. Balayage d'accents multi-positions
    3. Homophones des candidats trouves (via lexique)
    4. G2P phonetique (si disponible, prioritaire)
    5. Distance <= 2 en dernier recours (si aucun candidat)

    Args:
        mot: Mot a corriger.
        lexique: Lexique (existe, frequence, phone_de, homophones).
        max_n: Nombre max de suggestions.
        distance: Distance d'edition max (conserve pour compatibilite).
        g2p: Objet optionnel avec methode prononcer(mot) -> str | None.
    """
    low = mot.lower()
    seen: set[str] = set()

    def _freq(c: str) -> float:
        return lexique.frequence(c) if hasattr(lexique, "frequence") else 0.0

    # --- Phase 1 : distance 1 ---
    d1 = _edits_distance_1(low)
    valides_d1: list[tuple[str, float]] = []
    for c in d1:
        if c not in seen and lexique.existe(c):
            valides_d1.append((c, _freq(c)))
            seen.add(c)

    # Separer accent-only vs other au sein de d=1
    accent_d1 = [(c, f) for c, f in valides_d1 if _est_variante_accent(low, c)]
    other_d1 = [(c, f) for c, f in valides_d1 if not _est_variante_accent(low, c)]
    accent_d1.sort(key=lambda x: -x[1])
    other_d1.sort(key=lambda x: -x[1])

    # --- Phase 2 : balayage d'accents multi-positions ---
    accent_sweep = _variantes_accents(low, lexique)
    for c, f in accent_sweep:
        seen.add(c)

    # Combiner : accents sweep + accent_d1 + other_d1
    phase12: list[tuple[str, float]] = accent_sweep + accent_d1 + other_d1

    # --- Phase 3 : homophones des candidats trouves ---
    phase3 = _homophones_via_lexique(phase12, lexique, seen, low)

    combined_123 = phase12 + phase3

    # --- Phase 4 : G2P phonetique d<=1 (si disponible, quand d=1 graphemique a echoue) ---
    # Ne tourne que si les phases 1-3 n'ont rien trouve, pour eviter
    # de remplacer des bonnes corrections d=1 par des homophones incorrects.
    # Cherche d'abord le match exact, puis les variantes phone d=1
    # (deletions + substitutions vocaliques).
    phase4_g2p: list[tuple[str, float]] = []
    if g2p is not None and not combined_123 and hasattr(lexique, "homophones"):
        phone = g2p.prononcer(low) if hasattr(g2p, "prononcer") else None
        if phone:
            from lectura_correcteur._phones import generer_phones_d1

            # d=0 (match exact) + d=1 (variantes phonetiques)
            phones_a_tester = [phone] + generer_phones_d1(phone)
            phones_vus: set[str] = set()
            for p in phones_a_tester:
                if p in phones_vus:
                    continue
                phones_vus.add(p)
                for entry in lexique.homophones(p):
                    ortho = entry.get("ortho", "")
                    if not ortho:
                        continue
                    ortho_low = ortho.lower()
                    if ortho_low not in seen and ortho_low != low:
                        freq = entry.get("freq") or 0.0
                        phase4_g2p.append((ortho_low, freq))
                        seen.add(ortho_low)
            phase4_g2p.sort(key=lambda x: -x[1])

    combined_1234 = combined_123 + phase4_g2p

    # --- Phase 5 : distance 2 en dernier recours ---
    valides_d2: list[tuple[str, float]] = []
    if not combined_1234 and distance >= 2:
        count = 0
        for c in d1:
            if lexique.existe(c):
                continue
            count += 1
            if count > _MAX_D1_EXPAND:
                break
            for c2 in _edits_distance_1(c):
                if c2 not in seen and c2 != low and lexique.existe(c2):
                    valides_d2.append((c2, _freq(c2)))
                    seen.add(c2)
        valides_d2.sort(key=lambda x: -x[1])

    combined = combined_1234 + valides_d2

    return [c for c, _ in combined[:max_n]]
