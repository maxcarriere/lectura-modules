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
    # Reverse : accent -> base + cross-accent alternatives
    "à": ("a", "â"), "â": ("a", "à"),
    "é": ("e", "è", "ê"), "è": ("e", "é", "ê"), "ê": ("e", "é", "è"),
    "î": ("i",), "ô": ("o",), "ù": ("u", "û"), "û": ("u", "ù"), "ç": ("c",),
}


def _edit_distance_rapide(a: str, b: str) -> int:
    """Distance d'edition Damerau-Levenshtein (OSA).

    Comme Levenshtein, mais les transpositions de caracteres adjacents
    comptent pour 1 operation (au lieu de 2 substitutions).
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return abs(la - lb)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev_prev = None
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
            # Transposition adjacente
            if (
                i > 0 and j > 0
                and a[i] == b[j - 1] and a[i - 1] == b[j]
                and prev_prev is not None
            ):
                curr[j + 1] = min(curr[j + 1], prev_prev[j - 1] + 1)
        prev_prev = prev
        prev = curr
    return prev[lb]


def _est_variante_accent(original: str, candidat: str) -> bool:
    """Verifie si le candidat ne differe que par les accents."""
    if len(original) != len(candidat):
        return False
    return original.translate(_DESACCENTUER) == candidat.translate(_DESACCENTUER)


def _est_nom_propre_seul(mot: str, lexique) -> bool:
    """True si toutes les entrees du lexique pour ce mot sont des noms propres."""
    infos = lexique.info(mot)
    if not infos:
        return False
    return all(e.get("cgram") == "NOM PROPRE" for e in infos)


def _est_doublement_consonne(original: str, candidat: str) -> bool:
    """Verifie si le candidat ne differe que par un doublement/dedoublement de consonne."""
    if abs(len(original) - len(candidat)) != 1:
        return False
    short, long_ = (original, candidat) if len(original) < len(candidat) else (candidat, original)
    for i in range(len(long_)):
        rebuilt = long_[:i] + long_[i + 1:]
        if rebuilt == short:
            # Le caractere insere est le meme que son voisin
            if i > 0 and long_[i] == long_[i - 1]:
                return True
            if i < len(long_) - 1 and long_[i] == long_[i + 1]:
                return True
    return False


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


def _meilleure_variante_accent(mot: str, lexique, freq_actuelle: float) -> str | None:
    """Retourne la meilleure variante accentuee si elle est bien plus frequente.

    Ne propose que des variantes accent-only (meme longueur, meme squelette
    desaccentue). Seuil : freq_candidat > max(freq_actuelle * 3, 30).
    """
    variantes = _variantes_accents(mot, lexique)
    if not variantes:
        return None
    best_form, best_freq = variantes[0]  # triees par freq desc
    if not _est_variante_accent(mot.lower(), best_form.lower()):
        return None
    # Guard: if original has nearly zero frequency, it may be a proper name
    # entry — require higher absolute threshold
    _min_threshold = 10 if freq_actuelle >= 0.1 else 30
    # Guard: cross-accent changes (é↔è↔ê) between two common words
    # are dangerous (élevés/élèves, gène/gêne) — require higher ratio
    _mot_low = mot.lower()
    _best_low = best_form.lower()
    _is_cross_accent = (
        any(c in "éèê" for c in _mot_low)
        and any(c in "éèê" for c in _best_low)
        and _mot_low != _best_low
    )
    if _is_cross_accent and freq_actuelle > 2.0:
        # Both forms are common — require 10x ratio
        if best_freq < freq_actuelle * 10:
            return None
    # Guard: adding an accent to a word that already has accents creates
    # a different word (côte→côté). Removing accents (là→la) is fine.
    # Words without accents (etat→état) are spelling corrections.
    _n_acc_orig = sum(1 for c in _mot_low if c in "àâäéèêëïîôùûüÿç")
    _n_acc_best = sum(1 for c in _best_low if c in "àâäéèêëïîôùûüÿç")
    if (
        freq_actuelle >= 20
        and _n_acc_orig > 0
        and _n_acc_best > _n_acc_orig
        and best_freq < freq_actuelle * 10
    ):
        return None
    if best_freq > max(freq_actuelle * 3, _min_threshold):
        return best_form
    return None


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
    g2p=None, symspell=None,
) -> list[str]:
    """Pipeline de suggestions en 5 niveaux de priorite.

    1. Distance d'edition <= 1 (variantes d'accent prioritaires)
    2. Balayage d'accents multi-positions
    3. Homophones des candidats trouves (via lexique)
    4. G2P phonetique (si disponible, prioritaire)
    5. Distance <= 2 en dernier recours (si aucun candidat)

    Si un index SymSpell est fourni, la phase 1 utilise l'index au lieu de
    la generation brute-force. La phase 5 reste brute-force en fallback.

    Args:
        mot: Mot a corriger.
        lexique: Lexique (existe, frequence, phone_de, homophones).
        max_n: Nombre max de suggestions.
        distance: Distance d'edition max (conserve pour compatibilite).
        g2p: Objet optionnel avec methode prononcer(mot) -> str | None.
        symspell: Index SymSpell optionnel pour generation rapide.
    """
    low = mot.lower()
    seen: set[str] = set()

    def _freq(c: str) -> float:
        return lexique.frequence(c) if hasattr(lexique, "frequence") else 0.0

    # --- Phase 1 : distance 1 ---
    if symspell is not None:
        # SymSpell : lookup rapide, mais filtrer pour ne garder que d<=1
        sym_candidates = symspell.suggestions(low)
        d1 = set()  # pas besoin du brute-force d1
        valides_d1: list[tuple[str, float]] = []
        sym_d2: list[tuple[str, float]] = []  # candidats d=2 pour phase 5
        for c in sym_candidates:
            if c not in seen and lexique.existe(c):
                if _edit_distance_rapide(low, c) <= 1:
                    valides_d1.append((c, _freq(c)))
                else:
                    sym_d2.append((c, _freq(c)))
                seen.add(c)
    else:
        sym_d2 = []
        d1 = _edits_distance_1(low)
        valides_d1: list[tuple[str, float]] = []
        for c in d1:
            if c not in seen and lexique.existe(c):
                valides_d1.append((c, _freq(c)))
                seen.add(c)

    # Filtrer les noms propres quand le mot source est en minuscule :
    # un nom propre (ex: "Pomès") ne devrait pas corriger un mot commun.
    if low == mot.lower() and hasattr(lexique, "info"):
        valides_d1 = [
            (c, f) for c, f in valides_d1
            if not _est_nom_propre_seul(c, lexique)
        ]

    # Separer accent-only vs other au sein de d=1
    accent_d1 = [(c, f) for c, f in valides_d1 if _est_variante_accent(low, c)]
    other_d1 = [(c, f) for c, f in valides_d1 if not _est_variante_accent(low, c)]
    accent_d1.sort(key=lambda x: -x[1])
    other_d1.sort(key=lambda x: -x[1])

    # --- Phase 2 : balayage d'accents multi-positions ---
    accent_sweep = _variantes_accents(low, lexique)
    if low == mot.lower() and hasattr(lexique, "info"):
        accent_sweep = [
            (c, f) for c, f in accent_sweep
            if not _est_nom_propre_seul(c, lexique)
        ]
    for c, f in accent_sweep:
        seen.add(c)

    # Combiner : accents sweep + accent_d1 + other_d1
    phase12: list[tuple[str, float]] = accent_sweep + accent_d1 + other_d1

    # --- Phase 3 : homophones des candidats trouves ---
    phase3 = _homophones_via_lexique(phase12, lexique, seen, low)

    combined_123 = phase12 + phase3

    # --- Phase 4 : G2P phonetique d<=1 (si disponible) ---
    # Tourne quand les phases 1-3 ont trouve < 3 candidats, pour attraper
    # les cas ou d=1 graphemique est insuffisant (ex: farmacien->pharmacien).
    # Cherche d'abord le match exact, puis les variantes phone d=1
    # (deletions + substitutions vocaliques).
    phase4_g2p: list[tuple[str, float]] = []
    if g2p is not None and hasattr(lexique, "homophones"):
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

    # Marquer les candidats G2P phonetiques (pour le re-tri)
    _g2p_set: set[str] = {c for c, _ in phase4_g2p}

    combined_1234 = combined_123 + phase4_g2p

    # --- Phase 5 : distance 2 ---
    # SymSpell d=2 : toujours inclure (rapide, pre-filtre)
    # Brute-force d=2 : seulement en dernier recours (0 candidats)
    valides_d2: list[tuple[str, float]] = []
    if distance >= 2:
        if sym_d2:
            sym_d2.sort(key=lambda x: -x[1])
            valides_d2 = sym_d2
        elif not combined_1234:
            # Brute-force d=2 classique (dernier recours)
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

    # Assembler par priorite :
    #   1. Variantes accent (toujours fiables)
    #   2. d=1 edit hors homophones (other_d1, tries par freq)
    #   3. G2P phonetiques (tries par freq, signal fort)
    #   4. Homophones des d=1 (phase 3)
    #   5. d=2 SymSpell (tries par freq)
    combined = (
        accent_sweep + accent_d1     # 1. accents
        + other_d1                   # 2. d=1 edit
        + phase4_g2p                 # 3. G2P phonetiques
        + phase3                     # 4. homophones
        + valides_d2                 # 5. d=2
    )

    return [c for c, _ in combined[:max_n]]
