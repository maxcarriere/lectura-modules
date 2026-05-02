"""Resegmentation des tokens avec apostrophes/espaces manquants.

Pre-traitement pour le langage SMS/informel ou les apostrophes sont
souvent omises : "narrive" -> "n'arrive", "cest" -> "c'est", "jai" -> "j'ai".
"""

from __future__ import annotations

from typing import Any

from lectura_correcteur.orthographe._suggestions import _ACCENT_MAP

# Mots etrangers courants dont le split donnerait un faux positif
# (ex: "the" -> "t'hé", "she" -> "s'hé")
_MOTS_ETRANGERS_COURT = frozenset({
    "the", "she", "that", "this", "they",
    "card", "chard",
})

_CLITIQUES = [
    "qu'", "quelqu'", "lorsqu'", "puisqu'", "jusqu'",
    "presqu'", "aujourd'",
    "c'", "d'", "j'", "l'", "m'", "n'", "s'", "t'",
]

_CLITIQUES_SANS_APOS = [(c.replace("'", ""), c) for c in _CLITIQUES]
_CLITIQUES_SANS_APOS.sort(key=lambda x: -len(x[0]))

_VOYELLES = set("aeiouy\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb"
                "\u00ef\u00ee\u00f4\u00f9\u00fb\u00fc\u00e6\u0153")


def resegmenter(tokens: list[str], lexique: Any) -> list[str]:
    """Resegmente les tokens en separant les apostrophes manquantes."""
    result: list[str] = []

    for token in tokens:
        if lexique.existe(token):
            result.append(token)
            continue

        # Preserver les tokens avec trait d'union dont les deux parties
        # sont connues du lexique (ex: "vas-tu", "dit-on", "a-t-il").
        if "-" in token and not token.startswith("-") and not token.endswith("-"):
            parts = token.split("-")
            # Filtrer les particules intercalaires vides ou "t" euphonique
            meaningful = [p for p in parts if p.lower() not in ("", "t")]
            if meaningful and all(lexique.existe(p) for p in meaningful):
                result.append(token)
                continue

        if len(token) < 3:
            result.append(token)
            continue

        split = _tenter_split_clitique(token, lexique)
        if split:
            result.extend(split)
            continue

        split = _tenter_split_elargi(token, lexique)
        if split:
            result.extend(split)
        else:
            result.append(token)

    # Fusion de tokens adjacents ("beau coup" -> "beaucoup")
    result = _tenter_fusion(result, lexique)

    return result


def _trouver_forme_accent(mot: str, lexique: Any) -> str | None:
    """Retourne la forme accentuee du mot si elle existe dans le lexique.

    Teste les variantes mono-position (e→é, e→è, etc.) puis bi-position.
    Retourne la premiere forme trouvee ou None.
    """
    low = mot.lower()
    positions: list[int] = []
    for idx, ch in enumerate(low):
        if ch in _ACCENT_MAP:
            positions.append(idx)
    if not positions:
        return None
    # Mono-position
    for p in positions:
        ch = low[p]
        for alt in _ACCENT_MAP[ch]:
            chars = list(low)
            chars[p] = alt
            candidate = "".join(chars)
            if candidate != low and lexique.existe(candidate):
                return candidate
    # Bi-position
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
                    if candidate != low and lexique.existe(candidate):
                        return candidate
    return None


def _tenter_split_clitique(token: str, lexique: Any) -> list[str] | None:
    """Tente de separer un clitique du debut du token."""
    token_low = token.lower()

    # Guard: ne pas splitter les mots etrangers connus
    if token_low in _MOTS_ETRANGERS_COURT:
        return None

    for prefix_sans, prefix_avec in _CLITIQUES_SANS_APOS:
        if not token_low.startswith(prefix_sans):
            continue

        reste = token[len(prefix_sans):]

        # Prefixes longs (>=2 chars) : reste doit faire >= 3
        # Prefixes courts (1 char) : reste doit faire >= 2
        min_reste = 2 if len(prefix_sans) == 1 else 3
        if len(reste) < min_reste:
            continue

        if len(prefix_sans) == 1:
            first_char = reste[0].lower()
            if first_char not in _VOYELLES and first_char != "h":
                continue
            # Pour les prefixes d'1 char, exiger que le reste soit assez
            # frequent (evite les FP sur noms propres/mots etrangers OOV :
            # "mart" -> "m'art", "selim" -> "s'elim")
            # Seuil plus strict pour les restes courts (<=4 chars)
            if hasattr(lexique, "frequence"):
                _freq_reste = lexique.frequence(reste)
                _seuil_freq = 50.0 if len(reste) <= 4 else 10.0
                if _freq_reste < _seuil_freq:
                    continue

        reste_reel = reste
        if not lexique.existe(reste):
            reste_accent = _trouver_forme_accent(reste, lexique)
            if reste_accent:
                reste_reel = reste_accent
            else:
                continue
        clitique = token[:len(prefix_sans)] + "'"
        return [clitique, reste_reel]

    return None


_FUSIONS_CONNUES = frozenset({
    "beaucoup", "ensemble", "aujourd", "maintenant", "toujours",
    "autrefois", "longtemps", "aussitôt", "bientôt", "parfois",
    "quelquefois", "davantage", "cependant", "pourtant", "toutefois",
    "désormais", "dorénavant",
})

# Mapping: forme sans accent -> forme canonique dans le lexique
_COMPOSES_TRAIT: dict[str, str] = {
    "c'est-à-dire": "c'est-à-dire", "c'est-a-dire": "c'est-à-dire",
    "au-dessus": "au-dessus", "au-dessous": "au-dessous",
    "au-delà": "au-delà", "au-dela": "au-delà",
    "là-bas": "là-bas", "la-bas": "là-bas",
    "là-haut": "là-haut", "la-haut": "là-haut",
    "ci-dessus": "ci-dessus", "ci-dessous": "ci-dessous",
    "vis-à-vis": "vis-à-vis", "vis-a-vis": "vis-à-vis",
    "peut-on": "peut-on",
}


# Paires de tokens a remplacer par un seul mot
# (token1, token2) -> remplacement, avec guard optionnel sur le token suivant
_SPECIAL_MERGES: list[tuple[tuple[str, str], str, frozenset[str] | None]] = [
    # d'en→dans est gere en dur dans _tenter_fusion (blacklist)
]

# Expressions figees avec "d'en" (ne pas fusionner en "dans")
_DEN_EXPRESSIONS = frozenset({"haut", "bas", "face", "dessous", "dessus"})


def _tenter_fusion(tokens: list[str], lexique: Any) -> list[str]:
    """Fusionne les paires de tokens adjacents si le resultat est un mot connu."""
    result: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            # d'en → dans : fusionner sauf si suivi d'un infinitif ou expression
            _pair = (tokens[i].lower(), tokens[i + 1].lower())
            if _pair in (("d'", "en"), ("d\u2019", "en")):
                _merge_den = False
                if i + 2 < len(tokens):
                    _nxt_den = tokens[i + 2].lower()
                    if _nxt_den in _DEN_EXPRESSIONS:
                        pass  # expression figee → garder "d'en"
                    elif (
                        _nxt_den.endswith(("er", "ir", "oir", "re"))
                        and hasattr(lexique, "info")
                    ):
                        # Verbe infinitif → garder "d'en" (= de + en + VER)
                        _den_infos = lexique.info(tokens[i + 2])
                        _den_is_verb = any(
                            (e.get("cgram") or "").startswith(("VER", "AUX"))
                            for e in _den_infos
                        )
                        if not _den_is_verb:
                            _merge_den = True
                    else:
                        _merge_den = True
                if _merge_den:
                    result.append("dans")
                    i += 2
                    continue

            # Fusions speciales generiques
            _merged = False
            for (_t1, _t2), _repl, _guard_next in _SPECIAL_MERGES:
                if _pair == (_t1, _t2):
                    if _guard_next is None or (
                        i + 2 < len(tokens)
                        and tokens[i + 2].lower() in _guard_next
                    ):
                        result.append(_repl)
                        i += 2
                        _merged = True
                        break
            if _merged:
                continue

            fusionne = tokens[i] + tokens[i + 1]
            fusionne_low = fusionne.lower()
            # Fusion par liste connue (curatee, pas de garde token inconnu)
            if fusionne_low in _FUSIONS_CONNUES and lexique.existe(fusionne_low):
                result.append(fusionne_low)
                i += 2
                continue
            # Composes avec trait d'union ("la" + "bas" -> "là-bas")
            fusionne_trait = tokens[i] + "-" + tokens[i + 1]
            fusionne_trait_low = fusionne_trait.lower()
            if fusionne_trait_low in _COMPOSES_TRAIT:
                canon = _COMPOSES_TRAIT[fusionne_trait_low]
                if lexique.existe(canon):
                    result.append(canon)
                    i += 2
                    continue
            # Fusion generique: les deux tokens sont inconnus,
            # leur concat est connue et frequente
            if (
                not lexique.existe(tokens[i])
                and not lexique.existe(tokens[i + 1])
                and lexique.existe(fusionne_low)
            ):
                freq = lexique.frequence(fusionne_low) if hasattr(lexique, "frequence") else 0
                if freq >= 10.0:
                    result.append(fusionne_low)
                    i += 2
                    continue
        result.append(tokens[i])
        i += 1
    return result


def _tenter_split_elargi(token: str, lexique: Any) -> list[str] | None:
    """Split elargi pour prefixes multi-char avec reste court.

    Gere "quil" -> "qu'il" ou le reste (2 chars) est trop court pour
    la regle standard (min 3 pour prefixes multi-char).
    Garde la contrainte voyelle/h pour les prefixes d'1 char.
    """
    token_low = token.lower()

    # Guard: ne pas splitter les mots etrangers connus
    if token_low in _MOTS_ETRANGERS_COURT:
        return None

    for prefix_sans, prefix_avec in _CLITIQUES_SANS_APOS:
        if not token_low.startswith(prefix_sans):
            continue

        reste = token[len(prefix_sans):]

        if len(reste) < 2:
            continue

        # Pour les prefixes d'1 char, garder la contrainte voyelle/h
        # (ces cas sont deja geres par _tenter_split_clitique)
        if len(prefix_sans) == 1:
            first_char = reste[0].lower()
            if first_char not in _VOYELLES and first_char != "h":
                continue
            # Meme garde frequence que _tenter_split_clitique
            if hasattr(lexique, "frequence"):
                _freq_reste = lexique.frequence(reste)
                _seuil_freq = 50.0 if len(reste) <= 4 else 10.0
                if _freq_reste < _seuil_freq:
                    continue

        reste_reel = reste
        if not lexique.existe(reste):
            reste_accent = _trouver_forme_accent(reste, lexique)
            if reste_accent:
                reste_reel = reste_accent
            else:
                continue
        clitique = token[:len(prefix_sans)] + "'"
        return [clitique, reste_reel]

    return None
