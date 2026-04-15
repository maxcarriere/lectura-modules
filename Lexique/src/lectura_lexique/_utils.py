"""Utilitaires partages pour le lexique."""

from __future__ import annotations


def normaliser_ortho(mot: str) -> str:
    """Normalise un mot pour comparaison : strip + lower."""
    return mot.strip().lower()


# ---------------------------------------------------------------------------
# Variantes phonetiques proches
# ---------------------------------------------------------------------------

# Paires de phonemes français confondables (symétrique)
_CONFUSIONS: list[tuple[str, str]] = [
    # Voyelles mi-ouvertes / mi-fermées
    ("e", "ɛ"), ("o", "ɔ"), ("ø", "œ"),
    # A antérieur / postérieur
    ("a", "ɑ"),
    # Schwa
    ("ə", "ø"), ("ə", "œ"), ("ə", "e"),
    # Nasales proches
    ("ɛ̃", "œ̃"),
    # Consonnes voisées / sourdes
    ("s", "z"), ("ʃ", "ʒ"), ("f", "v"), ("t", "d"),
    ("p", "b"), ("k", "ɡ"),
    # Liquides
    ("ʁ", "l"),
    # Semi-voyelles
    ("w", "ɥ"),
]

# Construire un dict de substitutions
_SUBS: dict[str, list[str]] = {}
for _a, _b in _CONFUSIONS:
    _SUBS.setdefault(_a, []).append(_b)
    _SUBS.setdefault(_b, []).append(_a)


def generer_phones_proches(ipa: str, max_variantes: int = 8) -> list[str]:
    """Genere des variantes phonetiques proches d'une transcription IPA.

    Retourne toujours l'IPA original en premier, suivi de variantes
    obtenues par substitution d'un phoneme a la fois.

    Args:
        ipa: Transcription IPA (ex: "ʃa")
        max_variantes: Nombre maximal de variantes a retourner

    Returns:
        Liste de transcriptions IPA, l'original en premier
    """
    if not ipa:
        return [ipa]

    result = [ipa]
    seen = {ipa}

    # Tokeniser l'IPA en segments (graphèmes phonétiques)
    segments = _tokenize_ipa(ipa)

    for i, seg in enumerate(segments):
        if seg in _SUBS:
            for alt in _SUBS[seg]:
                variante = "".join(segments[:i] + [alt] + segments[i + 1:])
                if variante not in seen:
                    seen.add(variante)
                    result.append(variante)
                    if len(result) >= max_variantes:
                        return result

    return result


def _tokenize_ipa(ipa: str) -> list[str]:
    """Decoupe une chaine IPA en segments phonetiques.

    Gere les caractères combinants (nasalisation, longueur, etc.)
    """
    segments: list[str] = []
    i = 0
    combinants = set("̃ːˈˌ̥̤̰̪̺̻̼̊")  # diacritiques combinants courants
    while i < len(ipa):
        seg = ipa[i]
        i += 1
        # Absorber les diacritiques combinants
        while i < len(ipa) and ipa[i] in combinants:
            seg += ipa[i]
            i += 1
        segments.append(seg)
    return segments


def reverse_phone_ipa(phone: str) -> str:
    """Inverse une transcription IPA au niveau des phonemes.

    Exemple : 'ʃɔkɔla' → 'alɔkɔʃ'

    Utilise _tokenize_ipa pour gerer correctement les diacritiques.
    """
    if not phone:
        return ""
    segments = _tokenize_ipa(phone)
    segments.reverse()
    return "".join(segments)
