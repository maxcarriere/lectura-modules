"""Expansion des abreviations SMS avant l'analyse morpho.

Table de substitution token -> token(s). Gate par config.activer_sms.
N'expande que si le token n'est PAS dans le lexique.
"""

from __future__ import annotations

from typing import Any

# ~80 abreviations SMS courantes en francais
# Valeurs multi-mots : seront splitees en tokens multiples
SMS_TABLE: dict[str, str] = {
    # Salutations / politesse
    "bjr": "bonjour",
    "bsr": "bonsoir",
    "slt": "salut",
    "cc": "coucou",
    "dsl": "desole",
    "dslé": "desole",
    "stp": "s'il te plait",
    "svp": "s'il vous plait",
    "mrc": "merci",
    "mrci": "merci",
    # Pronoms / sujets courants
    "jsuis": "je suis",
    "chuis": "je suis",
    "jtm": "je t'aime",
    "jt": "je te",
    "ct": "c'etait",
    "c": "c'est",
    "g": "j'ai",
    # Adverbes / conjonctions
    "bcp": "beaucoup",
    "mtn": "maintenant",
    "pk": "pourquoi",
    "pq": "pourquoi",
    "pcq": "parce que",
    "pck": "parce que",
    "tjs": "toujours",
    "tjrs": "toujours",
    "tt": "tout",
    "tte": "toute",
    "tts": "toutes",
    "ms": "mais",
    "qd": "quand",
    "qnd": "quand",
    "ds": "dans",
    "pr": "pour",
    "ac": "avec",
    "avc": "avec",
    "avk": "avec",
    "jms": "jamais",
    "vrm": "vraiment",
    "vrmt": "vraiment",
    # Verbes / expressions
    "chpa": "je ne sais pas",
    "jsp": "je ne sais pas",
    "jcp": "je ne sais pas",
    "vs": "vous",
    "ns": "nous",
    "kan": "quand",
    "koi": "quoi",
    "kwa": "quoi",
    "ki": "qui",
    # Temps / lieu
    "ajd": "aujourd'hui",
    "auj": "aujourd'hui",
    "dem": "demain",
    "hier": "hier",
    # Adjectifs / noms courants
    "pb": "probleme",
    "pbm": "probleme",
    "rdv": "rendez-vous",
    "taf": "travail",
    "biz": "bisous",
    "bzou": "bisous",
    # Verbes
    "vla": "voila",
    "vlà": "voila",
    # Expressions numeriques
    "2m1": "demain",
    # Reductions phonetiques
    "fr": "faire",
    "ss": "suis",
    "lol": "lol",  # emprunts conserves
    "mdr": "mdr",
    # Negation reduite
    "pa": "pas",
    # Question
    "cmb": "combien",
    "cmn": "comment",
    "cmt": "comment",
    "kel": "quel",
    "kl": "quel",
}


def expander_sms(tokens: list[str], lexique: Any) -> list[str]:
    """Expande les abreviations SMS dans la liste de tokens.

    Pour chaque token, si le mot n'est pas dans le lexique et que sa forme
    minuscule est dans SMS_TABLE, remplace par l'expansion (potentiellement
    multi-mots via split).

    Args:
        tokens: Liste de tokens (mots + ponctuation).
        lexique: Objet lexique avec methode existe().

    Returns:
        Nouvelle liste de tokens avec les expansions.
    """
    result: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if not lexique.existe(tok) and low in SMS_TABLE:
            expansion = SMS_TABLE[low]
            result.extend(expansion.split())
        else:
            result.append(tok)
    return result
