"""Modele de Fujisaki pour generation de contours F0 lisses.

Decompose le F0 en composante de phrase (declination) et composantes
d'accent (montees/descentes locales), generant des courbes lisses et
physiologiquement plausibles — sans dependance supplementaire.

Reference : Fujisaki & Hirose 1984, Fujisaki 2004.
"""

from __future__ import annotations

import math

# Parametres francais (Jun & Fougeron, Fujisaki 2004)
ALPHA_DEFAULT = 2.0   # constante phrase (/s)
BETA_DEFAULT = 20.0   # constante accent (/s)


def phrase_response(t: float, alpha: float = ALPHA_DEFAULT) -> float:
    """Reponse impulsionnelle phrase Gp(t) = alpha^2 * t * exp(-alpha*t).

    Retourne 0 pour t < 0.
    """
    if t < 0.0:
        return 0.0
    return alpha * alpha * t * math.exp(-alpha * t)


def _ga_onset(t: float, beta: float) -> float:
    """Composante onset de la reponse echelon accent.

    Ga_on(t) = 1 - (1 + beta*t) * exp(-beta*t) pour t >= 0, sinon 0.
    """
    if t < 0.0:
        return 0.0
    return 1.0 - (1.0 + beta * t) * math.exp(-beta * t)


def accent_response(
    t: float, duration: float, beta: float = BETA_DEFAULT,
) -> float:
    """Reponse echelon accent Ga: onset a t=0, offset a t=duration.

    Ga(t) = Ga_on(t) - Ga_on(t - duration)
    Monte vers ~1.0 pendant la commande, puis redescend.
    """
    return _ga_onset(t, beta) - _ga_onset(t - duration, beta)


def accent_peak(duration: float, beta: float = BETA_DEFAULT) -> float:
    """Valeur maximale de Ga pour une commande de duree donnee.

    Le pic se produit a l'offset (t = duration). Utile pour compenser
    l'amplitude afin que la cible soit atteinte exactement.
    """
    return _ga_onset(duration, beta)


def generate_contour(
    fb: float,
    phrase_cmds: list[tuple[float, float]],
    accent_cmds: list[tuple[float, float, float]],
    eval_times: list[float],
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
) -> list[float]:
    """Evaluer le modele Fujisaki aux temps donnes.

    Args:
        fb: frequence de base (Hz) — plancher du contour.
        phrase_cmds: liste de (time, amplitude) pour les commandes phrase.
        accent_cmds: liste de (time, duration, amplitude) pour les
            commandes accent.
        eval_times: instants d'evaluation (secondes).
        alpha: constante de temps phrase.
        beta: constante de temps accent.

    Returns:
        Liste de F0 en Hz, meme longueur que eval_times.

    Le modele :
        ln(F0(t)) = ln(Fb) + SUM Ap * Gp(t - T0) + SUM Aa * Ga(t - T1, dur)
    """
    ln_fb = math.log(max(fb, 1.0))
    result: list[float] = []

    for t in eval_times:
        ln_f0 = ln_fb

        # Composantes phrase
        for t0, ap in phrase_cmds:
            ln_f0 += ap * phrase_response(t - t0, alpha)

        # Composantes accent
        for t1, dur, aa in accent_cmds:
            ln_f0 += aa * accent_response(t - t1, dur, beta)

        result.append(math.exp(ln_f0))

    return result
