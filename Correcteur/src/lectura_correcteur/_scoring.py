"""Scoring multi-facteurs des candidats de remplacement.

Facteurs :
- S_identite : bonus conservateur pour le mot original (conditionnel au contexte)
- S_freq : frequence lexicale normalisee (log10)
- S_edit : proximite edit-distance
- S_pos : concordance POS avec le CRF
- S_morpho : concordance genre/nombre avec le contexte (determinant voisin)
- S_phone : proximite phonetique
- S_contexte : adequation syntaxique (ex: PP apres AUX)
"""

from __future__ import annotations

import math
from typing import Any

from lectura_correcteur._azerty import ratio_adjacence_azerty
from lectura_correcteur._config import CorrecteurConfig
from lectura_correcteur._types import Candidat, MotAnalyse

# Poids par defaut (tunables)
W_IDENTITE = 0.15
W_FREQ = 0.08
W_EDIT = 0.08
W_POS = 0.14
W_MORPHO = 0.10
W_PHONE = 0.10
W_CTX = 0.30
W_AZERTY = 0.05

# POS compatibles (un POS CRF peut correspondre a plusieurs cgram lexique)
_POS_COMPAT: dict[str, set[str]] = {
    "VER": {"VER", "AUX"},
    "AUX": {"AUX", "VER"},
    "NOM": {"NOM"},
    "ADJ": {"ADJ"},
    "ADV": {"ADV"},
    "PRE": {"PRE"},
    "CON": {"CON"},
    "ART:def": {"ART:def"},
    "ART:ind": {"ART:ind"},
    "PRO:per": {"PRO:per"},
    "PRO:rel": {"PRO:rel"},
    "PRO:dem": {"PRO:dem"},
    "PRO:ind": {"PRO:ind"},
    "ADJ:pos": {"ADJ:pos"},
    "DET": {"DET", "ADJ:pos", "ART:def", "ART:ind"},
    "DET:dem": {"DET:dem"},
}


def extraire_contexte(
    analyses: list[MotAnalyse],
    idx: int,
    lexique: Any = None,
) -> dict[str, Any]:
    """Extrait le contexte POS/morpho des mots voisins.

    Retourne :
    - pos_gauche, pos_droite : POS des voisins
    - genre_det, nombre_det : genre/nombre du determinant le plus proche a gauche
    - aux_gauche : True si un auxiliaire precede (pour PP)
    """
    ctx: dict[str, Any] = {
        "pos_gauche": "",
        "pos_droite": "",
        "genre_det": "",
        "nombre_det": "",
        "aux_gauche": False,
        "aux_lemme": "",  # "avoir" ou "être"
    }

    if idx > 0:
        ctx["pos_gauche"] = analyses[idx - 1].pos

    if idx < len(analyses) - 1:
        ctx["pos_droite"] = analyses[idx + 1].pos

    # Chercher le determinant le plus proche a gauche
    for j in range(idx - 1, max(idx - 4, -1), -1):
        pos_j = analyses[j].pos
        if pos_j in ("ART:def", "ART:ind", "DET", "DET:dem", "ADJ:pos"):
            morpho_j = analyses[j].morpho
            ctx["genre_det"] = morpho_j.get("genre", "")
            ctx["nombre_det"] = morpho_j.get("nombre", "")
            break

    # Chercher un auxiliaire immediatement a gauche (pour PP)
    # Pattern: AUX [ADV]* VER  (ex: "a mangé", "a bien mangé", "n'a pas mangé")
    #
    # Strategie en 2 passes :
    # 1. Chercher un AUX a gauche et identifier son lemme (avoir vs être)
    # 2. Decider si le mot courant peut etre un verbe :
    #    - Apres "avoir" : check large (CRF + lexique), car PP est quasi-obligatoire
    #      et le CRF tague souvent le PP comme NOM (ex: "visite")
    #    - Apres "être" : check strict (CRF seul), car ADJ attribut est valide
    #      (ex: "content", "calme" ont des entrees VER mais sont des ADJ attributs)

    # Passe 1 : trouver l'AUX et son lemme
    aux_pos_j = None
    aux_lemme = ""
    for j in range(idx - 1, max(idx - 3, -1), -1):
        pos_j = analyses[j].pos
        if pos_j == "AUX":
            aux_pos_j = j
            if lexique is not None and hasattr(lexique, "info"):
                aux_forme = analyses[j].corrige.lower()
                for entry in lexique.info(aux_forme):
                    if entry.get("cgram") in ("AUX", "VER"):
                        aux_lemme = entry.get("lemme", "")
                        break
            break
        if pos_j != "ADV":
            break

    # Passe 2 : verifier si le mot courant peut etre un verbe
    if aux_pos_j is not None:
        mot_pos = analyses[idx].pos if idx < len(analyses) else ""
        mot_pourrait_etre_verbe = mot_pos in ("VER", "AUX")
        # Apres "avoir", etendre la detection aux mots ayant des entrees VER
        # dans le lexique (le CRF tague souvent le PP comme NOM)
        if not mot_pourrait_etre_verbe and aux_lemme == "avoir":
            if lexique is not None and idx < len(analyses):
                mot_pourrait_etre_verbe = _a_entree_verbe(
                    analyses[idx].corrige, lexique,
                )
        if mot_pourrait_etre_verbe:
            ctx["aux_gauche"] = True
            ctx["aux_lemme"] = aux_lemme

    return ctx


def _a_entree_participe(forme: str, lexique: Any) -> bool:
    """Verifie si le mot a au moins une entree participe dans le lexique."""
    if not hasattr(lexique, "info"):
        return False
    for entry in lexique.info(forme):
        mode = entry.get("mode", "")
        if mode and "participe" in str(mode).lower():
            return True
    return False


def _a_entree_verbe(forme: str, lexique: Any) -> bool:
    """Verifie si le mot a au moins une entree VER ou AUX dans le lexique."""
    if not hasattr(lexique, "info"):
        return False
    for entry in lexique.info(forme):
        if entry.get("cgram", "") in ("VER", "AUX"):
            return True
    return False


def _s_identite(
    candidat: Candidat,
    contexte: dict[str, Any],
    lexique: Any,
    *,
    freq_mot: float = -1.0,
) -> float:
    """Bonus identite, module par frequence et contexte.

    Un mot rare (freq basse) merite moins de confiance identite qu'un mot courant.
    Courbe : freq=1→0.30, freq=100→0.50, freq=1000→0.75, freq=10000→1.0.

    Apres un auxiliaire "avoir", si le mot n'est pas un PP, le bonus est annule.
    Apres "être", seuls les mots CRF-VER/AUX sont penalises (pas les ADJ/NOM
    qui ont des entrees VER secondaires comme "content"→contenter).
    """
    if candidat.source != "identite":
        return 0.0
    if contexte.get("aux_gauche"):
        est_pp = _a_entree_participe(candidat.forme, lexique)
        if est_pp:
            return 1.0  # PP apres AUX : identite maintenue
        aux_lemme = contexte.get("aux_lemme", "")
        if aux_lemme == "avoir":
            # Apres "avoir" : check large (CRF + lexique) car PP est quasi-obligatoire
            est_verbe = candidat.pos in ("VER", "AUX") or _a_entree_verbe(candidat.forme, lexique)
        else:
            # Apres "être" : check strict (CRF seul), ADJ attribut est valide
            est_verbe = candidat.pos in ("VER", "AUX")
        if est_verbe:
            return 0.0  # pas de bonus identite apres AUX si pas PP
    # Modulation par frequence du mot original
    if freq_mot >= 0.0:
        factor = min(1.0, math.log10(freq_mot + 1) / 4.0)
        factor = max(0.3, factor)
        return factor
    return 1.0


def _s_freq(candidat: Candidat) -> float:
    """Frequence normalisee : log10(freq + 1) / 6.0."""
    return math.log10(candidat.freq + 1) / 6.0


def _s_edit(candidat: Candidat) -> float:
    """Proximite edit-distance : 1.0 - d/3."""
    return max(0.0, 1.0 - candidat.edit_dist / 3.0)


def _s_pos(candidat: Candidat, pos_crf: str) -> float:
    """Concordance POS : 1.0 exact, 0.5 compatible, 0.0 sinon."""
    if not pos_crf or not candidat.pos:
        return 0.5  # pas d'info -> neutre
    if candidat.pos == pos_crf:
        return 1.0
    compat = _POS_COMPAT.get(pos_crf, set())
    if candidat.pos in compat:
        return 0.5
    # Verifier le sens inverse aussi
    compat_inv = _POS_COMPAT.get(candidat.pos, set())
    if pos_crf in compat_inv:
        return 0.5
    return 0.0


def _s_morpho(candidat: Candidat, contexte: dict[str, Any]) -> float:
    """Concordance genre/nombre avec le determinant voisin."""
    genre_det = contexte.get("genre_det", "")
    nombre_det = contexte.get("nombre_det", "")

    if not genre_det and not nombre_det:
        return 0.5  # pas de contexte -> neutre

    score = 0.0
    n_checks = 0

    if genre_det and candidat.genre:
        n_checks += 1
        if candidat.genre == genre_det:
            score += 1.0

    if nombre_det and candidat.nombre:
        n_checks += 1
        if candidat.nombre == nombre_det:
            score += 1.0

    if n_checks == 0:
        return 0.5

    return score / n_checks


def _s_phone(candidat: Candidat, phone_original: str) -> float:
    """Proximite phonetique : 1.0 identique, decroissance progressive."""
    if not phone_original or not candidat.phone:
        return 0.5  # pas d'info -> neutre
    if candidat.phone == phone_original:
        return 1.0

    d = _phone_distance(phone_original, candidat.phone)
    if d <= 1:
        return 0.5
    if d <= 2:
        return 0.25
    return 0.0


def _s_contexte(candidat: Candidat, contexte: dict[str, Any], lexique: Any) -> float:
    """Score contextuel syntaxique.

    - Apres "avoir" : favorise les PP (1.0), penalise les non-PP verbes (0.0),
      y compris via lexique (mots tagues NOM mais ayant des entrees VER).
    - Apres "être" : favorise les PP (1.0), penalise les CRF-VER non-PP (0.0),
      mais neutre (0.5) pour les ADJ/NOM (attribut possible).
    - Sans contexte particulier : neutre (0.5).
    """
    if contexte.get("aux_gauche"):
        # Verifier si le candidat est un PP
        if _a_entree_participe(candidat.forme, lexique):
            return 1.0
        aux_lemme = contexte.get("aux_lemme", "")
        if aux_lemme == "avoir":
            # Apres "avoir" : check large (CRF + lexique)
            if candidat.pos in ("VER", "AUX") or _a_entree_verbe(candidat.forme, lexique):
                return 0.0
        else:
            # Apres "être" : check strict (CRF seul)
            if candidat.pos in ("VER", "AUX"):
                return 0.0
        return 0.5
    return 0.5


def _s_azerty(candidat: Candidat, mot_original: str) -> float:
    """Score AZERTY : favorise les candidats dont les substitutions sont adjacentes."""
    return ratio_adjacence_azerty(mot_original, candidat.forme)


def _phone_distance(a: str, b: str) -> int:
    """Distance Levenshtein simplifiee pour les chaines phonetiques."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[lb]


def scorer_candidats(
    candidats: list[Candidat],
    mot_original: str,
    pos_crf: str,
    morpho_crf: dict[str, str],
    contexte: dict[str, Any],
    lexique: Any,
    *,
    dans_lexique: bool = True,
    config: CorrecteurConfig | None = None,
    g2p: Any | None = None,
) -> list[Candidat]:
    """Score les candidats et les trie par score decroissant.

    Args:
        candidats: Liste de Candidat (non scores).
        mot_original: Mot original (avant correction orthographe).
        pos_crf: POS assignee par le CRF.
        morpho_crf: Morpho du CRF.
        contexte: Contexte extrait par extraire_contexte().
        lexique: Objet lexique.
        dans_lexique: False pour les mots OOV (pas de bonus identite).
        config: Configuration du correcteur (pour activer_azerty).
        g2p: Objet G2P optionnel (pour estimer phone des OOV).

    Returns:
        Liste de Candidat tries par score decroissant.
    """
    if not candidats:
        return []

    # Phone de l'original pour comparaison
    phone_original = ""
    if hasattr(lexique, "phone_de"):
        phone_original = lexique.phone_de(mot_original) or ""
    if not phone_original and g2p is not None:
        phone_original = (
            g2p.prononcer(mot_original)
            if hasattr(g2p, "prononcer")
            else ""
        ) or ""

    # Apres "avoir", le CRF POS est souvent faux (ex: "visite" tague NOM
    # alors que c'est VER/PP). On neutralise S_pos pour ne pas penaliser
    # les candidats VER. Apres "être", on garde S_pos car ADJ est valide.
    neutraliser_pos = (
        contexte.get("aux_gauche", False)
        and contexte.get("aux_lemme", "") == "avoir"
    )

    azerty_actif = config is not None and config.activer_azerty

    # Pour les OOV, la frequence est un signal plus fort (le mot original
    # n'existe pas, donc un candidat frequent est probablement le bon).
    w_freq = W_FREQ if dans_lexique else 0.15

    freq_mot = lexique.frequence(mot_original) if hasattr(lexique, "frequence") else -1.0

    for c in candidats:
        s_id = _s_identite(c, contexte, lexique, freq_mot=freq_mot) if dans_lexique else 0.0
        s_fr = _s_freq(c)
        s_ed = _s_edit(c)
        # OOV identity: "being close to a non-existent word" has no value.
        if not dans_lexique and c.source == "identite":
            s_ed = 0.0
        s_po = 0.5 if neutraliser_pos else _s_pos(c, pos_crf)
        s_mo = _s_morpho(c, contexte)
        s_ph = _s_phone(c, phone_original)
        s_cx = _s_contexte(c, contexte, lexique)
        s_az = _s_azerty(c, mot_original) if azerty_actif else 0.5

        c.score = (
            W_IDENTITE * s_id
            + w_freq * s_fr
            + W_EDIT * s_ed
            + W_POS * s_po
            + W_MORPHO * s_mo
            + W_PHONE * s_ph
            + W_CTX * s_cx
            + W_AZERTY * s_az
        )

    candidats.sort(key=lambda x: -x.score)
    return candidats
