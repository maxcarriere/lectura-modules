"""Passe 2 — Correction par P2G sans ortho_words (pipeline V4).

Principe : apres la Passe 1 (orthographe pure), les erreurs restantes
sont phonetiquement transparentes (homophones, accords, participes).
On les detecte via un roundtrip G2P -> P2G *sans ortho_words*, ce qui
force le P2G a utiliser uniquement le contexte phonetique et son BiLSTM
pour reconstruire l'orthographe.

Le P2G sans ortho_words donne un meilleur tagging POS/Morpho sur texte
fautif (Number 86%, Person 93%, Tense 88%) et resout les homophones
grammaticaux (et/est, a/a, sont/son) que le V3 avec ortho_words echoue.

Etapes :
  2.1 G2P phonemisation (phones + POS initial)
  2.2 P2G sans ortho_words -> POS + Morpho + Ortho candidates
  2.3 Stockage morpho P2G dans MotV2
  2.4 Decision de correction
  2.5 Guards (elisions, accord sujet-verbe, mots proteges)
"""

from __future__ import annotations

import logging
from typing import Any

from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS
from lectura_correcteur._types import MotV2

logger = logging.getLogger(__name__)

# Mapping UD -> PM convention (pour enrichissement morpho)
_GENRE_MAP = {"Masc": "Masc", "Fem": "Fem", "m": "Masc", "f": "Fem"}
_NOMBRE_MAP = {"Sing": "Sing", "Plur": "Plur", "s": "Sing", "p": "Plur"}

# Pronoms sujet et la terminaison verbe attendue (indicatif present)
_PRONOUN_2SG = frozenset({"tu"})


def passe2_p2g(
    mots: list[MotV2],
    g2p_tagger: Any,
    p2g_adapter: Any,
    lexique: Any,
    config: Any,
) -> None:
    """Passe 2 V4 : correction par P2G sans ortho_words (in-place).

    Args:
        mots: liste de MotV2 (formes corrigees par passe 1)
        g2p_tagger: TaggerHybride ou G2PUnifieAdapter (tag_words_rich + prononcer)
        p2g_adapter: P2GAdapter (transcrire_complet)
        lexique: objet avec existe(), info(), frequence()
        config: CorrecteurV4Config
    """
    if not mots:
        return

    formes = [m.forme for m in mots]

    # 2.1 — G2P phonemisation
    tags = g2p_tagger.tag_words_rich(formes)
    _stocker_g2p(mots, tags, g2p_tagger)

    # Identifier les tokens elides (ancres)
    elision_mask = _marquer_elisions(mots)

    # 2.2 — P2G sans ortho_words
    p2g_result = _appeler_p2g(mots, p2g_adapter)
    if p2g_result is None:
        return

    # 2.3 — Stockage morpho P2G
    _stocker_morpho_p2g(mots, p2g_result)

    # 2.4 — Decisions de correction
    _decider_corrections(mots, p2g_result, lexique, elision_mask, config)

    # 2.5 — Guards
    _appliquer_guards(mots, formes, elision_mask)


# ---------------------------------------------------------------------------
# 2.1 — G2P phonemisation
# ---------------------------------------------------------------------------

def _stocker_g2p(mots: list[MotV2], tags: list[dict], g2p_tagger: Any) -> None:
    """Stocke les resultats G2P dans les MotV2 (phones + POS initial)."""
    for i, m in enumerate(mots):
        if i >= len(tags):
            break
        m.pos = tags[i].get("pos", "")
        m.confiance_pos = tags[i].get("confiance_pos", 1.0)
        m.pos_scores = tags[i].get("pos_scores", [])

        # Phone : depuis tag_words_rich (champ g2p) ou prononcer()
        phone = tags[i].get("g2p", "")
        if not phone and hasattr(g2p_tagger, "prononcer"):
            phone = g2p_tagger.prononcer(m.forme) or ""
        m.phone = phone


def _marquer_elisions(mots: list[MotV2]) -> list[bool]:
    """Marque les tokens elides comme ancres (ne pas corriger via P2G).

    Returns:
        Masque boolen par mot (True = elide/ancre).
    """
    mask: list[bool] = []
    for m in mots:
        is_elision = m.original.endswith(("'", "\u2019"))
        mask.append(is_elision)
    return mask


# ---------------------------------------------------------------------------
# 2.2 — P2G sans ortho_words
# ---------------------------------------------------------------------------

def _appeler_p2g(
    mots: list[MotV2],
    p2g_adapter: Any,
) -> dict[str, Any] | None:
    """Appelle P2G en mode sans ortho_words sur la phrase entiere.

    Returns:
        Dict brut du moteur P2G ou None si echec.
    """
    ipa_all: list[str] = []
    for m in mots:
        if m.phone:
            ipa_all.append(m.phone)
        else:
            # Mot sans phone : placeholder
            ipa_all.append(m.forme)

    try:
        result = p2g_adapter.transcrire_complet(
            ipa_all, ortho_words=None, k=5,
        )
    except Exception:
        logger.warning("P2G transcrire_complet echoue", exc_info=True)
        return None

    return result


# ---------------------------------------------------------------------------
# 2.3 — Stockage morpho P2G
# ---------------------------------------------------------------------------

def _stocker_morpho_p2g(
    mots: list[MotV2],
    p2g_result: dict[str, Any],
) -> None:
    """Remplit les champs morpho des MotV2 depuis les resultats P2G.

    La morpho P2G est plus fiable que celle du G2P sur texte fautif.
    """
    morpho = p2g_result.get("morpho", {})
    pos_list = p2g_result.get("pos", [])

    number_list = morpho.get("Number", [])
    gender_list = morpho.get("Gender", [])
    person_list = morpho.get("Person", [])

    for i, m in enumerate(mots):
        # POS depuis P2G (ecrase le G2P)
        if i < len(pos_list) and pos_list[i]:
            m.pos = pos_list[i]

        # Nombre
        if i < len(number_list):
            val = number_list[i]
            if val in _NOMBRE_MAP:
                m.nombre = _NOMBRE_MAP[val]
            elif val:
                m.nombre = val

        # Genre
        if i < len(gender_list):
            val = gender_list[i]
            if val in _GENRE_MAP:
                m.genre = _GENRE_MAP[val]
            elif val:
                m.genre = val

        # Personne
        if i < len(person_list):
            val = person_list[i]
            if val in ("1", "2", "3"):
                m.personne = val

        # PM tag
        m.pm_tag = f"{m.pos}|{m.nombre}|{m.genre}|{m.personne}"


# ---------------------------------------------------------------------------
# 2.4 — Decisions de correction
# ---------------------------------------------------------------------------

def _decider_corrections(
    mots: list[MotV2],
    p2g_result: dict[str, Any],
    lexique: Any,
    elision_mask: list[bool],
    config: Any,
) -> None:
    """Decide quelles corrections P2G appliquer.

    Pour chaque mot non-ancre :
    - Si p2g_ortho != forme ET p2g_ortho dans lexique ET confiance >= seuil → corriger
    - Bonus de confiance pour la forme originale
    """
    ortho_list = p2g_result.get("ortho", [])
    confiance_list = p2g_result.get("confiance", [])
    alternatives_list = p2g_result.get("alternatives", [])

    for i, m in enumerate(mots):
        # Skip tokens elides
        if elision_mask[i]:
            continue

        # Skip mots OOV (pas de roundtrip fiable)
        if not m.dans_lexique:
            continue

        # Skip mots proteges (function words)
        if _FUNCTION_WORD_POS.get(m.forme.lower()) is not None:
            continue

        if i >= len(ortho_list):
            break

        p2g_ortho = ortho_list[i].lower() if i < len(ortho_list) else ""
        confiance = confiance_list[i] if i < len(confiance_list) else 0.0

        if not p2g_ortho:
            continue

        # P2G confirme la forme courante → rien a faire
        if p2g_ortho == m.forme.lower():
            continue

        # La forme originale beneficie d'un bonus de confiance
        # Le P2G doit etre suffisamment confiant pour la remplacer
        seuil_effectif = config.seuil_confiance_p2g + config.bonus_forme_originale

        if confiance < seuil_effectif:
            continue

        # Verifier que la forme P2G est dans le lexique
        if not (hasattr(lexique, "existe") and lexique.existe(p2g_ortho)):
            # Chercher dans les alternatives
            found = False
            alts = alternatives_list[i] if i < len(alternatives_list) else []
            for alt_forme, alt_conf in alts:
                alt_low = alt_forme.lower()
                if alt_low == m.forme.lower():
                    continue  # c'est la forme actuelle
                if hasattr(lexique, "existe") and lexique.existe(alt_low):
                    if alt_conf >= config.seuil_confiance_p2g:
                        p2g_ortho = alt_low
                        confiance = alt_conf
                        found = True
                        break
            if not found:
                continue

        # Appliquer la correction
        old_forme = m.forme
        m.forme = p2g_ortho
        m.corrections.append(
            (2, "p2g.sans_ortho",
             f"{old_forme} -> {p2g_ortho} (conf={confiance:.2f})")
        )


# ---------------------------------------------------------------------------
# 2.5 — Guards
# ---------------------------------------------------------------------------

def _appliquer_guards(
    mots: list[MotV2],
    formes_originales: list[str],
    elision_mask: list[bool],
) -> None:
    """Annule les corrections qui violent les guards.

    Guards :
    - Elisions : ne jamais corriger les tokens elides
    - Accord sujet-verbe : ne pas casser un accord existant
    - Mots proteges : ne pas corriger les function words ancrees
    """
    for i, m in enumerate(mots):
        if m.forme == formes_originales[i].lower():
            continue  # pas de changement

        # Guard elisions (normalement deja filtre, double securite)
        if elision_mask[i]:
            _annuler_correction(m, formes_originales[i])
            continue

        # Guard accord sujet-verbe
        if _viole_accord_sujet_verbe(mots, i, formes_originales[i]):
            _annuler_correction(m, formes_originales[i])
            continue


def _viole_accord_sujet_verbe(
    mots: list[MotV2], idx: int, forme_originale: str,
) -> bool:
    """Verifie si la correction casse l'accord sujet-verbe.

    Cas detectes :
    - tu + verbe finissant en -s/-x : ne pas retirer le marqueur 2sg
    - tu + correction en -ez : c'est du 2pl (vous), pas 2sg
    """
    if idx == 0:
        return False

    m = mots[idx]
    low_orig = forme_originale.lower()
    low_new = m.forme.lower()

    # Chercher le sujet le plus proche a gauche
    sujet = mots[idx - 1].forme.lower()

    if sujet in _PRONOUN_2SG:
        # Tu + verbe : la forme doit finir par -s ou -x, pas -ez
        orig_has_s = low_orig.endswith(("s", "x"))
        new_has_s = low_new.endswith(("s", "x"))
        if orig_has_s and not new_has_s:
            return True  # la correction retire le marqueur 2sg

        # Tu + forme en -ez : c'est du 2pl (vous mangez), pas 2sg (tu manges)
        if low_new.endswith("ez") and not low_orig.endswith("ez"):
            return True

    return False


def _annuler_correction(mot: MotV2, forme_originale: str) -> None:
    """Annule la derniere correction de passe 2."""
    mot.forme = forme_originale.lower()
    mot.corrections = [c for c in mot.corrections if c[0] != 2]
