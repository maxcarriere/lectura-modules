"""Passe 2 — Correction par roundtrip G2P -> P2G (pipeline V3).

Principe : apres la Passe 1 (orthographe pure), toutes les erreurs restantes
sont phonetiquement transparentes (homophones, accords, participes).
On les detecte en faisant un roundtrip phonetique :

  forme (post-passe1) --G2P--> IPA --P2G--> ortho_candidate

Si le P2G reconstruit une forme differente avec confiance suffisante,
c'est probablement la bonne orthographe.

Etapes :
  2.1 G2P tagging (phones + POS + morpho)
  2.2 Ancrage (mots a haute confiance ou invariables)
  2.3 P2G roundtrip sur les mots non-ancres
  2.4 Decision de correction
  2.5 Guard conservatif
  2.6 Enrichissement morpho (depuis G2P)
"""

from __future__ import annotations

import logging

from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS
from lectura_correcteur._types import MotV2

logger = logging.getLogger(__name__)

# Mapping UD -> PM convention (pour enrichissement morpho)
_GENRE_MAP = {"Masc": "Masc", "Fem": "Fem", "m": "Masc", "f": "Fem"}
_NOMBRE_MAP = {"Sing": "Sing", "Plur": "Plur", "s": "Sing", "p": "Plur"}

# Mots-outils homophones avec des verbes courants : ne pas ancrer.
# Ce sont les erreurs les plus frequentes (son/sont, on/ont, etc.)
_FUNC_HOMOPHONE_VERBE = frozenset({
    "son", "sont",      # possessif / etre 3pl
    "on", "ont",        # pronom / avoir 3pl
    "est", "et",        # etre 3sg / conjonction
    "a",                # avoir 3sg / preposition a
    "sa",               # possessif / ca
    "ces", "ses",       # demonstratif / possessif
    "ce", "se",         # demonstratif / reflexif
})


def passe2_phonetique(
    mots: list[MotV2],
    g2p_tagger,
    p2g_adapter,
    lexique,
    pos_ngram,
    lm_homophones,
    config,
) -> None:
    """Passe 2 V3 : correction par roundtrip G2P -> P2G (in-place).

    Args:
        mots: liste de MotV2 (formes corrigees par passe 1)
        g2p_tagger: TaggerHybride ou G2PUnifieAdapter (tag_words_rich + prononcer)
        p2g_adapter: P2GAdapter (transcrire)
        lexique: objet avec existe(), info(), frequence()
        pos_ngram: PosNgram ou None (cross-check optionnel)
        lm_homophones: LMHomophones ou None (cross-check optionnel)
        config: CorrecteurV3Config
    """
    if not mots:
        return

    formes = [m.forme for m in mots]

    # 2.1 — G2P tagging
    tags = g2p_tagger.tag_words_rich(formes)
    _appliquer_overrides(mots, tags)
    _stocker_g2p(mots, tags, g2p_tagger)

    # 2.2 — Ancrage
    _ancrer(mots, lexique, lm_homophones, config)

    # 2.3-2.4 — P2G roundtrip + decisions
    _roundtrip_p2g(mots, p2g_adapter, lexique, config)

    # 2.5 — LM homophones (complementaire au P2G)
    _correction_homophones(mots, lm_homophones, config)

    # 2.6 — Guard conservatif
    _guard_conservatif(mots, lexique, pos_ngram, lm_homophones, formes, config)

    # 2.7 — Enrichissement morpho depuis G2P
    _enrichir_morpho(mots, tags)


# ---------------------------------------------------------------------------
# 2.1 — Overrides mots-outils (identique V2)
# ---------------------------------------------------------------------------

def _appliquer_overrides(mots: list[MotV2], tags: list[dict]) -> None:
    """Force le POS des mots-outils connus."""
    for i, m in enumerate(mots):
        if i >= len(tags):
            break
        override = _FUNCTION_WORD_POS.get(m.forme.lower())
        if override is not None:
            tags[i]["pos"] = override
            tags[i]["confiance_pos"] = 1.0
            if "pos_scores" in tags[i]:
                tags[i]["pos_scores"] = [(override, 1.0)]


def _stocker_g2p(mots: list[MotV2], tags: list[dict], g2p_tagger) -> None:
    """Stocke les resultats G2P dans les MotV2."""
    for i, m in enumerate(mots):
        if i >= len(tags):
            break
        m.pos = tags[i].get("pos", "")
        m.confiance_pos = tags[i].get("confiance_pos", 1.0)
        m.pos_scores = tags[i].get("pos_scores", [])

        # Phone : depuis le tag_words_rich (champ g2p) ou prononcer()
        phone = tags[i].get("g2p", "")
        if not phone and hasattr(g2p_tagger, "prononcer"):
            phone = g2p_tagger.prononcer(m.forme) or ""
        m.phone = phone


# ---------------------------------------------------------------------------
# 2.2 — Ancrage
# ---------------------------------------------------------------------------

def _ancrer(
    mots: list[MotV2],
    lexique,
    lm_homophones,
    config,
) -> None:
    """Determine quels mots sont ancres (pas soumis au P2G).

    Ancrage conservateur : seuls les mots sans ambiguite morphologique
    sont ancres. Les mots variables (NOM, VER, ADJ) passent par P2G
    car leur forme peut etre incorrecte meme si le POS est juste.
    """
    for m in mots:
        low = m.forme.lower()

        # Mot-outil avec override : toujours ancrer.
        # Le P2G ne peut pas desambiguiser ces homophones (son/sont,
        # et/est, a/as) et cree des faux positifs (est->ait, Ces->Ses).
        if _FUNCTION_WORD_POS.get(low) is not None:
            m.ancre_pos = True
            continue

        # Forme elidee (j', s', l', d', etc.)
        if m.original.endswith(("'", "\u2019")):
            m.ancre_pos = True
            continue

        # Mot OOV non corrige : pas de roundtrip fiable
        if not m.dans_lexique:
            m.ancre_pos = True
            continue

        # Les mots restants (dans le lexique, non function words)
        # passent par P2G pour verifier la forme morphologique.


# ---------------------------------------------------------------------------
# 2.3-2.4 — P2G roundtrip + decisions
# ---------------------------------------------------------------------------

def _roundtrip_p2g(
    mots: list[MotV2],
    p2g_adapter,
    lexique,
    config,
) -> None:
    """Execute le roundtrip P2G sur les mots non-ancres et decide.

    Passe TOUS les mots au P2G (pour le contexte de phrase complet)
    mais n'applique les corrections que sur les mots non-ancres.
    """
    # Tous les mots avec un phone valide (pour le contexte)
    ipa_all: list[str] = []
    ortho_all: list[str] = []
    valid_mask: list[bool] = []

    for m in mots:
        if m.phone:
            ipa_all.append(m.phone)
            ortho_all.append(m.forme)
            valid_mask.append(True)
        else:
            # Mot sans phone : utiliser la forme comme placeholder
            ipa_all.append(m.forme)
            ortho_all.append(m.forme)
            valid_mask.append(False)

    if not any(not m.ancre_pos and valid_mask[i] for i, m in enumerate(mots)):
        return

    # P2G en batch sur la phrase entiere (contexte complet)
    p2g_results = p2g_adapter.transcrire(ipa_all, ortho_all)

    # Decisions : seulement sur les mots non-ancres
    for i, m in enumerate(mots):
        if m.ancre_pos:
            continue
        if not valid_mask[i]:
            continue
        if i >= len(p2g_results):
            break

        r = p2g_results[i]

        p2g_top1 = r.get("ortho", "").lower()
        confiance = r.get("confiance", 0.0)
        alternatives = r.get("alternatives", [])

        if not p2g_top1:
            continue

        # P2G confirme la forme courante → rien a faire
        if p2g_top1 == m.forme.lower():
            continue

        # P2G propose une forme differente
        # Verifier que la forme est dans le lexique
        if not (hasattr(lexique, "existe") and lexique.existe(p2g_top1)):
            continue

        # Le P2G doit avoir une confiance suffisante
        if confiance < config.seuil_confiance_p2g:
            continue

        # Appliquer la correction (provisoire, guard peut annuler)
        old_forme = m.forme
        m.forme = p2g_top1
        m.corrections.append(
            (2, "p2g.roundtrip",
             f"{old_forme} -> {p2g_top1} (conf={confiance:.2f})")
        )


# ---------------------------------------------------------------------------
# 2.5 — LM homophones (complementaire au P2G)
# ---------------------------------------------------------------------------

def _correction_homophones(
    mots: list[MotV2],
    lm_homophones,
    config,
) -> None:
    """Corrige les homophones non geres par le P2G via LM trigramme.

    Le P2G excelle pour la morphologie (enfant→enfants, mange→mangent)
    mais manque parfois les homophones purs (son/sont, a/as, et/est).
    Le LM homophones utilise les trigrammes de contexte pour ces cas.

    S'applique uniquement aux mots non-ancres et non deja corriges par P2G.
    """
    if lm_homophones is None:
        return

    n = len(mots)
    for i, m in enumerate(mots):
        if m.ancre_pos:
            continue
        # Deja corrige par P2G → ne pas interferer
        if any(c[0] == 2 for c in m.corrections):
            continue
        # Seulement les homophones connus
        if not lm_homophones.est_homophone(m.forme):
            continue

        ctx_g = mots[i - 1].forme.lower() if i > 0 else None
        ctx_d = mots[i + 1].forme.lower() if i + 1 < n else None

        best, source = lm_homophones.meilleur_homophone(m.forme, ctx_g, ctx_d)

        if source != "LM":
            continue  # pas de donnees trigrammes → pas de correction
        if best.lower() == m.forme.lower():
            continue

        # Verifier que le LM est nettement meilleur
        score_best = lm_homophones.scorer(best, ctx_g, ctx_d)
        score_orig = lm_homophones.scorer(m.forme, ctx_g, ctx_d)
        if score_best <= score_orig:
            continue

        old_forme = m.forme
        m.forme = best.lower()
        m.corrections.append(
            (2, "lm.homophone",
             f"{old_forme} -> {best} (LM score={score_best} vs {score_orig})")
        )


# ---------------------------------------------------------------------------
# 2.6 — Guard conservatif
# ---------------------------------------------------------------------------

def _guard_conservatif(
    mots: list[MotV2],
    lexique,
    pos_ngram,
    lm_homophones,
    formes_originales: list[str],
    config,
) -> None:
    """Annule les corrections douteuses via cross-checks."""
    n = len(mots)

    for i, m in enumerate(mots):
        if m.ancre_pos:
            continue
        if m.forme == formes_originales[i].lower():
            continue  # pas de changement

        # Guard 0 : accord sujet-verbe — ne pas casser l'accord existant
        if _viole_accord_sujet_verbe(mots, i, formes_originales[i]):
            _annuler_correction(m, formes_originales[i])
            continue

        # Guard 1 : POS n-gram — la correction degrade-t-elle la sequence ?
        if config.activer_guard_pos_ngram and pos_ngram is not None:
            pos_tags = [mm.pos or "NOM" for mm in mots]

            # Score avec la correction
            score_new = _score_local_pos(pos_tags, i, pos_ngram)

            # Score avec l'original
            # Recuperer le POS de la forme originale
            orig_low = formes_originales[i].lower()
            orig_infos = lexique.info(orig_low) if hasattr(lexique, "info") else []
            orig_pos_set = {e.get("cgram") for e in orig_infos if e.get("cgram")}
            if orig_pos_set:
                # Tester chaque POS original, garder le meilleur
                best_orig_score = -100.0
                for op in orig_pos_set:
                    old_tags = list(pos_tags)
                    old_tags[i] = op
                    s = _score_local_pos(old_tags, i, pos_ngram)
                    if s > best_orig_score:
                        best_orig_score = s

                # Annuler si le score original est meilleur
                if best_orig_score > score_new:
                    _annuler_correction(m, formes_originales[i])
                    continue

        # Guard 2 : LM homophones (seulement pour les corrections LM,
        # pas P2G — le P2G utilise deja le contexte de phrase complet)
        if config.activer_guard_lm_homo and lm_homophones is not None:
            # Ne s'applique qu'aux corrections LM homophones (pas P2G)
            correction_lm = any(c[1] == "lm.homophone" for c in m.corrections)
            if correction_lm and lm_homophones.est_homophone(formes_originales[i].lower()):
                ctx_g = mots[i - 1].forme.lower() if i > 0 else None
                ctx_d = mots[i + 1].forme.lower() if i + 1 < n else None

                score_new = lm_homophones.scorer(m.forme, ctx_g, ctx_d)
                score_old = lm_homophones.scorer(
                    formes_originales[i].lower(), ctx_g, ctx_d,
                )
                if score_old > score_new:
                    _annuler_correction(m, formes_originales[i])
                    continue


# Pronoms sujet et la terminaison verbe attendue (indicatif present)
_PRONOUN_2SG = frozenset({"tu"})
_PRONOUN_1SG = frozenset({"je", "j"})
_PRONOUN_3PL = frozenset({"ils", "elles"})


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


def _score_local_pos(pos_tags: list[str], idx: int, pos_ngram) -> float:
    """Score POS n-gram local autour de la position idx."""
    n = len(pos_tags)
    score = 0.0

    # Trigrammes impliquant la position idx
    for start in range(max(0, idx - 2), min(n - 2, idx + 1)):
        if start + 2 < n:
            p1 = pos_tags[start] if start >= 0 else "BOS"
            p2 = pos_tags[start + 1]
            p3 = pos_tags[start + 2]
            score += pos_ngram.logp_trigram(p1, p2, p3)

    return score


def _annuler_correction(mot: MotV2, forme_originale: str) -> None:
    """Annule la derniere correction de passe 2."""
    mot.forme = forme_originale.lower()
    mot.corrections = [c for c in mot.corrections if c[0] != 2]


# ---------------------------------------------------------------------------
# 2.6 — Enrichissement morpho
# ---------------------------------------------------------------------------

def _enrichir_morpho(mots: list[MotV2], tags: list[dict]) -> None:
    """Remplit les champs morpho des MotV2 depuis le G2P."""
    for i, m in enumerate(mots):
        if i >= len(tags):
            break

        tag = tags[i]

        # Genre
        genre_raw = tag.get("genre", "")
        if genre_raw in _GENRE_MAP:
            m.genre = _GENRE_MAP[genre_raw]
        elif genre_raw:
            m.genre = genre_raw

        # Nombre
        nombre_raw = tag.get("nombre", "")
        if nombre_raw in _NOMBRE_MAP:
            m.nombre = _NOMBRE_MAP[nombre_raw]
        elif nombre_raw:
            m.nombre = nombre_raw

        # Personne
        personne = tag.get("personne", "")
        if personne in ("1", "2", "3"):
            m.personne = personne

        # PM tag
        m.pm_tag = f"{m.pos}|{m.nombre}|{m.genre}|{m.personne}"
