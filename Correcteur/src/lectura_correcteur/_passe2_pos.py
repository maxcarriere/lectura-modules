"""Passe 2 — Desambiguation POS Viterbi (homophones).

Resout les homophones qui changent de POS (a/a, est/et, son/sont...)
en combinant le G2P neural (confiance per-word) et le n-gram POS
(coherence de sequence) via un Viterbi trigramme.

Etapes :
  2.1 Tagging G2P
  2.2 Overrides mots-outils
  2.3 Ancrage POS (mots a haute confiance)
  2.4 Candidats homophones (mots non-ancres)
  2.5 Emissions (G2P + freq + conservatisme)
  2.6 Viterbi trigram POS
  2.7 Recuperation des formes
  2.8 Guard conservatif
"""

from __future__ import annotations

import logging
import math

from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS
from lectura_correcteur._types import MotV2
from lectura_correcteur._viterbi_core import viterbi_trigram

logger = logging.getLogger(__name__)

_EPS = 1e-6
_DEFAULT_LOGP = -15.0
_BACKOFF_TRIGRAM_THRESHOLD = -14.0


def passe2_pos(
    mots: list[MotV2],
    g2p_tagger,
    lexique,
    pos_ngram,
    lm_homophones,
    config,
) -> None:
    """Passe 2 : desambiguation POS via Viterbi trigramme (in-place).

    Args:
        mots: liste de MotV2 (forme deja corrigee par passe 1)
        g2p_tagger: TaggerProtocol avec tag_words_rich()
        lexique: objet avec existe(), info(), frequence(), phone_de(), homophones()
        pos_ngram: PosNgram (logp_trigram, logp_bigram)
        lm_homophones: LMHomophones (scorer, est_homophone)
        config: CorrecteurV2Config
    """
    if not mots:
        return

    formes = [m.forme for m in mots]

    # 2.1 — Tagging G2P
    tags = g2p_tagger.tag_words_rich(formes)

    # 2.2 — Overrides mots-outils
    _appliquer_overrides(mots, tags)

    # Stocker les resultats G2P dans les mots
    for i, m in enumerate(mots):
        if i < len(tags):
            m.pos = tags[i].get("pos", "")
            m.confiance_pos = tags[i].get("confiance_pos", 1.0)
            m.pos_scores = tags[i].get("pos_scores", [])

    # 2.3 — Ancrage POS
    _ancrer_pos(mots, lexique, lm_homophones, config)

    # 2.4-2.7 — Viterbi POS
    _viterbi_pos(mots, lexique, pos_ngram, lm_homophones, config)

    # 2.8 — Guard conservatif
    _guard_conservatif(mots, lm_homophones, formes, config)

    # 2.9 — Corrections directes LM homophones
    # Pour les mots non modifies par Viterbi, verifier si le LM homophones
    # propose fortement un autre homophone (ratio de score > seuil).
    _corriger_par_lm_homophones(mots, lm_homophones)


# ---------------------------------------------------------------------------
# 2.2 — Overrides mots-outils
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


# ---------------------------------------------------------------------------
# 2.3 — Ancrage POS
# ---------------------------------------------------------------------------

def _ancrer_pos(
    mots: list[MotV2],
    lexique,
    lm_homophones,
    config,
) -> None:
    """Determine quels mots sont ancres (POS fixe, pas de desambiguation)."""
    for m in mots:
        low = m.forme.lower()

        # Mot-outil avec override
        if _FUNCTION_WORD_POS.get(low) is not None:
            m.ancre_pos = True
            continue

        # Forme elidee (j', s', l', d', etc.) : toujours ancree
        if m.original.endswith(("'", "\u2019")):
            m.ancre_pos = True
            continue

        # G2P tres confiant
        if m.confiance_pos >= config.seuil_ancrage_pos:
            m.ancre_pos = True
            continue

        # POS unique dans le lexique et pas homophone
        infos = lexique.info(low) if hasattr(lexique, "info") else []
        cgrams = {e.get("cgram") for e in infos if e.get("cgram")}
        is_homo = (
            lm_homophones is not None
            and lm_homophones.est_homophone(low)
        )
        if len(cgrams) == 1 and not is_homo:
            m.ancre_pos = True
            continue

        # Corrige en Passe 1 (la correction fixe le POS)
        if m.source_ortho:
            m.ancre_pos = True
            continue

        # Mot OOV non corrige : pas de desambiguation POS
        # (pas de forme valide pour generer des homophones fiables)
        if not m.dans_lexique:
            m.ancre_pos = True
            continue


# ---------------------------------------------------------------------------
# 2.4-2.7 — Viterbi POS
# ---------------------------------------------------------------------------

def _viterbi_pos(
    mots: list[MotV2],
    lexique,
    pos_ngram,
    lm_homophones,
    config,
) -> None:
    """Execute le Viterbi POS et recupere les formes."""
    n = len(mots)

    # 2.4 — Construire les candidats (etats + emissions) par position
    all_states: list[list[str]] = []
    all_emissions: list[dict[str, float]] = []
    # Map pos -> (forme, freq) pour recuperer la forme apres Viterbi
    pos_to_forme: list[dict[str, tuple[str, float]]] = []

    for i, m in enumerate(mots):
        if m.ancre_pos:
            # Un seul etat
            all_states.append([m.pos] if m.pos else ["NOM"])
            em_val = config.w_g2p_emission * math.log(max(m.confiance_pos, _EPS))
            all_emissions.append({m.pos or "NOM": em_val})
            pos_to_forme.append({m.pos or "NOM": (m.forme, 1.0)})
        else:
            # Candidats : POS du mot lui-meme + homophones
            candidates = _build_pos_candidates(m, lexique, lm_homophones)
            states_i: list[str] = []
            emissions_i: dict[str, float] = {}
            formes_i: dict[str, tuple[str, float]] = {}

            for forme, pos, freq, is_original in candidates:
                if pos in emissions_i:
                    # Garder le candidat avec la meilleure emission
                    old_em = emissions_i[pos]
                    new_em = _compute_emission(
                        forme, pos, freq, is_original, m, config,
                    )
                    if new_em > old_em:
                        emissions_i[pos] = new_em
                        formes_i[pos] = (forme, freq)
                else:
                    states_i.append(pos)
                    emissions_i[pos] = _compute_emission(
                        forme, pos, freq, is_original, m, config,
                    )
                    formes_i[pos] = (forme, freq)

            if not states_i:
                states_i = [m.pos or "NOM"]
                emissions_i = {m.pos or "NOM": 0.0}
                formes_i = {m.pos or "NOM": (m.forme, 0.0)}

            all_states.append(states_i)
            all_emissions.append(emissions_i)
            pos_to_forme.append(formes_i)

    # 2.6 — Viterbi trigram POS
    def transition_fn(p1: str, p2: str, p3: str) -> float:
        tri = pos_ngram.logp_trigram(p1, p2, p3)
        if tri > _BACKOFF_TRIGRAM_THRESHOLD:
            return tri
        bi = pos_ngram.logp_bigram(p2, p3)
        return bi - 1.0

    results = viterbi_trigram(
        all_states, all_emissions, transition_fn,
        w_emission=1.0, w_transition=1.0,
    )

    # 2.7 — Recuperation des formes
    for i, (chosen_pos, confiance) in enumerate(results):
        m = mots[i]
        if m.ancre_pos:
            continue

        old_forme = m.forme
        old_pos = m.pos

        m.pos = chosen_pos
        m.confiance_pos = confiance

        # Recuperer la forme associee au POS choisi
        if chosen_pos in pos_to_forme[i]:
            new_forme, _freq = pos_to_forme[i][chosen_pos]
            if new_forme != old_forme.lower():
                m.forme = new_forme
                m.corrections.append(
                    (2, "pos.viterbi",
                     f"{old_forme}({old_pos}) -> {new_forme}({chosen_pos})")
                )


def _build_pos_candidates(
    mot: MotV2,
    lexique,
    lm_homophones,
) -> list[tuple[str, str, float, bool]]:
    """Construit les candidats (forme, pos, freq, is_original) pour un mot non-ancre.

    Sources :
    - POS du mot lui-meme (toujours)
    - POS des homophones (via lexique.homophones)
    """
    low = mot.forme.lower()
    candidates: list[tuple[str, str, float, bool]] = []
    seen: set[tuple[str, str]] = set()

    # POS du mot lui-meme
    infos = lexique.info(low) if hasattr(lexique, "info") else []
    pos_freq: dict[str, float] = {}
    for entry in infos:
        cgram = entry.get("cgram", "")
        if cgram:
            freq = float(entry.get("freq") or 0)
            pos_freq[cgram] = pos_freq.get(cgram, 0) + max(freq, 0.01)

    # Override mots-outils
    override = _FUNCTION_WORD_POS.get(low)
    if override and override not in pos_freq:
        pos_freq[override] = 100.0

    for pos, freq in pos_freq.items():
        key = (low, pos)
        if key not in seen:
            seen.add(key)
            candidates.append((low, pos, freq, True))

    # Homophones via LM homophones — seulement les paires connues du modele
    # de langue homophones. Les homophones phonetiques arbitraires (via phone_de)
    # produisent trop de faux positifs (guidé→guide, aie→hais, pratiqué→pratique).
    if lm_homophones is not None and lm_homophones.est_homophone(low):
        same_lemma = getattr(lm_homophones, "_same_lemma_pairs", set())
        for cand_ortho, _cand_freq in lm_homophones.candidats(low):
            ortho_low = cand_ortho.lower()
            if ortho_low == low:
                continue
            if (low, ortho_low) in same_lemma:
                continue

            # POS de l'homophone — filtrer les mots tres rares
            homo_infos = lexique.info(ortho_low) if hasattr(lexique, "info") else []
            homo_pos_freq: dict[str, float] = {}
            for e in homo_infos:
                cg = e.get("cgram", "")
                if cg:
                    f = float(e.get("freq") or 0)
                    homo_pos_freq[cg] = homo_pos_freq.get(cg, 0) + max(f, 0.01)

            for pos, freq in homo_pos_freq.items():
                if freq < 1.0:
                    continue
                key = (ortho_low, pos)
                if key not in seen:
                    seen.add(key)
                    candidates.append((ortho_low, pos, freq, False))

    # Fallback si aucun candidat
    if not candidates:
        candidates.append((low, mot.pos or "NOM", 0.01, True))

    return candidates


# ---------------------------------------------------------------------------
# 2.5 — Emissions
# ---------------------------------------------------------------------------

def _compute_emission(
    forme: str,
    pos: str,
    freq: float,
    is_original: bool,
    mot: MotV2,
    config,
) -> float:
    """Calcule le log-score d'emission pour un candidat POS.

    emission = w_g2p * log(g2p_prob[pos])
             + w_freq * log(freq)
             + w_conserv * (bonus si forme originale)
    """
    # G2P probability for this POS
    g2p_prob = 0.01  # default if POS not in pos_scores
    for p, prob in mot.pos_scores:
        if p == pos:
            g2p_prob = max(prob, _EPS)
            break

    em = config.w_g2p_emission * math.log(max(g2p_prob, _EPS))
    em += config.w_freq_emission * math.log(max(freq, 0.01) + _EPS)

    if is_original:
        em += config.w_conserv_emission

    return em


# ---------------------------------------------------------------------------
# 2.8 — Guard conservatif
# ---------------------------------------------------------------------------

def _guard_conservatif(
    mots: list[MotV2],
    lm_homophones,
    formes_originales: list[str],
    config,
) -> None:
    """Annule les corrections de faible confiance ou en conflit avec le LM.

    - Ne corriger que si confiance Viterbi >= seuil_correction_pos
    - Cross-check avec lm_homophones : si les deux divergent et
      confiance < 0.85, garder l'original
    """
    for i, m in enumerate(mots):
        if m.ancre_pos:
            continue
        if m.forme == formes_originales[i].lower():
            continue  # pas de changement

        # Guard : confiance trop basse
        if m.confiance_pos < config.seuil_correction_pos:
            _annuler_correction(m, formes_originales[i])
            continue

        # Guard : cross-check LM homophones
        if lm_homophones is not None and m.confiance_pos < 0.85:
            ctx_g = mots[i - 1].forme if i > 0 else None
            ctx_d = mots[i + 1].forme if i + 1 < len(mots) else None

            score_new = lm_homophones.scorer(m.forme, ctx_g, ctx_d)
            score_old = lm_homophones.scorer(
                formes_originales[i].lower(), ctx_g, ctx_d,
            )
            if score_old > score_new:
                _annuler_correction(m, formes_originales[i])


def _annuler_correction(mot: MotV2, forme_originale: str) -> None:
    """Annule la derniere correction de passe 2."""
    mot.forme = forme_originale.lower()
    # Retirer la derniere correction de passe 2 si presente
    mot.corrections = [c for c in mot.corrections if c[0] != 2]


# ---------------------------------------------------------------------------
# 2.9 — Corrections directes LM homophones
# ---------------------------------------------------------------------------

# Paires d'homophones fiables pour correction via LM trigramme.
# Seules ces paires sont traitees — les autres produisent trop de FP.
# Ces paires BYPASSENT l'ancrage POS et le filtre mots-outils,
# car les mots-outils ancres sont souvent les cibles d'erreurs homophones
# (sont/son, sa/ça, on/ont, etc.).
_PAIRES_HOMOPHONES_FIABLES: set[frozenset[str]] = {
    frozenset({"a", "à"}),
    frozenset({"se", "ce"}),
    frozenset({"sa", "ça"}),
    frozenset({"ou", "où"}),
    frozenset({"la", "là"}),
    frozenset({"sont", "son"}),
    frozenset({"on", "ont"}),
    frozenset({"quand", "quant"}),
    frozenset({"peu", "peut"}),
}

# Ratio minimum score_best/score_curr pour accepter une correction.
# 50x = tres conservateur, ne corrige que les cas tres clairs.
_LM_HOMO_RATIO_MIN = 50.0
# Score minimum du meilleur candidat (evite les contextes sans donnees).
_LM_HOMO_SCORE_MIN_BEST = 1000.0

def _corriger_par_lm_homophones(
    mots: list[MotV2],
    lm_homophones,
) -> None:
    """Correction directe par le LM homophones pour les paires fiables.

    Traite les homophones grammaticaux courants (a/à, sont/son, etc.)
    en utilisant le LM trigramme. BYPASS l'ancrage POS pour les paires
    whitelistees car ces mots sont souvent ancres comme mots-outils
    mais peuvent etre errones (ex: "sont" ancre AUX mais devrait etre "son").

    Critere : ratio score_best/score_curr >= 50 et score_best >= 1000.
    """
    if lm_homophones is None:
        return

    n = len(mots)
    for i, m in enumerate(mots):
        low = m.forme.lower()

        # Elision : ne pas corriger
        if m.original.endswith(("'", "\u2019")) or low.endswith("'"):
            continue

        # Mot corrige en passe 1 : ne pas re-corriger
        if m.source_ortho:
            continue

        if not lm_homophones.est_homophone(low):
            continue

        # Contexte gauche/droite
        ctx_g = mots[i - 1].forme.lower() if i > 0 else None
        ctx_d = mots[i + 1].forme.lower() if i + 1 < n else None

        best, _source = lm_homophones.meilleur_homophone(low, ctx_g, ctx_d)
        if best is None or best == low:
            continue

        # Verifier que la paire est dans les homophones fiables
        pair = frozenset({low, best.lower()})
        if pair not in _PAIRES_HOMOPHONES_FIABLES:
            continue

        # Scoring par ratio : score_best / score_curr >= ratio_min
        score_best = lm_homophones.scorer(best, ctx_g, ctx_d)
        score_curr = lm_homophones.scorer(low, ctx_g, ctx_d)

        if score_best < _LM_HOMO_SCORE_MIN_BEST:
            continue

        ratio = score_best / max(score_curr, 1.0)
        if ratio >= _LM_HOMO_RATIO_MIN:
            old_forme = m.forme
            m.forme = best
            m.corrections.append(
                (2, "pos.lm_homo",
                 f"{old_forme} -> {best} (curr={score_curr:.0f}, best={score_best:.0f}, ratio={ratio:.0f})")
            )
