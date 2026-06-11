"""Pass 2 : Analyse grammaticale complete — Viterbi contraint avec ancres.

Analyse chaque mot d'une phrase en POS + traits morphologiques (nombre,
genre, personne, temps, mode) avec un score de confiance.

Architecture :
1. Extraction des hypotheses PM par mot (lexique)
2. Identification des ancres (mots-outils, mots non-ambigus → 1 seul etat)
3. Viterbi trigramme contraint sur PM tags
4. Confiance = delta normalise entre meilleur et deuxieme meilleur score

Ce module est standalone : il ne modifie rien, il produit une analyse
que Pass 3 (regles de correction) pourra exploiter mecaniquement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS
from lectura_correcteur._viterbi_morpho import (
    _BOS,
    _DEFAULT_LOGP,
    _EPS,
    _GENRE_MAP,
    _NOMBRE_MAP,
    _edit_distance_simple,
    CandidatMorpho,
)


# ---------------------------------------------------------------------------
# Structure de sortie
# ---------------------------------------------------------------------------

@dataclass
class AnalyseMot:
    """Resultat d'analyse grammaticale pour un mot."""
    forme: str               # forme originale (input)
    forme_corrigee: str = "" # forme selectionnee par Viterbi (vide si inchangee)
    pos: str = ""           # "NOM", "VER", "ADJ", "ART", etc.
    nombre: str = ""        # "Sing", "Plur", "_"
    genre: str = ""         # "Masc", "Fem", "_"
    personne: str = ""      # "1", "2", "3", "_"
    temps: str = ""         # "pre", "imp", "pas", "fut", "_"
    mode: str = ""          # "ind", "sub", "con", "inf", "par", "_"
    confiance: float = 0.0  # 0.0-1.0
    ancre: bool = False     # True = haute confiance, 1 seul etat Viterbi
    pm_tag: str = ""        # ex: "NOM|Plur|Masc"
    candidats_pm: list[str] = field(default_factory=list)
    conflits: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mapping personne lexique → long
# ---------------------------------------------------------------------------

_PERSONNE_MAP = {"1": "1", "2": "2", "3": "3", "": "_", "_": "_"}
_TEMPS_MAP = {
    "pre": "pre", "imp": "imp", "pas": "pas", "fut": "fut",
    "present": "pre", "imparfait": "imp", "passe_simple": "pas", "futur": "fut",
    "": "_", "_": "_",
}
_MODE_MAP = {
    "ind": "ind", "sub": "sub", "con": "con", "inf": "inf", "par": "par",
    "indicatif": "ind", "subjonctif": "sub", "conditionnel": "con",
    "infinitif": "inf", "participe": "par",
    "": "_", "_": "_",
}

# Mapping etendu pour formes_de() qui retourne des valeurs longues
# ("singulier"/"pluriel"/"masculin"/"feminin") vs info() ("s"/"p"/"m"/"f")
_GENRE_MAP_EXT = {
    "m": "Masc", "f": "Fem", "": "_", "_": "_",
    "masculin": "Masc", "feminin": "Fem", "féminin": "Fem",
}
_NOMBRE_MAP_EXT = {
    "s": "Sing", "p": "Plur", "": "_", "_": "_",
    "singulier": "Sing", "pluriel": "Plur",
}


# ---------------------------------------------------------------------------
# Filtres homophones
# ---------------------------------------------------------------------------

# Fix 1 : frequence minimale pour qu'un homophone soit considere
_MIN_FREQ_HOMOPHONE = 1.0       # frequence absolue minimale
_MIN_FREQ_RATIO_HOMOPHONE = 0.01  # ratio min freq(candidat)/freq(original)

# Fix 2 : elisions qui ne doivent jamais etre corrigees par un homophone
_ELISIONS_PROTEGEES = frozenset({
    "l", "d", "s", "n", "m", "t", "j", "c",
    "qu", "jusqu", "lorsqu", "puisqu", "quelqu",
})


# ---------------------------------------------------------------------------
# 2a. Extraction des hypotheses par mot
# ---------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    """Supprime les accents d'une chaine."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# Variantes accent typiques pour retrouver un mot dans le lexique
_ACCENT_VARIANTS: dict[str, list[str]] = {
    "e": ["é", "è", "ê", "ë"],
    "a": ["à", "â"],
    "u": ["ù", "û", "ü"],
    "i": ["î", "ï"],
    "o": ["ô"],
    "c": ["ç"],
}


def _trouver_avec_accents(forme_sans_accents: str, lexique) -> list[dict]:
    """Cherche le mot dans le lexique en essayant des variantes avec accents.

    Strategie rapide : pour chaque position, tester les substitutions
    accent par accent. S'arrete au premier match.
    """
    low = forme_sans_accents.lower()
    stripped = _strip_accents(low)

    # Si le mot avec accents est different du stripped, il a deja des accents
    if stripped != low:
        return []

    # Essai 1 : chercher dans le lexique via existe() pour des variantes simples
    # On genere les variantes position par position (max 1 accent a la fois)
    for i, c in enumerate(low):
        if c in _ACCENT_VARIANTS:
            for repl in _ACCENT_VARIANTS[c]:
                variante = low[:i] + repl + low[i + 1:]
                infos = lexique.info(variante) if hasattr(lexique, "info") else []
                if infos:
                    return infos

    # Essai 2 : variantes avec 2 accents (ex: "eleve" → "élève")
    positions = [(i, c) for i, c in enumerate(low) if c in _ACCENT_VARIANTS]
    if len(positions) >= 2:
        for p1_idx in range(len(positions)):
            i1, c1 = positions[p1_idx]
            for r1 in _ACCENT_VARIANTS[c1]:
                v1 = low[:i1] + r1 + low[i1 + 1:]
                for p2_idx in range(p1_idx + 1, len(positions)):
                    i2, c2 = positions[p2_idx]
                    for r2 in _ACCENT_VARIANTS[c2]:
                        variante = v1[:i2] + r2 + v1[i2 + 1:]
                        infos = lexique.info(variante) if hasattr(lexique, "info") else []
                        if infos:
                            return infos

    return []


def _extraire_hypotheses(
    forme: str,
    lexique,
) -> list[CandidatMorpho]:
    """Toutes les hypotheses PM pour un mot depuis le lexique.

    PM tag = POS|Nombre|Genre|Personne (4 champs).
    Pas de bonus POS ici — on ne presuppose rien.
    Si le mot n'est pas dans le lexique, essaie des variantes avec accents.
    """
    low = forme.lower()
    infos = lexique.info(low) if hasattr(lexique, "info") else []

    # Fallback : mot sans accents → essayer avec accents
    if not infos:
        infos = _trouver_avec_accents(low, lexique)

    candidates: list[CandidatMorpho] = []
    seen_pm: set[str] = set()

    # Override POS avec sous-type pour mots-outils (ex: ART → ART:def)
    override_pos = _FUNCTION_WORD_POS.get(low)

    for entry in infos:
        cgram = entry.get("cgram", "")
        if not cgram:
            continue

        # Appliquer le sous-type du mot-outil si le cgram de base correspond
        # Ex: lexique donne "ART" pour "les", override = "ART:def"
        if override_pos and ":" not in cgram:
            override_base = override_pos.split(":")[0]
            if cgram == override_base:
                cgram = override_pos

        genre_raw = entry.get("genre", "") or ""
        nombre_raw = entry.get("nombre", "") or ""
        personne_raw = entry.get("personne", "") or ""
        freq = float(entry.get("freq") or 0)

        genre = _GENRE_MAP_EXT.get(genre_raw, "_")
        nombre = _NOMBRE_MAP_EXT.get(nombre_raw, "_")
        personne = _PERSONNE_MAP.get(personne_raw, "_")

        pm_tag = f"{cgram}|{nombre}|{genre}|{personne}"

        if pm_tag in seen_pm:
            for c in candidates:
                if c.pm_tag == pm_tag:
                    c.freq += max(freq, 0.01)
                    break
            continue

        seen_pm.add(pm_tag)
        candidates.append(CandidatMorpho(
            forme=low,
            pm_tag=pm_tag,
            pos=cgram,
            genre=genre,
            nombre=nombre,
            freq=max(freq, 0.01),
            emission=0.0,  # calcule apres
        ))

    # Recalculer emissions
    for c in candidates:
        c.emission = math.log(c.freq + _EPS)

    if not candidates:
        candidates.append(CandidatMorpho(
            forme=low, pm_tag="NOM|_|_|_", pos="NOM",
            genre="_", nombre="_", freq=0.01, emission=0.0,
        ))

    return candidates


# ---------------------------------------------------------------------------
# 2a-bis. Extraction des hypotheses elargies (morpho + homophones)
# ---------------------------------------------------------------------------

# Temps/modes verbaux : on ne corrige pas "mange" → "mangeait"
# (on garde le meme temps/mode pour eviter les fausses corrections)
_TEMPS_MODE_KEYS = ("temps", "mode")


def _extraire_hypotheses_elargies(
    forme: str,
    lexique,
    *,
    expand_morpho: bool = True,
    expand_homophones: bool = True,
    penalite_morpho: float = -3.0,
    penalite_homophone: float = -5.0,
) -> tuple[list[CandidatMorpho], dict[str, str]]:
    """Hypotheses PM elargies : forme originale + variantes morpho + homophones.

    Retourne:
        (candidates, pm_to_forme)
        pm_to_forme : {pm_tag: forme_selectionnee} pour retrouver la forme
                      apres le Viterbi.
    """
    # --- Niveau 1 : forme originale (identique a _extraire_hypotheses) ---
    base_candidates = _extraire_hypotheses(forme, lexique)

    candidates: list[CandidatMorpho] = list(base_candidates)
    seen_pm: set[str] = {c.pm_tag for c in candidates}
    pm_to_forme: dict[str, str] = {c.pm_tag: c.forme for c in candidates}

    low = forme.lower()
    infos = lexique.info(low) if hasattr(lexique, "info") else []
    if not infos:
        infos = _trouver_avec_accents(low, lexique)

    # --- Niveau 2 : variantes morphologiques (meme lemme) ---
    if expand_morpho and hasattr(lexique, "formes_de"):
        # Meilleure emission originale par POS de base
        # (pour ancrer les variantes relativement a l'original)
        best_orig_em_par_pos: dict[str, float] = {}
        for c in base_candidates:
            pb = c.pos.split(":")[0]
            if pb not in best_orig_em_par_pos or c.emission > best_orig_em_par_pos[pb]:
                best_orig_em_par_pos[pb] = c.emission

        # Collecter lemmes et POS de base de la forme originale
        lemmes_par_pos: dict[str, set[str]] = {}  # pos_base → {lemmes}
        temps_mode_orig: dict[str, tuple[str, str]] = {}  # pos_base → (temps, mode)
        for entry in infos:
            cgram = entry.get("cgram", "")
            lemme = entry.get("lemme", "")
            if not cgram or not lemme:
                continue
            pos_base = cgram.split(":")[0]
            if pos_base not in lemmes_par_pos:
                lemmes_par_pos[pos_base] = set()
            lemmes_par_pos[pos_base].add(lemme.lower())
            # Garder temps/mode de la forme originale pour filtrer les verbes
            if pos_base in ("VER", "AUX") and pos_base not in temps_mode_orig:
                t = entry.get("temps", "") or ""
                m = entry.get("mode", "") or ""
                temps_mode_orig[pos_base] = (t, m)

        for pos_base, lemmes in lemmes_par_pos.items():
            for lemme in lemmes:
                variantes = lexique.formes_de(lemme)
                if not variantes:
                    continue
                for var_entry in variantes:
                    var_forme = (var_entry.get("ortho", "") or "").lower()
                    if not var_forme or var_forme == low:
                        continue

                    var_cgram = var_entry.get("cgram", "")
                    if not var_cgram:
                        continue

                    # Restriction POS : meme categorie de base
                    if var_cgram.split(":")[0] != pos_base:
                        continue

                    # Restriction edit distance <= 3
                    if _edit_distance_simple(low, var_forme) > 3:
                        continue

                    # Restriction verbes : meme temps/mode
                    # formes_de() retourne des valeurs longues ("present"/"imparfait")
                    # vs info() qui retourne des codes courts ("pre"/"imp")
                    if pos_base in ("VER", "AUX"):
                        orig_tm = temps_mode_orig.get(pos_base, ("", ""))
                        var_t = _TEMPS_MAP.get(var_entry.get("temps", "") or "", "_")
                        var_m = _MODE_MAP.get(var_entry.get("mode", "") or "", "_")
                        orig_t_norm = _TEMPS_MAP.get(orig_tm[0], "_")
                        orig_m_norm = _MODE_MAP.get(orig_tm[1], "_")
                        if orig_t_norm != "_" and var_t != "_" and var_t != orig_t_norm:
                            continue
                        if orig_m_norm != "_" and var_m != "_" and var_m != orig_m_norm:
                            continue

                    # Utiliser les maps etendues (formes_de retourne "pluriel"/"masculin")
                    genre_raw = var_entry.get("genre", "") or ""
                    nombre_raw = var_entry.get("nombre", "") or ""
                    personne_raw = var_entry.get("personne", "") or ""

                    genre = _GENRE_MAP_EXT.get(genre_raw, "_")
                    nombre = _NOMBRE_MAP_EXT.get(nombre_raw, "_")
                    personne = _PERSONNE_MAP.get(personne_raw, "_")
                    pm_tag = f"{var_cgram}|{nombre}|{genre}|{personne}"

                    if pm_tag in seen_pm:
                        continue
                    seen_pm.add(pm_tag)

                    # Frequence : formes_de() retourne souvent freq=0
                    # → recuperer la vraie frequence via info()
                    freq = float(var_entry.get("freq") or 0)
                    if freq < 0.1 and hasattr(lexique, "info"):
                        var_infos = lexique.info(var_forme)
                        for vi in var_infos:
                            vi_cgram = vi.get("cgram", "")
                            if vi_cgram.split(":")[0] == pos_base:
                                vi_freq = float(vi.get("freq") or 0)
                                if vi_freq > freq:
                                    freq = vi_freq

                    # Emission = meilleure emission originale (meme POS) + penalite
                    # Cela garantit que le delta d'emission entre original et variante
                    # est exactement |penalite|, independamment des frequences.
                    anchor_em = best_orig_em_par_pos.get(pos_base, 0.0)
                    em = anchor_em + penalite_morpho

                    candidates.append(CandidatMorpho(
                        forme=var_forme,
                        pm_tag=pm_tag,
                        pos=var_cgram,
                        genre=genre,
                        nombre=nombre,
                        freq=max(freq, 0.01),
                        emission=em,
                    ))
                    pm_to_forme[pm_tag] = var_forme

    # --- Niveau 3 : homophones (meme prononciation, POS different) ---
    if expand_homophones and hasattr(lexique, "phone_de") and hasattr(lexique, "homophones"):
        # Fix 2 : proteger les elisions et mots hors lexique (noms propres, sigles)
        # Les elisions (l', d', s', n', etc.) ne doivent jamais etre "corrigees"
        if low in _ELISIONS_PROTEGEES:
            pass  # skip homophones entierement
        elif not infos:
            pass  # mot absent du lexique (nom propre, sigle) → pas d'homophones
        else:
            phone = lexique.phone_de(low)
            if phone:
                # Frequence max de la forme originale
                orig_max_freq = max(
                    (float(e.get("freq") or 0) for e in infos),
                    default=0.0,
                )

                # POS de base de la forme originale
                orig_pos_bases = {e.get("cgram", "").split(":")[0] for e in infos if e.get("cgram")}

                homophones = lexique.homophones(phone)
                for h_entry in homophones:
                    h_forme = (h_entry.get("ortho", "") or "").lower()
                    if not h_forme or h_forme == low:
                        continue

                    h_cgram = h_entry.get("cgram", "")
                    if not h_cgram:
                        continue

                    # Homophones : POS different de la forme originale
                    h_pos_base = h_cgram.split(":")[0]
                    if h_pos_base in orig_pos_bases:
                        continue

                    freq = float(h_entry.get("freq") or 0)

                    # Fix 1 : filtre de frequence absolue et relative
                    # Un homophone doit avoir une frequence minimale pour etre
                    # credible comme correction (elimine "quis", "cèle", etc.)
                    if freq < _MIN_FREQ_HOMOPHONE:
                        continue
                    if orig_max_freq > 0 and freq < orig_max_freq * _MIN_FREQ_RATIO_HOMOPHONE:
                        continue

                    genre_raw = h_entry.get("genre", "") or ""
                    nombre_raw = h_entry.get("nombre", "") or ""
                    personne_raw = h_entry.get("personne", "") or ""

                    genre = _GENRE_MAP.get(genre_raw, "_")
                    nombre = _NOMBRE_MAP.get(nombre_raw, "_")
                    personne = _PERSONNE_MAP.get(personne_raw, "_")
                    pm_tag = f"{h_cgram}|{nombre}|{genre}|{personne}"

                    if pm_tag in seen_pm:
                        continue
                    seen_pm.add(pm_tag)

                    # Fix 3 : penalite dynamique basee sur le ratio de frequences
                    # Un homophone courant (a/à) recoit ~penalite_homophone
                    # Un homophone rare (qui/quis) recoit une penalite beaucoup plus forte
                    best_orig_em = max((c.emission for c in base_candidates), default=0.0)
                    if orig_max_freq > 0 and freq > 0:
                        freq_ratio = min(freq / orig_max_freq, 1.0)
                        # log(ratio) est negatif ou nul : plus le candidat est rare, plus on penalise
                        freq_penalty = math.log(max(freq_ratio, 1e-6))
                    else:
                        freq_penalty = -10.0
                    em = best_orig_em + penalite_homophone + freq_penalty

                    candidates.append(CandidatMorpho(
                        forme=h_forme,
                        pm_tag=pm_tag,
                        pos=h_cgram,
                        genre=genre,
                        nombre=nombre,
                        freq=max(freq, 0.01),
                        emission=em,
                    ))
                    pm_to_forme[pm_tag] = h_forme

    return candidates, pm_to_forme


# ---------------------------------------------------------------------------
# 2b. Identification des ancres
# ---------------------------------------------------------------------------

def _est_ancre(
    forme: str,
    candidates: list[CandidatMorpho],
    lm_homophones=None,
) -> tuple[bool, str | None]:
    """Determine si un mot est une ancre (confiance ~1.0).

    Retourne (is_anchor, forced_pm_tag).
    """
    low = forme.lower()

    # 1. Mot-outil avec POS force
    override = _FUNCTION_WORD_POS.get(low)
    if override is not None:
        # Trouver le candidat PM correspondant a l'override
        for c in candidates:
            if c.pos == override or c.pos.startswith(override.split(":")[0]):
                return True, c.pm_tag
        # Si pas de candidat exact, forcer quand meme avec le premier du bon POS
        # Construire un PM tag generique
        return True, f"{override}|_|_|_"

    # 2. Un seul PM tag possible (non homophone)
    if len(candidates) == 1:
        if lm_homophones is None or not lm_homophones.est_homophone(low):
            return True, candidates[0].pm_tag

    # 3. Plusieurs candidats mais tous le meme cgram+nombre+genre (formes differentes du meme lemme)
    pm_tags_uniques = {c.pm_tag for c in candidates}
    if len(pm_tags_uniques) == 1:
        return True, candidates[0].pm_tag

    return False, None


def _trouver_pm_pour_override(override_pos: str, candidates: list[CandidatMorpho]) -> str:
    """Trouve le PM tag le plus probable pour un POS override."""
    base = override_pos.split(":")[0]
    # Chercher un candidat avec le meme POS de base
    matching = [c for c in candidates if c.pos.split(":")[0] == base]
    if matching:
        return max(matching, key=lambda c: c.freq).pm_tag
    return f"{override_pos}|_|_|_"


# ---------------------------------------------------------------------------
# G2P emission prior
# ---------------------------------------------------------------------------

# Mapping G2P FR morpho values → PM tag long values
_FR_NOMBRE_TO_LONG = {"s": "Sing", "p": "Plur"}
_FR_GENRE_TO_LONG = {"m": "Masc", "f": "Fem"}


def _compute_g2p_emissions(
    mots: list[str],
    tagger,
    candidates_par_pos: list[list[CandidatMorpho]],
) -> list[dict[str, float]]:
    """Compute G2P-based emission bonuses for Viterbi candidates.

    Uses the G2P neural model's POS + morpho predictions as a prior.
    For each PM candidate, computes:
    - POS bonus: log P_g2p(POS matches candidate POS)
    - Morpho bonus: small reward/penalty for matching nombre/genre/personne

    Returns:
        List of {pm_tag: bonus} dicts, one per position.
    """
    n = len(mots)
    g2p_em: list[dict[str, float]] = [{} for _ in range(n)]

    try:
        g2p_tags = tagger.tag_words_rich(mots)
    except Exception:
        return g2p_em

    for i in range(min(n, len(g2p_tags))):
        g2p = g2p_tags[i]
        g2p_conf = float(g2p.get("confiance_pos", 0.5))

        # POS probability distribution from G2P
        pos_probs: dict[str, float] = {}
        for pos_label, prob in g2p.get("pos_scores", []):
            pos_probs[pos_label] = prob

        # G2P morpho predictions (FR convention → long)
        g2p_nombre = _FR_NOMBRE_TO_LONG.get(str(g2p.get("nombre", "")), "_")
        g2p_genre = _FR_GENRE_TO_LONG.get(str(g2p.get("genre", "")), "_")
        g2p_personne = str(g2p.get("personne", "_")) or "_"

        for cand in candidates_par_pos[i]:
            parts = cand.pm_tag.split("|")
            cand_pos = parts[0] if len(parts) > 0 else ""
            cand_nombre = parts[1] if len(parts) > 1 else "_"
            cand_genre = parts[2] if len(parts) > 2 else "_"
            cand_personne = parts[3] if len(parts) > 3 else "_"

            # POS bonus: log P_g2p(matching POS)
            # Sum probabilities for exact match + base match (ART:def → ART)
            pos_prob = pos_probs.get(cand_pos, 0.0)
            if pos_prob == 0.0:
                cand_base = cand_pos.split(":")[0]
                for plabel, pprob in pos_probs.items():
                    if plabel.split(":")[0] == cand_base:
                        pos_prob += pprob

            if pos_prob > 0:
                bonus = math.log(max(pos_prob, 1e-4))
            else:
                bonus = -8.0  # floor for completely unseen POS

            # Morpho bonus: reward matching nombre/genre/personne
            morpho_bonus = 0.0
            if g2p_nombre != "_" and cand_nombre != "_":
                morpho_bonus += 0.5 if cand_nombre == g2p_nombre else -0.5
            if g2p_genre != "_" and cand_genre != "_":
                morpho_bonus += 0.5 if cand_genre == g2p_genre else -0.5
            if g2p_personne != "_" and cand_personne != "_":
                morpho_bonus += 0.5 if cand_personne == g2p_personne else -0.5

            # Weight morpho bonus by G2P confidence
            bonus += morpho_bonus * g2p_conf
            g2p_em[i][cand.pm_tag] = bonus

    return g2p_em


# ---------------------------------------------------------------------------
# 2c. Viterbi contraint
# ---------------------------------------------------------------------------

def _transition_pm_logp(pos_ngram, pm1: str, pm2: str, pm3: str) -> float:
    """Log-probabilite de transition PM avec backoff KN structure.

    Le modele Kneser-Ney gere le backoff en interne :
    trigram absent → backoff(pm1,pm2) + bigram(pm2,pm3)
    bigram absent  → backoff(pm2)     + unigram(pm3)
    """
    return pos_ngram.logp_pm_trigram(pm1, pm2, pm3)


def _viterbi_contraint(
    candidates_par_pos: list[list[CandidatMorpho]],
    ancres: list[tuple[bool, str | None]],
    pos_ngram,
    w_emission: float = 1.0,
    w_transition: float = 1.0,
    g2p_emissions: list[dict[str, float]] | None = None,
    w_g2p: float = 1.0,
) -> tuple[list[str], list[list[tuple[str, float]]]]:
    """Viterbi trigramme avec contraintes d'ancrage.

    Aux positions ancrees, un seul etat est considere.

    Returns:
        (pm_sequence, scores_par_position)
        scores_par_position[t] = [(pm_tag, score), ...] trie par score desc
    """
    n = len(candidates_par_pos)
    if n == 0:
        return [], []

    # Etats par position : 1 si ancre, sinon tous les candidats PM
    states: list[list[str]] = []
    emissions: list[dict[str, float]] = []

    for t in range(n):
        is_anchor, forced_pm = ancres[t]
        cands = candidates_par_pos[t]

        if is_anchor and forced_pm is not None:
            # Un seul etat force
            # Chercher l'emission du candidat correspondant
            em = 0.0
            for c in cands:
                if c.pm_tag == forced_pm:
                    em = c.emission
                    break
            states.append([forced_pm])
            emissions.append({forced_pm: em})
        else:
            pm_em: dict[str, float] = {}
            for c in cands:
                if c.pm_tag not in pm_em or c.emission > pm_em[c.pm_tag]:
                    pm_em[c.pm_tag] = c.emission
            states.append(list(pm_em.keys()))
            emissions.append(pm_em)

    # --- Viterbi trigramme ---

    # viterbi_scores[t] : {(pm_{t-1}, pm_t): score}
    viterbi_scores: list[dict[tuple[str, str], float]] = []
    backptrs: list[dict[tuple[str, str], tuple[str, str]]] = []

    # t=0
    v0: dict[tuple[str, str], float] = {}
    for c in states[0]:
        trans = _transition_pm_logp(pos_ngram, _BOS, _BOS, c)
        g2p_b = g2p_emissions[0].get(c, 0.0) if g2p_emissions else 0.0
        score = w_transition * trans + w_emission * emissions[0].get(c, 0.0) + w_g2p * g2p_b
        state = (_BOS, c)
        if state not in v0 or score > v0[state]:
            v0[state] = score
    viterbi_scores.append(v0)
    backptrs.append({})

    # Recurrence
    for t in range(1, n):
        vt: dict[tuple[str, str], float] = {}
        bt: dict[tuple[str, str], tuple[str, str]] = {}

        for (pp, p), prev_score in viterbi_scores[t - 1].items():
            for c in states[t]:
                trans = _transition_pm_logp(pos_ngram, pp, p, c)
                g2p_b = g2p_emissions[t].get(c, 0.0) if g2p_emissions else 0.0
                score = (
                    prev_score
                    + w_transition * trans
                    + w_emission * emissions[t].get(c, 0.0)
                    + w_g2p * g2p_b
                )
                new_state = (p, c)
                if new_state not in vt or score > vt[new_state]:
                    vt[new_state] = score
                    bt[new_state] = (pp, p)

        viterbi_scores.append(vt)
        backptrs.append(bt)

    # Meilleur etat final
    if not viterbi_scores[-1]:
        return [s[0] if s else "NOM|_|_|_" for s in states], [
            [(s, 0.0)] for s in [ss[0] if ss else "NOM|_|_|_" for ss in states]
        ]

    best_final_state = max(viterbi_scores[-1], key=viterbi_scores[-1].get)

    # Backtracking
    path_states: list[tuple[str, str]] = [best_final_state]
    for t in range(n - 1, 0, -1):
        prev_state = backptrs[t].get(path_states[-1])
        if prev_state is None:
            if viterbi_scores[t - 1]:
                prev_state = max(
                    viterbi_scores[t - 1],
                    key=viterbi_scores[t - 1].get,
                )
            else:
                prev_state = (_BOS, _BOS)
        path_states.append(prev_state)
    path_states.reverse()

    pm_sequence = [s[1] for s in path_states]

    # Scores par position pour calcul de confiance
    scores_par_pos: list[list[tuple[str, float]]] = []
    for t in range(n):
        # Collecter tous les scores pour cette position via les etats du Viterbi
        pm_scores: dict[str, float] = {}
        for (pp, p), score in viterbi_scores[t].items():
            if p not in pm_scores or score > pm_scores[p]:
                pm_scores[p] = score
        sorted_scores = sorted(pm_scores.items(), key=lambda x: -x[1])
        scores_par_pos.append(sorted_scores)

    return pm_sequence, scores_par_pos


# ---------------------------------------------------------------------------
# 2d. Calcul de confiance
# ---------------------------------------------------------------------------

def _calculer_confiance(
    scores: list[tuple[str, float]],
    is_anchor: bool,
) -> float:
    """Confiance normalisee basee sur le delta entre 1er et 2eme score."""
    if is_anchor:
        return 1.0
    if len(scores) <= 1:
        return 0.9  # un seul candidat non-ancre = assez sur

    best_score = scores[0][1]
    second_score = scores[1][1]

    # Delta normalise : plus la difference est grande, plus on est sur
    delta = best_score - second_score
    # Sigmoid-like : delta de 5+ → confiance ~0.95
    confiance = 1.0 / (1.0 + math.exp(-0.5 * delta))
    return max(0.05, min(0.99, confiance))


# ---------------------------------------------------------------------------
# Detection des conflits
# ---------------------------------------------------------------------------

def _forme_morpho_lexique(forme: str, lexique) -> dict[str, set[str]]:
    """Retourne les traits morpho possibles pour la forme telle qu'ecrite.

    Ex: "chat" → {"nombre": {"s"}, "genre": {"m"}}
        "petit" → {"nombre": {"s"}, "genre": {"m"}}
    """
    low = forme.lower()
    infos = lexique.info(low) if hasattr(lexique, "info") else []
    if not infos:
        return {}
    result: dict[str, set[str]] = {"nombre": set(), "genre": set()}
    for entry in infos:
        n = entry.get("nombre", "") or ""
        g = entry.get("genre", "") or ""
        if n:
            result["nombre"].add(n)
        if g:
            result["genre"].add(g)
    return result


# Determinants epicenes : pas de genre inherent.
# Le lexique peut leur assigner un genre arbitraire (Fem pour "les"),
# mais en realite ils s'accordent avec le nom qui suit.
_DET_EPICENES = frozenset({
    "les", "des", "ces", "mes", "tes", "ses", "nos", "vos", "leurs",
    "quelques", "plusieurs", "certains", "certaines",
})


def _detecter_conflits(
    analyses: list[AnalyseMot],
    lexique=None,
) -> None:
    """Detecte les conflits entre l'analyse Viterbi et la forme ecrite.

    Approche : le Viterbi determine le trait ATTENDU par le contexte.
    Si la forme ecrite ne peut PAS porter ce trait (d'apres le lexique),
    c'est un conflit = une erreur a corriger.

    Ex: Viterbi dit "chat" devrait etre Plur (contexte "les ___ mangent")
        mais "chat" n'existe qu'en Sing dans le lexique → CONFLIT.

    Detecte aussi les incoherences entre ancres et mots voisins.
    """
    n = len(analyses)

    # --- Conflits forme vs analyse Viterbi (requiert le lexique) ---
    if lexique is not None:
        for i in range(n):
            a = analyses[i]
            if a.ancre:
                continue  # les ancres sont fiables par definition

            morpho_forme = _forme_morpho_lexique(a.forme, lexique)
            if not morpho_forme:
                continue

            # Nombre : Viterbi dit Plur mais forme seulement Sing ?
            nombres_forme = morpho_forme.get("nombre", set())
            if nombres_forme and a.nombre in ("Sing", "Plur"):
                nombre_court = {"Sing": "s", "Plur": "p"}.get(a.nombre, "")
                if nombre_court and nombre_court not in nombres_forme:
                    a.conflits.append(
                        f"FORME_NOMBRE({a.forme} est {'/'.join(sorted(nombres_forme))}"
                        f" mais contexte={a.nombre})"
                    )

            # Genre : Viterbi dit Fem mais forme seulement Masc ?
            genres_forme = morpho_forme.get("genre", set())
            if genres_forme and a.genre in ("Masc", "Fem"):
                genre_court = {"Masc": "m", "Fem": "f"}.get(a.genre, "")
                if genre_court and genre_court not in genres_forme:
                    a.conflits.append(
                        f"FORME_GENRE({a.forme} est {'/'.join(sorted(genres_forme))}"
                        f" mais contexte={a.genre})"
                    )

    # --- Conflits structurels entre ancres et voisins ---
    for i in range(n):
        a = analyses[i]
        if not a.ancre:
            continue
        pos_base = a.pos.split(":")[0] if a.pos else ""

        # Ancre DET/ART + NOM/ADJ suivant : accord nombre/genre
        if pos_base in ("ART", "ADJ") and a.nombre in ("Sing", "Plur"):
            is_epicene = a.forme.lower() in _DET_EPICENES
            for j in range(i + 1, min(i + 4, n)):
                b = analyses[j]
                bpos = b.pos.split(":")[0] if b.pos else ""
                if bpos in ("NOM", "ADJ"):
                    if b.nombre in ("Sing", "Plur") and b.nombre != a.nombre:
                        b.conflits.append(
                            f"ACCORD_DET({a.forme}[ANCRE]={a.nombre}, "
                            f"{b.forme}={b.nombre})"
                        )
                    # Genre : seulement si le DET a un genre inherent
                    # (le/la/un/une/mon/ma/ton/ta/son/sa mais PAS les/des/ces/mes...)
                    if (
                        not is_epicene
                        and a.genre in ("Masc", "Fem")
                        and b.genre in ("Masc", "Fem")
                        and b.genre != a.genre
                    ):
                        b.conflits.append(
                            f"ACCORD_GENRE_DET({a.forme}[ANCRE]={a.genre}, "
                            f"{b.forme}={b.genre})"
                        )
                    if bpos == "NOM":
                        break
                elif bpos not in ("ADJ", "ADV"):
                    break

        # Ancre PRO + VER : accord nombre
        if pos_base == "PRO" and a.nombre in ("Sing", "Plur"):
            for j in range(i + 1, min(i + 4, n)):
                b = analyses[j]
                bpos = b.pos.split(":")[0] if b.pos else ""
                if bpos in ("VER", "AUX"):
                    if b.nombre in ("Sing", "Plur") and b.nombre != a.nombre:
                        b.conflits.append(
                            f"ACCORD_SV({a.forme}[ANCRE]={a.nombre}, "
                            f"{b.forme}={b.nombre})"
                        )
                    break
                elif bpos not in ("ADV", "PRO", "NOM"):
                    break

        # Ancre AUX(etre) + PP : accord nombre/genre avec sujet
        if a.pos == "AUX":
            # Chercher le sujet avant l'AUX
            sujet = None
            for k in range(i - 1, max(i - 5, -1), -1):
                sk = analyses[k]
                skpos = sk.pos.split(":")[0] if sk.pos else ""
                if skpos in ("NOM", "PRO"):
                    sujet = sk
                    break
                elif skpos not in ("ADV", "ADJ"):
                    break

            # Chercher le PP apres l'AUX
            for j in range(i + 1, min(i + 3, n)):
                b = analyses[j]
                bpos = b.pos.split(":")[0] if b.pos else ""
                if bpos == "VER" and b.mode in ("par", ""):
                    # PP doit accorder avec le sujet si AUX = etre
                    if sujet and sujet.nombre in ("Sing", "Plur"):
                        if b.nombre in ("Sing", "Plur") and b.nombre != sujet.nombre:
                            b.conflits.append(
                                f"ACCORD_PP({sujet.forme}={sujet.nombre}, "
                                f"AUX={a.forme}, {b.forme}={b.nombre})"
                            )
                    if sujet and sujet.genre in ("Masc", "Fem"):
                        if b.genre in ("Masc", "Fem") and b.genre != sujet.genre:
                            b.conflits.append(
                                f"ACCORD_PP_GENRE({sujet.forme}={sujet.genre}, "
                                f"AUX={a.forme}, {b.forme}={b.genre})"
                            )
                    break
                elif bpos not in ("ADV",):
                    break


# ---------------------------------------------------------------------------
# Extraction morpho enrichie depuis le lexique
# ---------------------------------------------------------------------------

def _extraire_morpho_complete(
    forme: str,
    pm_tag: str,
    lexique,
) -> dict[str, str]:
    """Extrait personne/temps/mode depuis les entrees lexique pour le PM tag choisi."""
    low = forme.lower()
    infos = lexique.info(low) if hasattr(lexique, "info") else []
    if not infos:
        return {}

    # Le PM tag donne POS|nombre|genre — on cherche l'entree correspondante
    parts = pm_tag.split("|")
    target_pos = parts[0] if len(parts) > 0 else ""
    target_nombre_long = parts[1] if len(parts) > 1 else "_"
    target_genre_long = parts[2] if len(parts) > 2 else "_"

    # Convertir en codes courts du lexique
    nombre_court = {"Sing": "s", "Plur": "p", "_": ""}.get(target_nombre_long, "")
    genre_court = {"Masc": "m", "Fem": "f", "_": ""}.get(target_genre_long, "")

    # Chercher l'entree la plus probable
    best_entry = None
    best_freq = -1
    for entry in infos:
        cgram = entry.get("cgram", "")
        if not cgram:
            continue
        # POS doit matcher (au moins la base)
        if cgram.split(":")[0] != target_pos.split(":")[0]:
            continue
        # Nombre et genre doivent matcher si specifies
        e_nombre = entry.get("nombre", "") or ""
        e_genre = entry.get("genre", "") or ""
        if nombre_court and e_nombre and e_nombre != nombre_court:
            continue
        if genre_court and e_genre and e_genre != genre_court:
            continue
        freq = float(entry.get("freq") or 0)
        if freq > best_freq:
            best_freq = freq
            best_entry = entry

    if best_entry is None and infos:
        # Fallback : entree la plus frequente du bon POS
        pos_entries = [e for e in infos if (e.get("cgram", "").split(":")[0] == target_pos.split(":")[0])]
        if pos_entries:
            best_entry = max(pos_entries, key=lambda e: float(e.get("freq") or 0))

    if best_entry is None:
        return {}

    result = {}
    for key in ("personne", "temps", "mode"):
        val = best_entry.get(key, "") or ""
        if val:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Post-filtre : coherence corrections / ancres voisines
# ---------------------------------------------------------------------------

def _filtrer_corrections_ancres(analyses: list[AnalyseMot]) -> None:
    """Rejette les corrections qui contredisent une ancre voisine.

    Regles :
    - Un ART/DET ancre impose son nombre (et genre si defini) aux NOM/ADJ
      adjacents a droite.
    - Un PRO:per ancre impose son nombre aux VER adjacents a droite.
    - Si la correction change un trait qui contredit l'ancre → on l'annule.

    Modifie ``analyses`` in-place (met forme_corrigee="" quand on rejette).
    """
    n = len(analyses)
    for i, am in enumerate(analyses):
        if not am.forme_corrigee:
            continue

        # Chercher l'ancre la plus proche a gauche (fenetre de 2)
        ancre = None
        for j in range(i - 1, max(i - 3, -1), -1):
            if analyses[j].ancre:
                ancre = analyses[j]
                break

        if ancre is None:
            continue

        pos_base = am.pos.split(":")[0]
        ancre_pos_base = ancre.pos.split(":")[0]

        # Regle 1 : ART/DET ancre → NOM/ADJ doit respecter nombre (et genre)
        if ancre_pos_base == "ART" and pos_base in ("NOM", "ADJ"):
            if ancre.nombre in ("Sing", "Plur") and am.nombre in ("Sing", "Plur"):
                if ancre.nombre != am.nombre:
                    am.forme_corrigee = ""
                    continue
            if ancre.genre in ("Masc", "Fem") and am.genre in ("Masc", "Fem"):
                if ancre.genre != am.genre:
                    am.forme_corrigee = ""
                    continue

        # Regle 2 : PRO:per ancre → VER doit respecter nombre
        if ancre_pos_base == "PRO" and pos_base == "VER":
            if ancre.nombre in ("Sing", "Plur") and am.nombre in ("Sing", "Plur"):
                if ancre.nombre != am.nombre:
                    am.forme_corrigee = ""
                    continue


# ---------------------------------------------------------------------------
# Point d'entree principal
# ---------------------------------------------------------------------------

def analyser_phrase(
    mots: list[str],
    lexique,
    pos_ngram,
    *,
    lm_homophones=None,
    tagger=None,
    seuil_ancre: float = 0.98,
    w_emission: float = 1.0,
    w_transition: float = 1.0,
    w_g2p: float = 1.0,
    expand_morpho: bool = True,
    expand_homophones: bool = True,
    penalite_morpho: float = -3.0,
    penalite_homophone: float = -5.0,
) -> list[AnalyseMot]:
    """Analyse grammaticale complete d'une phrase tokenisee.

    Args:
        mots: tokens (mots uniquement, pas de ponctuation)
        lexique: LexiqueProtocol avec info()
        pos_ngram: PosNgram avec logp_pm_trigram/bigram
        lm_homophones: LMHomophones pour filtrer les ancres (optionnel)
        tagger: G2PUnifieAdapter avec tag_words_rich() (optionnel).
            Fournit un prior neural sur POS+morpho comme bonus d'emission.
        seuil_ancre: seuil de confiance pour les ancres
        w_emission: poids des emissions dans le Viterbi
        w_transition: poids des transitions dans le Viterbi
        w_g2p: poids du prior G2P dans le Viterbi (defaut: 1.0)
        expand_morpho: activer les variantes morphologiques (meme lemme)
        expand_homophones: activer les variantes homophones (meme prononciation)
        penalite_morpho: penalite emission variantes morpho (defaut: -3.0)
        penalite_homophone: penalite emission homophones (defaut: -5.0)

    Returns:
        list[AnalyseMot] de meme longueur que mots
    """
    n = len(mots)
    if n == 0:
        return []

    use_expand = expand_morpho or expand_homophones

    # 2a. Hypotheses par mot
    candidates_par_pos: list[list[CandidatMorpho]] = []
    pm_to_forme_par_pos: list[dict[str, str]] = []

    if use_expand:
        for mot in mots:
            cands, pm_to_f = _extraire_hypotheses_elargies(
                mot, lexique,
                expand_morpho=expand_morpho,
                expand_homophones=expand_homophones,
                penalite_morpho=penalite_morpho,
                penalite_homophone=penalite_homophone,
            )
            candidates_par_pos.append(cands)
            pm_to_forme_par_pos.append(pm_to_f)
    else:
        for mot in mots:
            cands = _extraire_hypotheses(mot, lexique)
            candidates_par_pos.append(cands)
            pm_to_forme_par_pos.append({c.pm_tag: c.forme for c in cands})

    # 2b. Ancres — les ancres bloquent l'expansion (1 seul etat)
    ancres: list[tuple[bool, str | None]] = []
    for i, mot in enumerate(mots):
        is_anchor, forced_pm = _est_ancre(
            mot, candidates_par_pos[i], lm_homophones,
        )
        if is_anchor and use_expand:
            # Ancre : reduire aux candidats de la forme originale uniquement
            low = mot.lower()
            original_cands = [c for c in candidates_par_pos[i] if c.forme == low]
            if original_cands:
                candidates_par_pos[i] = original_cands
                pm_to_forme_par_pos[i] = {c.pm_tag: c.forme for c in original_cands}
        ancres.append((is_anchor, forced_pm))

    # 2c-bis. G2P emission prior (si tagger disponible)
    g2p_emissions = None
    if tagger is not None:
        g2p_emissions = _compute_g2p_emissions(
            mots, tagger, candidates_par_pos,
        )

    # 2c. Viterbi contraint
    pm_sequence, scores_par_pos = _viterbi_contraint(
        candidates_par_pos, ancres, pos_ngram,
        w_emission=w_emission, w_transition=w_transition,
        g2p_emissions=g2p_emissions, w_g2p=w_g2p,
    )

    # 2d. Construire les resultats
    analyses: list[AnalyseMot] = []
    for t in range(n):
        is_anchor, forced_pm = ancres[t]
        pm_tag = pm_sequence[t] if t < len(pm_sequence) else "NOM|_|_|_"
        parts = pm_tag.split("|")

        pos = parts[0] if len(parts) > 0 else ""
        nombre = parts[1] if len(parts) > 1 else "_"
        genre = parts[2] if len(parts) > 2 else "_"
        personne = parts[3] if len(parts) > 3 else "_"

        # Neutraliser le genre des DET epicenes (les, des, ces, mes, ...)
        # Le lexique leur assigne un genre arbitraire qu'il ne faut pas propager
        if mots[t].lower() in _DET_EPICENES:
            genre = "_"

        # Confiance
        scores = scores_par_pos[t] if t < len(scores_par_pos) else []
        confiance = _calculer_confiance(scores, is_anchor)

        # Forme corrigee : si le Viterbi a choisi un PM tag
        # dont la forme associee differe de la forme originale
        forme_corrigee = ""
        if use_expand:
            forme_viterbi = pm_to_forme_par_pos[t].get(pm_tag, mots[t].lower())
            if forme_viterbi != mots[t].lower():
                forme_corrigee = forme_viterbi

        # Morpho enrichie (temps, mode)
        # Pour la morpho, utiliser la forme corrigee si disponible
        forme_pour_morpho = forme_corrigee if forme_corrigee else mots[t]
        morpho = _extraire_morpho_complete(forme_pour_morpho, pm_tag, lexique)

        analyses.append(AnalyseMot(
            forme=mots[t],
            forme_corrigee=forme_corrigee,
            pos=pos,
            nombre=nombre,
            genre=genre,
            personne=personne if personne != "_" else morpho.get("personne", "_"),
            temps=morpho.get("temps", "_"),
            mode=morpho.get("mode", "_"),
            confiance=confiance,
            ancre=is_anchor,
            pm_tag=pm_tag,
            candidats_pm=[c.pm_tag for c in candidates_par_pos[t]],
        ))

    # Post-filtre : rejeter les corrections qui contredisent une ancre voisine
    if use_expand:
        _filtrer_corrections_ancres(analyses)

    # Detection des conflits
    _detecter_conflits(analyses, lexique=lexique)

    return analyses


# ---------------------------------------------------------------------------
# Utilitaires d'affichage
# ---------------------------------------------------------------------------

def formater_analyse(analyses: list[AnalyseMot], phrase: str = "") -> str:
    """Formate l'analyse grammaticale pour affichage lisible."""
    lines = []
    if phrase:
        lines.append(f'"{phrase}"')

    for a in analyses:
        flags = []
        if a.ancre:
            flags.append("[ANCRE]")
        if a.forme_corrigee:
            flags.append(f"[CORR: {a.forme} -> {a.forme_corrigee}]")
        for c in a.conflits:
            flags.append(f"<- {c}")

        personne_str = f"{a.personne}P" if a.personne not in ("_", "") else ""
        temps_str = a.temps if a.temps not in ("_", "") else ""
        mode_str = a.mode if a.mode not in ("_", "") else ""

        extra_parts = [s for s in (personne_str, temps_str, mode_str) if s]
        extra = " ".join(extra_parts)

        flags_str = "  ".join(flags)
        n_cands = len(a.candidats_pm)

        # Afficher la forme corrigee si disponible
        forme_affichee = a.forme_corrigee if a.forme_corrigee else a.forme
        line = (
            f"  {forme_affichee:<16s} {a.pos:<10s} {a.nombre:<5s} {a.genre:<5s}"
            f" {extra:<12s} conf={a.confiance:.2f}  ({n_cands} cand)"
        )
        if flags_str:
            line += f"  {flags_str}"
        lines.append(line)

    return "\n".join(lines)


def score_pm_sequence(analyses: list[AnalyseMot], pos_ngram) -> float:
    """Score PM de la sequence analysee (backoff KN structure)."""
    pm_tags = [a.pm_tag for a in analyses]
    if not pm_tags:
        return 0.0

    padded = [_BOS, _BOS] + pm_tags
    score = 0.0
    for i in range(2, len(padded)):
        score += pos_ngram.logp_pm_trigram(
            padded[i - 2], padded[i - 1], padded[i],
        )
    return score
