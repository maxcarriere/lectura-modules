"""Passe 3 — Correction morphologique via Viterbi PM (accords).

Les POS sont fixes par la Passe 2. La Passe 3 ne modifie que les traits
morphologiques (Number, Gender, Person) et les formes associees.

Cas d'usage :
- "les petites enfants" -> "les petits enfants" (genre)
- "il mange des pomme" -> "il mange des pommes" (nombre)
- "manger" vs "mange" pour infinitif/participe passe

Etapes :
  3.1 POS fixes (ancrage)
  3.2 Ancrage morpho (mots invariables)
  3.3 Candidats PM (mots non-ancres)
  3.4 Emissions
  3.5 Viterbi trigram PM
  3.6 Guards adaptes par type
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
_BACKOFF_PM_THRESHOLD = -14.0

# Mapping lexique short codes -> PM tag long form
_GENRE_MAP = {"m": "Masc", "f": "Fem", "": "_", "_": "_"}
_NOMBRE_MAP = {"s": "Sing", "p": "Plur", "": "_", "_": "_"}

# Mots-outils avec morpho invariable
_FUNCTION_WORD_MORPHO: dict[str, dict[str, str]] = {
    # Articles definis
    "le": {"nombre": "Sing", "genre": "Masc"},
    "la": {"nombre": "Sing", "genre": "Fem"},
    "les": {"nombre": "Plur", "genre": "_"},
    # Articles indefinis
    "un": {"nombre": "Sing", "genre": "Masc"},
    "une": {"nombre": "Sing", "genre": "Fem"},
    "des": {"nombre": "Plur", "genre": "_"},
    # Demonstratifs
    "ce": {"nombre": "Sing", "genre": "Masc"},
    "cet": {"nombre": "Sing", "genre": "Masc"},
    "cette": {"nombre": "Sing", "genre": "Fem"},
    "ces": {"nombre": "Plur", "genre": "_"},
    # Possessifs
    "mon": {"nombre": "Sing", "genre": "Masc"},
    "ma": {"nombre": "Sing", "genre": "Fem"},
    "ton": {"nombre": "Sing", "genre": "Masc"},
    "ta": {"nombre": "Sing", "genre": "Fem"},
    "son": {"nombre": "Sing", "genre": "Masc"},
    "sa": {"nombre": "Sing", "genre": "Fem"},
    "mes": {"nombre": "Plur", "genre": "_"},
    "tes": {"nombre": "Plur", "genre": "_"},
    "ses": {"nombre": "Plur", "genre": "_"},
    "nos": {"nombre": "Plur", "genre": "_"},
    "vos": {"nombre": "Plur", "genre": "_"},
    "leur": {"nombre": "Sing", "genre": "_"},
    "leurs": {"nombre": "Plur", "genre": "_"},
    # Pronoms
    "je": {"nombre": "Sing", "personne": "1"},
    "tu": {"nombre": "Sing", "personne": "2"},
    "il": {"nombre": "Sing", "genre": "Masc", "personne": "3"},
    "elle": {"nombre": "Sing", "genre": "Fem", "personne": "3"},
    "on": {"nombre": "Sing", "personne": "3"},
    "nous": {"nombre": "Plur", "personne": "1"},
    "vous": {"nombre": "Plur", "personne": "2"},
    "ils": {"nombre": "Plur", "genre": "Masc", "personne": "3"},
    "elles": {"nombre": "Plur", "genre": "Fem", "personne": "3"},
}


def passe3_morpho(
    mots: list[MotV2],
    lexique,
    pos_ngram,
    config,
) -> None:
    """Passe 3 : correction morphologique via Viterbi PM (in-place).

    Args:
        mots: liste de MotV2 (POS fixes par passe 2)
        lexique: objet avec info(), formes_de()
        pos_ngram: PosNgram (logp_pm_trigram, logp_pm_bigram)
        config: CorrecteurV2Config
    """
    if not mots:
        return

    # 3.2 — Ancrage morpho
    _ancrer_morpho(mots, lexique)

    # 3.3-3.5 — Viterbi PM
    _viterbi_morpho(mots, lexique, pos_ngram, config)

    # 3.6 — Guards adaptes par type
    _guards_morpho(mots, config)


# ---------------------------------------------------------------------------
# 3.2 — Ancrage morpho
# ---------------------------------------------------------------------------

def _ancrer_morpho(mots: list[MotV2], lexique) -> None:
    """Determine quels mots sont ancres morphologiquement."""
    for m in mots:
        low = m.forme.lower()

        # Forme elidee (s', l', d', n', m', etc.) : toujours ancree
        if m.original.endswith(("\u2019", "'")) or low.endswith("'"):
            m.pm_tag = f"{m.pos}|_|_|_"
            m.ancre_morpho = True
            continue

        # Mot tres court (1-2 chars) sans entree lexique riche : ancrer
        if len(low) <= 2:
            m.pm_tag = f"{m.pos}|_|_|_"
            m.ancre_morpho = True
            continue

        # Mot-outil avec morpho invariable
        if low in _FUNCTION_WORD_MORPHO:
            morpho = _FUNCTION_WORD_MORPHO[low]
            m.nombre = morpho.get("nombre", "_")
            m.genre = morpho.get("genre", "_")
            m.personne = morpho.get("personne", "_")
            m.pm_tag = f"{m.pos}|{m.nombre}|{m.genre}|{m.personne}"
            m.ancre_morpho = True
            continue

        # Morpho unique dans le lexique pour ce POS
        infos = lexique.info(low) if hasattr(lexique, "info") else []
        pm_tags_for_pos: set[str] = set()
        for entry in infos:
            cgram = entry.get("cgram", "")
            if cgram == m.pos:
                genre = _GENRE_MAP.get(entry.get("genre", "") or "", "_")
                nombre = _NOMBRE_MAP.get(entry.get("nombre", "") or "", "_")
                personne_raw = entry.get("personne", "") or ""
                personne = {"1": "1", "2": "2", "3": "3"}.get(personne_raw, "_")
                pm_tags_for_pos.add(f"{cgram}|{nombre}|{genre}|{personne}")

        if len(pm_tags_for_pos) == 1:
            pm = next(iter(pm_tags_for_pos))
            parts = pm.split("|")
            m.pm_tag = pm
            m.nombre = parts[1] if len(parts) > 1 else "_"
            m.genre = parts[2] if len(parts) > 2 else "_"
            m.personne = parts[3] if len(parts) > 3 else "_"
            m.ancre_morpho = True


# ---------------------------------------------------------------------------
# 3.3-3.5 — Viterbi PM
# ---------------------------------------------------------------------------

def _viterbi_morpho(
    mots: list[MotV2],
    lexique,
    pos_ngram,
    config,
) -> None:
    """Execute le Viterbi PM et met a jour les mots."""
    n = len(mots)

    all_states: list[list[str]] = []
    all_emissions: list[dict[str, float]] = []
    pm_to_forme: list[dict[str, tuple[str, str, str, str]]] = []
    # pm -> (forme, nombre, genre, personne)

    for i, m in enumerate(mots):
        if m.ancre_morpho:
            pm = m.pm_tag or f"{m.pos}|_|_|_"
            all_states.append([pm])
            all_emissions.append({pm: 0.0})
            pm_to_forme.append({pm: (m.forme, m.nombre, m.genre, m.personne)})
        else:
            candidates = _build_pm_candidates(m, lexique, config)
            states_i: list[str] = []
            emissions_i: dict[str, float] = {}
            formes_i: dict[str, tuple[str, str, str, str]] = {}

            for forme, pm_tag, freq, nombre, genre, personne in candidates:
                em = math.log(max(freq, 0.01) + _EPS)
                # Bonus si forme courante
                if forme == m.forme.lower():
                    em += 2.0
                if pm_tag in emissions_i:
                    if em > emissions_i[pm_tag]:
                        emissions_i[pm_tag] = em
                        formes_i[pm_tag] = (forme, nombre, genre, personne)
                else:
                    states_i.append(pm_tag)
                    emissions_i[pm_tag] = em
                    formes_i[pm_tag] = (forme, nombre, genre, personne)

            if not states_i:
                pm = f"{m.pos or 'NOM'}|_|_|_"
                states_i = [pm]
                emissions_i = {pm: 0.0}
                formes_i = {pm: (m.forme.lower(), "_", "_", "_")}

            all_states.append(states_i)
            all_emissions.append(emissions_i)
            pm_to_forme.append(formes_i)

    # Transition PM
    def transition_fn(pm1: str, pm2: str, pm3: str) -> float:
        tri = pos_ngram.logp_pm_trigram(pm1, pm2, pm3)
        if tri > _BACKOFF_PM_THRESHOLD:
            return tri
        bi = pos_ngram.logp_pm_bigram(pm2, pm3)
        return bi - 1.0

    results = viterbi_trigram(
        all_states, all_emissions, transition_fn,
        w_emission=1.0, w_transition=1.0,
    )

    # Appliquer les resultats
    for i, (chosen_pm, confiance) in enumerate(results):
        m = mots[i]
        if m.ancre_morpho:
            continue

        old_forme = m.forme

        if chosen_pm in pm_to_forme[i]:
            forme, nombre, genre, personne = pm_to_forme[i][chosen_pm]
            m.pm_tag = chosen_pm
            m.nombre = nombre
            m.genre = genre
            m.personne = personne

            if forme != old_forme.lower():
                # Stocker la correction provisoirement (guard peut annuler)
                m.forme = forme
                m.corrections.append(
                    (3, "morpho.viterbi",
                     f"{old_forme} -> {forme} ({chosen_pm})")
                )
        else:
            parts = chosen_pm.split("|")
            m.pm_tag = chosen_pm
            m.nombre = parts[1] if len(parts) > 1 else "_"
            m.genre = parts[2] if len(parts) > 2 else "_"
            m.personne = parts[3] if len(parts) > 3 else "_"


def _build_pm_candidates(
    mot: MotV2,
    lexique,
    config,
) -> list[tuple[str, str, float, str, str, str]]:
    """Construit les candidats PM pour un mot non-ancre.

    Returns:
        list[(forme, pm_tag, freq, nombre, genre, personne)]
    """
    low = mot.forme.lower()
    pos = mot.pos
    candidates: list[tuple[str, str, float, str, str, str]] = []
    seen_pm: set[str] = set()

    # Candidats directs : entrees du lexique pour cette forme + POS
    infos = lexique.info(low) if hasattr(lexique, "info") else []
    for entry in infos:
        cgram = entry.get("cgram", "")
        if not cgram:
            continue
        # Accepter le POS exact ou le meme POS de base
        if cgram != pos and cgram.split(":")[0] != (pos or "").split(":")[0]:
            continue

        genre_raw = entry.get("genre", "") or ""
        nombre_raw = entry.get("nombre", "") or ""
        personne_raw = entry.get("personne", "") or ""
        freq = float(entry.get("freq") or 0)

        genre = _GENRE_MAP.get(genre_raw, "_")
        nombre = _NOMBRE_MAP.get(nombre_raw, "_")
        personne = {"1": "1", "2": "2", "3": "3"}.get(personne_raw, "_")
        pm_tag = f"{cgram}|{nombre}|{genre}|{personne}"

        if pm_tag not in seen_pm:
            seen_pm.add(pm_tag)
            candidates.append((low, pm_tag, max(freq, 0.01), nombre, genre, personne))

    # Variantes morphologiques (meme lemme, meme POS, distance <= 2)
    # Ne varier que genre/nombre/personne, pas mode/temps (eviter
    # les swaps conjugue <-> infinitif comme travaille <-> travailler).
    if config.morpho_use_variants and hasattr(lexique, "formes_de"):
        lemmes: set[str] = set()
        # Collecter le mode/temps du mot original pour filtrer
        orig_modes: set[str] = set()
        for entry in infos:
            cgram = entry.get("cgram", "")
            lemme = entry.get("lemme", "")
            if lemme and cgram:
                pos_base = cgram.split(":")[0]
                pos_actuel_base = (pos or "").split(":")[0]
                if pos_base == pos_actuel_base:
                    lemmes.add(lemme.lower())
                    mode = entry.get("mode", "") or ""
                    temps = entry.get("temps", "") or ""
                    if mode or temps:
                        orig_modes.add(f"{mode}|{temps}")

        seen_formes: set[str] = {low}
        for lemme in lemmes:
            variantes = lexique.formes_de(lemme)
            if not variantes:
                continue
            for var_entry in variantes:
                var_forme = var_entry.get("ortho", "").lower()
                if not var_forme or var_forme in seen_formes:
                    continue
                cgram = var_entry.get("cgram", "")
                if not cgram:
                    continue
                if cgram.split(":")[0] != (pos or "").split(":")[0]:
                    continue
                if _edit_distance(low, var_forme) > 2:
                    continue

                # Guard : ne pas changer de mode/temps (ex: conjugue <-> infinitif)
                var_mode = var_entry.get("mode", "") or ""
                var_temps = var_entry.get("temps", "") or ""
                if orig_modes:
                    var_mt = f"{var_mode}|{var_temps}"
                    if var_mt not in orig_modes:
                        continue

                genre_raw = var_entry.get("genre", "") or ""
                nombre_raw = var_entry.get("nombre", "") or ""
                personne_raw = var_entry.get("personne", "") or ""
                freq = float(var_entry.get("freq") or 0)

                genre = _GENRE_MAP.get(genre_raw, "_")
                nombre = _NOMBRE_MAP.get(nombre_raw, "_")
                personne = {"1": "1", "2": "2", "3": "3"}.get(personne_raw, "_")
                pm_tag = f"{cgram}|{nombre}|{genre}|{personne}"

                if pm_tag not in seen_pm:
                    seen_pm.add(pm_tag)
                    candidates.append(
                        (var_forme, pm_tag, max(freq, 0.01), nombre, genre, personne)
                    )
                    seen_formes.add(var_forme)

    # Fallback
    if not candidates:
        pm_tag = f"{pos or 'NOM'}|_|_|_"
        candidates.append((low, pm_tag, 0.01, "_", "_", "_"))

    return candidates


def _edit_distance(a: str, b: str) -> int:
    """Distance d'edition Levenshtein simple."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 3:
        return abs(la - lb)
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[lb]


# ---------------------------------------------------------------------------
# 3.6 — Guards adaptes par type
# ---------------------------------------------------------------------------

def _guards_morpho(mots: list[MotV2], config) -> None:
    """Annule les corrections morpho de faible delta n-gram.

    Seuils adaptes par type de changement :
    - Number : delta > seuil_delta_nombre
    - Gender : delta > seuil_delta_genre
    - Person : delta > seuil_delta_personne
    - Inf/PP : delta > seuil_delta_inf_pp
    """
    for m in mots:
        if m.ancre_morpho:
            continue
        # Verifier s'il y a eu une correction passe 3
        corr_p3 = [c for c in m.corrections if c[0] == 3]
        if not corr_p3:
            continue

        # Determiner le type de changement a partir du pm_tag
        # et annuler si le delta est insuffisant
        # Pour l'instant, on accepte toutes les corrections car le Viterbi
        # a deja fait le scoring. Les seuils de delta seront calcules
        # quand on aura les metriques de benchmark.
