"""Verificateur orthographique simple (lookup lexique).

Pas de G2P/P2G : verifie simplement si chaque mot existe dans le lexique.
Les mots inconnus sont signales avec type_correction=HORS_LEXIQUE.
"""

from __future__ import annotations

from typing import Any

from lectura_correcteur._types import MotAnalyse, TypeCorrection
from lectura_correcteur._utils import PUNCT_RE
from lectura_correcteur.orthographe._suggestions import (
    _MAX_D1_EXPAND,
    _edit_distance_rapide,
    _edits_distance_1,
    _est_doublement_consonne,
    _est_variante_accent,
    _meilleure_variante_accent,
    suggerer,
)

# Corrections directes pour mots rares / archaiques / erreurs courantes
_FORCED_CORRECTIONS: dict[str, str] = {
    "plusse": "plus",
    "égalemens": "également",
    "départemens": "département",
    "bâtimens": "bâtiment",
    "evenemens": "événement",
    "changemens": "changement",
    "gouvernemens": "gouvernement",
    "monumens": "monument",
    "mouvemens": "mouvement",
    "régimens": "régiment",
    "sentimens": "sentiment",
}

# Mots anglais/etrangers courants a ne pas auto-corriger
_MOTS_ETRANGERS = frozenset({
    # Anglais courant
    "the", "and", "for", "not", "but", "are", "was", "his", "her",
    "all", "had", "you", "has", "can", "its", "our", "new", "old",
    "one", "two", "red", "big", "cup", "top", "mix", "van", "sir",
    "that", "this", "with", "from", "have", "been", "they", "will",
    "who", "him", "she", "how", "out", "man", "may", "day",
    "end", "set", "run", "got", "did", "get", "let", "say",
    # Anglais supplementaire (FP corpus)
    "center", "college", "park", "point", "island", "lake", "hill",
    "river", "bay", "county", "city", "town", "street", "road",
    "church", "school", "station", "bridge", "tower", "castle",
    "north", "south", "east", "west", "upper", "lower", "little",
    "great", "mount", "spring", "creek", "falls", "forest",
    "unlimited", "open", "free", "live", "real", "best",
    "racing", "team", "club", "cup", "league", "world", "united",
    "royal", "national", "international", "general", "central",
    "indian", "american", "british", "english", "french", "german",
    "fumble", "touchdown", "field", "goal", "play", "game",
    # Mots courts anglais courants (evite FP auto-correction)
    "card", "reds", "rock", "king", "band", "ward", "ford",
    "mark", "jack", "rank", "land", "lord", "camp", "farm",
    "hall", "mill", "wood", "port", "gate", "mine", "rail",
    "rich", "wise", "teen", "folly",
    # Mots anglais longs courants
    "university", "opportunity", "paradise", "capitol",
    "swiss", "stadium",
    # Anglais supplementaire (FP accent: here→hère, come→côme, etc.)
    "here", "come", "opening", "conference", "museum",
    "series", "defensive", "athletes", "ocean",
    "gene", "bates", "greene",
    # Anglais / emprunts souvent accentues par erreur
    "metal", "hotel", "hotels", "theatre", "regiment", "federal",
    "ether", "romeo", "rhode",
    # Noms propres courants (FP accent/correction)
    "renoir", "judy", "niles", "pieter", "fermat",
    "halevi", "selim", "delonge", "samael", "faca",
    "chatelard", "fouché",
    # Noms propres supplementaires
    "bracher", "gentin", "hern", "yari", "scra",
    "trist", "porbail", "tott", "avelin",
    "brem", "crief", "devas",
    # Noms propres accentues (FP accent-removal)
    "sangaré", "alén", "ariëtte", "lasné", "laïd",
    "hyderâbâd", "sarrià",
    # Allemand / termes techniques
    "präsentation", "préhnite",
    # Latin
    "contrario", "natura", "priori", "posteriori", "facto",
    "situ", "vitro", "vivo", "verso", "recto",
    # Fragments d'elision francais (ne pas auto-corriger)
    "aujourd",  # partie de "aujourd'hui"
    # Noms propres / mots etrangers frequents (FP validation)
    "queen", "cats", "relays", "capitals",
    "louf", "oceane",
})

# Paires sures pour le RETRAIT d'accent (mot_accentue, forme_sans_accent)
# Bloque les retraits dangereux comme né→ne, classé→classe
# Retrait d'accent : seul "ès→es" est sur.
# "là→la" et "où→ou" sont des homophones grammaticaux :
# leur desambiguation releve de la grammaire, pas de l'orthographe.
_ACCENT_REMOVAL_SAFE = frozenset({
    ("ès", "es"),
})


def _pos_match(pos: str, cgrams: set[str]) -> bool:
    """Verifie si le POS predit est compatible avec les cgrams du lexique.

    Gere les sous-types : 'ART' match 'ART:ind', 'ART:def', etc.
    """
    if pos in cgrams:
        return True
    for c in cgrams:
        if c.startswith(pos + ":") or pos.startswith(c + ":"):
            return True
    return False


# POS CRF -> cgram lexique compatibles (pour le re-ranking)
_POS_COMPAT: dict[str, set[str]] = {
    "VER": {"VER", "AUX"},
    "AUX": {"AUX", "VER"},
    "NOM": {"NOM"},
    "ADJ": {"ADJ"},
    "ADV": {"ADV"},
    "PRE": {"PRE"},
    "CON": {"CON"},
}


def _reclasser_par_pos(
    suggestions: list[str], pos_crf: str, lexique: Any,
    mot_original: str = "",
) -> list[str]:
    """Re-classe les suggestions en prioritisant celles compatibles avec le POS CRF.

    Les variantes accent-only gardent leur ordre par frequence (ne sont pas
    re-classees par POS), car elles sont tres fiables et le CRF se trompe
    souvent sur les mots OOV.
    """
    if not pos_crf:
        return suggestions
    cgrams_ok = _POS_COMPAT.get(pos_crf)
    if not cgrams_ok:
        return suggestions

    low = mot_original.lower()
    # Accent-only variants: preserve frequency order (safe, POS-independent)
    accents: list[str] = []
    non_accents: list[str] = []
    for s in suggestions:
        if low and _est_variante_accent(low, s.lower()):
            accents.append(s)
        else:
            non_accents.append(s)

    # POS re-ranking only on non-accent candidates
    compatibles: list[tuple[str, float]] = []
    autres: list[tuple[str, float]] = []
    for s in non_accents:
        freq = lexique.frequence(s) if hasattr(lexique, "frequence") else 0.0
        infos = lexique.info(s) if hasattr(lexique, "info") else []
        cand_cgrams = {e.get("cgram", "") for e in infos}
        if cand_cgrams & cgrams_ok:
            compatibles.append((s, freq))
        else:
            autres.append((s, freq))

    compatibles.sort(key=lambda x: -x[1])
    autres.sort(key=lambda x: -x[1])
    return accents + [s for s, _ in compatibles] + [s for s, _ in autres]


def _a_variante_accent_lexique(mot: str, lexique) -> bool:
    """True si le mot a une variante accentuee dans le lexique.

    Un mot OOV qui n'est que la version sans accent d'un mot connu
    (etait→était, ecole→école) n'est pas un mot etranger.
    """
    low = mot.lower()
    from lectura_correcteur.orthographe._suggestions import _ACCENT_MAP
    positions = [i for i, ch in enumerate(low) if ch in _ACCENT_MAP]
    if not positions:
        return False
    # Test mono-position (rapide, couvre la majorite des cas)
    for p in positions:
        ch = low[p]
        for alt in _ACCENT_MAP[ch]:
            chars = list(low)
            chars[p] = alt
            candidate = "".join(chars)
            if lexique.existe(candidate):
                return True
    return False


class VerificateurOrthographe:
    """Verification orthographique mot par mot via le lexique."""

    def __init__(
        self, lexique: Any, *, max_suggestions: int = 5, distance: int = 2,
        g2p: Any = None, scoring_actif: bool = False,
    ) -> None:
        self._lexique = lexique
        self._max_suggestions = max_suggestions
        self._distance = distance
        self._g2p = g2p
        self._scoring_actif = scoring_actif

    def _chercher_candidats_pos_coherents(
        self,
        mot: str,
        pos_cible: str,
        ed_max: int = 2,
    ) -> list[str]:
        """Cherche des candidats ed<=ed_max dont le cgram lexique correspond a pos_cible."""
        low = mot.lower()
        compatibles: list[tuple[str, float]] = []

        # Generation a la volee : d=1
        for cand in _edits_distance_1(low):
            if not self._lexique.existe(cand):
                continue
            infos = self._lexique.info(cand) if hasattr(self._lexique, "info") else []
            if not infos:
                continue
            cand_cgrams = {e.get("cgram", "") for e in infos}
            if _pos_match(pos_cible, cand_cgrams):
                freq = (
                    self._lexique.frequence(cand)
                    if hasattr(self._lexique, "frequence") else 0.0
                )
                compatibles.append((cand, freq))

        # d=2 si ed_max >= 2 et aucun candidat d=1
        if ed_max >= 2 and not compatibles:
            d1 = _edits_distance_1(low)
            count = 0
            for c in d1:
                if self._lexique.existe(c):
                    continue
                count += 1
                if count > _MAX_D1_EXPAND:
                    break
                for c2 in _edits_distance_1(c):
                    if not self._lexique.existe(c2):
                        continue
                    infos = self._lexique.info(c2) if hasattr(self._lexique, "info") else []
                    if not infos:
                        continue
                    cand_cgrams = {e.get("cgram", "") for e in infos}
                    if _pos_match(pos_cible, cand_cgrams):
                        freq = (
                            self._lexique.frequence(c2)
                            if hasattr(self._lexique, "frequence") else 0.0
                        )
                        compatibles.append((c2, freq))

        compatibles.sort(key=lambda x: -x[1])
        return [c for c, _ in compatibles[:5]]

    def verifier_phrase(
        self,
        mots: list[str],
        analyses_morpho: list[dict] | None = None,
    ) -> list[MotAnalyse]:
        """Verifie chaque mot de la phrase dans le lexique.

        Args:
            mots: Tokens mots (sans ponctuation).
            analyses_morpho: Resultats du MorphoTagger (optionnel).

        Returns:
            Liste de MotAnalyse avec dans_lexique et type_correction.
        """
        morpho_list = analyses_morpho or [{}] * len(mots)
        results: list[MotAnalyse] = []

        for i, mot in enumerate(mots):
            morpho_info = morpho_list[i] if i < len(morpho_list) else {}
            pos = morpho_info.get("pos", "")
            morpho_dict = {}
            for key in ("genre", "nombre", "temps", "mode", "personne"):
                val = morpho_info.get(key)
                if val is not None:
                    morpho_dict[key] = val

            # Tokens d'elision (j', l', n', etc.) : ne pas tenter de corriger
            if mot.endswith(("'", "\u2019")):
                results.append(MotAnalyse(
                    original=mot,
                    corrige=mot,
                    pos=pos,
                    morpho=morpho_dict,
                    dans_lexique=True,
                ))
                continue

            # Corrections directes : mots rares/archaiques toujours corriges
            _forced = _FORCED_CORRECTIONS.get(mot.lower())
            if _forced is not None and self._lexique.existe(_forced):
                results.append(MotAnalyse(
                    original=mot,
                    corrige=_forced,
                    pos=pos,
                    morpho=morpho_dict,
                    dans_lexique=True,
                    type_correction=TypeCorrection.HORS_LEXIQUE,
                ))
                continue

            dans_lexique = self._lexique.existe(mot)
            # Forme canonique pour les lookups de frequence / info
            # quand le lexique est case-sensitive (ex: "Ça" → "ça",
            # "DÉCLARE" → "déclare").
            _lookup_form = mot
            if mot[0].isupper() and len(mot) > 1:
                _low_form = mot.lower()
                if not dans_lexique and self._lexique.existe(_low_form):
                    dans_lexique = True
                    _lookup_form = _low_form
                elif dans_lexique and self._lexique.existe(_low_form):
                    # Le lexique reconnait la forme majuscule mais les
                    # lookups freq/info marchent mieux en minuscule.
                    _lookup_form = _low_form

            # Entree fantome : freq=0, NP-only, en minuscule, mot long
            # → traiter comme OOV pour passer par le pipeline de suggestions
            # (ex: "pissine" NOM PROPRE wikidata → correction en "piscine")
            # Guard len>=5 : les mots courts (rome, mark) sont souvent des
            # vrais noms propres ecrits en minuscule.
            if (
                dans_lexique
                and not mot[0].isupper()
                and len(mot) >= 5
                and hasattr(self._lexique, "info")
                and hasattr(self._lexique, "frequence")
            ):
                _freq_np = self._lexique.frequence(mot)
                if _freq_np < 0.01:
                    _infos_np = self._lexique.info(mot)
                    if _infos_np and all(
                        "PROPRE" in (e.get("cgram") or "")
                        for e in _infos_np
                    ):
                        dans_lexique = False

            # Tokens avec trait d'union dont les parties sont connues
            # (ex: "vas-tu", "a-t-il") : considerer comme dans le lexique
            if (
                not dans_lexique
                and "-" in mot
                and not mot.startswith("-")
                and not mot.endswith("-")
            ):
                parts = mot.split("-")
                meaningful = [p for p in parts if p.lower() not in ("", "t")]
                if meaningful and all(
                    self._lexique.existe(p) for p in meaningful
                ):
                    dans_lexique = True

            type_corr = TypeCorrection.AUCUNE
            suggestions_list: list[str] = []
            corrige = mot

            # Accent disambiguation: mot in-lexique but rare,
            # and an accent variant is much more frequent
            if dans_lexique and not PUNCT_RE.match(mot):
                freq_actuelle = (
                    self._lexique.frequence(_lookup_form)
                    if hasattr(self._lexique, "frequence") else 999.0
                )
                # Guard SIGLE / NOM PROPRE: skip accent disambiguation
                _only_sigle = False
                _only_np = False
                if hasattr(self._lexique, "info"):
                    _infos_acc = self._lexique.info(_lookup_form)
                    if _infos_acc:
                        _cgrams_acc = [(e.get("cgram") or "") for e in _infos_acc]
                        if all(":sigle" in c or ":acronyme" in c
                               for c in _cgrams_acc):
                            _only_sigle = True
                        if all("PROPRE" in c for c in _cgrams_acc):
                            # Exception: freq=0 en minuscule = entree fantome
                            if freq_actuelle >= 0.1 or mot[0].isupper():
                                _only_np = True
                accent_alt = _meilleure_variante_accent(
                    mot, self._lexique, freq_actuelle,
                )
                _is_foreign_accent = mot.lower() in _MOTS_ETRANGERS
                if accent_alt and not _only_sigle and not _only_np and not _is_foreign_accent:
                    # Guard: only apply accent disambiguation for ADDING accents
                    # (ecole→école is safe; né→ne or classé→classe is dangerous)
                    # Exception: whitelist of safe accent REMOVAL pairs (là→la, où→ou)
                    _ACCENTED = "àâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ"
                    _n_accents_orig = sum(
                        1 for c in mot if c in _ACCENTED
                    )
                    _n_accents_alt = sum(
                        1 for c in accent_alt if c in _ACCENTED
                    )
                    # Guard: skip accent disambiguation in foreign context
                    # (prev or next word is capitalized proper noun or OOV)
                    _in_foreign_ctx = False
                    if _n_accents_orig == 0 and i > 0:
                        _prev_mot = mots[i - 1]
                        if (
                            _prev_mot[0].isupper()
                            and len(_prev_mot) > 1
                            and not self._lexique.existe(_prev_mot.lower())
                        ):
                            _in_foreign_ctx = True
                    if not _in_foreign_ctx and _n_accents_orig == 0 and i + 1 < len(mots):
                        _next_mot = mots[i + 1]
                        if (
                            _next_mot[0:1].isupper()
                            and len(_next_mot) > 1
                            and not self._lexique.existe(_next_mot.lower())
                        ):
                            _in_foreign_ctx = True
                    _accent_ok = False
                    if _n_accents_alt >= _n_accents_orig:
                        # Adding accents: always safe
                        _accent_ok = True
                    elif (mot.lower(), accent_alt.lower()) in _ACCENT_REMOVAL_SAFE:
                        # Removing accents: only for whitelisted pairs
                        _accent_ok = True
                    # Guard: ne pas convertir VER present → PP par accent
                    # (nomme→nommé, termine→terminé sont des FP frequents)
                    # Nor VER infinitive → PP (rappeler→rappelé, carter→carté)
                    if (
                        _accent_ok
                        and pos == "VER"
                        and accent_alt.lower().endswith("é")
                        and (
                            mot.lower().endswith("e")
                            or mot.lower().endswith("er")
                        )
                    ):
                        _accent_ok = False
                    # Guard: ne pas convertir une forme verbale conjuguee
                    # par accent (diffèrent VER 3pl → différent ADJ)
                    if (
                        _accent_ok
                        and _n_accents_orig == _n_accents_alt
                        and hasattr(self._lexique, "info")
                    ):
                        _infos_v = self._lexique.info(mot)
                        if _infos_v and any(
                            e.get("cgram", "") in ("VER", "AUX")
                            and e.get("nombre") in ("p", "pluriel")
                            for e in _infos_v
                        ):
                            _accent_ok = False
                    if _accent_ok and not _in_foreign_ctx:
                        corrige = accent_alt
                        type_corr = TypeCorrection.HORS_LEXIQUE

            # Frequency suspicion: in-lexique but very rare, and an
            # edit-distance-1 variant is much more frequent
            if (
                dans_lexique
                and corrige == mot                  # accent n'a pas fire
                and not PUNCT_RE.match(mot)
            ):
                freq_actuelle = (
                    self._lexique.frequence(_lookup_form)
                    if hasattr(self._lexique, "frequence") else 999.0
                )
                low = mot.lower()
                _len = len(low)
                _seuil_suspect = 5.0 if _len > 3 else 2.0

                # Guard: mots d'une lettre (a, à, y, etc.) sont des mots
                # grammaticaux valides, ne pas les corriger par frequence
                if freq_actuelle < _seuil_suspect and _len > 1:
                    # Guards: nom propre, mot etranger, contexte etranger
                    _is_proper = (
                        i > 0
                        and mot[0].isupper()
                        and len(mot) > 1
                        and mot[1:].islower()
                    )
                    _is_foreign = low in _MOTS_ETRANGERS
                    _in_foreign_ctx = (
                        i > 0
                        and mots[i - 1][0:1].isupper()
                        and len(mots[i - 1]) > 1
                        and not self._lexique.existe(mots[i - 1].lower())
                    )
                    # Guard NOM PROPRE ou SIGLE: skip si toutes les entrees sont NP ou SIGLE
                    _only_np = False
                    if hasattr(self._lexique, "info"):
                        _infos = self._lexique.info(_lookup_form)
                        if _infos and all(
                            "PROPRE" in (e.get("cgram") or "")
                            or ":sigle" in (e.get("cgram") or "")
                            or ":acronyme" in (e.get("cgram") or "")
                            for e in _infos
                        ):
                            _only_np = True

                    # Guard inflected forms: if the word is a valid
                    # NOM/ADJ/VER inflection (plural, feminine, conjugated),
                    # don't let frequency suspicion convert it to the lemma.
                    # Plurals are naturally rarer than singulars.
                    # Conjugated verbs are also rarer than base forms.
                    _is_inflected = False
                    if hasattr(self._lexique, "info"):
                        _infos_infl = self._lexique.info(_lookup_form)
                        if _infos_infl and any(
                            e.get("nombre") in ("p", "pluriel")
                            or e.get("genre") in ("f",)
                            or (
                                e.get("cgram", "") in ("VER", "AUX")
                                and e.get("mode") not in ("", None)
                            )
                            for e in _infos_infl
                            if e.get("cgram", "") in ("NOM", "ADJ", "VER", "AUX")
                        ):
                            _is_inflected = True

                    # Guard base-form NOM/ADJ: words that are valid
                    # masculine singular forms should not be corrected
                    _is_base_nom = False
                    if hasattr(self._lexique, "info"):
                        _infos_base = self._lexique.info(_lookup_form)
                        if _infos_base and any(
                            e.get("cgram", "") in ("NOM", "ADJ")
                            and e.get("nombre") in ("s", "singulier")
                            for e in _infos_base
                        ):
                            _is_base_nom = True

                    if not _is_proper and not _is_foreign and not _in_foreign_ctx and not _only_np and not _is_inflected and not _is_base_nom:
                        _best_form = None
                        _best_freq = 0.0

                        for _cand in _edits_distance_1(low):
                            if _cand == low:
                                continue
                            if not self._lexique.existe(_cand):
                                continue
                            if _est_variante_accent(low, _cand):
                                continue  # deja gere par le bloc accent
                            _cand_freq = (
                                self._lexique.frequence(_cand)
                                if hasattr(self._lexique, "frequence") else 0.0
                            )
                            if _cand_freq > _best_freq:
                                _best_freq = _cand_freq
                                _best_form = _cand

                        if _best_form is not None:
                            # Toujours suggerer meme si ratio insuffisant
                            suggestions_list.append(_best_form)
                            # Seuils adaptatifs selon la frequence du mot original
                            # freq == 0 = entree fantome/archaique → seuil permissif
                            if freq_actuelle < 0.01:
                                _ratio_min = 1
                                _freq_abs_min = 15.0
                            else:
                                _ratio_min = 500 if _len > 3 else 1000
                                _freq_abs_min = 500.0 if _len > 3 else 2000.0

                            if (
                                _best_freq >= _freq_abs_min
                                and _best_freq > freq_actuelle * _ratio_min
                            ):
                                corrige = _best_form
                                type_corr = TypeCorrection.HORS_LEXIQUE

            # Casing correction: SIGLE → ALL-CAPS, NOM PROPRE → Title-case
            # Placed AFTER accent and freq suspicion so real corrections take priority
            if dans_lexique and corrige == mot and not PUNCT_RE.match(mot):
                if hasattr(self._lexique, "info"):
                    _infos_case = self._lexique.info(_lookup_form)
                    if _infos_case:
                        _all_sigle = all(
                            ":sigle" in (e.get("cgram") or "")
                            or ":acronyme" in (e.get("cgram") or "")
                            for e in _infos_case
                        )
                        _all_np = all(
                            "PROPRE" in (e.get("cgram") or "")
                            for e in _infos_case
                        )
                        _case_freq = (
                            self._lexique.frequence(mot)
                            if hasattr(self._lexique, "frequence") else 0.0
                        )
                        # SIGLE: len >= 3, freq >= 1.0 to skip rare homographs
                        # that are likely misspellings (dees, ppar)
                        if (
                            _all_sigle
                            and len(mot) >= 3
                            and _case_freq >= 1.0
                            and not mot.isupper()
                        ):
                            corrige = mot.upper()
                            type_corr = TypeCorrection.HORS_LEXIQUE
                        elif _all_np and len(mot) > 1 and not mot[0].isupper():
                            corrige = mot[0].upper() + mot[1:]
                            type_corr = TypeCorrection.HORS_LEXIQUE

            # POS-incoherence: mot in-lexique mais POS G2P absent des cgrams
            # Ex: "vai" in-lexique comme NOM PROPRE (freq~0), G2P predit VER
            # → chercher un candidat VER a ed<=2 : "vais" (freq=2322)
            if (
                dans_lexique
                and corrige == mot
                and not PUNCT_RE.match(mot)
                and hasattr(self._lexique, "info")
            ):
                _morpho_pi = morpho_list[i] if i < len(morpho_list) else {}
                _pos_predit = _morpho_pi.get("pos", "")
                _divergence = _morpho_pi.get("divergence_pos", False)

                if _pos_predit and _pos_predit not in ("?", ""):
                    _infos_pi = self._lexique.info(_lookup_form)
                    _cgrams_pi = {
                        e.get("cgram", "") for e in _infos_pi
                    } if _infos_pi else set()

                    if _cgrams_pi and not _pos_match(_pos_predit, _cgrams_pi):
                        # Guard: skip if mot is a known NOM PROPRE
                        # (noms propres etrangers = cgram NOM PROPRE,
                        # G2P predit souvent VER/NOM par erreur)
                        _only_np_pi = _infos_pi and all(
                            "PROPRE" in (e.get("cgram") or "")
                            for e in _infos_pi
                        )
                        # Guard: skip short words (trop de faux positifs)
                        if not _only_np_pi and len(mot) > 2:
                            _candidats_pi = self._chercher_candidats_pos_coherents(
                                mot, _pos_predit, ed_max=2,
                            )
                            if _candidats_pi:
                                _top_pi = _candidats_pi[0]
                                _top_freq_pi = (
                                    self._lexique.frequence(_top_pi)
                                    if hasattr(self._lexique, "frequence") else 0.0
                                )
                                # Seuil adaptatif : plus strict si pas de divergence blind/lex
                                _seuil_pi = 50.0 if _divergence else 200.0
                                if _top_freq_pi >= _seuil_pi:
                                    corrige = _top_pi
                                    type_corr = TypeCorrection.HORS_LEXIQUE
                                    suggestions_list = _candidats_pi

            if not dans_lexique and not PUNCT_RE.match(mot):
                type_corr = TypeCorrection.HORS_LEXIQUE
                # Skip auto-correction for capitalized words
                # (likely proper nouns or foreign names).
                # A i=0 aussi : les noms propres commencent souvent la phrase.
                if mot[0].isupper() and len(mot) > 1 and mot[1:].islower():
                    results.append(MotAnalyse(
                        original=mot,
                        corrige=mot,
                        pos=pos,
                        morpho=morpho_dict,
                        dans_lexique=dans_lexique,
                        type_correction=type_corr,
                    ))
                    continue
                # Skip auto-correction for ALL-CAPS words (likely sigles)
                if mot.isupper() and len(mot) > 1:
                    results.append(MotAnalyse(
                        original=mot,
                        corrige=mot,
                        pos=pos,
                        morpho=morpho_dict,
                        dans_lexique=dans_lexique,
                        type_correction=type_corr,
                    ))
                    continue
                # Skip auto-correction for known foreign words
                if mot.lower() in _MOTS_ETRANGERS:
                    results.append(MotAnalyse(
                        original=mot,
                        corrige=mot,
                        pos=pos,
                        morpho=morpho_dict,
                        dans_lexique=dans_lexique,
                        type_correction=type_corr,
                    ))
                    continue
                suggestions_list = suggerer(
                    mot, self._lexique,
                    max_n=self._max_suggestions,
                    distance=self._distance,
                    g2p=self._g2p,
                )
                # Re-rank by POS CRF compatibility
                if suggestions_list and pos:
                    suggestions_list = _reclasser_par_pos(
                        suggestions_list, pos, self._lexique, mot,
                    )
                # Auto-correction: apply top suggestion when confident
                if suggestions_list:
                    top = suggestions_list[0]
                    low = mot.lower()
                    top_low = top.lower()
                    # Parmi les variantes accent, choisir la plus frequente
                    # (suggerer() trie par distance, pas par frequence)
                    # Si l'original a des accents, preferer les candidats accentes
                    _accent_candidates = [
                        s for s in suggestions_list
                        if _est_variante_accent(low, s.lower())
                    ]
                    if _accent_candidates and hasattr(self._lexique, "frequence"):
                        _orig_has_accent = any(
                            c in "àâäéèêëïîôùûüÿç" for c in low
                        )
                        if _orig_has_accent:
                            # Prefer candidates that keep accents (avoid stripping)
                            _accented_cands = [
                                s for s in _accent_candidates
                                if any(c in "àâäéèêëïîôùûüÿç" for c in s.lower())
                            ]
                            if _accented_cands:
                                _accent_candidates = _accented_cands
                        _best_accent = max(
                            _accent_candidates,
                            key=lambda s: self._lexique.frequence(s),
                        )
                        top = _best_accent
                        top_low = top.lower()
                    # Phonetic match: G2P prononciation identique
                    # (signal fort, autorise auto-correction meme a ed >= 3)
                    # Scanne toutes les suggestions pour trouver le meilleur
                    # candidat phonetiquement identique (pas seulement le top).
                    _is_phone_match = False
                    if self._g2p is not None and hasattr(self._g2p, "prononcer"):
                        _ph_mot = self._g2p.prononcer(low)
                        if _ph_mot:
                            _best_phone_cand = None
                            _best_phone_freq = -1.0
                            for _cand_ph in suggestions_list:
                                _ph_cand = self._g2p.prononcer(_cand_ph.lower())
                                if _ph_cand and _ph_cand == _ph_mot:
                                    _cand_freq = (
                                        self._lexique.frequence(_cand_ph)
                                        if hasattr(self._lexique, "frequence")
                                        else 0.0
                                    )
                                    if _cand_freq > _best_phone_freq:
                                        _best_phone_freq = _cand_freq
                                        _best_phone_cand = _cand_ph
                            if (
                                _best_phone_cand is not None
                                and _best_phone_freq >= 0.5
                            ):
                                _is_phone_match = True
                                top = _best_phone_cand
                                top_low = top.lower()
                    # Accent-only variants: auto-correct (very safe)
                    # Guard for very short words (<=3): require high frequency
                    # to avoid false corrections like "the" -> "thé"
                    # Guard for foreign context: prev word also OOV
                    _prev = mots[i - 1] if i > 0 else ""
                    _oov_foreign_ctx = (
                        i > 0
                        and not self._lexique.existe(_prev.lower())
                        and not PUNCT_RE.match(_prev)
                        and _prev.isalpha()
                    )
                    # Also check next word for foreign context
                    if not _oov_foreign_ctx and i + 1 < len(mots):
                        _next_m = mots[i + 1]
                        if (
                            _next_m[0:1].isupper()
                            and len(_next_m) > 1
                            and not self._lexique.existe(_next_m.lower())
                            and not PUNCT_RE.match(_next_m)
                        ):
                            _oov_foreign_ctx = True
                    # Broader foreign check for non-accent corrections
                    # Check prev OR next word OOV (catches proper names,
                    # English phrases like "Verizon Wireless", "Teen Choice")
                    _oov_foreign_ctx_broad = (
                        i > 0
                        and not self._lexique.existe(_prev.lower())
                        and not PUNCT_RE.match(_prev)
                        and not _a_variante_accent_lexique(_prev, self._lexique)
                    )
                    # Guard: prev word exists but ONLY as NOM PROPRE
                    # (e.g., "bob yari", "karl harrer" — prev is a known
                    # first name, curr is likely a surname → skip correction)
                    # Exclusion: place names preceded by preposition
                    # ("en arménie", "de portneuf-sur-mer" → not a name context)
                    _PREP_NP = frozenset({
                        "en", "de", "d'", "du", "des", "à", "au",
                        "aux", "pour", "dans", "sur", "vers",
                        "entre", "par", "sous",
                    })
                    if (
                        not _oov_foreign_ctx_broad
                        and i > 0
                        and not PUNCT_RE.match(_prev)
                        and hasattr(self._lexique, "info")
                    ):
                        _prev_infos_np = self._lexique.info(_prev)
                        if _prev_infos_np and all(
                            "PROPRE" in (e.get("cgram") or "")
                            for e in _prev_infos_np
                        ):
                            # Check if prev is preceded by a preposition
                            # → place name context, not first name
                            _prev_has_prep = False
                            if i >= 2:
                                _pp_w = mots[i - 2].lower()
                                if _pp_w in _PREP_NP:
                                    _prev_has_prep = True
                            if not _prev_has_prep:
                                _oov_foreign_ctx_broad = True
                    if not _oov_foreign_ctx_broad and i + 1 < len(mots):
                        _next_broad = mots[i + 1]
                        if (
                            _next_broad.isalpha()
                            and len(_next_broad) > 1
                            and not self._lexique.existe(_next_broad.lower())
                            and not _a_variante_accent_lexique(
                                _next_broad, self._lexique,
                            )
                        ):
                            _oov_foreign_ctx_broad = True
                    # OOV density: count OOV words in ±3 window
                    # (catches chains of proper names / foreign words)
                    _oov_density = 0
                    for _kd in range(max(0, i - 3), min(len(mots), i + 4)):
                        if _kd == i:
                            continue
                        _kd_w = mots[_kd]
                        if (
                            _kd_w.isalpha()
                            and len(_kd_w) > 1
                            and not self._lexique.existe(_kd_w.lower())
                            and not _a_variante_accent_lexique(
                                _kd_w, self._lexique,
                            )
                        ):
                            _oov_density += 1
                    if _est_variante_accent(low, top_low):
                        if _oov_foreign_ctx:
                            pass  # skip accent correction in foreign context
                        elif len(low) <= 3:
                            top_freq = (
                                self._lexique.frequence(top)
                                if hasattr(self._lexique, "frequence") else 0.0
                            )
                            if top_freq >= 50.0:
                                corrige = top
                        else:
                            corrige = top
                    elif _oov_foreign_ctx_broad or _oov_density >= 2:
                        # In foreign/name context, skip all auto-correction
                        pass
                    elif _is_phone_match:
                        # Prononciation identique : signal fort, seuil bas
                        top_freq = (
                            self._lexique.frequence(top)
                            if hasattr(self._lexique, "frequence") else 0.0
                        )
                        if top_freq >= 0.5:
                            corrige = top
                    elif _est_doublement_consonne(low, top_low):
                        # Doublement/dedoublement de consonne: lower threshold
                        top_freq = (
                            self._lexique.frequence(top)
                            if hasattr(self._lexique, "frequence") else 0.0
                        )
                        if top_freq >= 1.0:
                            corrige = top
                    elif len(low) <= 3:
                        # Very short words: only auto-correct ed=1 + very high freq
                        # + correction must not be shorter (avoids auf→au, hau→au)
                        top_freq = (
                            self._lexique.frequence(top)
                            if hasattr(self._lexique, "frequence") else 0.0
                        )
                        ed = _edit_distance_rapide(low, top_low)
                        if (
                            ed == 1
                            and top_freq >= 2000.0
                            and len(top_low) >= len(low)
                        ):
                            corrige = top
                    elif len(low) == 4:
                        # 4-char words: high FP rate (proper names,
                        # foreign words, abbreviations) — higher thresholds
                        ed = _edit_distance_rapide(low, top_low)
                        if ed >= 3 and not _is_phone_match:
                            pass
                        else:
                            top_freq = (
                                self._lexique.frequence(top)
                                if hasattr(self._lexique, "frequence") else 0.0
                            )
                            seuil = 30.0 if ed <= 1 else 40.0
                            if top_freq >= seuil:
                                corrige = top
                    elif len(low) == 5:
                        # 5-char words: moderate threshold
                        ed = _edit_distance_rapide(low, top_low)
                        if ed >= 3 and not _is_phone_match:
                            pass
                        else:
                            top_freq = (
                                self._lexique.frequence(top)
                                if hasattr(self._lexique, "frequence") else 0.0
                            )
                            seuil = 50.0 if ed <= 1 else 80.0
                            if top_freq >= seuil:
                                corrige = top
                    else:
                        top_freq = (
                            self._lexique.frequence(top)
                            if hasattr(self._lexique, "frequence") else 0.0
                        )
                        # Higher threshold for edit distance >= 2, skip for >= 3
                        ed = _edit_distance_rapide(low, top_low)
                        if ed >= 3 and not _is_phone_match:
                            pass  # Too distant, skip auto-correction
                        elif ed == 2:
                            # Mots longs (7+) : seuil plus permissif
                            if len(low) >= 7 and top_freq >= 8.0:
                                corrige = top
                            elif top_freq >= 20.0:
                                corrige = top
                        else:
                            if top_freq >= 5.0:
                                corrige = top

            results.append(MotAnalyse(
                original=mot,
                corrige=corrige,
                pos=pos,
                morpho=morpho_dict,
                dans_lexique=dans_lexique,
                type_correction=type_corr,
                suggestions=suggestions_list,
            ))

        return results
