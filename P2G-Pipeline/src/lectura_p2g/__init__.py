"""Lectura P2G — Pipeline P2G complet du francais (IPA -> orthographe).

Orchestre : Graphemiseur (modele core) + Formules + Noms propres.
Couche 2, en miroir de lectura-g2p pour le G2P.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md

Installation :
    pip install lectura-p2g

Exemple rapide::

    from lectura_p2g import analyser, creer_engine
    engine = creer_engine()
    result = analyser(["bɔ̃ʒuʁ"], engine=engine)
    print(result["ortho"])  # ['bonjour']
"""

from __future__ import annotations

import json
import unicodedata
from importlib import resources

__version__ = "4.0.0"

# ── Re-exports depuis le graphemiseur (couche 1) ─────────────────────

from lectura_graphemiseur import creer_engine, tokeniser_ipa, ipa_phrase_vers_chars
from lectura_graphemiseur import corriger_p2g, corriger_phrase_v2, corriger_phrase_v3


# ── Pluriels sans -s (importe depuis le graphemiseur) ────────────────

from lectura_graphemiseur._chargeur import invariables_pluriel as _load_no_plural_s

_NO_PLURAL_S = _load_no_plural_s()


# ── Reconnaissance de formules (nombres, sigles) ─────────────────────

def _appliquer_formules(
    result: list[str],
    ipa_words: list[str],
    pos_tags: list[str],
    n: int,
    _in_lex,
    formule_mode: str = "num",
) -> set[int]:
    """Detecte et remplace les formules (nombres, sigles) dans la sortie P2G.

    Utilise lectura_formules.detect_number_spans et detect_sigle_spans
    pour identifier les sequences IPA correspondant a des formules.

    Parameters
    ----------
    formule_mode : str
        Mode d'affichage des formules :
        - ``"num"``   : chiffres/symboles (35, SNCF) — defaut
        - ``"texte"`` : texte epele (trente-cinq, esse-enne-ce-effe)

    Modifie ``result`` in-place.
    Returns: set de positions traitees (a proteger des corrections suivantes).
    """
    try:
        from lectura_formules import detect_number_spans, detect_sigle_spans
    except ImportError:
        return set()

    use_num = (formule_mode != "texte")
    formule_positions: set[int] = set()

    # ── Nombres (min_span=1 pour les nombres simples aussi) ──
    number_spans = detect_number_spans(ipa_words, min_span=1, max_span=8)
    for start, end, formula_result in number_spans:
        if use_num and formula_result.display_num:
            display = formula_result.display_num
        else:
            display = formula_result.display_fr
        display_words = display.split()
        span_len = end - start
        for k in range(span_len):
            if k < len(display_words):
                result[start + k] = display_words[k]
            else:
                result[start + k] = ""
            formule_positions.add(start + k)

    # ── Sigles (min_span=2, lettres epelees) ──
    sigle_spans = detect_sigle_spans(ipa_words, min_span=2, max_span=8)
    for start, end, formula_result in sigle_spans:
        # Pas de chevauchement avec les nombres
        if any(i in formule_positions for i in range(start, end)):
            continue
        # Sigles : toujours afficher l'acronyme (SNCF), quel que soit le mode
        result[start] = formula_result.display_num or formula_result.display_fr
        for k in range(1, end - start):
            result[start + k] = ""
        for k in range(end - start):
            formule_positions.add(start + k)

    # ── Regles contextuelles post-formules ──
    for start, end, formula_result in number_spans:
        valeur = formula_result.valeur
        if not isinstance(valeur, (int, float)):
            continue

        next_idx = end
        if next_idx >= n or next_idx in formule_positions:
            continue

        next_pos = pos_tags[next_idx] if next_idx < len(pos_tags) else ""

        # Regle: nombre + "en" (POS=NOM) -> "ans"
        # Corrige l'homophone /an/ quand le contexte est "X ans".
        # AVANT la regle pluriel, car sinon "en" -> "ens" empeche le match.
        if next_pos == "NOM" and result[next_idx].lower() == "en":
            result[next_idx] = "ans"

        # Regle: nombre > 1 -> NOM/ADJ suivant doit etre pluriel
        if valeur > 1 and next_pos in ("NOM", "ADJ"):
            curr = result[next_idx]
            if (
                not curr.endswith(("s", "x", "z"))
                and len(curr) > 1
                and curr.lower() not in _NO_PLURAL_S
            ):
                candidate = curr + "s"
                if _in_lex(candidate):
                    result[next_idx] = candidate

    return formule_positions


# ── Capitalisation des noms propres ──────────────────────────────────

def _capitaliser_noms_propres(
    result: list[str],
    ipa_words: list[str] | None,
    phone_lexicon: object | None,
    skip_positions: set[int],
) -> None:
    """Capitalise les mots exclusivement NOM PROPRE dans le phone_lexicon.

    Si le phone_lexicon contient une entree NOM PROPRE correspondant
    au mot predit, et qu'il n'existe aucun sens commun pour la meme
    ortho, on capitalise (le modele ne gere pas la casse).

    Modifie ``result`` in-place.
    """
    if phone_lexicon is None or ipa_words is None:
        return
    if not hasattr(phone_lexicon, "all_entries"):
        return

    n = len(result)
    for i in range(n):
        if i in skip_positions:
            continue
        if i >= len(ipa_words):
            continue
        phone = ipa_words[i]
        word = result[i]
        # Pour les elisions (j'..., l'...), verifier la base
        if "'" in word:
            parts = word.split("'", 1)
            ortho_base = parts[1]
        else:
            ortho_base = word
        if "'" in phone:
            apo_idx = phone.index("'")
            base_phone = phone[apo_idx + 1:]
        else:
            base_phone = phone
        if not base_phone or not ortho_base:
            continue
        entries = phone_lexicon.all_entries(base_phone)
        if not entries:
            continue
        # Capitaliser si l'ortho predit correspond exclusivement a un
        # NOM PROPRE dans le lexique (pas de sens commun pour la meme ortho).
        # Ex: "antoine" n'existe qu'en NOM PROPRE -> capitaliser.
        # Ex: "paris" existe en NOM + NOM PROPRE -> ne pas capitaliser.
        # Ex: "appelle" existe en VER -> ne pas capitaliser.
        ortho_lower = ortho_base.lower()
        has_propre = False
        has_commun = False
        for e in entries:
            if (e.get("ortho") or "").lower() != ortho_lower:
                continue
            if (e.get("cgram") or "").strip() == "NOM PROPRE":
                has_propre = True
            else:
                has_commun = True
        is_propre = has_propre and not has_commun
        if is_propre:
            if "'" in word:
                parts = word.split("'", 1)
                result[i] = parts[0] + "'" + parts[1].capitalize()
            else:
                result[i] = word.capitalize()


# ── Reconnaissance d'entites (noms propres notables) ───────────────

def _distance_ipa(a: str, b: str) -> int:
    """Distance Damerau-Levenshtein sur chaines IPA (NFC, sans espaces)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 3:
        return abs(la - lb)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev_prev = None
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
            if (
                i > 0 and j > 0
                and a[i] == b[j - 1] and a[i - 1] == b[j]
                and prev_prev is not None
            ):
                curr[j + 1] = min(curr[j + 1], prev_prev[j - 1] + 1)
        prev_prev = prev
        prev = curr
    return prev[lb]


def _seuil_ipa(longueur: int) -> int:
    """Seuil de distance IPA adaptatif selon la longueur de la chaine."""
    if longueur <= 4:
        return 1
    if longueur <= 8:
        return 2
    return 3


_ENTITES: dict | None = None

# Seuil de notoriete pour autoriser une distance IPA > 0
_NOTORIETE_DIST1 = 200


def _charger_entites() -> dict:
    """Charge entites.json depuis le package (singleton)."""
    global _ENTITES
    if _ENTITES is not None:
        return _ENTITES

    pkg = resources.files("lectura_p2g") / "data" / "entites.json"
    data = json.loads(pkg.read_text(encoding="utf-8"))

    # Index exact par (ipa, nb_mots) pour lookup O(1)
    # Garde l'entite la plus notoire en cas de collision
    index_exact: dict[tuple[str, int], int] = {}
    for idx, ent in enumerate(data["entites"]):
        key = (ent["ipa"], ent["nb_mots"])
        if key not in index_exact or ent["notoriete"] > data["entites"][index_exact[key]]["notoriete"]:
            index_exact[key] = idx
    data["index_exact"] = index_exact

    # Index par nb_mots pour les entites tres notoires (fallback dist <= 1)
    index_nb_mots: dict[int, list[int]] = {}
    for idx, ent in enumerate(data["entites"]):
        if ent["notoriete"] >= _NOTORIETE_DIST1:
            k = ent["nb_mots"]
            index_nb_mots.setdefault(k, []).append(idx)
    data["index_nb_mots"] = index_nb_mots

    _ENTITES = data
    return _ENTITES


def _detecter_entites(
    result: list[str],
    ipa_words: list[str],
    phone_lexicon: object | None,
    skip_positions: set[int],
) -> None:
    """Detecte et corrige les noms propres via le dictionnaire d'entites.

    Passe finale : cherche des sequences de mots OOV consecutifs
    et les compare au dictionnaire d'entites notables par IPA.

    Strategie conservative pour minimiser les faux positifs :
      - Match exact IPA (distance 0) pour toutes les entites
      - Tolerance distance <= 1 uniquement pour les entites tres notoires
        (notoriete >= 200)
      - Mono-mots : match exact seulement, notoriete >= 100

    Modifie result in-place.
    """
    if ipa_words is None:
        return

    try:
        data = _charger_entites()
    except (FileNotFoundError, json.JSONDecodeError):
        return

    entites = data["entites"]
    index_exact = data["index_exact"]
    index_ipa = data["index_ipa"]
    index_nb_mots = data["index_nb_mots"]
    n = len(result)

    # Determiner les positions OOV
    has_lex = phone_lexicon is not None and hasattr(phone_lexicon, "all_entries")
    oov: set[int] = set()
    for i in range(n):
        if i in skip_positions:
            continue
        if i >= len(ipa_words):
            continue
        if not ipa_words[i]:
            continue
        if has_lex:
            phone = ipa_words[i]
            if "'" in phone:
                phone = phone[phone.index("'") + 1:]
            entries = phone_lexicon.all_entries(phone)
            if entries:
                has_commun = any(
                    (e.get("cgram") or "").strip() != "NOM PROPRE"
                    for e in entries
                )
                if has_commun:
                    continue
        oov.add(i)

    if not oov:
        return

    # Grouper les OOV consecutifs en spans
    sorted_oov = sorted(oov)
    spans: list[tuple[int, int]] = []
    start = sorted_oov[0]
    end = start + 1
    for pos in sorted_oov[1:]:
        if pos == end:
            end = pos + 1
        else:
            spans.append((start, end))
            start = pos
            end = pos + 1
    spans.append((start, end))

    matched: set[int] = set()

    for span_start, span_end in spans:
        span_len = span_end - span_start

        # ── Multi-mots (k >= 2) ──
        for k in range(min(span_len, 5), 1, -1):
            for offset in range(span_len - k + 1):
                s = span_start + offset
                e = s + k
                if any(i in matched for i in range(s, e)):
                    continue

                ipa_parts = []
                for i in range(s, e):
                    ph = ipa_words[i] if i < len(ipa_words) else ""
                    if "'" in ph:
                        ph = ph[ph.index("'") + 1:]
                    ipa_parts.append(ph)
                ipa_concat = unicodedata.normalize("NFC", "".join(ipa_parts))
                if not ipa_concat:
                    continue

                # 1) Match exact (distance 0)
                exact_key = (ipa_concat, k)
                best_ent = None
                if exact_key in index_exact:
                    best_ent = entites[index_exact[exact_key]]

                # 2) Fallback : distance <= 1 pour entites tres notoires
                if best_ent is None:
                    candidats_nb = set(index_nb_mots.get(k, []))
                    if candidats_nb:
                        candidats_bg: dict[int, int] = {}
                        for ci in range(len(ipa_concat) - 1):
                            bg = ipa_concat[ci: ci + 2]
                            for idx in index_ipa.get(bg, []):
                                if idx in candidats_nb:
                                    candidats_bg[idx] = candidats_bg.get(idx, 0) + 1
                        if candidats_bg:
                            best_dist = 2  # max distance 1
                            for idx in sorted(candidats_bg, key=candidats_bg.get, reverse=True):
                                ent = entites[idx]
                                d = _distance_ipa(ipa_concat, ent["ipa"])
                                if d < best_dist:
                                    best_dist = d
                                    best_ent = ent
                                    if d == 0:
                                        break

                if best_ent is not None:
                    label_words = best_ent["label"].split()
                    for ki in range(k):
                        if ki < len(label_words):
                            result[s + ki] = label_words[ki]
                        else:
                            result[s + ki] = ""
                        matched.add(s + ki)

        # ── Mono-mots : match exact seulement, notoriete >= 100 ──
        for i in range(span_start, span_end):
            if i in matched:
                continue
            ph = ipa_words[i] if i < len(ipa_words) else ""
            if "'" in ph:
                ph = ph[ph.index("'") + 1:]
            ipa_single = unicodedata.normalize("NFC", ph)
            if not ipa_single:
                continue
            exact_key = (ipa_single, 1)
            if exact_key not in index_exact:
                continue
            ent = entites[index_exact[exact_key]]
            if ent["notoriete"] < 100:
                continue
            result[i] = ent["label"]
            matched.add(i)

    skip_positions.update(matched)


# ── Fusion de mots composes ──────────────────────────────────────────

# Phones correspondant a des clitiques francais (j', d', l', s', etc.)
# Ne pas fusionner quand le premier phone est un clitique.
_CLITIC_PHONES = frozenset({"d", "l", "s", "n", "m", "t", "k", "ʒ"})

# Seuil de frequence au-dessus duquel un compose est toujours fusionne
_COMPOSE_FREQ_SEUIL = 5.0


def _est_compose_interne(ortho: str) -> bool:
    """Verifie si l'ortho a un tiret ou apostrophe INTERNE (pas en pos 0/-1)."""
    if len(ortho) < 3:
        return False
    inner = ortho[1:-1]
    return "-" in inner or "'" in inner


def _fusionner_composes(
    result: list[str],
    ipa_words: list[str],
    phone_lexicon: object,
    skip_positions: set[int],
) -> None:
    """Fusionne des mots IPA consecutifs quand ils forment un compose du lexique.

    Detecte les cas ou le tokeniseur a separe un mot compose
    (aujourd'hui, peut-etre, n'est-ce pas, etats-unis, etc.)
    en mots individuels, et les re-fusionne quand l'IPA concatene
    correspond a un compose du lexique.

    Filtres conservateurs pour eviter les faux positifs :
      - Le compose doit avoir un tiret ou apostrophe INTERNE
      - Le premier/dernier phone ne doit pas etre un clitique isolé
      - Au moins un mot predit doit etre OOV (pas d'ortho valide
        pour son phone dans le lexique)

    Modifie result in-place.
    """
    n = len(result)
    if n < 2:
        return
    if not hasattr(phone_lexicon, "exists"):
        return

    matched: set[int] = set()

    for k in range(min(5, n), 1, -1):
        for start in range(n - k + 1):
            if any(i in matched or i in skip_positions for i in range(start, start + k)):
                continue

            # Construire l'IPA fusionne
            phones: list[str] = []
            for i in range(start, start + k):
                if i >= len(ipa_words):
                    break
                ph = ipa_words[i]
                if not ph:
                    break
                if "'" in ph:
                    ph = ph[ph.index("'") + 1:]
                phones.append(ph)
            if len(phones) != k:
                continue

            # Exclure les clitiques en debut/fin
            if phones[0] in _CLITIC_PHONES:
                continue
            if phones[-1] in _CLITIC_PHONES:
                continue

            fused = "".join(phones)
            if not fused or not phone_lexicon.exists(fused):
                continue

            # Trouver le meilleur compose interne
            best_ortho = phone_lexicon.best_ortho(fused)
            if not best_ortho:
                continue

            if _est_compose_interne(best_ortho):
                compound_freq = phone_lexicon.best_freq(fused)
            else:
                # best_ortho n'est pas un compose — chercher dans toutes les entrees
                entries = phone_lexicon.all_entries(fused)
                compound_entries = [
                    e for e in entries if _est_compose_interne(e.get("ortho", ""))
                ]
                if not compound_entries:
                    continue
                best_entry = max(compound_entries, key=lambda e: e.get("freq", 0) or 0)
                best_ortho = best_entry["ortho"]
                compound_freq = best_entry.get("freq", 0) or 0

            # Decision : fusionner seulement si au moins un mot predit
            # n'a pas d'ortho valide pour son phone (OOV).
            # Evite les faux positifs quand les mots individuels sont corrects
            # (ex: "état major" vs "état-major").
            all_valid = True
            for i in range(start, start + k):
                ph = ipa_words[i]
                if "'" in ph:
                    ph = ph[ph.index("'") + 1:]
                entries_i = phone_lexicon.all_entries(ph)
                pred_lower = result[i].lower()
                if not any((e.get("ortho") or "").lower() == pred_lower for e in entries_i):
                    all_valid = False
                    break
            if all_valid:
                continue

            # Appliquer le compose
            result[start] = best_ortho
            for i in range(1, k):
                result[start + i] = ""
            for i in range(k):
                matched.add(start + i)

    skip_positions.update(matched)


# ── Pipeline complet ─────────────────────────────────────────────────

def corriger_phrase_pipeline(
    ortho_words: list[str],
    pos_tags: list[str],
    morpho_features: dict[str, list[str]],
    ipa_words: list[str] | None = None,
    phone_lexicon: object | None = None,
    formule_mode: str = "num",
    **kwargs,
) -> list[str]:
    """Pipeline post-traitement complet P2G (couche 2).

    Orchestre les etapes de correction :
      - Etape 0   : formules (nombres, sigles via lectura_formules)
      - Etape 0b  : fusion de mots composes (aujourd'hui, peut-etre, etc.)
      - Etapes 1-1c : coherence morpho + accents (delegue au graphemiseur)
      - Etape 1d  : noms propres (capitalisation via phone_lexicon)
      - Etape 2   : entites notables (dictionnaire IPA)

    Parameters
    ----------
    ortho_words : list[str]
        Mots ortho predits par le modele P2G.
    pos_tags : list[str]
        POS tags predits.
    morpho_features : dict[str, list[str]]
        Features morpho predites.
    ipa_words : list[str] | None
        Mots IPA en entree (pour formules et noms propres).
    phone_lexicon : object | None
        PhoneLexicon pour la detection des noms propres.
    formule_mode : str
        Mode d'affichage des formules :
        - ``"num"``   : chiffres/symboles (35, SNCF) — defaut
        - ``"texte"`` : texte epele (trente-cinq)
    **kwargs
        Arguments supplementaires passes a corriger_phrase_v3
        (lexique, lexique_index, freq_map, lex_candidates).

    Returns
    -------
    list[str]
        Mots ortho corriges.
    """
    result = list(ortho_words)
    n = len(result)

    lexique = kwargs.get("lexique")
    _in_lex = (lambda w: w.lower() in lexique) if lexique is not None else (lambda w: True)

    # Etape 0 : formules (nombres, sigles)
    formule_positions: set[int] = set()
    if ipa_words is not None and len(ipa_words) == n:
        formule_positions = _appliquer_formules(
            result, ipa_words, pos_tags, n, _in_lex,
            formule_mode=formule_mode,
        )

    # Etape 0b : fusion de mots composes
    if ipa_words is not None and len(ipa_words) == n and phone_lexicon is not None:
        _fusionner_composes(result, ipa_words, phone_lexicon, formule_positions)

    # Etapes 1-1c : coherence morpho + accents (delegue au graphemiseur core)
    result = corriger_phrase_v3(
        result, pos_tags, morpho_features,
        skip_positions=formule_positions, **kwargs,
    )

    # Etape 1d : noms propres
    _capitaliser_noms_propres(result, ipa_words, phone_lexicon, formule_positions)

    # Etape 2 : entites notables (dictionnaire IPA)
    if ipa_words is not None and len(ipa_words) == n:
        _detecter_entites(result, ipa_words, phone_lexicon, formule_positions)

    return result


# ── Facade simplifiee ────────────────────────────────────────────────

def analyser(
    ipa_words: list[str],
    *,
    engine: object | None = None,
    formule_mode: str = "num",
) -> dict:
    """Analyse P2G complete d'une liste de mots IPA.

    Pipeline : inference P2G → formules → coherence morpho → noms propres.

    Parameters
    ----------
    ipa_words : list[str]
        Liste de mots en IPA.
    engine : object | None
        Moteur P2G neural. Si None, en cree un automatiquement.
    formule_mode : str
        Mode d'affichage des formules :
        - ``"num"``   : chiffres/symboles (35, SNCF) — defaut
        - ``"texte"`` : texte epele (trente-cinq)

    Returns
    -------
    dict
        {"ipa_words": [...], "ortho": [...], "pos": [...], "morpho": {...}}
    """
    if engine is None:
        engine = creer_engine()

    result = engine.analyser(ipa_words)
    result["ortho"] = corriger_phrase_pipeline(
        result["ortho"], result["pos"], result["morpho"],
        ipa_words=ipa_words,
        phone_lexicon=getattr(engine, "phone_lexicon", None),
        formule_mode=formule_mode,
    )
    return result
