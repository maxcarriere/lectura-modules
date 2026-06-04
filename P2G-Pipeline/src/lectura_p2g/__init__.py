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

__version__ = "1.0.1"

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
) -> set[int]:
    """Detecte et remplace les formules (nombres, sigles) dans la sortie P2G.

    Utilise lectura_formules.detect_number_spans et detect_sigle_spans
    pour identifier les sequences IPA correspondant a des formules.
    Remplace les mots P2G par le texte correct (display_fr).

    Modifie ``result`` in-place.
    Returns: set de positions traitees (a proteger des corrections suivantes).
    """
    try:
        from lectura_formules import detect_number_spans, detect_sigle_spans
    except ImportError:
        return set()

    formule_positions: set[int] = set()

    # ── Nombres (min_span=1 pour les nombres simples aussi) ──
    number_spans = detect_number_spans(ipa_words, min_span=1, max_span=8)
    for start, end, formula_result in number_spans:
        display_words = formula_result.display_fr.split()
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
        # Pour les sigles, display_num contient l'acronyme (ex: "SNCF")
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


# ── Pipeline complet ─────────────────────────────────────────────────

def corriger_phrase_pipeline(
    ortho_words: list[str],
    pos_tags: list[str],
    morpho_features: dict[str, list[str]],
    ipa_words: list[str] | None = None,
    phone_lexicon: object | None = None,
    **kwargs,
) -> list[str]:
    """Pipeline post-traitement complet P2G (couche 2).

    Orchestre les trois etapes :
      - Etape 0  : formules (nombres, sigles via lectura_formules)
      - Etapes 1-1c : coherence morpho + accents (delegue au graphemiseur)
      - Etape 1d : noms propres (capitalisation via phone_lexicon)

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
        )

    # Etapes 1-1c : coherence morpho + accents (delegue au graphemiseur core)
    result = corriger_phrase_v3(
        result, pos_tags, morpho_features,
        skip_positions=formule_positions, **kwargs,
    )

    # Etape 1d : noms propres
    _capitaliser_noms_propres(result, ipa_words, phone_lexicon, formule_positions)

    return result


# ── Facade simplifiee ────────────────────────────────────────────────

def analyser(
    ipa_words: list[str],
    *,
    engine: object | None = None,
) -> dict:
    """Analyse P2G complete d'une liste de mots IPA.

    Pipeline : inference P2G → formules → coherence morpho → noms propres.

    Parameters
    ----------
    ipa_words : list[str]
        Liste de mots en IPA.
    engine : object | None
        Moteur P2G neural. Si None, en cree un automatiquement.

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
    )
    return result
