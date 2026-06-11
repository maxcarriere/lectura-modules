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

__version__ = "4.5.3"

# ── Re-exports depuis le graphemiseur (couche 1) ─────────────────────

from lectura_graphemiseur import creer_engine, tokeniser_ipa, ipa_phrase_vers_chars
from lectura_graphemiseur import corriger_p2g, corriger_phrase_v2, corriger_phrase_v3


# ── Corrections P2G (lookup IPA → ortho pour phones sans ambiguite) ──

_P2G_CORRECTIONS: dict[str, str] | None = None


def _charger_corrections(etendu: bool = False) -> dict[str, str]:
    """Charge le fichier de corrections P2G (singleton).

    Parameters
    ----------
    etendu : bool
        Si True, charge le fichier etendu (toutes les phones sans ambiguite).
        Si False, charge le fichier base (freq > 0 uniquement).
    """
    global _P2G_CORRECTIONS
    if _P2G_CORRECTIONS is not None:
        return _P2G_CORRECTIONS

    name = "p2g_corrections_etendu.json" if etendu else "p2g_corrections.json"
    pkg = resources.files("lectura_p2g") / "data" / name
    try:
        _P2G_CORRECTIONS = json.loads(pkg.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Fallback : essayer le fichier base si l'etendu est absent
        if etendu:
            try:
                pkg_base = resources.files("lectura_p2g") / "data" / "p2g_corrections.json"
                _P2G_CORRECTIONS = json.loads(pkg_base.read_text(encoding="utf-8"))
            except FileNotFoundError:
                _P2G_CORRECTIONS = {}
        else:
            _P2G_CORRECTIONS = {}
    return _P2G_CORRECTIONS


def _appliquer_corrections(
    result: list[str],
    ipa_words: list[str],
    corrections: dict[str, str],
    skip_positions: set[int],
) -> None:
    """Applique les corrections P2G (lookup IPA → ortho sans ambiguite).

    Remplace le mot predit par l'ortho du lexique quand le phone
    n'a qu'une seule orthographe possible.

    Modifie result in-place.
    """
    matched: set[int] = set()
    for i in range(len(result)):
        if i in skip_positions:
            continue
        if i >= len(ipa_words):
            continue
        phone = ipa_words[i]
        if not phone:
            continue
        # Ne pas corriger les elisions (deja traitees)
        if "'" in phone:
            continue
        phone_nfc = unicodedata.normalize("NFC", phone)
        ortho = corrections.get(phone_nfc)
        if ortho is not None and ortho.lower() != result[i].lower():
            result[i] = ortho
            matched.add(i)
    skip_positions.update(matched)


# ── Filtrage ponctuation ─────────────────────────────────────────────

_PUNCT_CHARS = frozenset(".,;:!?…«»\"'()[]{}–—-/")


def _est_ponctuation(token: str) -> bool:
    """Un token est ponctuation si tous ses caracteres sont Unicode P/S ou dans _PUNCT_CHARS."""
    if not token:
        return False
    for ch in token:
        if ch in _PUNCT_CHARS:
            continue
        cat = unicodedata.category(ch)
        if not (cat.startswith("P") or cat.startswith("S")):
            return False
    return True


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
    formule_tolerance: str = "exact",
    number_mode: str = "auto",
    stt_mode: str = "auto",
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
    formule_tolerance : str
        Tolerance de reconnaissance IPA :
        - ``"exact"`` : IPA exact (defaut, pour texte IPA propre)
        - ``"stt"``   : tolerant STT (normalisation vocalique, Levenshtein)
    number_mode : str
        Mode de detection des nombres passe a ``detect_number_spans`` :
        - ``"num"``   : agressif (tous les nombres isoles convertis)
        - ``"texte"`` : pas de conversion numerique
        - ``"auto"``  : rejette les homophones ambigus isoles (sept/cette,
          cent/sang, vingt/vain, un/article) mais garde les spans >= 2 mots.

    Modifie ``result`` in-place.
    Returns: set de positions traitees (a proteger des corrections suivantes).
    """
    try:
        from lectura_formules import detect_number_spans, detect_sigle_spans, detect_formula_spans
    except ImportError:
        return set()

    stt = (formule_tolerance == "stt")

    if stt:
        try:
            from lectura_formules import detect_formula_spans_stt, detect_formule_spans_stt
        except ImportError:
            stt = False

    use_num = (formule_mode != "texte")

    # Controle fin des detecteurs selon stt_mode
    if stt_mode == "formule":
        # Plus permissif : tout detecter, pas de garde ambiguite
        _detect_math = True
        _detect_typed = True
        _detect_numbers = True
        number_mode = "num"   # override : pas de garde 2 mots
    elif stt_mode == "texte":
        # Texte brut : sigles uniquement
        _detect_math = False
        _detect_typed = False
        _detect_numbers = False
    else:  # "auto" (defaut)
        # Pas de formules math (evite lettres grecques/symboles)
        _detect_math = False
        _detect_typed = True
        _detect_numbers = True

    formule_positions: set[int] = set()

    # ── Formules math (avant nombres, pour eviter la fragmentation) ──
    if _detect_math:
        try:
            if stt:
                math_spans = detect_formula_spans_stt(ipa_words, min_span=3, max_span=20)
            else:
                math_spans = detect_formula_spans(ipa_words, min_span=3, max_span=20)
        except Exception:
            math_spans = []
    else:
        math_spans = []
    for start, end, formula_result in math_spans:
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

    # ── Formules typees STT (heures, dates, monnaie, %, fractions, nombres) ──
    # En mode STT, detect_formule_spans_stt remplace detect_number_spans
    # car il couvre tous les types y compris les nombres.
    if stt and _detect_typed:
        try:
            typed_spans = detect_formule_spans_stt(
                ipa_words, min_span=1, max_span=15,
                permissive=(number_mode == "num"),
            )
        except Exception:
            typed_spans = []
        for start, end, formula_result in typed_spans:
            if any(i in formule_positions for i in range(start, end)):
                continue
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

    # ── Nombres (min_span=1 pour les nombres simples aussi) ──
    number_spans = []
    if _detect_numbers:
        # Phones ambigus nombre/mot courant — rejeter les spans courts (1-2 mots)
        # contenant ces phones sans contexte numerique clair
        _AMBIG_NUM_PHONES = frozenset({
            "sɑ̃",     # cent / sans, sang, sent, sens
            "sɛt",     # sept / cette, cet
            "vɛ̃",      # vingt / vain, vin
            "ɛ̃",       # un / article indéfini
            "dø",      # deux / de (prep)
            "mil",     # mille / mil (céréale)
            "kat",     # quatre (match partiel katʁ)
            "twa",     # trois (match partiel tʁwa) / toi
        })
        number_spans = detect_number_spans(ipa_words, min_span=1, max_span=15, mode=number_mode)
    for start, end, formula_result in number_spans:
        # Pas de chevauchement avec les formules math
        if any(i in formule_positions for i in range(start, end)):
            continue
        span_len = end - start
        # Filtre IPA : rejeter les matches partiels (phone detecté ≠ phone reel)
        if span_len == 1 and hasattr(formula_result, "phone"):
            detected_phone = unicodedata.normalize("NFC", formula_result.phone)
            actual_phone = unicodedata.normalize("NFC", ipa_words[start])
            if detected_phone != actual_phone and detected_phone != actual_phone + "ə":
                continue
        # Filtre ambiguïté : pour les spans courts (1-2 mots), rejeter si
        # un phone est ambigu nombre/mot courant.
        # En mode "num", le filtre est desactive (conversion agressive).
        if number_mode != "num" and span_len <= 2:
            has_ambig = any(
                unicodedata.normalize("NFC", ipa_words[j]) in _AMBIG_NUM_PHONES
                for j in range(start, end) if j < len(ipa_words)
            )
            if has_ambig:
                continue
        if use_num and formula_result.display_num:
            display = formula_result.display_num
        else:
            display = formula_result.display_fr
        display_words = display.split()
        for k in range(span_len):
            if k < len(display_words):
                result[start + k] = display_words[k]
            else:
                result[start + k] = ""
            formule_positions.add(start + k)

    # ── Sigles (min_span=3 pour eviter les faux positifs 2-lettres
    #    comme ɛl|a → "LA", i|a → "IA" qui sont des mots courants) ──
    sigle_spans = detect_sigle_spans(ipa_words, min_span=3, max_span=15)
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
    if _detect_numbers:
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
            # Exception : pour k >= 3 (composés 3+ mots comme arc-en-ciel),
            # les faux positifs sont rares, donc on fusionne toujours.
            if k < 3:
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


# ── Elisions (clitiques isolés devant voyelle) ──────────────────────

# Mapping IPA clitique → prefixe ortho (meme table que le graphemiseur)
_ELISION_IPA_TO_ORTHO = {
    "ʒ": "j",      # je → j'ai
    "l": "l",      # le/la → l'homme
    "d": "d",      # de → d'accord
    "n": "n",      # ne → n'est
    "s": "s",      # se → s'en
    "m": "m",      # me → m'appelle
    "t": "t",      # te → t'en
    "k": "qu",     # que → qu'il
}

# Voyelles IPA (debut de mot suivant)
_VOYELLES_IPA = frozenset("aeɛiouøœəɑɔyɑ̃ɛ̃ɔ̃œ̃")


def _est_debut_voyelle(phone: str) -> bool:
    """Le phone commence par une voyelle IPA."""
    if not phone:
        return False
    # Les nasales composees (ɑ̃, ɛ̃, etc.) : le 1er char est une voyelle
    return phone[0] in _VOYELLES_IPA


def _fusionner_elisions(
    result: list[str],
    ipa_words: list[str],
    skip_positions: set[int],
) -> None:
    """Fusionne un clitique isolé avec le mot suivant par apostrophe.

    Detecte les cas ou un clitique (l, d, s, n, j, etc.) est un token
    IPA isolé suivi d'un mot commençant par une voyelle, et fusionne :
      result[i] = "l", result[i+1] = "homme" → result[i] = "l'homme", result[i+1] = ""

    Modifie result in-place.
    """
    n = len(result)
    if n < 2:
        return

    matched: set[int] = set()
    i = 0
    while i < n - 1:
        if i in skip_positions or i in matched:
            i += 1
            continue
        if i + 1 in skip_positions or i + 1 in matched:
            i += 1
            continue
        if i >= len(ipa_words) or i + 1 >= len(ipa_words):
            i += 1
            continue

        phone = ipa_words[i]
        next_phone = ipa_words[i + 1]

        ortho_prefix = _ELISION_IPA_TO_ORTHO.get(phone)
        if ortho_prefix is not None and _est_debut_voyelle(next_phone):
            # Fusionner : prefix' + mot suivant
            result[i] = ortho_prefix + "'" + result[i + 1]
            result[i + 1] = ""
            matched.add(i)
            matched.add(i + 1)
            i += 2
            continue
        i += 1

    skip_positions.update(matched)


# ── Composes IPA (mots compacts sans separateur) ─────────────────────

_COMPOSES_IPA: dict[str, str] = {
    # aujourd'hui (+ variantes STT sans /d/, avec /u/ ou /y/)
    "oʒuʁdɥi": "aujourd'hui",
    "oʒuʁɥi": "aujourd'hui",
    "oʒuʁdyi": "aujourd'hui",
    "oʒuʁyi": "aujourd'hui",
    # lorsque / lorsqu'… (+ variantes STT /ɔ/ au lieu de /o/)
    "loʁsk": "lorsque",
    "loʁskə": "lorsque",
    "lɔʁsk": "lorsque",
    "lɔʁskə": "lorsque",
    "loʁskil": "lorsqu'il",
    "lɔʁskil": "lorsqu'il",
    "loʁskɛl": "lorsqu'elle",
    "lɔʁskɛl": "lorsqu'elle",
    "loʁskɔ̃": "lorsqu'on",
    "lɔʁskɔ̃": "lorsqu'on",
    # puisque / puisqu'… (+ variantes STT /ɥi/ → /yi/)
    "pɥisk": "puisque",
    "pɥiskə": "puisque",
    "pyisk": "puisque",
    "pɥiskil": "puisqu'il",
    "pyiskil": "puisqu'il",
    "pɥiskɛl": "puisqu'elle",
    "pɥiskɔ̃": "puisqu'on",
    # quelqu'un(e)
    "kɛlkœ̃": "quelqu'un",
    "kɛlkyn": "quelqu'une",
    # jusque / jusqu'… (+ variantes STT)
    "ʒyska": "jusque",
    "ʒyskə": "jusque",
    "ʒyskisi": "jusqu'ici",
    "ʒysko": "jusqu'au",
    "ʒyskoz": "jusqu'aux",
    "ʒyskalɔʁ": "jusqu'alors",
    # peut-etre
    "pøtɛtʁ": "peut-être",
    "pøtɛtʁə": "peut-être",
    # c'est-a-dire
    "sɛtadiʁ": "c'est-à-dire",
    "sɛtadiʁə": "c'est-à-dire",
    # est-ce que / qu'est-ce que
    "ɛskə": "est-ce que",
    "ɛstkə": "est-ce que",
    "kɛskə": "qu'est-ce que",
    "kɛstkə": "qu'est-ce que",
    # parce que
    "paʁskə": "parce que",
    "paʁsk": "parce que",
    # presqu'ile / presque
    "pʁɛskil": "presqu'île",
    # quelquefois
    "kɛlkəfwa": "quelquefois",
}

# Normaliser les cles en NFC
_COMPOSES_IPA = {
    unicodedata.normalize("NFC", k): v for k, v in _COMPOSES_IPA.items()
}


def _corriger_composes_mono(
    result: list[str],
    ipa_words: list[str],
    phone_lexicon: object | None,
    skip_positions: set[int],
) -> None:
    """Corrige les mots composes entres sans separateur (IPA compact).

    Couche 1 : lookup exact dans _COMPOSES_IPA (O(1)).
    Couche 2 : lookup dans phone_lexicon pour les entrees dont l'ortho
               contient une apostrophe ou un tiret interne.

    Skip les phones qui contiennent deja une apostrophe (deja traites
    par _split_elision).

    Modifie result in-place.
    """
    n = len(result)
    matched: set[int] = set()
    for i in range(n):
        if i in skip_positions:
            continue
        if i >= len(ipa_words):
            continue
        phone = ipa_words[i]
        if not phone or "'" in phone:
            continue

        phone_nfc = unicodedata.normalize("NFC", phone)

        # Couche 1 : dictionnaire curate
        ortho = _COMPOSES_IPA.get(phone_nfc)
        if ortho is not None:
            result[i] = ortho
            matched.add(i)
            continue

        # Couche 2 : phone_lexicon — chercher les entrees compose
        if phone_lexicon is None or not hasattr(phone_lexicon, "all_entries"):
            continue

        found = False
        for phone_try in (phone_nfc, phone_nfc + "ə"):
            entries = phone_lexicon.all_entries(phone_try)
            if not entries:
                continue
            for e in entries:
                o = e.get("ortho", "")
                if len(o) < 5:
                    continue
                # Verifier que c'est un vrai compose (pas une elision simple)
                # Les deux parties autour du separateur doivent etre >= 2 chars
                sep_pos = -1
                for sep in ("'", "-"):
                    idx = o.find(sep, 1)
                    if idx > 0 and idx < len(o) - 1:
                        sep_pos = idx
                        break
                if sep_pos < 0:
                    continue
                prefix, suffix = o[:sep_pos], o[sep_pos + 1:]
                if len(prefix) >= 2 and len(suffix) >= 2:
                    result[i] = o
                    matched.add(i)
                    found = True
                    break
            if found:
                break

    skip_positions.update(matched)


# ── Pipeline complet ─────────────────────────────────────────────────

def corriger_phrase_pipeline(
    ortho_words: list[str],
    pos_tags: list[str],
    morpho_features: dict[str, list[str]],
    ipa_words: list[str] | None = None,
    phone_lexicon: object | None = None,
    formule_mode: str = "num",
    formule_tolerance: str = "exact",
    number_mode: str = "auto",
    corrections_p2g: bool | str = True,
    stt_mode: str = "auto",
    **kwargs,
) -> list[str]:
    """Pipeline post-traitement complet P2G (couche 2).

    Orchestre les etapes de correction :
      - Etape 0    : formules (nombres, sigles via lectura_formules)
      - Etape 0a   : elisions (clitiques isolés devant voyelle)
      - Etape 0b   : fusion de mots composes (aujourd'hui, peut-etre, etc.)
      - Etape 0b2  : composes mono-mots (IPA compact sans separateur)
      - Etape 0b3  : corrections P2G (lookup IPA → ortho sans ambiguite)
      - Etape 0c   : correction neuronale (correcteur P2G, optionnel)
      - Etapes 1-1c: coherence morpho + accents (delegue au graphemiseur)
      - Etape 1d   : rescoring homophones par n-gramme POS (si pos_ngram fourni)
      - Etape 1e   : noms propres (capitalisation via phone_lexicon)
      - Etape 2    : entites notables (dictionnaire IPA)

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
    formule_tolerance : str
        Tolerance de reconnaissance IPA :
        - ``"exact"`` : IPA exact (defaut)
        - ``"stt"``   : tolerant STT (normalisation vocalique, Levenshtein)
    number_mode : str
        Mode de detection des nombres :
        - ``"num"``   : agressif (tous les nombres isoles convertis)
        - ``"texte"`` : pas de conversion numerique
        - ``"auto"``  : rejette les homophones ambigus isoles
    corrections_p2g : bool | str
        Active les corrections P2G (lookup IPA → ortho) :
        - ``True``      : fichier base (freq > 0) — defaut
        - ``"etendu"``  : fichier etendu (toutes les phones sans ambiguite)
        - ``False``     : desactive
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
            formule_tolerance=formule_tolerance,
            number_mode=number_mode,
            stt_mode=stt_mode,
        )

    # Etape 0a : elisions (clitiques isolés devant voyelle)
    if ipa_words is not None and len(ipa_words) == n:
        _fusionner_elisions(result, ipa_words, formule_positions)

    # Etape 0b : fusion de mots composes
    if ipa_words is not None and len(ipa_words) == n and phone_lexicon is not None:
        _fusionner_composes(result, ipa_words, phone_lexicon, formule_positions)

    # Etape 0b2 : composes mono-mots (IPA compact sans separateur)
    if ipa_words is not None and len(ipa_words) == n:
        _corriger_composes_mono(result, ipa_words, phone_lexicon, formule_positions)

    # Etape 0b3 : corrections P2G (lookup IPA → ortho sans ambiguite)
    if corrections_p2g and ipa_words is not None and len(ipa_words) == n:
        etendu = (corrections_p2g == "etendu")
        corrections = _charger_corrections(etendu=etendu)
        if corrections:
            _appliquer_corrections(result, ipa_words, corrections, formule_positions)

    # Etape 0c : correction neuronale (optionnel)
    correcteur = kwargs.pop("correcteur", None)
    if correcteur is not None:
        try:
            corrected = correcteur.corriger(result)
            for i in range(n):
                if i not in formule_positions and corrected[i]:
                    result[i] = corrected[i]
        except Exception:
            pass  # Degradation gracieuse

    # Etapes 1-1d : coherence morpho + accents + rescoring homophones
    result = corriger_phrase_v3(
        result, pos_tags, morpho_features,
        skip_positions=formule_positions, **kwargs,
    )

    # Etape 1e : noms propres
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
    formule_tolerance: str = "exact",
    number_mode: str = "auto",
    corrections_p2g: bool | str = True,
    stt_mode: str = "auto",
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
    formule_tolerance : str
        Tolerance de reconnaissance IPA :
        - ``"exact"`` : IPA exact (defaut)
        - ``"stt"``   : tolerant STT (normalisation vocalique, Levenshtein)
    number_mode : str
        Mode de detection des nombres :
        - ``"num"``   : agressif (tous les nombres isoles convertis)
        - ``"texte"`` : pas de conversion numerique
        - ``"auto"``  : rejette les homophones ambigus isoles
    corrections_p2g : bool | str
        Active les corrections P2G (lookup IPA → ortho) :
        - ``True``      : fichier base (freq > 0) — defaut
        - ``"etendu"``  : fichier etendu (toutes les phones)
        - ``False``     : desactive

    Returns
    -------
    dict
        {"ipa_words": [...], "ortho": [...], "pos": [...], "morpho": {...}}
    """
    if engine is None:
        engine = creer_engine()

    # ── Filtrage ponctuation : extraire avant inference ──
    punct_map: dict[int, str] = {}
    clean_words: list[str] = []
    for idx, tok in enumerate(ipa_words):
        if _est_ponctuation(tok):
            punct_map[idx] = tok
        else:
            clean_words.append(tok)

    if clean_words:
        result = engine.analyser(clean_words)
        result["ortho"] = corriger_phrase_pipeline(
            result["ortho"], result["pos"], result["morpho"],
            ipa_words=clean_words,
            phone_lexicon=getattr(engine, "phone_lexicon", None),
            formule_mode=formule_mode,
            formule_tolerance=formule_tolerance,
            number_mode=number_mode,
            corrections_p2g=corrections_p2g,
            stt_mode=stt_mode,
            correcteur=getattr(engine, "correcteur", None),
        )
    else:
        result = {"ipa_words": [], "ortho": [], "pos": [], "morpho": {}}

    # ── Reinjecter la ponctuation aux bonnes positions ──
    if punct_map:
        morpho_keys = list(result.get("morpho", {}).keys())
        n_clean = len(result.get("ortho", []))
        for pos in sorted(punct_map.keys()):
            tok = punct_map[pos]
            result.setdefault("ipa_words", []).insert(pos, tok)
            result["ortho"].insert(pos, tok)
            result["pos"].insert(pos, "PUNCT")
            for k in morpho_keys:
                result["morpho"][k].insert(pos, "_")

    return result
