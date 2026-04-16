"""Post-traitement pour le modèle P2G unifié.

Deux stratégies complémentaires :
- corriger_p2g() : correction par mot via prédictions morpho (utile pour v1)
- corriger_phrase_v2() : correction contextuelle inter-mots + lexique (utile pour v2)
"""

from __future__ import annotations

_HOMOPHONES_POS: dict[tuple[str, str], str] | None = None
_PLUR_DET: frozenset[str] | None = None
_SING_DET: frozenset[str] | None = None
_NO_PLURAL_S: frozenset[str] | None = None


def _get_homophones_pos() -> dict[tuple[str, str], str]:
    global _HOMOPHONES_POS
    if _HOMOPHONES_POS is None:
        from lectura_p2g._chargeur import homophones_pos
        _HOMOPHONES_POS = homophones_pos()
    return _HOMOPHONES_POS


def _get_plur_det() -> frozenset[str]:
    global _PLUR_DET
    if _PLUR_DET is None:
        from lectura_p2g._chargeur import determinants_pluriel
        _PLUR_DET = determinants_pluriel()
    return _PLUR_DET


def _get_no_plural_s() -> frozenset[str]:
    global _NO_PLURAL_S
    if _NO_PLURAL_S is None:
        from lectura_p2g._chargeur import invariables_pluriel
        _NO_PLURAL_S = invariables_pluriel()
    return _NO_PLURAL_S


# ── Correction par mot (v1, morpho-based) ──────────────────────────

def corriger_p2g(
    ortho: str,
    pos: str = "",
    morpho: dict[str, str] | None = None,
) -> str:
    """Corrige l'orthographe P2G en utilisant les prédictions morpho.

    Règles appliquées :
    - Number=Plur : ajoute -s si absent (sauf si finit par s/x/z)
    - Number=Sing : retire -s final si en trop
    - Gender=Fem  : ajoute -e si participe/adj sans -e final
    - Gender=Masc : retire -e final si en trop sur participe/adj
    - Person=3 + Number=Plur + VerbForm=Fin : terminaison -ent
    """
    if not ortho:
        return ortho

    # ── Correction homophones POS-aware (priorité haute) ──
    key = (ortho.lower(), pos)
    homophones = _get_homophones_pos()
    if key in homophones:
        return homophones[key]

    if morpho is None:
        return ortho

    number = morpho.get("Number", "_")
    gender = morpho.get("Gender", "_")
    person = morpho.get("Person", "_")
    verbform = morpho.get("VerbForm", "_")

    # Ne pas toucher les mots fonctionnels
    if pos in ("PRE", "CON", "ART:def", "ART:ind", "PRO:rel", "PRO:dem",
               "PRO:per", "ADV", "INT"):
        return ortho

    result = ortho

    # ── Verbes 3pl : ajouter -nt sur forme en -e ──
    if (
        number == "Plur"
        and person == "3"
        and verbform == "Fin"
        and pos in ("VER", "AUX")
    ):
        if result.endswith("e") and not result.endswith(("ent", "nt")):
            result = result + "nt"
        return result

    # ── Ne pas modifier les verbes conjugués (1sg/2sg finissent en -s/-x) ──
    if verbform == "Fin" and pos in ("VER", "AUX"):
        return result

    # ── Féminin + Pluriel : mot doit finir par -es ──
    if number == "Plur" and gender == "Fem" and pos in ("ADJ", "VER", "NOM"):
        if result.endswith("es"):
            return result  # déjà correct
        if result.endswith("e"):
            return result + "s"  # ajouter -s
        if result.endswith("s"):
            return result  # finit déjà par -s, ne pas ajouter -e
        # Ni -e ni -s : ajouter -es
        return result + "es"

    # ── Pluriel seul : ajouter -s ──
    if number == "Plur" and pos in ("NOM", "ADJ"):
        if not result.endswith(("s", "x", "z")):
            return result + "s"
        return result

    # ── Féminin seul (singulier) : ajouter -e ──
    if (
        gender == "Fem"
        and number != "Plur"
        and pos in ("ADJ", "VER", "NOM")
        and verbform in ("Part", "_")
        and not result.endswith(("e", "ée"))
    ):
        return result + "e"

    # ── Masculin : retirer -ée → -é (participe) ──
    if (
        gender == "Masc"
        and pos in ("VER", "ADJ")
        and verbform == "Part"
        and result.endswith("ée")
    ):
        return result[:-1]

    return result


# ── Correction contextuelle inter-mots (v2) ────────────────────────

def corriger_phrase_v2(
    ortho_words: list[str],
    pos_tags: list[str],
    lexique: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Corrige une phrase en exploitant le contexte inter-mots.

    Règles appliquées (uniquement en ajout, jamais en suppression) :
    - Dét. pluriel + NOM/ADJ sans -s → ajouter -s (si forme plurielle dans lexique)
    - Dét. pluriel + ADJ + NOM sans -s → idem
    - ils/elles + VER en -e → ajouter -nt (si forme conjuguée dans lexique)

    Le lexique (set de formes en minuscules) sert de filtre de sécurité.
    Sans lexique, les règles sont quand même appliquées mais sans vérification.
    """
    if not ortho_words:
        return ortho_words

    result = list(ortho_words)
    n = len(result)
    plur_det = _get_plur_det()
    no_plural_s = _get_no_plural_s()

    for i in range(n):
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]

        # ── Règle 1 : Dét. pluriel → NOM/ADJ doit avoir -s ──
        if i > 0 and pos in ("NOM", "ADJ"):
            prev = result[i - 1].lower()
            if (
                prev in plur_det
                and not curr.endswith(("s", "x", "z"))
                and len(curr) > 1
                and curr.lower() not in no_plural_s
            ):
                candidate = curr + "s"
                if lexique is None or candidate.lower() in lexique:
                    result[i] = candidate

        # ── Règle 2 : Dét. pluriel + ADJ + NOM ──
        if (
            i > 1
            and pos == "NOM"
            and pos_tags[i - 1] == "ADJ"
        ):
            prev2 = result[i - 2].lower()
            if (
                prev2 in plur_det
                and not result[i].endswith(("s", "x", "z"))
                and len(result[i]) > 1
                and result[i].lower() not in no_plural_s
            ):
                candidate = result[i] + "s"
                if lexique is None or candidate.lower() in lexique:
                    result[i] = candidate

        # ── Règle 3 : ils/elles + VER en -e → -ent ──
        if (
            i > 0
            and pos in ("VER", "AUX")
            and result[i - 1].lower() in ("ils", "elles")
            and curr.endswith("e")
            and not curr.endswith(("ent", "nt"))
        ):
            candidate = curr + "nt"
            if lexique is None or candidate.lower() in lexique:
                result[i] = candidate

    return result


def corriger_phrase(
    ortho_words: list[str],
    pos_tags: list[str],
    morpho_features: dict[str, list[str]],
) -> list[str]:
    """Corrige tous les mots d'une phrase (v1, morpho-based)."""
    result = []
    n = len(ortho_words)
    for i in range(n):
        word_morpho = {
            feat: vals[i] if i < len(vals) else "_"
            for feat, vals in morpho_features.items()
        }
        pos = pos_tags[i] if i < len(pos_tags) else ""
        corrected = corriger_p2g(ortho_words[i], pos=pos, morpho=word_morpho)
        result.append(corrected)
    return result
