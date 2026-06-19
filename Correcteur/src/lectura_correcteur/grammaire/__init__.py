"""Sous-module grammaire : regles d'accord, conjugaison, homophones, PP, negation."""

from lectura_correcteur.grammaire._accord import verifier_accords
from lectura_correcteur.grammaire._conjugaison import verifier_conjugaisons
from lectura_correcteur.grammaire._homophones import (
    detecter_homophones_p2g,
    verifier_homophones,
)
from lectura_correcteur.grammaire._negation import verifier_negation
from lectura_correcteur.grammaire._participe import (
    verifier_participes_passes,
    verifier_pp_accord_etre,
)

__all__ = [
    "appliquer_grammaire",
    "detecter_homophones_p2g",
    "verifier_accords",
    "verifier_conjugaisons",
    "verifier_homophones",
    "verifier_negation",
    "verifier_participes_passes",
    "verifier_pp_accord_etre",
]


def appliquer_grammaire(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
    *,
    activer_negation: bool = False,
    pos_confiance: list[float] | None = None,
    pm_guidance=None,
    skip_homophones: bool = False,
    activer_homophones_gram: bool = True,
    activer_accords: bool = True,
    activer_conjugaisons: bool = True,
    activer_participes: bool = True,
    activer_pp_etre: bool = True,
) -> tuple[list[str], list]:
    """Applique toutes les regles grammaticales sur la phrase.

    Args:
        pos_confiance: Confiance POS par position (optionnel).
            Si fourni, les regles homophones peuvent skip quand
            la confiance est trop faible.
        pm_guidance: Liste d'AccordGuidance du module PM (optionnel).
            Si fourni, les regles d'accord sautent les positions
            deja corrigees par le module PM.
        skip_homophones: Si True, sauter verifier_homophones (V1).
            Utilise par V5 quand detecter_homophones_p2g a deja tourne.
        activer_homophones_gram: Activer les homophones contextuels.
        activer_accords: Activer les regles d'accord.
        activer_conjugaisons: Activer les regles de conjugaison.
        activer_participes: Activer les regles de participe passe.
        activer_pp_etre: Activer l'accord PP avec sujet (aux. etre).

    Returns:
        Tuple (mots_corriges, liste_de_Correction).
    """
    from lectura_correcteur._types import Correction, TypeCorrection

    result = list(mots)
    origs = originaux if originaux else mots
    corrections: list[Correction] = []

    # 1. Homophones contextuels (AVANT accords/conjugaison pour que est→et
    #    ait priorite sur est→sont dans les cas de coordination)
    if activer_homophones_gram and not skip_homophones:
        result_homo, corr_homo = verifier_homophones(
            result, pos_tags, morpho, lexique, origs,
            pos_confiance=pos_confiance,
        )
        result = result_homo
        corrections.extend(corr_homo)

    # 2. Accords (det+nom, det+adj+nom, det+nom+ver, genre)
    if activer_accords:
        result_acc, corr_acc = verifier_accords(
            result, pos_tags, morpho, lexique, origs,
            pm_guidance=pm_guidance,
        )
        result = result_acc
        corrections.extend(corr_acc)

    # 3. Conjugaisons (pronom+verbe)
    if activer_conjugaisons:
        result_conj, corr_conj = verifier_conjugaisons(
            result, pos_tags, morpho, lexique, origs,
        )
        result = result_conj
        corrections.extend(corr_conj)

    # 4. Participes passes (auxiliaire + infinitif -> PP)
    if activer_participes:
        result_pp, corr_pp = verifier_participes_passes(
            result, pos_tags, morpho, lexique, origs,
        )
        result = result_pp
        corrections.extend(corr_pp)

    # 4b. Accord PP avec sujet quand auxiliaire = etre
    if activer_pp_etre:
        result_ppetre, corr_ppetre = verifier_pp_accord_etre(
            result, pos_tags, morpho, lexique, origs,
        )
        result = result_ppetre
        corrections.extend(corr_ppetre)

    # 5. Negation (inserer ne)
    if activer_negation:
        result_neg, corr_neg = verifier_negation(
            result, pos_tags, morpho, lexique, origs,
        )
        result = result_neg
        corrections.extend(corr_neg)

    # 6. Elision : pronom/article + voyelle → forme elidee
    # "je ai" → "j' ai" (reconstruction collera "j'ai")
    # Ne tourne que si au moins une regle de grammaire a produit des corrections.
    # Sinon les corrections ortho (accent, etc.) declenchent des elisions parasites.
    _any_grammar_active = (
        activer_homophones_gram or activer_accords or activer_conjugaisons
        or activer_participes or activer_pp_etre or activer_negation
    )
    _ELISION = {
        "je": "j'", "me": "m'", "te": "t'", "se": "s'",
        "le": "l'", "la": "l'", "de": "d'", "ne": "n'",
        "que": "qu'",
    }
    _VOYELLES = frozenset("aeiouyàâéèêëïîôùûüæœh")
    if not _any_grammar_active:
        return result, corrections
    for _i_el in range(len(result) - 1):
        _low_el = result[_i_el].lower()
        _elided = _ELISION.get(_low_el)
        if not _elided:
            continue
        _next_el = result[_i_el + 1]
        if not _next_el or _next_el[0].lower() not in _VOYELLES:
            continue
        # Guard: don't elide if already elided
        if result[_i_el].endswith(("'", "\u2019")):
            continue
        # Guard: "le"/"la" — only elide articles (not pronom COI)
        if _low_el in ("le", "la"):
            _pos_el = pos_tags[_i_el] if _i_el < len(pos_tags) else ""
            if not _pos_el.startswith(("ART", "DET")):
                continue
        # Guard: don't elide before OOV words (English loanwords,
        # proper nouns) — "le user" should NOT become "l'user"
        if lexique is not None and not lexique.existe(_next_el.lower()):
            continue
        # Guard: only elide when the NEXT word was actually modified
        # by grammar rules (je avoir→ai) or is a known French word.
        # This avoids elision in phrases that weren't touched.
        _was_corrected_el = (
            _i_el + 1 < len(origs)
            and result[_i_el + 1].lower() != origs[_i_el + 1].lower()
        )
        if not _was_corrected_el:
            continue
        _ancien_el = result[_i_el]
        # Preserve case of first letter
        if _ancien_el[0].isupper():
            _elided = _elided[0].upper() + _elided[1:]
        result[_i_el] = _elided
        corrections.append(Correction(
            index=_i_el,
            original=_ancien_el,
            corrige=_elided,
            type_correction=TypeCorrection.GRAMMAIRE,
            regle="syntaxe.elision",
            explication=f"Elision '{_ancien_el}' -> '{_elided}' devant voyelle",
        ))

    return result, corrections
