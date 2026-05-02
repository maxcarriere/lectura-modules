"""Coherence contextuelle par bigrams POS suspects.

Module experimental (OFF par defaut) qui detecte des sequences POS
anormales et propose des remplacements homophones pour les resoudre.

Exemple : "et beau" (CON ADJ) est valide, mais "et dans" (CON PRE)
est suspect -> si "et" peut etre "est" (AUX), "AUX PRE" est valide.
"""

from __future__ import annotations

from typing import Any

from lectura_correcteur._types import Correction, MotAnalyse, TypeCorrection

# Bigrams POS suspects (pos_gauche, pos_droite)
# Ces sequences ne se trouvent normalement pas dans du francais correct.
_BIGRAMS_SUSPECTS: set[tuple[str, str]] = {
    ("CON", "PRE"),       # "et dans" suspect -> "est dans"
    ("CON", "ART:def"),   # "et le" suspect si c'est "est le"
    ("CON", "ART:ind"),   # "et un" suspect si c'est "est un"
    ("PRE", "CON"),       # "a et" suspect -> "à et" (homophone a/à)
    ("CON", "VER"),       # "et mange" suspect -> "est mange" (cas rares)
}

# Bigrams valides utilises pour valider un remplacement
_BIGRAMS_VALIDES: set[tuple[str, str]] = {
    ("AUX", "PRE"),
    ("AUX", "ART:def"),
    ("AUX", "ART:ind"),
    ("AUX", "ADJ"),
    ("AUX", "NOM"),
    ("AUX", "VER"),
    ("AUX", "ADV"),
    ("PRE", "ART:def"),
    ("PRE", "ART:ind"),
    ("PRE", "NOM"),
    ("PRO:per", "VER"),
    ("PRO:per", "AUX"),
    ("ART:def", "NOM"),
    ("ART:def", "ADJ"),
    ("ART:ind", "NOM"),
    ("ART:ind", "ADJ"),
    ("ADV", "VER"),
    ("ADV", "ADJ"),
    ("NOM", "VER"),
    ("NOM", "ADJ"),
    ("VER", "PRE"),
    ("VER", "ART:def"),
    ("VER", "ART:ind"),
    ("VER", "NOM"),
    ("VER", "ADJ"),
    ("VER", "ADV"),
    ("ADJ", "NOM"),
}


def appliquer_coherence(
    analyses: list[MotAnalyse],
    lexique: Any,
) -> list[Correction]:
    """Detecte les bigrams POS suspects et propose des corrections.

    Pour chaque bigram suspect, cherche un homophone du mot suspect
    dont le POS resoudrait le bigram (le rendrait valide).
    """
    corrections: list[Correction] = []

    for i in range(len(analyses) - 1):
        bigram = (analyses[i].pos, analyses[i + 1].pos)
        if bigram not in _BIGRAMS_SUSPECTS:
            continue

        # Tester un remplacement pour le mot de gauche
        remplacement = _chercher_homophone_valide(
            analyses[i], analyses[i + 1].pos, "droite", lexique,
        )
        if remplacement:
            orig = analyses[i].corrige
            analyses[i].corrige = remplacement[0]
            analyses[i].pos = remplacement[1]
            if analyses[i].type_correction == TypeCorrection.AUCUNE:
                analyses[i].type_correction = TypeCorrection.GRAMMAIRE
            corrections.append(Correction(
                index=i,
                original=orig,
                corrige=remplacement[0],
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="coherence.pos",
                explication="COHERENCE_POS",
            ))

    return corrections


def _chercher_homophone_valide(
    analyse: MotAnalyse,
    pos_voisin: str,
    cote: str,
    lexique: Any,
) -> tuple[str, str] | None:
    """Cherche un homophone dont le POS forme un bigram valide avec le voisin."""
    if not hasattr(lexique, "phone_de") or not hasattr(lexique, "homophones"):
        return None

    phone = lexique.phone_de(analyse.corrige)
    if not phone:
        return None

    meilleur: tuple[str, str, float] | None = None

    for entry in lexique.homophones(phone):
        ortho = entry.get("ortho", "")
        cgram = entry.get("cgram", "")
        if not ortho or not cgram:
            continue
        if ortho.lower() == analyse.corrige.lower():
            continue

        # Verifier que le bigram serait valide
        if cote == "droite":
            nouveau_bigram = (cgram, pos_voisin)
        else:
            nouveau_bigram = (pos_voisin, cgram)

        if nouveau_bigram in _BIGRAMS_VALIDES:
            freq = float(entry.get("freq") or 0)
            if meilleur is None or freq > meilleur[2]:
                meilleur = (ortho, cgram, freq)

    if meilleur:
        return (meilleur[0], meilleur[1])
    return None


def verifier_coherence_post_corrections(
    analyses: list[MotAnalyse],
    lexique: Any,
    tagger: Any,
) -> list[Correction]:
    """Re-verification post-corrections : accords et sujet-verbe.

    Apres les couches ortho+grammaire, re-tagger les formes corrigees
    et verifier la coherence genre/nombre dans les groupes nominaux.
    """
    corrections: list[Correction] = []
    if not analyses or tagger is None:
        return corrections

    # Re-tagger les formes corrigees
    decided_words = [a.corrige for a in analyses]
    retags = tagger.tag_words(decided_words)

    # Mettre a jour les POS des analyses
    for j, tag in enumerate(retags):
        if j < len(analyses) and tag.get("pos"):
            analyses[j].pos = tag["pos"]

    # Re-verification accords DET→NOM genre/nombre
    n = len(analyses)
    for i in range(n - 1):
        pos_i = analyses[i].pos
        pos_j = analyses[i + 1].pos

        # DET + NOM : verifier coherence nombre
        if pos_i.startswith(("ART", "DET")) and pos_j in ("NOM", "ADJ"):
            mot_det = analyses[i].corrige.lower()
            mot_nom = analyses[i + 1].corrige.lower()
            # Determinant pluriel + nom sans marque plurielle
            _PLUR_DETS = frozenset({
                "les", "des", "ces", "ses", "mes", "tes",
                "nos", "vos", "leurs",
            })
            _SING_DETS = frozenset({
                "le", "la", "un", "une", "ce", "cette", "cet",
                "son", "sa", "mon", "ma", "ton", "ta",
            })
            if mot_det in _PLUR_DETS:
                # NOM devrait etre au pluriel
                if not mot_nom.endswith(("s", "x", "z")):
                    # Chercher la forme plurielle dans le lexique
                    if hasattr(lexique, "info"):
                        cand_pl = mot_nom + "s"
                        if lexique.existe(cand_pl):
                            orig = analyses[i + 1].corrige
                            analyses[i + 1].corrige = cand_pl
                            if analyses[i + 1].type_correction == TypeCorrection.AUCUNE:
                                analyses[i + 1].type_correction = TypeCorrection.GRAMMAIRE
                            corrections.append(Correction(
                                index=i + 1,
                                original=orig,
                                corrige=cand_pl,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="coherence.accord_nombre",
                                explication="Accord nombre DET pluriel + NOM",
                            ))
            elif mot_det in _SING_DETS:
                # NOM devrait etre au singulier
                if mot_nom.endswith("s") and len(mot_nom) > 2:
                    cand_sg = mot_nom[:-1]
                    if hasattr(lexique, "existe") and lexique.existe(cand_sg):
                        orig = analyses[i + 1].corrige
                        analyses[i + 1].corrige = cand_sg
                        if analyses[i + 1].type_correction == TypeCorrection.AUCUNE:
                            analyses[i + 1].type_correction = TypeCorrection.GRAMMAIRE
                        corrections.append(Correction(
                            index=i + 1,
                            original=orig,
                            corrige=cand_sg,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="coherence.accord_nombre",
                            explication="Accord nombre DET singulier + NOM",
                        ))

    return corrections
