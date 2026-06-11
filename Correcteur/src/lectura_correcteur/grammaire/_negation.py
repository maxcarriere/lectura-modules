"""Regles de negation : verbe + pas/plus/jamais sans ne -> inserer ne."""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection

_NEGATIFS = frozenset({"pas", "plus", "jamais", "rien", "personne"})

_VOYELLES = frozenset("aeiouyàâäéèêëïîôùûüæœ")


def verifier_negation(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Insere 'ne' quand il manque dans une negation.

    Pattern : verbe + pas/plus/jamais/rien/personne sans ne avant le verbe.
    Ex: "je mange pas" -> "je ne mange pas"

    Returns the word list with insertions applied, and Corrections
    with original="" for each insertion (index is in the output list).
    Processes in reverse order to preserve indices.
    """
    if not mots:
        return mots, []

    result = list(mots)
    pos = list(pos_tags)
    corrections: list[Correction] = []
    n = len(result)

    # Collecter les positions d'insertion
    insertions: list[int] = []

    for i in range(n):
        mot_negatif = result[i].lower()
        if mot_negatif not in _NEGATIFS:
            continue

        # Guard: "pas" = NOM (le pas, un pas, des pas, quelques pas)
        if mot_negatif == "pas":
            _pas_pos = pos[i] if i < len(pos) else ""
            if _pas_pos == "NOM":
                continue
            # Fallback lexical : DET/ART/ADJ avant "pas" → NOM
            if i >= 1:
                _prev_pos_pas = pos[i - 1] if i - 1 < len(pos) else ""
                _prev_low_pas = result[i - 1].lower()
                if (
                    _prev_pos_pas.startswith(("ART", "DET", "ADJ"))
                    or _prev_low_pas in (
                        "le", "un", "des", "les", "ses", "ces", "quelques",
                        "son", "ce", "chaque", "premier", "premiers",
                        "aucun", "moindre", "deux", "trois", "quatre",
                    )
                ):
                    continue

        # Guard: "personne" = NOM (une personne, sa personne)
        if mot_negatif == "personne":
            _pers_pos = pos[i] if i < len(pos) else ""
            if _pers_pos == "NOM":
                continue
            if i >= 1:
                _prev_pos_pers = pos[i - 1] if i - 1 < len(pos) else ""
                if _prev_pos_pers.startswith(("ART", "DET", "ADJ")):
                    continue

        # Guard: "jamais" en emploi emphatique apres "que" + subjonctif
        # "que tu aies jamais", "que l'humanite ait jamais eue"
        if mot_negatif == "jamais":
            # Pre-scan: chercher "que"/"qu'" dans les 6 mots avant le mot negatif
            _que_before_jamais = False
            for _kq in range(i - 1, max(-1, i - 7), -1):
                _wq = result[_kq].lower()
                if _wq in ("que", "qu'", "qu\u2019"):
                    _que_before_jamais = True
                    break
            if _que_before_jamais:
                continue

        # Guard: "plus" comparatif/superlatif — ne PAS inserer "ne"
        if mot_negatif == "plus":
            if i + 1 < n:
                next_pos = pos[i + 1] if i + 1 < len(pos) else ""
                next_word = result[i + 1].lower() if i + 1 < n else ""
                # plus + ADJ/ADV/que → comparatif/superlatif, pas negation
                if next_pos in ("ADJ", "ADV") or next_word == "que":
                    continue
            # Guard: DET/PRE + plus = superlatif/comparatif
            # "le plus", "la plus", "les plus", "au plus", "de plus", "en plus"
            if i >= 1:
                _prev_plus = result[i - 1].lower()
                if _prev_plus in ("le", "la", "les", "au", "de", "en", "du"):
                    continue

        # Chercher le verbe juste avant (i-1 ou i-2 si pronom intercale)
        verbe_idx = None
        if i >= 1:
            p = pos[i - 1] if i - 1 < len(pos) else ""
            if p in ("VER", "AUX"):
                verbe_idx = i - 1
            elif i >= 2:
                p2 = pos[i - 2] if i - 2 < len(pos) else ""
                if p2 in ("VER", "AUX"):
                    verbe_idx = i - 2

        if verbe_idx is None:
            continue

        # Verifier qu'il n'y a pas deja "ne"/"n'" avant le verbe
        # Chercher dans les 3 positions precedant le verbe pour
        # couvrir les pronoms intercales : "ne se nourrissent pas",
        # "ne le mange pas", "ne me les donne pas"
        deja_ne = False
        for _k in range(verbe_idx - 1, max(-1, verbe_idx - 4), -1):
            _av = result[_k].lower()
            if _av in ("ne", "n'", "n\u2019"):
                deja_ne = True
                break
            # Seuls les pronoms clitiques peuvent s'intercaler
            if _av not in (
                "se", "s'", "me", "m'", "te", "t'",
                "le", "la", "l'", "les", "lui", "leur",
                "nous", "vous", "en", "y",
            ):
                break

        # Guard: "sans" avant le verbe → negation implicite
        # "sans parler a personne", "sans rien dire", "sans jamais verifier"
        _sans_before = False
        if verbe_idx is not None:
            for _ks in range(verbe_idx - 1, max(-1, verbe_idx - 4), -1):
                _ws = result[_ks].lower()
                if _ws == "sans":
                    _sans_before = True
                    break
                if _ws not in (
                    "se", "s'", "me", "m'", "te", "t'",
                    "le", "la", "l'", "les", "lui", "leur",
                    "nous", "vous", "en", "y", "ne", "n'",
                ):
                    break
        if _sans_before:
            continue

        # Guard: "c'est pas" / "c'etait pas" = negation familiere
        # (le "ne" est volontairement omis, ne pas le reinserser)
        if verbe_idx is not None and not deja_ne:
            _vw = result[verbe_idx].lower()
            if _vw in ("est", "es", "as", "a", "était", "etait") and verbe_idx >= 1:
                _before_v = result[verbe_idx - 1].lower()
                if _before_v in ("c'", "c\u2019", "t'", "t\u2019"):
                    continue

        if not deja_ne:
            insertions.append(verbe_idx)

    # Inserer en ordre inverse pour preserver les indices
    for verbe_idx in reversed(insertions):
        verbe = result[verbe_idx]
        if verbe and verbe[0].lower() in (_VOYELLES | {"h"}):
            # Elision : ne + aime → n'aime (modifier le verbe en place)
            result[verbe_idx] = "n'" + verbe
            corrections.append(Correction(
                index=verbe_idx,
                original=verbe,
                corrige="n'" + verbe,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="negation.elision",
                explication="Negation incomplete -> elision n'",
            ))
        else:
            result.insert(verbe_idx, "ne")
            pos.insert(verbe_idx, "ADV")
            corrections.append(Correction(
                index=verbe_idx,
                original="",
                corrige="ne",
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="negation.insertion",
                explication="Negation incomplete -> insertion de 'ne'",
            ))

    return result, corrections
