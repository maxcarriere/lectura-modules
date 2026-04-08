"""Regles de conjugaison : pronom sujet + verbe.

Simplifie par rapport au POC : pas de lookup par phone (pas de IPA).
Correction par suffixe direct.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import (
    AUXILIAIRES,
    IRREGULIERS_FORMES_FAUSSES,
    PRONOM_PERSONNE,
    SUJETS_3PL,
    generer_candidats_1pl,
    generer_candidats_2pl,
    generer_candidats_3pl,
    generer_candidats_singulier,
)

_TRANSPARENTS_AUX = frozenset({
    "ne", "n'", "pas", "plus", "jamais", "rien", "point", "y", "en",
})

# Terminaisons attendues par personne (indicatif present, 1er groupe)
_SUFFIXES_ATTENDUS: dict[str, list[str]] = {
    "1": ["e", "s"],      # je mange, je finis
    "2": ["es", "s"],     # tu manges, tu finis
    "3": ["e", "t", "d"],  # il mange, il finit, il prend
    "1p": ["ons"],        # nous mangeons
    "2p": ["ez"],         # vous mangez
    "3p": ["ent"],        # ils mangent
}


def verifier_conjugaisons(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Applique les regles de conjugaison sur la phrase.

    Regles :
    3. ils/elles + VER en -e -> -ent
    5. Pronom sujet + VER -> forcer conjugaison correcte (par suffixe)
    """
    if not mots:
        return mots, []

    origs = originaux if originaux else mots
    result = list(mots)
    corrections: list[Correction] = []
    n = len(result)

    for i in range(n):
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]

        # Regle 3a : Formes fausses (allent->vont, etc.) — AVANT le check POS
        # car le tagger peut mal etiqueter ces formes (NOM au lieu de VER)
        # Verifie aussi le mot original (avant correction orthographique)
        if i > 0:
            prev_is_3pl = (
                result[i - 1].lower() in SUJETS_3PL
                or (i - 1 < len(origs) and origs[i - 1].lower() in SUJETS_3PL)
            )
            if prev_is_3pl:
                faux_candidate = IRREGULIERS_FORMES_FAUSSES.get(curr.lower())
                # Fallback : verifier le mot original (avant correction ortho)
                if faux_candidate is None and i < len(origs):
                    faux_candidate = IRREGULIERS_FORMES_FAUSSES.get(origs[i].lower())
                if faux_candidate is not None and (lexique is None or lexique.existe(faux_candidate)):
                    ancien = result[i]
                    result[i] = transferer_casse(curr, faux_candidate)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="ils/elles + forme fausse -> 3pl",
                    ))
                    continue

        # Regle 3 : ils/elles + VER -> 3e pluriel
        if i > 0 and pos in ("VER", "AUX"):
            prev_is_3pl = (
                result[i - 1].lower() in SUJETS_3PL
                or (i - 1 < len(origs) and origs[i - 1].lower() in SUJETS_3PL)
            )

            if prev_is_3pl and not curr.lower().endswith(("ent", "nt")):
                candidats = generer_candidats_3pl(curr)
                for candidate in candidats:
                    if lexique is None or lexique.existe(candidate):
                        ancien = result[i]
                        result[i] = candidate
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=candidate,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="ils/elles + verbe -> 3pl",
                        ))
                        break
                else:
                    candidate = None
                if candidate is not None and result[i] != curr:
                    continue

        # Regle 5 : Pronom sujet + VER -> correction conjugaison
        # Ne pas appliquer si un auxiliaire precede (laisser la regle PP)
        if i > 0 and pos in ("VER", "AUX"):
            _skip_aux = False
            for _j in range(i - 1, max(-1, i - 4), -1):
                _w = result[_j].lower()
                if _w in AUXILIAIRES:
                    _skip_aux = True
                    break
                if _w not in _TRANSPARENTS_AUX:
                    break
            if _skip_aux:
                pass  # Laisser la regle des participes gerer ce cas
            elif (pronom_info := _trouver_pronom_sujet(result, origs, i)) is not None:
                personne, nombre = pronom_info
                # Essayer d'abord par lexique (imparfait/futur)
                temps = _detecter_temps_from_suffixe(curr)
                correction = None
                if temps is not None:
                    correction = _corriger_par_lexique(
                        curr, personne, nombre, temps, lexique,
                    )
                # Fallback: deriver directement quand lemmatisation echoue
                if correction is None and temps is not None:
                    correction = _deriver_forme_nombre(
                        curr, personne, nombre, temps, lexique,
                    )
                # Sinon fallback sur suffixe (present)
                if correction is None:
                    correction = _corriger_par_suffixe(
                        curr, personne, nombre, lexique,
                    )
                if correction and correction.lower() != curr.lower():
                    ancien = result[i]
                    result[i] = transferer_casse(curr, correction)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication=f"Conjugaison P{personne}",
                    ))

    return result, corrections


def _trouver_pronom_sujet(
    mots: list[str], origs: list[str], idx_verbe: int,
) -> tuple[str, str] | None:
    """Cherche le pronom sujet le plus proche avant le verbe."""
    for j in range(idx_verbe - 1, max(-1, idx_verbe - 4), -1):
        mot = mots[j].lower()
        orig = origs[j].lower() if j < len(origs) else ""
        for candidate in (mot, orig):
            if candidate in PRONOM_PERSONNE:
                return PRONOM_PERSONNE[candidate]
    return None


def _detecter_temps_from_suffixe(mot: str) -> str | None:
    """Detecte le temps d'une forme verbale par son suffixe.

    Retourne "Imp" (imparfait) ou "Fut" (futur) ou None.
    """
    low = mot.lower()
    # Imparfait : -aient (avant -ais/-ait pour eviter collision)
    if low.endswith("aient"):
        return "Imp"
    if low.endswith(("ais", "ait")) and len(low) > 3:
        return "Imp"
    if low.endswith(("ions", "iez")) and len(low) > 4:
        return "Imp"
    # Futur : -ront, -rons, -rez (avant -ra/-rai/-ras)
    if low.endswith(("ront", "rons", "rez")) and len(low) > 4:
        return "Fut"
    if low.endswith(("rai", "ras")) and len(low) > 4:
        return "Fut"
    if low.endswith("ra") and len(low) > 3:
        return "Fut"
    return None


def _lemmatiser_verbe(mot: str, temps: str) -> str | None:
    """Retrouve l'infinitif a partir d'une forme conjuguee.

    Heuristique par suffixe pour imparfait et futur.
    """
    low = mot.lower()
    if temps == "Imp":
        # Ordre : suffixes longs d'abord
        for suf in ("aient", "ions", "iez", "ais", "ait"):
            if low.endswith(suf) and len(low) > len(suf):
                radical = low[:-len(suf)]
                # 2e groupe : finissait → finiss → finir
                if radical.endswith("iss"):
                    return radical[:-3] + "ir"
                # 3e groupe : dormait → dorm → dormir
                if radical.endswith(("m", "t", "v", "n")) and not radical.endswith("e"):
                    return radical + "ir"
                # 1er groupe : mangeait → mange → manger
                # radical se termine deja par 'e' (mange) → juste ajouter 'r'
                if radical.endswith("e"):
                    return radical + "r"
                return radical + "er"
    if temps == "Fut":
        for suf in ("ront", "rons", "rez", "rai", "ras", "ra"):
            if low.endswith(suf) and len(low) > len(suf) + 1:
                radical = low[:-len(suf)]
                # Le radical du futur = infinitif pour les reguliers
                # manger(a) → radical=manger, finir(a) → radical=finir
                if radical.endswith("er") or radical.endswith("ir"):
                    return radical
                # 2e groupe : finira → radical=fini → finir
                if radical.endswith("i"):
                    return radical + "r"
                # 1er groupe : mangera → radical=mange → manger
                if radical.endswith("e"):
                    return radical + "r"
                return radical + "er"
    return None


def _corriger_par_lexique(
    mot: str, personne: str, nombre: str, temps: str, lexique,
) -> str | None:
    """Corrige par lookup dans lexique.conjuguer().

    Cherche l'infinitif, puis la bonne forme conjuguee.
    Gere deux formats de cles :
    - MockLexique : "1s", "2s", "3s", "1p", "2p", "3p"
    - Lexique reel : "1", "2", "3" (generalement une seule forme par personne)
    """
    if lexique is None:
        return None

    infinitif = _lemmatiser_verbe(mot, temps)
    if infinitif is None:
        return None

    conj = lexique.conjuguer(infinitif)
    if not conj:
        return None

    temps_key = "imparfait" if temps == "Imp" else "futur"
    indicatif = conj.get("indicatif", {})
    table = indicatif.get(temps_key, {})
    if not table:
        return None

    # Essayer les cles du format MockLexique ("1s", "3p", etc.)
    key_sn = personne + ("s" if nombre in ("", "s") else "p")
    forme = table.get(key_sn)
    if forme and forme.lower() != mot.lower():
        if lexique.existe(forme):
            return forme

    # Essayer la cle simple (format reel : "1", "2", "3")
    forme_base = table.get(personne)
    if forme_base:
        # La forme dans la table peut etre sing ou plur.
        # On doit deriver la bonne forme selon le nombre.
        candidat = _deriver_forme_nombre(forme_base, personne, nombre, temps, lexique)
        if candidat and candidat.lower() != mot.lower():
            return candidat

    return None


def _deriver_forme_nombre(
    forme: str, personne: str, nombre: str, temps: str, lexique,
) -> str | None:
    """Derive la forme sing/plur a partir d'une forme de base du lexique.

    Le lexique reel stocke souvent une seule forme par personne.
    Par ex. imparfait P3 = "mangeaient" (3pl). Si on veut 3s, on derive "mangeait".
    """
    low = forme.lower()

    # Determiner si la forme de base est sing ou plur
    is_plur = low.endswith(("ent", "ons", "ez", "ont"))
    want_plur = nombre == "p"

    if is_plur == want_plur:
        # La forme correspond deja au nombre voulu
        if lexique is None or lexique.existe(forme):
            return forme

    if temps == "Imp":
        if is_plur and not want_plur:
            # 3pl -> sing : mangeaient -> mangeait (P3s), mangeais (P1/P2s)
            if low.endswith("aient"):
                if personne in ("1", "2"):
                    cand = low[:-4] + "is"
                else:
                    cand = low[:-3] + "t"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ions"):
                cand = low[:-4] + "ais"
                if lexique is None or lexique.existe(cand):
                    return cand
        elif not is_plur and want_plur:
            # sing -> plur
            if low.endswith("ait"):
                if personne == "1":
                    cand = low[:-3] + "ions"
                elif personne == "2":
                    cand = low[:-3] + "iez"
                else:
                    cand = low[:-1] + "ent"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ais"):
                radical = low[:-3]
                if personne == "1":
                    suffixes = ["ions"]
                elif personne == "2":
                    suffixes = ["iez"]
                else:
                    suffixes = ["aient"]
                for suf in suffixes:
                    cand = radical + suf
                    if lexique is None or lexique.existe(cand):
                        return cand
                    # 1er groupe : mangeais → mang + ions (drop 'e')
                    if radical.endswith("e"):
                        cand2 = radical[:-1] + suf
                        if lexique is None or lexique.existe(cand2):
                            return cand2

    if temps == "Fut":
        if is_plur and not want_plur:
            # ront -> ra (P3s), rai (P1s), ras (P2s)
            if low.endswith("ront"):
                if personne == "1":
                    cand = low[:-4] + "rai"
                elif personne == "2":
                    cand = low[:-4] + "ras"
                else:
                    cand = low[:-4] + "ra"
                if lexique is None or lexique.existe(cand):
                    return cand
        elif not is_plur and want_plur:
            # sing -> plur
            if low.endswith("ra") and not low.endswith("ira"):
                if personne == "1":
                    cand = low[:-1] + "ons"
                elif personne == "2":
                    cand = low[:-1] + "ez"
                else:
                    cand = low[:-2] + "ront"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ra"):
                # For 2e groupe : finira -> finiront/finirons/finirez
                if personne == "1":
                    cand = low[:-1] + "ons"
                elif personne == "2":
                    cand = low[:-1] + "ez"
                else:
                    cand = low[:-2] + "ront"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("rai"):
                if personne == "1":
                    cand = low[:-2] + "ons"
                elif personne == "2":
                    cand = low[:-2] + "ez"
                else:
                    cand = low[:-2] + "ont"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ras"):
                if personne == "1":
                    cand = low[:-3] + "rons"
                elif personne == "2":
                    cand = low[:-3] + "rez"
                else:
                    cand = low[:-3] + "ront"
                if lexique is None or lexique.existe(cand):
                    return cand

    return None


def _corriger_par_suffixe(
    mot: str, personne: str, nombre: str, lexique,
) -> str | None:
    """Corrige la conjugaison par ajustement de suffixe.

    Pour la 2e personne : "mange" -> "manges"
    Pour la 3e pluriel : "mange" -> "mangent"
    """
    low = mot.lower()
    key = personne + nombre  # ex: "3p", "2", "1"

    # Tu + verbe en -e sans -s final -> ajouter -s
    if key == "2" and low.endswith("e") and not low.endswith("es"):
        candidate = mot + "s"
        if lexique is None or lexique.existe(candidate):
            return candidate

    # Tu + verbe en -ent (3pl) -> singulariser + s
    if key == "2" and low.endswith("ent") and len(low) > 3:
        for candidate in generer_candidats_singulier(mot, "2"):
            if lexique is None or lexique.existe(candidate):
                return candidate

    # Je (P1) : -ent -> singulariser, -es -> retirer s
    if key == "1":
        for candidate in generer_candidats_singulier(mot, "1"):
            if lexique is None or lexique.existe(candidate):
                return candidate

    # Il/elle (P3s) : -ent -> singulariser
    if key in ("3", "3s") and low.endswith("ent") and len(low) > 3:
        for candidate in generer_candidats_singulier(mot, "3"):
            if lexique is None or lexique.existe(candidate):
                return candidate

    # 3e pluriel : generer candidats (1er, 2e, 3e groupe)
    if key == "3p" and not low.endswith(("ent", "nt")):
        candidats = generer_candidats_3pl(mot)
        for candidate in candidats:
            if lexique is None or lexique.existe(candidate):
                return candidate

    # Nous (P1p) : generer candidats 1re pluriel
    if key == "1p" and not low.endswith("ons"):
        for candidate in generer_candidats_1pl(mot):
            if lexique is None or lexique.existe(candidate):
                return candidate

    # Vous (P2p) : generer candidats 2e pluriel
    if key == "2p" and not low.endswith("ez"):
        for candidate in generer_candidats_2pl(mot):
            if lexique is None or lexique.existe(candidate):
                return candidate

    return None
