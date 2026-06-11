"""Correction d'accord guidee par n-gram PM (POS+Morpho).

Utilise le scoring PM n-gram pour detecter les sequences improbables
dans les groupes nominaux (DET + ADJ? + NOM + ADJ? + VER?), generer
des hypotheses de correction concurrentes, les scorer et appliquer
la meilleure si le delta de score est suffisant.

Quatre phases :
1. Detection des groupes d'accord (ancres sur DET)
2. Generation d'hypotheses (adapter au DET, changer le DET, cascade NOM+VER)
3. Scoring par PM n-gram (fenetre locale de trigrammes)
4. Selection (delta minimum pour accepter)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lectura_correcteur._pos_ngram import PosNgram
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import (
    INVARIABLES,
    PLUR_DET,
    SING_DET,
    SING_FEM_DET,
    SING_MASC_DET,
    generer_candidats_3pl,
    generer_candidats_pluriel,
    generer_candidats_singulier_nom,
)
from lectura_correcteur.grammaire._accord import _ACCORD_EXCLUS

if TYPE_CHECKING:
    pass


# =====================================================================
# Dataclasses
# =====================================================================

@dataclass
class AccordGroupe:
    """Groupe d'accord ancre sur un determinant."""

    debut: int
    fin: int               # exclusif
    det_idx: int
    nom_idx: int | None = None
    adj_indices: list[int] = field(default_factory=list)
    ver_idx: int | None = None
    ancre_nombre: str = ""   # "Sing" ou "Plur"
    ancre_genre: str = "_"   # "Masc", "Fem", "_"
    violation: bool = False


@dataclass
class HypotheseMorpho:
    """Hypothese de correction pour un groupe d'accord."""

    formes: dict[int, str] = field(default_factory=dict)    # idx -> forme corrigee
    pm_tags: dict[int, str] = field(default_factory=dict)    # idx -> PM tag
    score: float = 0.0
    description: str = ""


@dataclass
class AccordGuidance:
    """Resultat : correction guidee par PM n-gram."""

    index: int
    forme_suggeree: str
    pm_tag: str
    nombre_suggere: str    # "Sing" ou "Plur"
    genre_suggere: str     # "Masc", "Fem", "_"
    confiance: float = 0.0


# POS consideres comme nominaux/adjectivaux
_POS_NOM = frozenset({"NOM", "NAM"})
_POS_ADJ = frozenset({"ADJ"})
_POS_VER = frozenset({"VER", "AUX"})
_POS_DET = frozenset({"DET", "DET:ART", "DET:dem", "ADJ:dem", "ADJ:pos", "DET:pos"})


# =====================================================================
# Phase 1 : Detection des groupes d'accord
# =====================================================================

def _detecter_groupes(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    pos_ngram: PosNgram,
    seuil_violation: float,
) -> list[AccordGroupe]:
    """Scanne la sequence pour trouver les groupes ancres sur un DET."""
    n = len(mots)
    groupes: list[AccordGroupe] = []

    for i in range(n):
        pos_i = pos_tags[i] if i < len(pos_tags) else ""
        low_i = mots[i].lower()

        # Doit etre un DET reconnu
        if not _est_det(pos_i, low_i):
            continue

        # Determiner le nombre ancre du DET
        if low_i in PLUR_DET:
            ancre_nombre = "Plur"
        elif low_i in SING_DET:
            ancre_nombre = "Sing"
        else:
            continue

        # Genre ancre du DET
        if low_i in SING_FEM_DET:
            ancre_genre = "Fem"
        elif low_i in SING_MASC_DET:
            ancre_genre = "Masc"
        else:
            ancre_genre = "_"

        # Scanner a droite : ADJ? NOM? ADJ? (fenetre max 4 mots)
        nom_idx = None
        adj_indices: list[int] = []
        ver_idx = None
        fin = i + 1

        for j in range(i + 1, min(i + 5, n)):
            pos_j = pos_tags[j] if j < len(pos_tags) else ""
            low_j = mots[j].lower()

            if low_j in _ACCORD_EXCLUS:
                break

            if pos_j in _POS_NOM or pos_j == "NOM PROPRE":
                if pos_j == "NOM PROPRE":
                    break  # ne pas toucher aux noms propres
                nom_idx = j
                fin = j + 1
            elif pos_j in _POS_ADJ:
                adj_indices.append(j)
                fin = j + 1
            elif pos_j in _POS_VER and nom_idx is not None:
                # VER apres le NOM (sujet-verbe)
                ver_idx = j
                fin = j + 1
                break
            elif pos_j in _POS_DET or _est_det(pos_j, low_j):
                break  # nouveau groupe nominal
            else:
                break

        if nom_idx is None and not adj_indices:
            continue

        # Verifier les desaccords
        violation = False
        nombre_morpho = morpho.get("nombre", [])
        genre_morpho = morpho.get("genre", [])

        members = adj_indices[:]
        if nom_idx is not None:
            members.append(nom_idx)

        for midx in members:
            low_m = mots[midx].lower()
            if low_m in INVARIABLES or low_m in _ACCORD_EXCLUS:
                continue
            # Extraire le nombre du membre depuis le lexique
            m_nombre = _nombre_lexique(low_m, lexique)
            if not m_nombre:
                m_nombre = nombre_morpho[midx] if midx < len(nombre_morpho) else "_"
                m_nombre = _norm_nombre(m_nombre)
            if m_nombre and m_nombre != "_" and m_nombre != ancre_nombre:
                violation = True
                break

        # Verification VER aussi
        if not violation and ver_idx is not None:
            low_v = mots[ver_idx].lower()
            if low_v not in INVARIABLES and low_v not in _ACCORD_EXCLUS:
                v_nombre = _nombre_lexique(low_v, lexique)
                if not v_nombre:
                    v_nombre = nombre_morpho[ver_idx] if ver_idx < len(nombre_morpho) else "_"
                    v_nombre = _norm_nombre(v_nombre)
                if v_nombre and v_nombre != "_" and v_nombre != ancre_nombre:
                    violation = True

        if not violation:
            continue

        # Confirmer via PM bigram : au moins un bigram improbable dans le groupe
        pm_tags_local = _construire_pm_tags(mots, pos_tags, morpho, lexique)
        bigram_suspect = False
        for j in range(i, min(fin - 1, n - 1)):
            lp = pos_ngram.logp_pm_bigram(
                pm_tags_local[j] if j < len(pm_tags_local) else "_|_|_",
                pm_tags_local[j + 1] if j + 1 < len(pm_tags_local) else "_|_|_",
            )
            if lp < seuil_violation:
                bigram_suspect = True
                break

        if not bigram_suspect:
            continue

        groupes.append(AccordGroupe(
            debut=i,
            fin=fin,
            det_idx=i,
            nom_idx=nom_idx,
            adj_indices=adj_indices,
            ver_idx=ver_idx,
            ancre_nombre=ancre_nombre,
            ancre_genre=ancre_genre,
            violation=True,
        ))

    return groupes


# =====================================================================
# Phase 2 : Generation d'hypotheses
# =====================================================================

def _generer_hypotheses(
    groupe: AccordGroupe,
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
) -> list[HypotheseMorpho]:
    """Genere des hypotheses de correction pour un groupe avec violation."""
    from lectura_correcteur._tags import _appliquer_nombre

    hypotheses: list[HypotheseMorpho] = []

    # --- Hypothese D (nul) : garder tel quel ---
    hypotheses.append(HypotheseMorpho(
        formes={},
        pm_tags={},
        description="nul",
    ))

    ancre = groupe.ancre_nombre
    genre = groupe.ancre_genre

    # Positions a infliger
    nom_adj_indices = list(groupe.adj_indices)
    if groupe.nom_idx is not None:
        nom_adj_indices.append(groupe.nom_idx)

    # --- Hypothese A : adapter NOM/ADJ/VER au DET (DET est correct) ---
    hyp_a_formes: dict[int, str] = {}
    hyp_a_tags: dict[int, str] = {}
    nombre_cible = "pluriel" if ancre == "Plur" else "singulier"

    for midx in nom_adj_indices:
        low_m = mots[midx].lower()
        if low_m in INVARIABLES or low_m in _ACCORD_EXCLUS:
            continue
        m_nombre = _nombre_lexique(low_m, lexique)
        if m_nombre == ancre:
            continue  # deja correct
        new_forme = _appliquer_nombre(low_m, nombre_cible, lexique)
        # Verifier que la nouvelle forme existe dans le lexique
        if new_forme.lower() != low_m and lexique.existe(new_forme.lower()):
            hyp_a_formes[midx] = transferer_casse(mots[midx], new_forme)
            pos_m = pos_tags[midx] if midx < len(pos_tags) else "NOM"
            hyp_a_tags[midx] = PosNgram.make_pm_tag(pos_m, ancre, genre)

    # Propager au VER si present
    if groupe.ver_idx is not None and hyp_a_formes:
        _ajouter_ver_hypothese(
            groupe.ver_idx, ancre, mots, pos_tags, lexique,
            hyp_a_formes, hyp_a_tags,
        )

    if hyp_a_formes:
        hypotheses.append(HypotheseMorpho(
            formes=hyp_a_formes,
            pm_tags=hyp_a_tags,
            description=f"adapter_au_det({ancre})",
        ))

    # --- Hypothese B : changer le DET (NOM est correct) ---
    if groupe.nom_idx is not None:
        nom_low = mots[groupe.nom_idx].lower()
        nom_nombre = _nombre_lexique(nom_low, lexique)
        if nom_nombre and nom_nombre != "_" and nom_nombre != ancre:
            new_det = _det_pour_nombre(mots[groupe.det_idx].lower(), nom_nombre, genre, lexique)
            if new_det and new_det.lower() != mots[groupe.det_idx].lower():
                hyp_b_formes: dict[int, str] = {
                    groupe.det_idx: transferer_casse(mots[groupe.det_idx], new_det),
                }
                det_pm = PosNgram.make_pm_tag(
                    pos_tags[groupe.det_idx] if groupe.det_idx < len(pos_tags) else "DET",
                    nom_nombre,
                    genre,
                )
                hyp_b_tags: dict[int, str] = {groupe.det_idx: det_pm}

                # Aussi adapter les ADJ au NOM
                for aidx in groupe.adj_indices:
                    low_a = mots[aidx].lower()
                    if low_a in INVARIABLES or low_a in _ACCORD_EXCLUS:
                        continue
                    a_nombre = _nombre_lexique(low_a, lexique)
                    if a_nombre != nom_nombre:
                        nb_cible = "pluriel" if nom_nombre == "Plur" else "singulier"
                        new_a = _appliquer_nombre(low_a, nb_cible, lexique)
                        if new_a.lower() != low_a and lexique.existe(new_a.lower()):
                            hyp_b_formes[aidx] = transferer_casse(mots[aidx], new_a)
                            pos_a = pos_tags[aidx] if aidx < len(pos_tags) else "ADJ"
                            hyp_b_tags[aidx] = PosNgram.make_pm_tag(pos_a, nom_nombre, genre)

                hypotheses.append(HypotheseMorpho(
                    formes=hyp_b_formes,
                    pm_tags=hyp_b_tags,
                    description=f"changer_det({nom_nombre})",
                ))

    # --- Hypothese C : cascade NOM+VER (meme que A mais force la propagation) ---
    # Deja couverte par hyp A si ver_idx est present

    return hypotheses


def _ajouter_ver_hypothese(
    ver_idx: int,
    nombre_cible_tag: str,
    mots: list[str],
    pos_tags: list[str],
    lexique,
    hyp_formes: dict[int, str],
    hyp_tags: dict[int, str],
) -> None:
    """Ajoute la correction du VER a une hypothese si necessaire."""
    from lectura_correcteur._tags import _appliquer_nombre

    low_v = mots[ver_idx].lower()
    if low_v in INVARIABLES or low_v in _ACCORD_EXCLUS:
        return

    v_nombre = _nombre_lexique(low_v, lexique)
    if v_nombre == nombre_cible_tag:
        return  # deja correct

    if nombre_cible_tag == "Plur":
        candidats = generer_candidats_3pl(low_v)
        for c in candidats:
            if lexique.existe(c):
                hyp_formes[ver_idx] = transferer_casse(mots[ver_idx], c)
                pos_v = pos_tags[ver_idx] if ver_idx < len(pos_tags) else "VER"
                hyp_tags[ver_idx] = PosNgram.make_pm_tag(pos_v, "Plur", "_")
                return
    else:
        # Plur -> Sing : utiliser _appliquer_nombre ou generateur
        new_v = _appliquer_nombre(low_v, "singulier", lexique)
        if new_v.lower() != low_v and lexique.existe(new_v.lower()):
            hyp_formes[ver_idx] = transferer_casse(mots[ver_idx], new_v)
            pos_v = pos_tags[ver_idx] if ver_idx < len(pos_tags) else "VER"
            hyp_tags[ver_idx] = PosNgram.make_pm_tag(pos_v, "Sing", "_")


# =====================================================================
# Phase 3 : Scoring par PM n-gram
# =====================================================================

def _scorer_hypothese(
    pos_ngram: PosNgram,
    pm_tags_base: list[str],
    hypothese: HypotheseMorpho,
    groupe: AccordGroupe,
) -> float:
    """Score une hypothese via trigrammes PM sur une fenetre locale."""
    pm_tags = list(pm_tags_base)
    for idx, pm_tag in hypothese.pm_tags.items():
        if idx < len(pm_tags):
            pm_tags[idx] = pm_tag

    # Fenetre : [debut-1, fin+1] avec padding BOS/EOS
    padded = [PosNgram.BOS, PosNgram.BOS] + pm_tags + [PosNgram.EOS]
    # Indices dans padded : debut_real = groupe.debut + 2, fin_real = groupe.fin + 2
    start = max(2, groupe.debut + 2)
    end = min(len(padded), groupe.fin + 3)

    score = 0.0
    for k in range(start, end):
        score += pos_ngram.logp_pm_trigram(padded[k - 2], padded[k - 1], padded[k])
    return score


# =====================================================================
# Phase 4 : Selection
# =====================================================================

def _selectionner(
    hypotheses: list[HypotheseMorpho],
    seuil_delta: float,
) -> HypotheseMorpho | None:
    """Selectionne la meilleure hypothese si le delta est suffisant."""
    if not hypotheses:
        return None

    hypotheses.sort(key=lambda h: -h.score)
    best = hypotheses[0]

    # Hypothese nulle (pas de correction)
    nul = next((h for h in hypotheses if not h.formes), None)

    if not best.formes:
        return None  # La meilleure est de ne rien faire

    if nul is not None and (best.score - nul.score) < seuil_delta:
        return None  # Pas assez confiant

    return best


# =====================================================================
# Point d'entree
# =====================================================================

def guider_accords_pm(
    mots: list[str],
    pos_tags: list[str],
    lexique,
    pos_ngram: PosNgram,
    *,
    seuil_violation: float = -10.0,
    seuil_delta: float = 2.0,
) -> list[AccordGuidance]:
    """Detecte les violations d'accord et retourne des corrections guidees par PM n-gram."""
    if not mots:
        return []

    # Extraire morpho depuis le lexique pour construire les PM tags
    morpho = _extraire_morpho_lexique(mots, pos_tags, lexique)

    # Phase 1 : detecter les groupes d'accord avec violation
    groupes = _detecter_groupes(
        mots, pos_tags, morpho, lexique, pos_ngram, seuil_violation,
    )

    if not groupes:
        return []

    # Construire les PM tags de base
    pm_tags_base = _construire_pm_tags(mots, pos_tags, morpho, lexique)

    guidances: list[AccordGuidance] = []
    handled_indices: set[int] = set()

    for groupe in groupes:
        # Eviter les chevauchements
        group_indices = set(range(groupe.debut, groupe.fin))
        if group_indices & handled_indices:
            continue

        # Phase 2 : generer des hypotheses
        hypotheses = _generer_hypotheses(
            groupe, mots, pos_tags, morpho, lexique,
        )

        # Phase 3 : scorer chaque hypothese
        for hyp in hypotheses:
            hyp.score = _scorer_hypothese(pos_ngram, pm_tags_base, hyp, groupe)

        # Phase 4 : selectionner
        best = _selectionner(hypotheses, seuil_delta)
        if best is None:
            continue

        # Convertir en AccordGuidance
        nul = next((h for h in hypotheses if not h.formes), None)
        delta = (best.score - nul.score) if nul else best.score

        for idx, forme in best.formes.items():
            if forme.lower() == mots[idx].lower():
                continue
            pm_tag = best.pm_tags.get(idx, "_|_|_")
            parts = pm_tag.split("|")
            nombre_s = parts[1] if len(parts) > 1 else "_"
            genre_s = parts[2] if len(parts) > 2 else "_"
            guidances.append(AccordGuidance(
                index=idx,
                forme_suggeree=forme,
                pm_tag=pm_tag,
                nombre_suggere=nombre_s,
                genre_suggere=genre_s,
                confiance=delta,
            ))
            handled_indices.add(idx)

        handled_indices.update(group_indices)

    return guidances


# =====================================================================
# Utilitaires
# =====================================================================

def _est_det(pos: str, low: str) -> bool:
    """Determine si un mot est un DET reconnu."""
    if pos in _POS_DET:
        return True
    return low in PLUR_DET or low in SING_DET


def _nombre_lexique(mot: str, lexique) -> str:
    """Retourne le nombre ("Sing"/"Plur") d'un mot via le lexique, ou "" si ambigu."""
    infos = lexique.info(mot)
    if not infos:
        return ""
    nombres = set()
    for entry in infos:
        nb = entry.get("nombre", "")
        if nb in ("singulier", "s"):
            nombres.add("Sing")
        elif nb in ("pluriel", "p"):
            nombres.add("Plur")
    if len(nombres) == 1:
        return nombres.pop()
    return ""  # ambigu ou inconnu


def _norm_nombre(val: str) -> str:
    """Normalise une valeur nombre vers 'Sing'/'Plur'/'_'."""
    if val in ("Sing", "Plur"):
        return val
    if val in ("singulier", "s"):
        return "Sing"
    if val in ("pluriel", "p"):
        return "Plur"
    return "_"


def _norm_genre(val: str) -> str:
    """Normalise une valeur genre vers 'Masc'/'Fem'/'_'."""
    if val in ("Masc", "Fem"):
        return val
    if val in ("masculin", "m"):
        return "Masc"
    if val in ("feminin", "féminin", "f"):
        return "Fem"
    return "_"


def _extraire_morpho_lexique(
    mots: list[str],
    pos_tags: list[str],
    lexique,
) -> dict[str, list[str]]:
    """Extrait les features morpho depuis le lexique pour chaque mot."""
    n = len(mots)
    nombres: list[str] = []
    genres: list[str] = []

    for i in range(n):
        low = mots[i].lower()
        infos = lexique.info(low)
        nb, gn = "_", "_"
        if infos:
            for entry in infos:
                e_nb = entry.get("nombre", "")
                e_gn = entry.get("genre", "")
                if e_nb:
                    nb = _norm_nombre(e_nb)
                if e_gn:
                    gn = _norm_genre(e_gn)
                break  # premiere entree
        # Fallback pour DET connus
        if nb == "_":
            if low in PLUR_DET:
                nb = "Plur"
            elif low in SING_DET:
                nb = "Sing"
        if gn == "_":
            if low in SING_FEM_DET:
                gn = "Fem"
            elif low in SING_MASC_DET:
                gn = "Masc"

        nombres.append(nb)
        genres.append(gn)

    return {"nombre": nombres, "genre": genres}


def _construire_pm_tags(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
) -> list[str]:
    """Construit la sequence de PM tags pour toute la phrase."""
    n = len(mots)
    nombres = morpho.get("nombre", ["_"] * n)
    genres = morpho.get("genre", ["_"] * n)
    pm_tags: list[str] = []
    for i in range(n):
        pos = pos_tags[i] if i < len(pos_tags) else "_"
        nb = nombres[i] if i < len(nombres) else "_"
        gn = genres[i] if i < len(genres) else "_"
        pm_tags.append(PosNgram.make_pm_tag(pos, nb, gn))
    return pm_tags


def _det_pour_nombre(
    det_low: str,
    nombre_cible: str,
    genre: str,
    lexique,
) -> str | None:
    """Retourne le DET correspondant au nombre cible."""
    # Mapping pluriel -> singulier
    _plur_to_sing: dict[str, dict[str, str]] = {
        "les": {"Masc": "le", "Fem": "la", "_": "le"},
        "des": {"Masc": "un", "Fem": "une", "_": "un"},
        "ces": {"Masc": "ce", "Fem": "cette", "_": "ce"},
        "ses": {"Masc": "son", "Fem": "sa", "_": "son"},
        "mes": {"Masc": "mon", "Fem": "ma", "_": "mon"},
        "tes": {"Masc": "ton", "Fem": "ta", "_": "ton"},
        "nos": {"_": "notre", "Masc": "notre", "Fem": "notre"},
        "vos": {"_": "votre", "Masc": "votre", "Fem": "votre"},
        "leurs": {"_": "leur", "Masc": "leur", "Fem": "leur"},
        "aux": {"Masc": "au", "Fem": "à la", "_": "au"},
    }
    # Mapping singulier -> pluriel
    _sing_to_plur: dict[str, str] = {
        "le": "les", "la": "les", "l'": "les", "l": "les",
        "un": "des", "une": "des",
        "ce": "ces", "cet": "ces", "cette": "ces",
        "son": "ses", "sa": "ses",
        "mon": "mes", "ma": "mes",
        "ton": "tes", "ta": "tes",
        "notre": "nos", "votre": "vos",
        "leur": "leurs",
        "au": "aux", "du": "des",
        "chaque": "les",
    }

    if nombre_cible == "Sing" and det_low in _plur_to_sing:
        variants = _plur_to_sing[det_low]
        return variants.get(genre, variants.get("_"))

    if nombre_cible == "Plur" and det_low in _sing_to_plur:
        return _sing_to_plur[det_low]

    return None
