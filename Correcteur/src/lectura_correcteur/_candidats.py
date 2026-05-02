"""Generation unifiee de candidats de remplacement.

Trois tiers :
- Tier 1 : identite + homophones (tous les mots, meme in-lexique)
- Tier 2 : edit distance d=1/d=2, G2P phone d<=1 (mots hors-lexique)
- Tier 3 : variantes morphologiques (mots in-lexique, contexte suspect)
"""

from __future__ import annotations

from typing import Any

from lectura_correcteur._azerty import generer_variantes_azerty
from lectura_correcteur._config import CorrecteurConfig
from lectura_correcteur._types import Candidat
from lectura_correcteur.orthographe._suggestions import (
    ALPHABET_FR,
    _edits_distance_1,
    _MAX_D1_EXPAND,
)


def _edit_distance(a: str, b: str) -> int:
    """Distance d'edition Levenshtein simple entre deux chaines."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[lb]


def generer_candidats(
    mot: str,
    dans_lexique: bool,
    pos_crf: str,
    morpho_crf: dict[str, str],
    lexique: Any,
    g2p: Any | None = None,
    config: CorrecteurConfig | None = None,
    suggestions: list[str] | None = None,
) -> list[Candidat]:
    """Genere les candidats de remplacement pour un mot.

    Args:
        mot: Mot a analyser (forme corrigee par l'orthographe).
        dans_lexique: True si le mot existe dans le lexique.
        pos_crf: POS assignee par le CRF.
        morpho_crf: Morpho du CRF (genre, nombre, temps, ...).
        lexique: Objet lexique.
        g2p: Objet G2P optionnel.
        config: Configuration du correcteur.
        suggestions: Suggestions du verificateur (OOV). Si fourni et OOV,
            utilise comme Tier 2 au lieu de recalculer d=1/d=2/G2P.

    Returns:
        Liste de Candidat (non scores, score=0.0).
    """
    low = mot.lower()
    seen: set[str] = set()
    candidats: list[Candidat] = []
    cfg = config or CorrecteurConfig()

    # --- Tier 1 : identite + homophones ---
    candidats.extend(_tier1_identite_homophones(low, lexique, seen))

    # --- Tier 2 : edit distance ---
    # Actif pour les mots hors-lexique, et aussi pour les mots in-lexique
    # dont la frequence est en dessous du seuil de suspicion.
    suspicion_freq = (
        dans_lexique
        and cfg.seuil_freq_suspicion > 0.0
        and hasattr(lexique, "frequence")
        and lexique.frequence(low) < cfg.seuil_freq_suspicion
    )
    # Guards : ne pas activer la suspicion pour les noms propres, sigles,
    # formes flechies et NOM/ADJ singuliers de base.
    if suspicion_freq and hasattr(lexique, "info"):
        infos = lexique.info(low)
        if infos:
            # Guard NOM PROPRE / SIGLE
            if all(
                "PROPRE" in (e.get("cgram") or "")
                or (e.get("cgram") or "") == "SIGLE"
                for e in infos
            ):
                suspicion_freq = False
            # Guard forme flechie (pluriel, feminin, verbe conjugue)
            if suspicion_freq and any(
                e.get("nombre") in ("p", "pluriel")
                or e.get("genre") in ("f",)
                or (
                    e.get("cgram", "") in ("VER", "AUX")
                    and e.get("mode") not in ("", None)
                )
                for e in infos
                if e.get("cgram", "") in ("NOM", "ADJ", "VER", "AUX")
            ):
                suspicion_freq = False
            # Guard NOM/ADJ singulier de base
            if suspicion_freq and any(
                e.get("cgram", "") in ("NOM", "ADJ")
                and e.get("nombre") in ("s", "singulier")
                for e in infos
            ):
                suspicion_freq = False
    if not dans_lexique or suspicion_freq:
        if suggestions is not None:
            candidats.extend(
                _tier2_from_suggestions(low, suggestions, lexique, seen),
            )
        else:
            candidats.extend(_tier2_edit_distance(low, lexique, g2p, seen))
        # Toujours tenter le canal G2P en complement (les suggestions
        # SymSpell ne couvrent pas forcement les candidats phonetiques)
        if g2p is not None and suggestions is not None:
            candidats.extend(_tier2_g2p(low, lexique, g2p, seen))
        # Candidats par substitution AZERTY (typos clavier)
        if cfg.activer_azerty:
            candidats.extend(_tier2_azerty(low, lexique, seen))

    # --- Tier 3 : variantes morphologiques (in-lexique) ---
    if dans_lexique:
        candidats.extend(_tier3_morpho(low, pos_crf, lexique, seen))

    return candidats


def _tier1_identite_homophones(
    mot: str,
    lexique: Any,
    seen: set[str],
) -> list[Candidat]:
    """Tier 1 : le mot lui-meme + ses homophones."""
    candidats: list[Candidat] = []

    # Identite
    freq_id = lexique.frequence(mot) if hasattr(lexique, "frequence") else 0.0
    infos_id = lexique.info(mot) if hasattr(lexique, "info") else []
    pos_id = ""
    phone_id = ""
    lemme_id = ""
    genre_id = ""
    nombre_id = ""
    if infos_id:
        best = max(infos_id, key=lambda e: float(e.get("freq", 0)))
        pos_id = best.get("cgram", "")
        phone_id = best.get("phone", "")
        lemme_id = best.get("lemme", "")
        genre_id = best.get("genre", "")
        nombre_id = best.get("nombre", "")

    candidats.append(Candidat(
        forme=mot,
        source="identite",
        freq=freq_id,
        edit_dist=0,
        pos=pos_id,
        phone=phone_id,
        lemme=lemme_id,
        genre=genre_id,
        nombre=nombre_id,
    ))
    seen.add(mot)

    # Homophones
    if hasattr(lexique, "phone_de") and hasattr(lexique, "homophones"):
        phone = lexique.phone_de(mot)
        if phone:
            for entry in lexique.homophones(phone):
                ortho = entry.get("ortho", "")
                if not ortho:
                    continue
                ortho_low = ortho.lower()
                if ortho_low in seen:
                    continue
                seen.add(ortho_low)
                candidats.append(Candidat(
                    forme=ortho_low,
                    source="homophone",
                    freq=float(entry.get("freq") or 0),
                    edit_dist=_edit_distance(mot, ortho_low),
                    pos=entry.get("cgram", ""),
                    phone=entry.get("phone", ""),
                    lemme=entry.get("lemme", ""),
                    genre=entry.get("genre", ""),
                    nombre=entry.get("nombre", ""),
                ))

    return candidats


def _tier2_from_suggestions(
    mot: str,
    suggestions: list[str],
    lexique: Any,
    seen: set[str],
) -> list[Candidat]:
    """Tier 2 : candidats depuis les suggestions du verificateur.

    Enrichit chaque suggestion avec POS/morpho/phone depuis lexique.info().
    """
    candidats: list[Candidat] = []
    for s in suggestions:
        s_low = s.lower()
        if s_low in seen:
            continue
        seen.add(s_low)
        freq = lexique.frequence(s_low) if hasattr(lexique, "frequence") else 0.0
        infos = lexique.info(s_low) if hasattr(lexique, "info") else []
        pos = ""
        phone = ""
        lemme = ""
        genre = ""
        nombre = ""
        if infos:
            best = max(infos, key=lambda e: float(e.get("freq", 0)))
            pos = best.get("cgram", "")
            phone = best.get("phone", "")
            lemme = best.get("lemme", "")
            genre = best.get("genre", "")
            nombre = best.get("nombre", "")
        candidats.append(Candidat(
            forme=s_low,
            source="ortho_suggestion",
            freq=freq,
            edit_dist=_edit_distance(mot, s_low),
            pos=pos,
            phone=phone,
            lemme=lemme,
            genre=genre,
            nombre=nombre,
        ))
    return candidats


def _tier2_edit_distance(
    mot: str,
    lexique: Any,
    g2p: Any | None,
    seen: set[str],
) -> list[Candidat]:
    """Tier 2 : candidats par edit distance (d=1, d=2, G2P)."""
    candidats: list[Candidat] = []

    # d=1
    d1_set = _edits_distance_1(mot)
    d1_valides: list[Candidat] = []
    for c in d1_set:
        if c in seen or not lexique.existe(c):
            continue
        seen.add(c)
        freq = lexique.frequence(c) if hasattr(lexique, "frequence") else 0.0
        infos = lexique.info(c) if hasattr(lexique, "info") else []
        pos = ""
        phone = ""
        lemme = ""
        genre = ""
        nombre = ""
        if infos:
            best = max(infos, key=lambda e: float(e.get("freq", 0)))
            pos = best.get("cgram", "")
            phone = best.get("phone", "")
            lemme = best.get("lemme", "")
            genre = best.get("genre", "")
            nombre = best.get("nombre", "")
        d1_valides.append(Candidat(
            forme=c,
            source="ortho_d1",
            freq=freq,
            edit_dist=1,
            pos=pos,
            phone=phone,
            lemme=lemme,
            genre=genre,
            nombre=nombre,
        ))
    d1_valides.sort(key=lambda x: -x.freq)
    candidats.extend(d1_valides)

    # G2P phone d<=1
    if g2p is not None and hasattr(lexique, "homophones"):
        phone_mot = g2p.prononcer(mot) if hasattr(g2p, "prononcer") else None
        if phone_mot:
            from lectura_correcteur._phones import generer_phones_d1

            phones_a_tester = [phone_mot] + generer_phones_d1(phone_mot)
            phones_vus: set[str] = set()
            for p in phones_a_tester:
                if p in phones_vus:
                    continue
                phones_vus.add(p)
                for entry in lexique.homophones(p):
                    ortho = entry.get("ortho", "")
                    if not ortho:
                        continue
                    ortho_low = ortho.lower()
                    if ortho_low in seen:
                        continue
                    seen.add(ortho_low)
                    candidats.append(Candidat(
                        forme=ortho_low,
                        source="g2p",
                        freq=float(entry.get("freq") or 0),
                        edit_dist=_edit_distance(mot, ortho_low),
                        pos=entry.get("cgram", ""),
                        phone=entry.get("phone", ""),
                        lemme=entry.get("lemme", ""),
                        genre=entry.get("genre", ""),
                        nombre=entry.get("nombre", ""),
                    ))

    # d=2 (dernier recours, si aucun candidat d=1/g2p)
    if not candidats:
        d2_valides: list[Candidat] = []
        count = 0
        for c in d1_set:
            if lexique.existe(c):
                continue
            count += 1
            if count > _MAX_D1_EXPAND:
                break
            for c2 in _edits_distance_1(c):
                if c2 in seen or c2 == mot or not lexique.existe(c2):
                    continue
                seen.add(c2)
                freq = lexique.frequence(c2) if hasattr(lexique, "frequence") else 0.0
                infos = lexique.info(c2) if hasattr(lexique, "info") else []
                pos = ""
                phone = ""
                lemme = ""
                genre = ""
                nombre = ""
                if infos:
                    best = max(infos, key=lambda e: float(e.get("freq", 0)))
                    pos = best.get("cgram", "")
                    phone = best.get("phone", "")
                    lemme = best.get("lemme", "")
                    genre = best.get("genre", "")
                    nombre = best.get("nombre", "")
                d2_valides.append(Candidat(
                    forme=c2,
                    source="ortho_d2",
                    freq=freq,
                    edit_dist=2,
                    pos=pos,
                    phone=phone,
                    lemme=lemme,
                    genre=genre,
                    nombre=nombre,
                ))
        d2_valides.sort(key=lambda x: -x.freq)
        candidats.extend(d2_valides)

    return candidats


def _tier2_g2p(
    mot: str,
    lexique: Any,
    g2p: Any,
    seen: set[str],
) -> list[Candidat]:
    """Tier 2 G2P : candidats phonetiques (phone exacte + d<=1)."""
    candidats: list[Candidat] = []
    if not hasattr(lexique, "homophones"):
        return candidats
    phone_mot = g2p.prononcer(mot) if hasattr(g2p, "prononcer") else None
    if not phone_mot:
        return candidats

    from lectura_correcteur._phones import generer_phones_d1

    phones_a_tester = [phone_mot] + generer_phones_d1(phone_mot)
    phones_vus: set[str] = set()
    for p in phones_a_tester:
        if p in phones_vus:
            continue
        phones_vus.add(p)
        for entry in lexique.homophones(p):
            ortho = entry.get("ortho", "")
            if not ortho:
                continue
            ortho_low = ortho.lower()
            if ortho_low in seen:
                continue
            seen.add(ortho_low)
            candidats.append(Candidat(
                forme=ortho_low,
                source="g2p",
                freq=float(entry.get("freq") or 0),
                edit_dist=_edit_distance(mot, ortho_low),
                pos=entry.get("cgram", ""),
                phone=entry.get("phone", ""),
                lemme=entry.get("lemme", ""),
                genre=entry.get("genre", ""),
                nombre=entry.get("nombre", ""),
            ))
    return candidats


def _tier2_azerty(
    mot: str,
    lexique: Any,
    seen: set[str],
) -> list[Candidat]:
    """Tier 2 AZERTY : candidats par substitution de touches voisines."""
    candidats: list[Candidat] = []
    for variante in generer_variantes_azerty(mot):
        if variante in seen:
            continue
        if not lexique.existe(variante):
            continue
        seen.add(variante)
        freq = lexique.frequence(variante) if hasattr(lexique, "frequence") else 0.0
        infos = lexique.info(variante) if hasattr(lexique, "info") else []
        pos = ""
        phone = ""
        lemme = ""
        genre = ""
        nombre = ""
        if infos:
            best = max(infos, key=lambda e: float(e.get("freq", 0)))
            pos = best.get("cgram", "")
            phone = best.get("phone", "")
            lemme = best.get("lemme", "")
            genre = best.get("genre", "")
            nombre = best.get("nombre", "")
        candidats.append(Candidat(
            forme=variante,
            source="azerty",
            freq=freq,
            edit_dist=1,
            pos=pos,
            phone=phone,
            lemme=lemme,
            genre=genre,
            nombre=nombre,
        ))
    return candidats


def _tier3_morpho(
    mot: str,
    pos_crf: str,
    lexique: Any,
    seen: set[str],
) -> list[Candidat]:
    """Tier 3 : variantes morphologiques du meme lemme.

    Explore TOUS les lemmes possibles du mot (pas seulement le plus frequent),
    car le CRF peut se tromper de POS (ex: "visite" tague NOM mais a aussi
    un lemme VER "visiter").
    """
    candidats: list[Candidat] = []

    if not hasattr(lexique, "formes_de"):
        return candidats

    # Collecter tous les lemmes possibles depuis toutes les entrees lexicales
    lemmes: set[str] = set()
    if hasattr(lexique, "info"):
        for entry in lexique.info(mot):
            l = entry.get("lemme", "")
            if l:
                lemmes.add(l.lower())
    # Fallback : lemme_de retourne le lemme le plus frequent
    if not lemmes and hasattr(lexique, "lemme_de"):
        l = lexique.lemme_de(mot)
        if l:
            lemmes.add(l.lower())

    if not lemmes:
        return candidats

    for lemme in lemmes:
        formes = lexique.formes_de(lemme)
        for entry in formes:
            ortho = entry.get("ortho", "")
            if not ortho:
                continue
            ortho_low = ortho.lower()
            if ortho_low in seen:
                continue
            seen.add(ortho_low)
            candidats.append(Candidat(
                forme=ortho_low,
                source="morpho",
                freq=float(entry.get("freq") or 0),
                edit_dist=_edit_distance(mot, ortho_low),
                pos=entry.get("cgram", ""),
                phone=entry.get("phone", ""),
                lemme=entry.get("lemme", lemme),
                genre=entry.get("genre", ""),
                nombre=entry.get("nombre", ""),
            ))

    return candidats
