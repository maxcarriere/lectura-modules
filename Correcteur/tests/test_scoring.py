"""Tests pour le scoring multi-facteurs."""

from lectura_correcteur._scoring import (
    extraire_contexte,
    scorer_candidats,
    _s_identite,
    _s_freq,
    _s_edit,
    _s_pos,
    _s_morpho,
    _s_phone,
    _s_contexte,
    _a_entree_verbe,
)
from lectura_correcteur._types import Candidat, MotAnalyse


class _NullLex:
    def info(self, mot):
        return []


class _VisiteLex:
    """Lexique ou 'visite' a des entrees NOM et VER, 'avons' est AUX avoir."""
    def info(self, mot):
        if mot == "visite":
            return [
                {"cgram": "NOM", "freq": 50.0},
                {"cgram": "VER", "freq": 15.0, "lemme": "visiter"},
            ]
        if mot == "visité":
            return [{"cgram": "VER", "freq": 10.0, "mode": "participe"}]
        if mot == "avons":
            return [{"cgram": "AUX", "lemme": "avoir"}]
        if mot == "est":
            return [{"cgram": "AUX", "lemme": "être"}]
        return []


_NULL_CTX = {"aux_gauche": False, "genre_det": "", "nombre_det": ""}


# --- Tests des facteurs individuels ---

def test_s_identite():
    c_id = Candidat(forme="chat", source="identite", pos="NOM")
    c_other = Candidat(forme="chats", source="homophone")
    assert _s_identite(c_id, _NULL_CTX, _NullLex()) == 1.0
    assert _s_identite(c_other, _NULL_CTX, _NullLex()) == 0.0


def test_s_identite_reduit_apres_avoir():
    """Apres avoir, un verbe non-PP perd son bonus identite."""
    c_ver = Candidat(forme="mange", source="identite", pos="VER")
    ctx_avoir = {"aux_gauche": True, "aux_lemme": "avoir"}
    assert _s_identite(c_ver, ctx_avoir, _NullLex()) == 0.0


def test_s_identite_reduit_apres_avoir_via_lexique():
    """Apres avoir, un NOM ayant VER dans le lexique perd aussi le bonus."""
    c_visite = Candidat(forme="visite", source="identite", pos="NOM")
    ctx_avoir = {"aux_gauche": True, "aux_lemme": "avoir"}
    assert _s_identite(c_visite, ctx_avoir, _VisiteLex()) == 0.0


def test_s_identite_maintenu_apres_etre_adj():
    """Apres être, un ADJ/NOM avec entree VER garde son bonus (attribut)."""
    c_visite = Candidat(forme="visite", source="identite", pos="NOM")
    ctx_etre = {"aux_gauche": True, "aux_lemme": "être"}
    # NOM n'est pas VER/AUX par CRF -> bonus maintenu apres être
    assert _s_identite(c_visite, ctx_etre, _VisiteLex()) == 1.0


def test_s_identite_reduit_apres_etre_ver():
    """Apres être, un mot CRF-VER non-PP perd son bonus."""
    c_ver = Candidat(forme="mange", source="identite", pos="VER")
    ctx_etre = {"aux_gauche": True, "aux_lemme": "être"}
    assert _s_identite(c_ver, ctx_etre, _NullLex()) == 0.0


def test_s_identite_maintenu_pour_pp():
    """Apres AUX, un PP garde son bonus identite."""
    class PPLex:
        def info(self, mot):
            return [{"mode": "participe", "cgram": "VER"}]

    c_pp = Candidat(forme="mangé", source="identite", pos="VER")
    ctx_avoir = {"aux_gauche": True, "aux_lemme": "avoir"}
    assert _s_identite(c_pp, ctx_avoir, PPLex()) == 1.0


def test_s_freq():
    c_high = Candidat(forme="le", source="identite", freq=890.0)
    c_low = Candidat(forme="xyz", source="identite", freq=0.0)
    assert _s_freq(c_high) > _s_freq(c_low)
    assert _s_freq(c_low) == 0.0  # log10(0+1)/6 = 0


def test_s_edit():
    c0 = Candidat(forme="chat", source="identite", edit_dist=0)
    c1 = Candidat(forme="chats", source="ortho_d1", edit_dist=1)
    c2 = Candidat(forme="chant", source="ortho_d2", edit_dist=2)
    assert _s_edit(c0) == 1.0
    assert abs(_s_edit(c1) - 0.6667) < 0.01
    assert abs(_s_edit(c2) - 0.3333) < 0.01


def test_s_pos_exact():
    c = Candidat(forme="chat", source="identite", pos="NOM")
    assert _s_pos(c, "NOM") == 1.0


def test_s_pos_compatible():
    c = Candidat(forme="est", source="homophone", pos="AUX")
    assert _s_pos(c, "VER") == 0.5


def test_s_pos_incompatible():
    c = Candidat(forme="chat", source="identite", pos="NOM")
    assert _s_pos(c, "VER") == 0.0


def test_s_morpho_concordant():
    c = Candidat(forme="chat", source="identite", genre="m", nombre="s")
    ctx = {"genre_det": "m", "nombre_det": "s"}
    assert _s_morpho(c, ctx) == 1.0


def test_s_morpho_discordant():
    c = Candidat(forme="chats", source="identite", genre="m", nombre="p")
    ctx = {"genre_det": "f", "nombre_det": "s"}
    assert _s_morpho(c, ctx) == 0.0


def test_s_morpho_partiel():
    c = Candidat(forme="chat", source="identite", genre="m", nombre="s")
    ctx = {"genre_det": "m", "nombre_det": "p"}
    assert _s_morpho(c, ctx) == 0.5  # genre ok, nombre ko


def test_s_morpho_pas_de_contexte():
    c = Candidat(forme="chat", source="identite", genre="m", nombre="s")
    ctx = {"genre_det": "", "nombre_det": ""}
    assert _s_morpho(c, ctx) == 0.5  # neutre


def test_s_phone_identique():
    c = Candidat(forme="chat", source="identite", phone="ʃa")
    assert _s_phone(c, "ʃa") == 1.0


def test_s_phone_different():
    c = Candidat(forme="table", source="identite", phone="tabl")
    assert _s_phone(c, "ʃa") == 0.0


def test_s_contexte_pp_apres_aux():
    """PP apres AUX -> score contexte = 1.0."""
    class PPLex:
        def info(self, mot):
            return [{"mode": "participe"}]

    c = Candidat(forme="mangé", source="morpho", pos="VER")
    ctx = {"aux_gauche": True, "aux_lemme": "avoir"}
    assert _s_contexte(c, ctx, PPLex()) == 1.0


def test_s_contexte_present_apres_avoir():
    """Verbe present apres avoir -> score contexte = 0.0."""
    c = Candidat(forme="mange", source="identite", pos="VER")
    ctx = {"aux_gauche": True, "aux_lemme": "avoir"}
    assert _s_contexte(c, ctx, _NullLex()) == 0.0


def test_s_contexte_nom_avec_ver_apres_avoir():
    """NOM avec VER dans le lexique, apres avoir -> score contexte = 0.0."""
    c = Candidat(forme="visite", source="identite", pos="NOM")
    ctx = {"aux_gauche": True, "aux_lemme": "avoir"}
    assert _s_contexte(c, ctx, _VisiteLex()) == 0.0


def test_s_contexte_nom_avec_ver_apres_etre():
    """NOM avec VER dans le lexique, apres être -> score contexte = 0.5 (attribut)."""
    c = Candidat(forme="visite", source="identite", pos="NOM")
    ctx = {"aux_gauche": True, "aux_lemme": "être"}
    # NOM n'est pas VER/AUX par CRF -> neutre apres être
    assert _s_contexte(c, ctx, _VisiteLex()) == 0.5


def test_s_contexte_neutre():
    """Pas de contexte special -> 0.5."""
    c = Candidat(forme="chat", source="identite", pos="NOM")
    ctx = {"aux_gauche": False}
    assert _s_contexte(c, ctx, _NullLex()) == 0.5


# --- Tests _a_entree_verbe ---

def test_a_entree_verbe():
    assert _a_entree_verbe("visite", _VisiteLex()) is True
    assert _a_entree_verbe("chat", _NullLex()) is False


# --- Tests extraire_contexte ---

def test_extraire_contexte_basique():
    analyses = [
        MotAnalyse(original="le", corrige="le", pos="ART:def",
                   morpho={"genre": "m", "nombre": "s"}),
        MotAnalyse(original="chat", corrige="chat", pos="NOM"),
        MotAnalyse(original="mange", corrige="mange", pos="VER"),
    ]
    ctx = extraire_contexte(analyses, 1)
    assert ctx["pos_gauche"] == "ART:def"
    assert ctx["pos_droite"] == "VER"
    assert ctx["genre_det"] == "m"
    assert ctx["nombre_det"] == "s"


def test_extraire_contexte_aux_gauche():
    """AUX 'a' + VER -> aux_gauche detecte (sans lexique, pas de aux_lemme)."""
    analyses = [
        MotAnalyse(original="il", corrige="il", pos="PRO:per"),
        MotAnalyse(original="a", corrige="a", pos="AUX"),
        MotAnalyse(original="mange", corrige="mange", pos="VER"),
    ]
    ctx = extraire_contexte(analyses, 2)
    assert ctx["aux_gauche"] is True


def test_extraire_contexte_aux_gauche_via_lexique_avoir():
    """Apres avoir, NOM avec VER dans le lexique -> aux_gauche detecte."""
    analyses = [
        MotAnalyse(original="avons", corrige="avons", pos="AUX"),
        MotAnalyse(original="visite", corrige="visite", pos="NOM"),
    ]
    ctx = extraire_contexte(analyses, 1, lexique=_VisiteLex())
    assert ctx["aux_gauche"] is True
    assert ctx["aux_lemme"] == "avoir"


def test_extraire_contexte_aux_gauche_etre_nom():
    """Apres être, NOM avec VER dans le lexique -> aux_gauche PAS detecte (ADJ attribut)."""
    analyses = [
        MotAnalyse(original="est", corrige="est", pos="AUX"),
        MotAnalyse(original="visite", corrige="visite", pos="NOM"),
    ]
    ctx = extraire_contexte(analyses, 1, lexique=_VisiteLex())
    # "visite" est NOM par CRF, et après "être" on ne fait pas le check lexique VER
    assert ctx["aux_gauche"] is False


def test_extraire_contexte_aux_gauche_pur_nom():
    """Mot tague NOM sans entree VER -> aux_gauche pas detecte."""
    analyses = [
        MotAnalyse(original="est", corrige="est", pos="AUX"),
        MotAnalyse(original="chat", corrige="chat", pos="NOM"),
    ]
    ctx = extraire_contexte(analyses, 1, lexique=_NullLex())
    assert ctx["aux_gauche"] is False


def test_extraire_contexte_premier_mot():
    analyses = [
        MotAnalyse(original="chat", corrige="chat", pos="NOM"),
        MotAnalyse(original="mange", corrige="mange", pos="VER"),
    ]
    ctx = extraire_contexte(analyses, 0)
    assert ctx["pos_gauche"] == ""
    assert ctx["pos_droite"] == "VER"


# --- Tests scorer_candidats ---

def test_scorer_identite_favorise(mock_lexique):
    """L'identite a un bonus qui la maintient en tete quand rien ne justifie le remplacement."""
    candidats = [
        Candidat(forme="chat", source="identite", freq=45.0, edit_dist=0,
                 pos="NOM", phone="ʃa", genre="m", nombre="s"),
        Candidat(forme="chats", source="homophone", freq=12.0, edit_dist=1,
                 pos="NOM", phone="ʃa", genre="m", nombre="p"),
    ]
    scored = scorer_candidats(
        candidats, "chat", "NOM", {},
        {"genre_det": "", "nombre_det": "", "aux_gauche": False},
        mock_lexique,
    )
    assert scored[0].forme == "chat"
    assert scored[0].score > scored[1].score


def test_scorer_pp_apres_avoir():
    """Apres avoir, un PP doit passer devant le present."""
    class PPLexique:
        def info(self, mot):
            if mot == "mangé":
                return [{"mode": "participe", "cgram": "VER"}]
            return [{"cgram": "VER"}]  # pas participe
        def phone_de(self, mot):
            return "mɑ̃ʒe" if mot in ("mangé", "manger") else "mɑ̃ʒ"
        def frequence(self, mot):
            return 15.0 if mot == "mange" else 8.0

    candidats = [
        Candidat(forme="mange", source="identite", freq=15.0, edit_dist=0,
                 pos="VER", phone="mɑ̃ʒ", genre="", nombre="s"),
        Candidat(forme="mangé", source="morpho", freq=8.0, edit_dist=1,
                 pos="VER", phone="mɑ̃ʒe", genre="m", nombre="s"),
    ]
    scored = scorer_candidats(
        candidats, "mange", "VER", {},
        {"genre_det": "", "nombre_det": "", "aux_gauche": True,
         "aux_lemme": "avoir", "pos_gauche": "AUX", "pos_droite": ""},
        PPLexique(),
    )
    # mangé should be ranked first after avoir
    assert scored[0].forme == "mangé"


def test_scorer_pp_nom_apres_avoir():
    """Apres avoir, un NOM (visite) doit ceder au PP (visité)."""
    candidats = [
        Candidat(forme="visite", source="identite", freq=50.0, edit_dist=0,
                 pos="NOM", phone="vizit", genre="f", nombre="s"),
        Candidat(forme="visité", source="morpho", freq=10.0, edit_dist=1,
                 pos="VER", phone="vizite", genre="m", nombre="s"),
    ]
    scored = scorer_candidats(
        candidats, "visite", "NOM", {},
        {"genre_det": "", "nombre_det": "", "aux_gauche": True,
         "aux_lemme": "avoir", "pos_gauche": "AUX", "pos_droite": ""},
        _VisiteLex(),
    )
    assert scored[0].forme == "visité"


def test_scorer_candidats_vide():
    class MinimalLex:
        def phone_de(self, mot): return None
        def info(self, mot): return []

    scored = scorer_candidats([], "chat", "NOM", {}, {}, MinimalLex())
    assert scored == []


# --- Tests dans_lexique=False (OOV) ---

def test_scorer_oov_pas_de_bonus_identite():
    """Pour un mot OOV, S_identite=0 : l'identite a un score plus bas."""
    class SimpleLex:
        def phone_de(self, mot): return None
        def info(self, mot): return []

    def _make_candidats():
        return [
            Candidat(forme="chet", source="identite", freq=0.0, edit_dist=0,
                     pos="NOM", phone=""),
            Candidat(forme="chat", source="ortho_suggestion", freq=200.0,
                     edit_dist=1, pos="NOM", phone="ʃa"),
        ]

    ctx = {"genre_det": "", "nombre_det": "", "aux_gauche": False}

    # Avec dans_lexique=False, l'identite perd W_IDENTITE*1.0 = 0.20
    scored_oov = scorer_candidats(
        _make_candidats(), "chet", "NOM", {}, ctx, SimpleLex(),
        dans_lexique=False,
    )
    id_oov = next(c for c in scored_oov if c.source == "identite")

    # Avec dans_lexique=True, l'identite garde son bonus
    scored_inlex = scorer_candidats(
        _make_candidats(), "chet", "NOM", {}, ctx, SimpleLex(),
        dans_lexique=True,
    )
    id_inlex = next(c for c in scored_inlex if c.source == "identite")

    # Le score identite OOV est inferieur de W_IDENTITE (0.20)
    assert id_inlex.score - id_oov.score > 0.19
    # Et la correction a un meilleur classement relatif en mode OOV
    corr_oov = next(c for c in scored_oov if c.forme == "chat")
    assert corr_oov.score > id_oov.score


def test_scorer_in_lexique_identite_conservee():
    """Pour un mot in-lexique, S_identite=1.0 : l'identite reste favorisee."""
    class SimpleLex:
        def phone_de(self, mot): return None
        def info(self, mot): return []

    candidats = [
        Candidat(forme="chat", source="identite", freq=45.0, edit_dist=0,
                 pos="NOM", phone="ʃa"),
        Candidat(forme="chats", source="homophone", freq=12.0, edit_dist=1,
                 pos="NOM", phone="ʃa"),
    ]
    scored = scorer_candidats(
        candidats, "chat", "NOM", {},
        {"genre_det": "", "nombre_det": "", "aux_gauche": False},
        SimpleLex(),
        dans_lexique=True,
    )
    assert scored[0].forme == "chat"
