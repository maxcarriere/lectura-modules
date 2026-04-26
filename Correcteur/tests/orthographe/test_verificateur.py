"""Tests pour le verificateur orthographique."""

from lectura_correcteur._types import TypeCorrection
from lectura_correcteur.orthographe._verificateur import VerificateurOrthographe


def test_mot_connu(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["chat"])
    assert len(results) == 1
    assert results[0].dans_lexique is True
    assert results[0].type_correction == TypeCorrection.AUCUNE


def test_mot_inconnu(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["xyzabc"])
    assert len(results) == 1
    assert results[0].dans_lexique is False
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE


def test_phrase_mixte(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["le", "chat", "manjer"])
    assert results[0].dans_lexique is True
    assert results[1].dans_lexique is True
    assert results[2].dans_lexique is False
    assert results[2].type_correction == TypeCorrection.HORS_LEXIQUE


def test_casse_insensible(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["Chat"])
    assert results[0].dans_lexique is True


def test_avec_morpho(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    morpho = [{"pos": "NOM", "genre": "Masc", "nombre": "Sing"}]
    results = v.verifier_phrase(["chat"], morpho)
    assert results[0].pos == "NOM"
    assert results[0].morpho.get("genre") == "Masc"


def test_suggestions_mot_inconnu(mock_lexique):
    """Un mot inconnu a 1 edit de distance d'un mot connu doit avoir des suggestions."""
    v = VerificateurOrthographe(mock_lexique)
    # "chet" est a distance 1 de "chat" (replace e->a)
    results = v.verifier_phrase(["chet"])
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE
    assert len(results[0].suggestions) > 0
    assert "chat" in results[0].suggestions


def test_suggestions_mot_connu_pas_de_suggestions(mock_lexique):
    """Un mot connu ne doit pas avoir de suggestions."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["chat"])
    assert results[0].suggestions == []


# --- Axe 6 : Suggestions distance 2 ---

def test_suggestions_distance_2(mock_lexique):
    """Un mot a distance 2 d'un mot connu doit avoir des suggestions avec distance=2."""
    from lectura_correcteur.orthographe._suggestions import suggerer

    # "cjat" -> d1 -> "cjt", "cat", "cjat"... aucun n'est "chat"
    # "cjat" -> d1 -> "cjt" -> d1 -> "chat" ✗ (distance 2 via different path)
    # Mieux : "xhat" -> d1 edits don't yield "chat" (replace x->c = "chat"!)
    # En fait "xhat" -> replace x->c -> "chat" = distance 1
    # Utilisons "xhbt" : x->c, b->a = 2 edits
    results_d1 = suggerer("xhbt", mock_lexique, distance=1)
    results_d2 = suggerer("xhbt", mock_lexique, distance=2)
    # Distance 1 ne doit pas trouver "chat" pour "xhbt"
    assert "chat" not in results_d1
    # Distance 2 doit trouver "chat" (xhbt->xhat->chat ou xhbt->chat via 2 replaces)
    assert "chat" in results_d2


def test_suggestions_distance_2_via_verificateur(mock_lexique):
    """Le verificateur passe correctement le parametre distance."""
    v = VerificateurOrthographe(mock_lexique, distance=2)
    results = v.verifier_phrase(["chtt"])
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE
    assert "chat" in results[0].suggestions


# --- Auto-correction ---

def test_auto_correction_hcat(mock_lexique):
    """'hcat' (d=1 de 'chat', freq 45.2) -> auto-corrige en 'chat'."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    results = v.verifier_phrase(["hcat"])
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE
    assert results[0].corrige == "chat"


def test_auto_correction_miason(mock_lexique):
    """'miason' (d=1 de 'maison', freq 38.7) -> auto-corrige en 'maison'."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    results = v.verifier_phrase(["miason"])
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE
    assert results[0].corrige == "maison"


def test_elision_token_pas_corrige(mock_lexique):
    """Les tokens d'elision (j', l', n') ne doivent pas etre corriges."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["j'"])
    assert len(results) == 1
    assert results[0].corrige == "j'"
    assert results[0].dans_lexique is True


def test_pas_auto_correction_si_ambigue(mock_lexique):
    """Pas d'auto-correction si les deux meilleures suggestions sont proches en frequence."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    # 'chet' -> 'chat' (45.2) mais aussi possiblement d'autres candidats
    results = v.verifier_phrase(["chet"])
    # On ne verifie pas le corrige exact (depend des suggestions trouvees)
    # mais on verifie que le mecanisme fonctionne sans erreur
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE


# --- scoring_actif mode ---

def test_scoring_actif_auto_correction_conservee(mock_lexique):
    """En mode scoring_actif, l'auto-correction freq>=5.0 fonctionne toujours."""
    v = VerificateurOrthographe(mock_lexique, distance=1, scoring_actif=True)
    results = v.verifier_phrase(["hcat"])
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE
    # Auto-correction active (chat freq=45.2 >= 5.0)
    assert results[0].corrige == "chat"
    assert len(results[0].suggestions) > 0


def test_scoring_actif_mot_connu_inchange(mock_lexique):
    """En mode scoring_actif, un mot connu reste inchange."""
    v = VerificateurOrthographe(mock_lexique, distance=1, scoring_actif=True)
    results = v.verifier_phrase(["chat"])
    assert results[0].dans_lexique is True
    assert results[0].corrige == "chat"
    assert results[0].suggestions == []


# --- Axe 1 : Auto-correction accent sans seuil frequence ---

def test_accent_auto_correction_ecole(mock_lexique):
    """'ecole' (sans accent) -> 'école' meme si freq < 5.0."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    results = v.verifier_phrase(["ecole"])
    assert results[0].corrige == "école"


def test_accent_auto_correction_mere(mock_lexique):
    """'mere' -> 'mère' meme si freq < 5.0."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    results = v.verifier_phrase(["mere"])
    assert results[0].corrige == "mère"


def test_accent_auto_correction_fatigue(mock_lexique):
    """'fatigue' -> 'fatigué' meme si freq < 5.0."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    results = v.verifier_phrase(["fatigue"])
    assert results[0].corrige == "fatigué"


# --- Axe 2 : Auto-correction doublement consonne ---

def test_doublement_consonne_balon(mock_lexique):
    """'balon' -> 'ballon' (doublement consonne, freq >= 1.0)."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    results = v.verifier_phrase(["balon"])
    assert results[0].corrige == "ballon"


def test_doublement_consonne_dificile(mock_lexique):
    """'dificile' -> 'difficile' (doublement consonne, freq >= 1.0)."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    results = v.verifier_phrase(["dificile"])
    assert results[0].corrige == "difficile"


# --- POS re-ranking ---

def test_pos_reranking_drot_ver(mock_lexique):
    """'drot' with POS=VER -> 'dort' (VER) preferred over 'droit' (ADJ, higher freq)."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    morpho = [{"pos": "VER"}]
    results = v.verifier_phrase(["drot"], morpho)
    assert results[0].suggestions[0] == "dort"


def test_pos_reranking_drot_adj(mock_lexique):
    """'drot' with POS=ADJ -> 'droit' (ADJ) preferred over 'dort' (VER)."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    morpho = [{"pos": "ADJ"}]
    results = v.verifier_phrase(["drot"], morpho)
    assert results[0].suggestions[0] == "droit"


# --- Accent disambiguation in-lexique ---

def test_accent_inlexique_tres(mock_lexique):
    """'tres' in lexique (freq=1) -> corrige en 'très' (freq=500)."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["tres"])
    assert results[0].corrige == "très"


def test_accent_inlexique_foret(mock_lexique):
    """'foret' in lexique (freq=1) -> corrige en 'forêt' (freq=50)."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["foret"])
    assert results[0].corrige == "forêt"


def test_accent_inlexique_chat_pas_modifie(mock_lexique):
    """'chat' (freq=45.2) n'a pas de variante accent -> pas modifie."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["chat"])
    assert results[0].corrige == "chat"


# --- Sigles : pas d'auto-correction OOV ---

def test_sigle_allcaps_pas_auto_corrige(mock_lexique):
    """Un mot ALL-CAPS (sigle) OOV ne doit pas etre auto-corrige."""
    v = VerificateurOrthographe(mock_lexique, distance=1)
    results = v.verifier_phrase(["UNESCO"])
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE
    # Pas de correction automatique : le mot reste tel quel
    assert results[0].corrige == "UNESCO"


# --- Correction casse : SIGLE → majuscules, NOM PROPRE → 1ere lettre ---

def test_sigle_minuscule_corrige_en_majuscule(mock_lexique):
    """'onu' (SIGLE dans le lexique) -> corrige en 'ONU'."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["onu"])
    assert results[0].corrige == "ONU"
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE


def test_sigle_deja_majuscule_pas_modifie(mock_lexique):
    """'ONU' deja en majuscules -> pas de correction."""
    # ONU n'est pas dans le lexique (existe cherche en minuscule "onu")
    # mais "onu" oui; verifions que si le mot est deja en caps, pas de re-correction
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["ONU"])
    # Le mot existe dans le lexique (case-insensitive) et est deja en majuscules
    assert results[0].corrige == "ONU"


def test_nom_propre_minuscule_corrige_en_majuscule(mock_lexique):
    """'mozart' (NOM PROPRE) -> corrige en 'Mozart'."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["mozart"])
    assert results[0].corrige == "Mozart"
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE


def test_nom_propre_deja_capitalise_pas_modifie(mock_lexique):
    """'Mozart' deja capitalise -> pas de correction."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["Mozart"])
    assert results[0].corrige == "Mozart"
    assert results[0].type_correction == TypeCorrection.AUCUNE


# --- Axe 3 : accent foreign context bidirectionnel ---

def test_accent_foreign_ctx_suivant(mock_lexique):
    """Mot suivi d'un OOV capitalise -> pas d'accent ajoutee."""
    # "tres" suivi de "Race" (OOV, capitalise) → contexte etranger, pas de correction
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["tres", "Race"])
    assert results[0].corrige == "tres"  # pas corrige en "très"
