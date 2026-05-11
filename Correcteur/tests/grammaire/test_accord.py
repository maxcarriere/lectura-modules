"""Tests pour les regles d'accord."""

from lectura_correcteur._types import TypeCorrection
from lectura_correcteur.grammaire._accord import verifier_accords


def test_pluriel_nom_apres_det(mock_lexique):
    """'les enfant' -> 'les enfants'."""
    mots = ["les", "enfant"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["les", "enfants"]
    assert len(corrections) == 1
    assert corrections[0].type_correction == TypeCorrection.GRAMMAIRE


def test_pluriel_deja_present(mock_lexique):
    """'les enfants' -> pas de correction."""
    mots = ["les", "enfants"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["les", "enfants"]
    assert len(corrections) == 0


def test_restaurer_ils(mock_lexique):
    """'il' (quand original etait 'ils') -> 'ils'."""
    mots = ["il", "mange"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique, originaux=["ils", "mange"],
    )
    assert result[0].lower() == "ils"


def test_regle4_det_nom_ver(mock_lexique):
    """'les enfants mange' -> 'les enfants mangent'."""
    mots = ["les", "enfants", "mange"]
    pos = ["ART:def", "NOM", "VER"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["les", "enfants", "mangent"]


def test_invariable_pas_modifie(mock_lexique):
    """Les mots invariables ne prennent pas de -s."""
    mots = ["les", "chose"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["les", "chose"]


def test_adj_post_nominal_pluriel(mock_lexique):
    """'Des maisons blanche' -> 'Des maisons blanches'."""
    mots = ["Des", "maisons", "blanche"]
    pos = ["ART:ind", "NOM", "ADJ"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["Des", "maisons", "blanches"]
    assert len(corrections) >= 1
    assert any(c.corrige == "blanches" for c in corrections)


def test_copule_pluriel_adj(mock_lexique):
    """'sont content' -> 'sont contents'."""
    mots = ["sont", "content"]
    pos = ["AUX", "ADJ"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["sont", "contents"]
    assert len(corrections) >= 1
    assert any(c.corrige == "contents" for c in corrections)


# --- Axe 4 : Accord en genre ---

def test_genre_fem_det_adj(mock_lexique):
    """'la petit fille' -> 'la petite fille'."""
    mots = ["la", "petit", "fille"]
    pos = ["ART:def", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "petite"
    assert len(corrections) >= 1
    assert any(c.corrige == "petite" for c in corrections)


def test_genre_fem_det_nom(mock_lexique):
    """'la grand maison' -> 'la grande maison'."""
    mots = ["la", "grand", "maison"]
    pos = ["ART:def", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "grande"
    assert any(c.corrige == "grande" for c in corrections)


def test_genre_pas_de_correction_si_feminin(mock_lexique):
    """'la petite fille' -> pas de correction (deja feminin)."""
    mots = ["la", "petite", "fille"]
    pos = ["ART:def", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["la", "petite", "fille"]
    # Pas de correction de genre (petite est deja feminin)
    assert not any(c.explication.startswith("Accord en genre") for c in corrections)


# --- Axe 5 : Accord sujet-verbe a distance ---

def test_sujet_verbe_distance(mock_lexique):
    """'les enfants de la voisine mange' -> '... mangent'."""
    mots = ["les", "enfants", "de", "la", "voisine", "mange"]
    pos = ["ART:def", "NOM", "PRE", "ART:def", "NOM", "VER"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[5] == "mangent"
    assert any(c.corrige == "mangent" for c in corrections)


def test_sujet_sing_pas_correction(mock_lexique):
    """'la voisine mange' -> pas de correction (singulier correct)."""
    mots = ["la", "voisine", "mange"]
    pos = ["ART:def", "NOM", "VER"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[2] == "mange"
    assert not any(c.corrige == "mangent" for c in corrections)


# --- Axe genre : NOM fem + ADJ masc ---

def test_voitures_vert_vertes(mock_lexique):
    """'les voitures vert' -> 'les voitures vertes' (genre + pluriel)."""
    mots = ["les", "voitures", "vert"]
    pos = ["ART:def", "NOM", "ADJ"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[2] == "vertes"


def test_la_petit_fille(mock_lexique):
    """'la petit fille' -> 'la petite fille' (Regle 6 det fem)."""
    mots = ["la", "petit", "fille"]
    pos = ["ART:def", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "petite"


# --- Accord sujet-verbe a distance : du/au ---

def test_sujet_distant_du(mock_lexique):
    """'les chats du jardin mange' -> '... mangent'."""
    mots = ["les", "chats", "du", "jardin", "mange"]
    pos = ["ART:def", "NOM", "ART:def", "NOM", "VER"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[4] == "mangent"


def test_sujet_distant_au(mock_lexique):
    """'les enfants au jardin mange' -> '... mangent'."""
    mots = ["les", "enfants", "au", "jardin", "mange"]
    pos = ["ART:def", "NOM", "ART:def", "NOM", "VER"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[4] == "mangent"


# --- Regle 8 : Coherence DET↔NOM en genre ---

def test_le_petit_fille(mock_lexique):
    """'le petit fille' -> 'la petite fille' (NOM unambigu fem, DET+ADJ corriges)."""
    mots = ["le", "petit", "fille"]
    pos = ["ART:def", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    # NOM "fille" unambigu fem → flip DET + feminiser ADJ
    assert result[0] == "la"
    assert result[1] == "petite"
    assert result[2] == "fille"
    assert len(corrections) >= 2


def test_un_belle_maison(mock_lexique):
    """'un belle maison' -> 'une belle maison' (Rule 8 now handles un/une)."""
    mots = ["un", "belle", "maison"]
    pos = ["ART:ind", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    # un→une car maison est 100% feminin
    assert result[0] == "une"
    assert result[1] == "belle"
    assert result[2] == "maison"


def test_le_grosse_chat(mock_lexique):
    """'le grosse chat' -> 'le gros chat' (DET+NOM masc, ADJ de-feminisee)."""
    mots = ["le", "grosse", "chat"]
    pos = ["ART:def", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    # DET+NOM masc agree → de-feminiser l'ADJ
    assert result[0] == "le"
    assert result[1] == "gros"
    assert result[2] == "chat"
    assert len(corrections) >= 1


def test_le_grosse_enfant_ambigu(mock_lexique):
    """'le grosse enfant' with ambiguous NOM (m+f) → correct ADJ only.
    DET 'le' signals masc intent, NOM accepts masc → ADJ aligns to masc."""
    from tests.conftest import MockLexique
    formes = {
        "le": [{"ortho": "le", "cgram": "ART:def", "freq": 890, "genre": "m", "nombre": "s"}],
        "la": [{"ortho": "la", "cgram": "ART:def", "freq": 720, "genre": "f", "nombre": "s"}],
        "grosse": [{"ortho": "grosse", "cgram": "ADJ", "freq": 20, "genre": "f", "nombre": "s"}],
        "gros": [{"ortho": "gros", "cgram": "ADJ", "freq": 30, "genre": "m", "nombre": "s"}],
        "enfant": [
            {"ortho": "enfant", "cgram": "NOM", "freq": 30, "genre": "m", "nombre": "s"},
            {"ortho": "enfant", "cgram": "NOM", "freq": 25, "genre": "f", "nombre": "s"},
        ],
    }
    lex = MockLexique(formes=formes)
    mots = ["le", "grosse", "enfant"]
    pos = ["ART:def", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, lex)
    # NOM ambigu (m+f) but DET masc → Fix 1: ADJ aligns to DET+NOM(m)
    assert result[0] == "le"
    assert result[1] == "gros"
    assert result[2] == "enfant"
    assert len(corrections) >= 1


def test_le_fille_no_adj_still_excluded(mock_lexique):
    """'le fille' (DET+NOM, no ADJ) → no correction (le/la guard still active)."""
    mots = ["le", "fille"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    # Without ADJ intercale, le/la exclusion still applies (no Fix 1/2)
    assert result[0] == "le"
    assert result[1] == "fille"


# --- Pluriels irreguliers (-al -> -aux, -eau -> -eaux) ---

def test_les_cheval_chevaux(mock_lexique):
    """'les cheval' -> 'les chevaux' (pluriel -al -> -aux)."""
    mots = ["les", "cheval"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "chevaux"
    assert any(c.corrige == "chevaux" for c in corrections)


def test_les_journal_journaux(mock_lexique):
    """'les journal' -> 'les journaux' (pluriel -al -> -aux)."""
    mots = ["les", "journal"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "journaux"
    assert any(c.corrige == "journaux" for c in corrections)


def test_les_gateau_gateaux(mock_lexique):
    """'les gâteau' -> 'les gâteaux' (pluriel -eau -> -eaux)."""
    mots = ["les", "gâteau"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "gâteaux"
    assert any(c.corrige == "gâteaux" for c in corrections)


# --- A1 : Chiffres arabes comme declencheur pluriel ---

def test_chiffre_pluriel_nom(mock_lexique):
    """'29 mort' -> '29 morts' (chiffre >= 2 pluralise le NOM)."""
    mots = ["29", "mort"]
    pos = ["NUM", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "morts"
    assert any(c.corrige == "morts" for c in corrections)


def test_chiffre_1_pas_de_pluriel(mock_lexique):
    """'1 mort' -> pas de pluriel (chiffre = 1)."""
    mots = ["1", "mort"]
    pos = ["NUM", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "mort"
    assert not any(c.corrige == "morts" for c in corrections)


def test_chiffre_nom_deja_pluriel(mock_lexique):
    """'3 morts' -> pas de correction (deja pluriel)."""
    mots = ["3", "morts"]
    pos = ["NUM", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "morts"
    assert not any(c.original == "morts" for c in corrections)


# --- A2 : même n'est plus exclu des accords ---

def test_meme_pluriel_apres_det(mock_lexique):
    """'les même règles' -> 'les mêmes règles' (même accorde au pluriel)."""
    mots = ["les", "même"]
    pos = ["ART:def", "ADJ"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "mêmes"
    assert any(c.corrige == "mêmes" for c in corrections)


def test_memes_singulier_apres_det(mock_lexique):
    """'la mêmes' -> 'la même' (mêmes singularise apres DET singulier)."""
    mots = ["la", "mêmes"]
    pos = ["ART:def", "ADJ"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "même"


# --- A3 : un/une genre correction ---

def test_une_segment_un_segment(mock_lexique):
    """'une segment' -> 'un segment' (segment est 100% masculin)."""
    mots = ["une", "segment"]
    pos = ["ART:ind", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[0] == "un"
    assert any(c.corrige == "un" for c in corrections)


def test_un_maison_une_maison(mock_lexique):
    """'un maison' -> 'une maison' (maison est 100% feminin)."""
    mots = ["un", "maison"]
    pos = ["ART:ind", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[0] == "une"
    assert any(c.corrige == "une" for c in corrections)


def test_un_enfant_pas_de_correction(mock_lexique):
    """'un enfant' -> pas de correction (enfant n'est pas 100% feminin)."""
    mots = ["un", "enfant"]
    pos = ["ART:ind", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[0] == "un"
    assert not any(c.regle == "accord.genre_det" for c in corrections)


# --- A4 : ADJ pre-nominal + NOM pluriel ---

def test_de_jolie_femmes(mock_lexique):
    """'de jolie femmes' -> 'de jolies femmes' (ADJ pre-nominal pluralise)."""
    mots = ["de", "jolie", "femmes"]
    pos = ["PRE", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "jolies"
    assert any(c.corrige == "jolies" for c in corrections)


def test_de_jolies_femmes_pas_de_correction(mock_lexique):
    """'de jolies femmes' -> pas de correction (ADJ deja pluriel)."""
    mots = ["de", "jolies", "femmes"]
    pos = ["PRE", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[1] == "jolies"
    assert not any(c.original == "jolies" for c in corrections)
