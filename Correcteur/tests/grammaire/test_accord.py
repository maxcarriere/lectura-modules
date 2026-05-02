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
    """'le petit fille' -> DET le/la excluded from Rule 8 (too many FP).
    Pre-nominal ADJ not corrected (Rule 7 is post-nominal only)."""
    mots = ["le", "petit", "fille"]
    pos = ["ART:def", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    # le/la excluded from DET genre correction (Rule 8)
    assert result[0] == "le"
    # Pre-nominal ADJ not feminized (Rule 7 is post-nominal only)
    assert result[1] == "petit"
    assert result[2] == "fille"


def test_un_belle_maison(mock_lexique):
    """'un belle maison' -> un/une excluded from Rule 8 (too many FP with ambiguous NOM genre)."""
    mots = ["un", "belle", "maison"]
    pos = ["ART:ind", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    # un/une excluded from DET genre correction (Rule 8) like le/la
    assert result[0] == "un"
    assert result[1] == "belle"
    assert result[2] == "maison"


def test_le_grosse_chat(mock_lexique):
    """'le grosse chat' -> le/la excluded from Rule 8, ADJ not de-feminized."""
    mots = ["le", "grosse", "chat"]
    pos = ["ART:def", "ADJ", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result[0] == "le"
    # le/la excluded from Rule 8, so ADJ stays as-is
    assert result[1] == "grosse"
    assert result[2] == "chat"


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
