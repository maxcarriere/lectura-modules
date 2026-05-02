"""Tests pour la resegmentation."""

from lectura_correcteur.orthographe._resegmentation import resegmenter


def test_narrive_split(mock_lexique):
    """'narrive' devrait etre resegmente en ["n'", "arrive"]."""
    tokens = ["narrive"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["n'", "arrive"]


def test_mot_connu_pas_resegmente(mock_lexique):
    """Un mot connu du lexique ne doit pas etre resegmente."""
    tokens = ["chat"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["chat"]


def test_mot_court_pas_resegmente(mock_lexique):
    """Un token de moins de 3 caracteres ne doit pas etre resegmente."""
    tokens = ["ab"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["ab"]


def test_pas_de_faux_positif(mock_lexique):
    """Un mot inconnu qui ne commence pas par un clitique reste tel quel."""
    tokens = ["xyzabc"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["xyzabc"]


def test_lhomme_split(mock_lexique):
    """'lhomme' -> ["l'", "homme"] via split elargi (consonne h)."""
    tokens = ["lhomme"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["l'", "homme"]


def test_cest_split(mock_lexique):
    """'cest' -> ["c'", "est"] via split elargi."""
    tokens = ["cest"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["c'", "est"]


def test_quil_split(mock_lexique):
    """'quil' -> ["qu'", "il"] via split elargi."""
    tokens = ["quil"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["qu'", "il"]


# --- Axe 3 : Fusion de tokens ---

def test_fusion_beaucoup(mock_lexique):
    """'beau' + 'coup' -> 'beaucoup' (fusion connue)."""
    tokens = ["beau", "coup"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["beaucoup"]


def test_fusion_generique_frequente(mock_lexique):
    """Generic fusion: two unknown tokens whose concat is known & frequent."""
    # "ensem" + "ble" are both unknown, "ensemble" has freq=50 >= 10
    tokens = ["ensem", "ble"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["ensemble"]


def test_fusion_connue_deux_mots_connus(mock_lexique):
    """Fusion connue meme si les deux tokens sont connus (beau+coup)."""
    tokens = ["beau", "coup"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["beaucoup"]


def test_fusion_compose_trait_union(mock_lexique):
    """'peut' + 'être' reste separe (peut être = verbe pouvoir + etre)."""
    tokens = ["peut", "être"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["peut", "être"]


def test_fusion_compose_sans_accent(mock_lexique):
    """'peut' + 'etre' reste separe ('pouvoir etre', pas l'adverbe)."""
    tokens = ["peut", "etre"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["peut", "etre"]


def test_fusion_pas_de_faux_positif(mock_lexique):
    """Deux tokens connus hors liste ne doivent pas etre fusionnes."""
    tokens = ["le", "chat"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["le", "chat"]


# --- Tokens avec trait d'union (inversions verbe-pronom) ---

def test_vas_tu_preserve(mock_lexique):
    """'vas-tu' ne doit pas etre corrompu (les deux parties sont connues)."""
    tokens = ["vas-tu"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["vas-tu"]


def test_a_t_il_preserve(mock_lexique):
    """'a-t-il' preserve (t euphonique filtre, a et il sont connus)."""
    tokens = ["a-t-il"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["a-t-il"]
