"""Tests d'integration du pipeline complet de correction."""

from lectura_correcteur import Correcteur, TypeCorrection
from lectura_correcteur._config import CorrecteurConfig


def test_phrase_vide(mock_lexique):
    c = Correcteur(mock_lexique)
    result = c.corriger("")
    assert result.phrase_corrigee == ""
    assert result.n_corrections == 0


def test_phrase_simple(mock_lexique):
    c = Correcteur(mock_lexique)
    result = c.corriger("Le chat dort")
    assert result.phrase_corrigee is not None
    assert len(result.phrase_corrigee) > 0


def test_grammaire_pluriel(mock_lexique):
    """'les enfant' -> 'les enfants' via regles grammaire (appel direct)."""
    from lectura_correcteur.grammaire import appliquer_grammaire

    mots = ["les", "enfant"]
    pos = ["ART:def", "NOM"]
    result, corrections = appliquer_grammaire(
        mots, pos, {}, mock_lexique,
    )
    assert "enfants" in [m.lower() for m in result]


def test_resultat_structure(mock_lexique):
    c = Correcteur(mock_lexique)
    result = c.corriger("Le chat dort")
    assert hasattr(result, "phrase_originale")
    assert hasattr(result, "phrase_corrigee")
    assert hasattr(result, "mots")
    assert hasattr(result, "n_corrections")
    assert hasattr(result, "corrections")


def test_syntaxe_majuscule(mock_lexique):
    """La premiere lettre doit etre en majuscule."""
    c = Correcteur(mock_lexique)
    result = c.corriger("le chat")
    assert result.phrase_corrigee[0].isupper()


def test_sous_modules_independants():
    """Les sous-modules doivent etre importables independamment."""
    from lectura_correcteur.grammaire import verifier_accords
    from lectura_correcteur.orthographe import VerificateurOrthographe
    from lectura_correcteur.syntaxe import verifier_ponctuation

    assert callable(verifier_accords)
    assert callable(verifier_ponctuation)
    assert VerificateurOrthographe is not None


def test_config_sans_grammaire(mock_lexique):
    """Avec grammaire desactivee, pas de corrections grammaticales."""
    config = CorrecteurConfig(activer_grammaire=False)
    c = Correcteur(mock_lexique, config=config)
    result = c.corriger("Le chat dort")
    assert result.phrase_corrigee is not None


def test_mot_analyse_sans_ipa(mock_lexique):
    """MotAnalyse ne doit pas avoir de champ ipa/alternatives."""
    c = Correcteur(mock_lexique)
    result = c.corriger("Le chat dort")
    for m in result.mots:
        assert not hasattr(m, "ipa")
        assert not hasattr(m, "alternatives")


def test_pipeline_utilise_crf(mock_lexique):
    """Le pipeline doit utiliser le MorphoTagger CRF pour les POS."""
    c = Correcteur(mock_lexique)
    result = c.corriger("Le chat mange")
    # Les mots doivent avoir des POS assigns par le CRF
    for m in result.mots:
        assert m.pos != "" or m.original in ("Le", "le")


def test_integration_3e_groupe(mock_lexique):
    """Test 3e groupe conjugation via grammaire directement (CRF POS-indep)."""
    from lectura_correcteur.grammaire import appliquer_grammaire

    mots = ["Ils", "dort", "bien"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = appliquer_grammaire(
        mots, pos, {}, mock_lexique, originaux=mots,
    )
    assert result[1] == "dorment"
    assert any(c.corrige == "dorment" for c in corrections)


# --- Axe 1 : Injection tagger ---

def test_tagger_injection(mock_lexique):
    """Le pipeline utilise un tagger injecte au lieu du CRF embarque."""
    class MockTagger:
        def tokenize(self, text):
            return [(t, not t.isspace()) for t in text.split()]
        def tag_words(self, words):
            return [{"pos": "NOM"}] * len(words)

    c = Correcteur(mock_lexique, tagger=MockTagger())
    result = c.corriger("chat maison")
    # Le mock tagger doit etre utilise (pas d'erreur CRF)
    assert result.phrase_corrigee is not None
    assert len(result.mots) == 2
    # Tous les POS doivent etre NOM (du mock)
    assert all(m.pos == "NOM" for m in result.mots)


# --- Axe 2 : Injection tokeniseur ---

def test_tokeniseur_injection(mock_lexique):
    """Le pipeline utilise un tokeniseur injecte."""
    from types import SimpleNamespace
    from enum import Enum

    class TokType(Enum):
        mot = "mot"
        separateur = "separateur"
        ponctuation = "ponctuation"

    class MockTokeniseur:
        def tokeniser(self, text):
            tokens = []
            for part in text.split():
                tokens.append(SimpleNamespace(
                    text=part, type=TokType.mot, sep_type="",
                ))
            return tokens

    class MockTagger:
        def tokenize(self, text):
            raise AssertionError("Should not be called when tokeniseur is set")
        def tag_words(self, words):
            return [{"pos": "NOM"}] * len(words)

    c = Correcteur(
        mock_lexique,
        tagger=MockTagger(),
        tokeniseur=MockTokeniseur(),
    )
    result = c.corriger("chat maison")
    assert result.phrase_corrigee is not None
    assert len(result.mots) == 2


# --- Chantier 3 : Fallback POS via lexique ---

def test_pos_fallback_dort(mock_lexique):
    """Le POS fallback corrige 'dort' de NOM → VER quand le lexique dit VER seul."""
    class MockTagger:
        def tokenize(self, text):
            return [(t, True) for t in text.split()]
        def tag_words(self, words):
            # Le CRF tague "dort" comme NOM par erreur
            tags = []
            for w in words:
                if w.lower() == "dort":
                    tags.append({"pos": "NOM"})
                elif w.lower() == "les":
                    tags.append({"pos": "ART:def"})
                elif w.lower() in ("chats", "chat"):
                    tags.append({"pos": "NOM"})
                elif w.lower() == "du":
                    tags.append({"pos": "ART:def"})
                elif w.lower() == "voisin":
                    tags.append({"pos": "NOM"})
                else:
                    tags.append({"pos": "NOM"})
            return tags

    c = Correcteur(mock_lexique, tagger=MockTagger())
    result = c.corriger("les chats du voisin dort")
    # Le fallback POS doit corriger "dort" de NOM → VER
    # Ensuite la regle d'accord sujet-verbe doit le conjuguer en "dorment"
    assert "dorment" in result.phrase_corrigee.lower()
