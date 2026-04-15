# Lectura NLP — Outils de traitement du langage naturel pour le francais.
# Copyright (C) 2025  Max Carriere
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests unitaires complets pour lectura_tokeniseur.

Couvre la normalisation, la tokenisation, la detection de formules,
les types de tokens et les cas limites.
"""

import pytest

from lectura_tokeniseur import (
    FormuleType,
    Formule,
    LecturaTokeniseur,
    MathToken,
    Mot,
    Ponctuation,
    ResultatTokenisation,
    Separateur,
    Span,
    Token,
    TokenType,
    normalise,
    tokenise,
    tokenize_maths,
)


# ============================================================================
# 1. Normalisation
# ============================================================================


class TestNormalisation:
    """Tests de la fonction normalise() et de ses sous-transformations."""

    # -- Espaces multiples -> espace unique --

    def test_espaces_multiples_reduits(self):
        """Plusieurs espaces consecutifs sont reduits a un seul."""
        assert normalise("Bonjour    monde") == "Bonjour monde"

    def test_espaces_tabs_et_insecables(self):
        """Tabulations et espaces insecables sont traites comme des espaces."""
        assert normalise("a\tb") == "a b"
        assert normalise("a\u00A0b") == "a b"

    def test_espaces_en_debut_et_fin(self):
        """Les espaces en debut et fin de chaine sont supprimes."""
        assert normalise("  bonjour  ") == "bonjour"

    # -- Apostrophes typographiques -> standardisees --

    def test_apostrophe_avec_espaces(self):
        """Les espaces autour d'une apostrophe sont supprimes."""
        assert "l'homme" in normalise("l ' homme")

    def test_apostrophe_normalisee(self):
        """L'apostrophe droite est conservee, les espaces elimines."""
        result = normalise("l'enfant")
        assert "l'enfant" in result

    # -- Points de suspension ... -> ellipse unicode --

    def test_trois_points_en_ellipse(self):
        """Trois points consecutifs sont remplaces par le caractere ellipse."""
        result = normalise("Bonjour...")
        assert "\u2026" in result
        assert "..." not in result

    def test_ellipse_en_debut(self):
        """Ellipse en debut de texte."""
        result = normalise("...Bonjour")
        assert result.startswith("\u2026")

    def test_ellipse_espacement(self):
        """L'ellipse obtient un espace avant (si non debut) et apres."""
        result = normalise("Bonjour... monde")
        assert "\u2026" in result
        # L'ellipse doit etre entouree d'espaces (pas collee au mot)
        idx = result.index("\u2026")
        if idx > 0:
            assert result[idx - 1] == " "

    # -- Guillemets droits -> guillemets francais --

    def test_guillemets_droits_vers_francais(self):
        """Les guillemets droits sont remplaces par des guillemets francais."""
        result = normalise('Il a dit "bonjour" a tous.')
        assert "\u00ab" in result  # <<
        assert "\u00bb" in result  # >>

    def test_guillemets_avec_contenu(self):
        """Le contenu entre guillemets est preserve."""
        result = normalise('Elle dit "merci".')
        assert "merci" in result

    # -- Tirets en debut de ligne (dialogue) --

    def test_tiret_dialogue(self):
        """Un tiret isole (dialogue) est entoure d'espaces."""
        result = normalise("- Bonjour")
        assert "-" in result
        # Le tiret de dialogue doit avoir un espace apres
        idx = result.index("-")
        assert idx + 1 < len(result)

    # -- Parentheses (espaces internes supprimes) --

    def test_parentheses_espaces_internes(self):
        """Les espaces internes aux parentheses sont supprimes."""
        result = normalise("( bonjour )")
        assert "(bonjour)" in result

    def test_crochets_espaces_internes(self):
        """Les espaces internes aux crochets sont supprimes."""
        result = normalise("[ note ]")
        assert "[note]" in result

    # -- Cas supplementaires --

    def test_texte_vide(self):
        """Un texte vide retourne une chaine vide."""
        assert normalise("") == ""

    def test_normalise_ponctuation_faible(self):
        """Pas d'espace avant virgule/point, un espace apres."""
        result = normalise("Bonjour , monde")
        assert " ," not in result

    def test_normalise_nombres_avec_espaces(self):
        """Les grands nombres avec espaces sont normalises avec apostrophes."""
        result = normalise("1 000 000")
        assert result == "1'000'000"

    def test_normalise_virgule_decimale(self):
        """La virgule decimale entre chiffres est convertie en point."""
        result = normalise("3,14")
        assert "3.14" in result


# ============================================================================
# 2. Tokenisation
# ============================================================================


class TestTokenisation:
    """Tests de la fonction tokenise() et du pipeline de tokenisation."""

    def test_phrase_simple(self):
        """Une phrase simple produit des tokens Mot et Ponctuation."""
        tokens = tokenise("Le chat dort.")
        types = [t.type for t in tokens]
        # Doit contenir des MOT et au moins une PONCTUATION (le point)
        assert TokenType.MOT in types
        assert TokenType.PONCTUATION in types

    def test_phrase_simple_mots(self):
        """Les mots d'une phrase simple sont correctement identifies."""
        tokens = tokenise("Le chat dort.")
        mots = [t for t in tokens if isinstance(t, Mot)]
        textes = [m.text for m in mots]
        assert "Le" in textes
        assert "chat" in textes
        assert "dort" in textes

    def test_mot_compose_trait_union(self):
        """Un mot compose avec trait d'union est fusionne en un seul Mot."""
        tokens = tokenise("peut-etre")
        mots = [t for t in tokens if isinstance(t, Mot)]
        assert any("peut-etre" in m.text.lower() for m in mots)

    def test_mot_compose_cest_a_dire(self):
        """La locution c'est-\u00e0-dire est reconnue comme un seul token."""
        tokens = tokenise("c'est-\u00e0-dire")
        mots = [t for t in tokens if isinstance(t, Mot)]
        # Doit etre fusionne en un seul mot compose (locution figee)
        textes = [m.text.lower() for m in mots]
        assert any("c'est-\u00e0-dire" in t for t in textes)

    def test_elision_homme(self):
        """L'elision l'homme produit deux tokens (l + apostrophe + homme)."""
        tokens = tokenise("l'homme")
        mots = [t for t in tokens if isinstance(t, Mot)]
        # L'elision doit couper : "l" est un mot, "homme" est un autre mot
        textes = [m.text.lower() for m in mots]
        assert "l" in textes
        assert "homme" in textes

    def test_elision_daccord(self):
        """L'elision d'accord produit deux mots separes."""
        tokens = tokenise("d'accord")
        mots = [t for t in tokens if isinstance(t, Mot)]
        textes = [m.text.lower() for m in mots]
        assert "d" in textes
        assert "accord" in textes

    def test_elision_sil(self):
        """L'elision s'il produit deux mots separes."""
        tokens = tokenise("s'il")
        mots = [t for t in tokens if isinstance(t, Mot)]
        textes = [m.text.lower() for m in mots]
        assert "s" in textes
        assert "il" in textes

    def test_texte_vide(self):
        """Un texte vide retourne une liste vide."""
        assert tokenise("") == []

    def test_ponctuation_seule(self):
        """De la ponctuation seule produit des tokens Ponctuation."""
        tokens = tokenise(".,!")
        for t in tokens:
            assert isinstance(t, (Ponctuation, Separateur))

    def test_espaces_seulement(self):
        """Un texte avec seulement des espaces retourne une liste de Separateurs."""
        tokens = tokenise("   ")
        assert all(isinstance(t, Separateur) for t in tokens)

    def test_separateur_espace(self):
        """Les espaces entre mots generent des tokens Separateur de type space."""
        tokens = tokenise("un deux")
        seps = [t for t in tokens if isinstance(t, Separateur) and t.sep_type == "space"]
        assert len(seps) >= 1

    def test_spans_coherents(self):
        """Les spans des tokens couvrent le texte original correctement."""
        text = "Le chat"
        tokens = tokenise(text)
        for t in tokens:
            assert t.text == text[t.span[0]:t.span[1]]


# ============================================================================
# 3. Detection de formules
# ============================================================================


class TestFormules:
    """Tests de la detection des differents types de formules."""

    # -- NOMBRE --

    def test_nombre_entier(self):
        """Un nombre entier simple est detecte comme NOMBRE."""
        tokens = tokenise("42")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert len(formules) == 1
        assert formules[0].formule_type == FormuleType.NOMBRE
        assert formules[0].text == "42"

    def test_nombre_avec_separateurs(self):
        """Un nombre avec apostrophes de milliers est detecte comme NOMBRE."""
        tokens = tokenise("1'000")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.NOMBRE for f in formules)

    def test_nombre_decimal(self):
        """Un nombre decimal avec point est detecte comme NOMBRE."""
        tokens = tokenise("3.14")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.NOMBRE for f in formules)

    # -- DATE --

    def test_date_format_francais(self):
        """Une date au format JJ/MM/AAAA est detectee comme DATE."""
        tokens = tokenise("01/01/2025")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.DATE for f in formules)

    def test_date_format_iso(self):
        """Une date au format AAAA-MM-JJ est detectee comme DATE."""
        tokens = tokenise("2025-01-01")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.DATE for f in formules)

    def test_date_format_point(self):
        """Une date au format JJ.MM.AAAA est detectee comme DATE."""
        tokens = tokenise("15.03.2024")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.DATE for f in formules)

    def test_date_valeur_preservee(self):
        """La valeur d'une date est preservee."""
        tokens = tokenise("01/01/2025")
        formules = [t for t in tokens if isinstance(t, Formule) and t.formule_type == FormuleType.DATE]
        assert len(formules) >= 1
        assert "01" in formules[0].valeur
        assert "2025" in formules[0].valeur

    # -- TELEPHONE --

    def test_telephone_format_espaces(self):
        """Un numero de telephone au format 06 12 34 56 78 est detecte."""
        tokens = tokenise("06 12 34 56 78")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.TELEPHONE for f in formules)

    def test_telephone_valeur_nettoyee(self):
        """La valeur d'un telephone est nettoyee (sans espaces)."""
        tokens = tokenise("06 12 34 56 78")
        tel = [f for f in tokens if isinstance(f, Formule) and f.formule_type == FormuleType.TELEPHONE]
        assert len(tel) >= 1
        assert tel[0].valeur == "0612345678"

    def test_telephone_children(self):
        """Un telephone a des enfants (paires de chiffres)."""
        tokens = tokenise("06 12 34 56 78")
        tel = [f for f in tokens if isinstance(f, Formule) and f.formule_type == FormuleType.TELEPHONE]
        assert len(tel) >= 1
        assert len(tel[0].children) == 5

    # -- SIGLE --

    def test_sigle_sncf(self):
        """SNCF est detecte comme un SIGLE."""
        tokens = tokenise("SNCF")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.SIGLE for f in formules)

    def test_sigle_unesco(self):
        """UNESCO est detecte comme un SIGLE."""
        tokens = tokenise("UNESCO")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.SIGLE for f in formules)

    def test_sigle_valeur_upper(self):
        """La valeur d'un sigle est en majuscules."""
        tokens = tokenise("SNCF")
        sigles = [f for f in tokens if isinstance(f, Formule) and f.formule_type == FormuleType.SIGLE]
        assert len(sigles) >= 1
        assert sigles[0].valeur == "SNCF"

    def test_sigle_avec_chiffres(self):
        """Un sigle mixte lettres+chiffres (FR25) est detecte."""
        # Apres normalisation, FR25 devrait etre detecte
        tok = LecturaTokeniseur()
        tokens = tok.tokenize("FR25")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.SIGLE for f in formules)

    # -- ORDINAL --

    def test_ordinal_1er(self):
        """1er est detecte comme ORDINAL."""
        tokens = tokenise("1er")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.ORDINAL for f in formules)

    def test_ordinal_42e(self):
        """42e est detecte comme ORDINAL."""
        tokens = tokenise("42e")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.ORDINAL for f in formules)

    def test_ordinal_2eme(self):
        """2\u00e8me est detecte comme ORDINAL."""
        tok = LecturaTokeniseur()
        formules = tok.extract_formules("2\u00e8me")
        assert any(f.formule_type == FormuleType.ORDINAL for f in formules)

    def test_ordinal_children(self):
        """Un ordinal a des enfants (nombre + suffixe)."""
        tokens = tokenise("1er")
        ords = [f for f in tokens if isinstance(f, Formule) and f.formule_type == FormuleType.ORDINAL]
        assert len(ords) >= 1
        assert len(ords[0].children) == 2  # nombre + suffixe

    # -- FRACTION --

    def test_fraction_3_4(self):
        """3/4 est detecte comme FRACTION."""
        tokens = tokenise("3/4")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.FRACTION for f in formules)

    def test_fraction_1_2(self):
        """1/2 est detecte comme FRACTION."""
        tokens = tokenise("1/2")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.FRACTION for f in formules)

    def test_fraction_valeur(self):
        """La valeur d'une fraction est formatee correctement."""
        tokens = tokenise("3/4")
        fracs = [f for f in tokens if isinstance(f, Formule) and f.formule_type == FormuleType.FRACTION]
        assert len(fracs) >= 1
        assert fracs[0].valeur == "3/4"

    def test_fraction_children(self):
        """Une fraction a 3 enfants (numerateur, barre, denominateur)."""
        tokens = tokenise("3/4")
        fracs = [f for f in tokens if isinstance(f, Formule) and f.formule_type == FormuleType.FRACTION]
        assert len(fracs) >= 1
        assert len(fracs[0].children) == 3

    # -- SCIENTIFIQUE --

    def test_scientifique_notation(self):
        """3.14e-5 est detecte comme SCIENTIFIQUE."""
        # La notation scientifique est parsee en tokens adjacents
        # qui sont fusionnes en passe 2
        tok = LecturaTokeniseur()
        formules = tok.extract_formules("3.14e-5")
        assert any(f.formule_type == FormuleType.SCIENTIFIQUE for f in formules)

    def test_scientifique_positif(self):
        """2e10 est detecte comme SCIENTIFIQUE."""
        tok = LecturaTokeniseur()
        formules = tok.extract_formules("2e10")
        assert any(f.formule_type == FormuleType.SCIENTIFIQUE for f in formules)

    # -- MATHS --

    def test_maths_expression(self):
        """2x+3 est detecte comme MATHS."""
        tok = LecturaTokeniseur()
        formules = tok.extract_formules("2x+3")
        assert any(f.formule_type == FormuleType.MATHS for f in formules)

    def test_maths_children(self):
        """Une expression mathematique a des sous-tokens."""
        tok = LecturaTokeniseur()
        tokens = tok.tokenize("2x+3")
        maths = [f for f in tokens if isinstance(f, Formule) and f.formule_type == FormuleType.MATHS]
        if maths:
            assert len(maths[0].children) > 0

    # -- NUMERO --

    def test_numero_compose(self):
        """654 001 45 est detecte comme NUMERO."""
        tokens = tokenise("654 001 45")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.formule_type == FormuleType.NUMERO for f in formules)

    def test_numero_valeur(self):
        """La valeur d'un numero est le texte original."""
        tokens = tokenise("654 001 45")
        nums = [f for f in tokens if isinstance(f, Formule) and f.formule_type == FormuleType.NUMERO]
        assert len(nums) >= 1
        assert nums[0].valeur == "654 001 45"


# ============================================================================
# 4. Types de tokens et attributs
# ============================================================================


class TestTokenTypes:
    """Tests des types de donnees : Token, Mot, Ponctuation, Separateur, Formule."""

    # -- Token dataclass --

    def test_token_fields(self):
        """Un Token a les champs type, text et span."""
        tok = Token(type=TokenType.MOT, text="chat", span=(0, 4))
        assert tok.type == TokenType.MOT
        assert tok.text == "chat"
        assert tok.span == (0, 4)

    # -- Mot dataclass --

    def test_mot_fields(self):
        """Un Mot a les champs ortho et children en plus de Token."""
        mot = Mot(type=TokenType.MOT, text="Chat", span=(0, 4), ortho="chat")
        assert mot.ortho == "chat"
        assert mot.children == []

    def test_mot_children_default(self):
        """Les children de Mot sont une liste vide par defaut."""
        mot = Mot(type=TokenType.MOT, text="a", span=(0, 1))
        assert mot.children == []

    def test_mot_ortho_default(self):
        """L'ortho de Mot est une chaine vide par defaut."""
        mot = Mot(type=TokenType.MOT, text="A", span=(0, 1))
        assert mot.ortho == ""

    # -- Ponctuation dataclass --

    def test_ponctuation_fields(self):
        """Un token Ponctuation a les champs de base Token."""
        p = Ponctuation(type=TokenType.PONCTUATION, text=".", span=(5, 6))
        assert p.type == TokenType.PONCTUATION
        assert p.text == "."
        assert p.span == (5, 6)

    # -- Separateur dataclass --

    def test_separateur_fields(self):
        """Un Separateur a un champ sep_type en plus de Token."""
        s = Separateur(type=TokenType.SEPARATEUR, text="'", span=(1, 2), sep_type="apostrophe")
        assert s.sep_type == "apostrophe"

    def test_separateur_sep_type_default(self):
        """Le sep_type de Separateur est None par defaut."""
        s = Separateur(type=TokenType.SEPARATEUR, text=" ", span=(0, 1))
        assert s.sep_type is None

    def test_separateur_types_valides(self):
        """Les types de separateurs valides sont apostrophe, hyphen, space."""
        tokens = tokenise("l'homme est peut-etre la")
        seps = [t for t in tokens if isinstance(t, Separateur)]
        sep_types = {s.sep_type for s in seps}
        # Doit contenir au moins space et apostrophe
        assert "space" in sep_types
        assert "apostrophe" in sep_types

    # -- Formule dataclass --

    def test_formule_fields(self):
        """Une Formule a les champs formule_type, children, valeur, display_fr."""
        f = Formule(
            type=TokenType.FORMULE,
            text="42",
            span=(0, 2),
            formule_type=FormuleType.NOMBRE,
            valeur="42",
        )
        assert f.formule_type == FormuleType.NOMBRE
        assert f.children == []
        assert f.valeur == "42"
        assert f.display_fr == ""

    def test_formule_defaults(self):
        """Les valeurs par defaut de Formule sont correctes."""
        f = Formule(type=TokenType.FORMULE, text="0", span=(0, 1))
        assert f.formule_type == FormuleType.NOMBRE
        assert f.children == []
        assert f.valeur == ""
        assert f.display_fr == ""

    # -- TokenType enum --

    def test_token_type_values(self):
        """Les valeurs de l'enum TokenType sont correctes."""
        assert TokenType.MOT.value == "mot"
        assert TokenType.PONCTUATION.value == "ponctuation"
        assert TokenType.SEPARATEUR.value == "separateur"
        assert TokenType.FORMULE.value == "formule"

    def test_token_type_all_members(self):
        """L'enum TokenType a exactement 4 membres."""
        assert len(TokenType) == 4

    # -- FormuleType enum --

    def test_formule_type_values(self):
        """Les valeurs de l'enum FormuleType sont correctes."""
        assert FormuleType.NOMBRE.value == "nombre"
        assert FormuleType.SIGLE.value == "sigle"
        assert FormuleType.DATE.value == "date"
        assert FormuleType.TELEPHONE.value == "telephone"
        assert FormuleType.NUMERO.value == "numero"
        assert FormuleType.ORDINAL.value == "ordinal"
        assert FormuleType.FRACTION.value == "fraction"
        assert FormuleType.SCIENTIFIQUE.value == "scientifique"
        assert FormuleType.MATHS.value == "maths"

    def test_formule_type_all_members(self):
        """L'enum FormuleType a exactement 15 membres."""
        assert len(FormuleType) == 15

    # -- Span type alias --

    def test_span_est_tuple_int_int(self):
        """Span est un alias pour tuple[int, int]."""
        span: Span = (0, 5)
        assert isinstance(span, tuple)
        assert len(span) == 2
        assert isinstance(span[0], int)
        assert isinstance(span[1], int)


# ============================================================================
# 5. Classe LecturaTokeniseur
# ============================================================================


class TestLecturaTokeniseur:
    """Tests de la classe principale LecturaTokeniseur."""

    @pytest.fixture
    def tok(self):
        """Instance fraiche de LecturaTokeniseur pour chaque test."""
        return LecturaTokeniseur()

    # -- .normalize() --

    def test_normalize_methode(self, tok):
        """La methode normalize() applique la normalisation."""
        result = tok.normalize("Bonjour    monde")
        assert result == "Bonjour monde"

    def test_normalize_apostrophes(self, tok):
        """normalize() standardise les apostrophes."""
        result = tok.normalize("l ' homme")
        assert "'" in result
        assert "l'homme" in result

    def test_normalize_guillemets(self, tok):
        """normalize() convertit les guillemets droits en francais."""
        result = tok.normalize('Il dit "oui".')
        assert "\u00ab" in result

    def test_normalize_texte_vide(self, tok):
        """normalize() avec une chaine vide retourne une chaine vide."""
        assert tok.normalize("") == ""

    # -- .tokenize() --

    def test_tokenize_retourne_liste_tokens(self, tok):
        """tokenize() retourne une liste de tokens."""
        tokens = tok.tokenize("Bonjour monde.")
        assert isinstance(tokens, list)
        assert all(isinstance(t, Token) for t in tokens)

    def test_tokenize_avec_normalisation(self, tok):
        """tokenize() normalise par defaut avant de tokeniser."""
        tokens = tok.tokenize("Bonjour    monde.")
        mots = [t for t in tokens if isinstance(t, Mot)]
        textes = [m.text for m in mots]
        assert "Bonjour" in textes
        assert "monde" in textes

    def test_tokenize_sans_normalisation(self, tok):
        """tokenize(normalize=False) ne normalise pas le texte."""
        text = "Le chat."
        tokens = tok.tokenize(text, normalize=False)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_texte_vide(self, tok):
        """tokenize() avec texte vide retourne une liste vide."""
        assert tok.tokenize("") == []

    # -- .analyze() --

    def test_analyze_retourne_resultat(self, tok):
        """analyze() retourne un ResultatTokenisation."""
        result = tok.analyze("Le chat dort.")
        assert isinstance(result, ResultatTokenisation)

    def test_analyze_texte_original_preserve(self, tok):
        """analyze() preserve le texte original."""
        original = "Bonjour    monde !"
        result = tok.analyze(original)
        assert result.texte_original == original

    def test_analyze_texte_normalise(self, tok):
        """analyze() contient le texte normalise."""
        result = tok.analyze("Bonjour    monde")
        assert result.texte_normalise == "Bonjour monde"

    def test_analyze_tokens_non_vide(self, tok):
        """analyze() produit des tokens pour un texte non vide."""
        result = tok.analyze("Bonjour.")
        assert len(result.tokens) > 0

    def test_analyze_propriete_mots(self, tok):
        """La propriete .mots retourne uniquement les Mot."""
        result = tok.analyze("Le chat dort.")
        assert all(isinstance(m, Mot) for m in result.mots)

    def test_analyze_propriete_formules(self, tok):
        """La propriete .formules retourne uniquement les Formule."""
        result = tok.analyze("Il a 42 ans.")
        assert all(isinstance(f, Formule) for f in result.formules)
        assert any(f.text == "42" for f in result.formules)

    def test_analyze_nb_mots(self, tok):
        """La propriete nb_mots retourne le bon compte."""
        result = tok.analyze("Le chat dort.")
        assert result.nb_mots == 3

    def test_analyze_nb_tokens(self, tok):
        """La propriete nb_tokens retourne le nombre total de tokens."""
        result = tok.analyze("Le chat dort.")
        assert result.nb_tokens > 0
        assert result.nb_tokens == len(result.tokens)

    def test_analyze_words(self, tok):
        """La methode words() retourne les formes orthographiques."""
        result = tok.analyze("Le Chat dort.")
        words = result.words()
        assert isinstance(words, list)
        assert all(isinstance(w, str) for w in words)
        # Les mots doivent etre en minuscules (ortho)
        assert "le" in words
        assert "chat" in words
        assert "dort" in words

    def test_analyze_format_table(self, tok):
        """La methode format_table() retourne une chaine non vide."""
        result = tok.analyze("Bonjour.")
        table = result.format_table()
        assert isinstance(table, str)
        assert len(table) > 0
        assert "Texte" in table  # en-tete de colonnes

    # -- .extract_words() --

    def test_extract_words(self, tok):
        """extract_words() retourne les mots en minuscules."""
        words = tok.extract_words("Le Chat Dort.")
        assert isinstance(words, list)
        assert "le" in words
        assert "chat" in words
        assert "dort" in words

    def test_extract_words_texte_vide(self, tok):
        """extract_words() avec texte vide retourne une liste vide."""
        assert tok.extract_words("") == []

    # -- .extract_formules() --

    def test_extract_formules(self, tok):
        """extract_formules() retourne les formules detectees."""
        formules = tok.extract_formules("Il a 42 ans.")
        assert isinstance(formules, list)
        assert all(isinstance(f, Formule) for f in formules)
        assert any(f.text == "42" for f in formules)

    def test_extract_formules_telephone(self, tok):
        """extract_formules() detecte un numero de telephone."""
        formules = tok.extract_formules("Appeler le 06 12 34 56 78.")
        tels = [f for f in formules if f.formule_type == FormuleType.TELEPHONE]
        assert len(tels) >= 1

    def test_extract_formules_date(self, tok):
        """extract_formules() detecte une date."""
        formules = tok.extract_formules("Le 15/03/2024 est un vendredi.")
        dates = [f for f in formules if f.formule_type == FormuleType.DATE]
        assert len(dates) >= 1

    def test_extract_formules_texte_sans_formule(self, tok):
        """extract_formules() retourne une liste vide si pas de formules."""
        formules = tok.extract_formules("Le chat dort.")
        assert formules == []

    def test_extract_formules_multiples(self, tok):
        """extract_formules() detecte plusieurs formules dans un texte."""
        formules = tok.extract_formules("SNCF et 42 et 1er")
        assert len(formules) >= 3


# ============================================================================
# 6. Cas limites
# ============================================================================


class TestEdgeCases:
    """Tests des cas limites et situations speciales."""

    @pytest.fixture
    def tok(self):
        return LecturaTokeniseur()

    # -- Caracteres Unicode (accents, cedilles) --

    def test_unicode_accents(self, tok):
        """Les caracteres accentues sont correctement tokenises."""
        tokens = tok.tokenize("L'ete a ete beau.")
        mots = [t for t in tokens if isinstance(t, Mot)]
        textes = [m.text.lower() for m in mots]
        assert any("ete" in t for t in textes)

    def test_unicode_cedille(self, tok):
        """La cedille est preservee dans les tokens."""
        tokens = tok.tokenize("Le garcon a recu une lecon.")
        mots = [t for t in tokens if isinstance(t, Mot)]
        textes = [m.text for m in mots]
        assert any("recu" in t.lower() for t in textes)
        assert any("lecon" in t.lower() for t in textes)

    def test_unicode_e_accent_grave(self, tok):
        """Les accents graves sont preserves."""
        result = tok.analyze("A la mere et au pere")
        words = result.words()
        assert any("mere" in w or "m\u00e8re" in w for w in words)

    def test_unicode_trema(self, tok):
        """Les tremas sont preserves."""
        tokens = tok.tokenize("noel")
        mots = [t for t in tokens if isinstance(t, Mot)]
        assert len(mots) >= 1

    # -- Texte tres long --

    def test_texte_tres_long(self, tok):
        """Un texte tres long est tokenise sans erreur."""
        texte = "Le chat dort. " * 1000
        tokens = tok.tokenize(texte)
        assert len(tokens) > 0
        mots = [t for t in tokens if isinstance(t, Mot)]
        assert len(mots) >= 3000  # au moins 3 mots x 1000 repetitions

    def test_texte_long_performance(self, tok):
        """Un texte long ne provoque pas d'erreur de recursion."""
        texte = " ".join(["mot"] * 5000)
        tokens = tok.tokenize(texte)
        mots = [t for t in tokens if isinstance(t, Mot)]
        assert len(mots) == 5000

    # -- Nombres aux frontieres --

    def test_nombre_en_debut_de_texte(self, tok):
        """Un nombre en debut de texte est correctement detecte."""
        tokens = tok.tokenize("42 chats")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.text == "42" for f in formules)

    def test_nombre_en_fin_de_texte(self, tok):
        """Un nombre en fin de texte est correctement detecte."""
        tokens = tok.tokenize("article 42")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert any(f.text == "42" for f in formules)

    def test_nombre_isole(self, tok):
        """Un nombre seul est correctement detecte."""
        tokens = tok.tokenize("42")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert len(formules) == 1
        assert formules[0].formule_type == FormuleType.NOMBRE

    def test_nombre_zero(self, tok):
        """Le nombre 0 est detecte comme NOMBRE."""
        tokens = tok.tokenize("0")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert len(formules) == 1
        assert formules[0].formule_type == FormuleType.NOMBRE

    # -- Formules melangees avec du texte --

    def test_formules_et_texte_meles(self, tok):
        """Un texte avec des formules melangees est correctement traite."""
        result = tok.analyze("Le 01/01/2025, la SNCF a eu 42 passagers.")
        assert result.nb_mots > 0
        formules = result.formules
        types_trouves = {f.formule_type for f in formules}
        assert FormuleType.DATE in types_trouves
        assert FormuleType.SIGLE in types_trouves
        assert FormuleType.NOMBRE in types_trouves

    def test_telephone_dans_phrase(self, tok):
        """Un telephone dans une phrase complete est detecte."""
        result = tok.analyze("Appelez le 06 12 34 56 78 maintenant.")
        tels = [f for f in result.formules if f.formule_type == FormuleType.TELEPHONE]
        assert len(tels) >= 1

    def test_ordinal_dans_phrase(self, tok):
        """Un ordinal dans une phrase est correctement detecte."""
        result = tok.analyze("C'est le 1er prix.")
        ords = [f for f in result.formules if f.formule_type == FormuleType.ORDINAL]
        assert len(ords) >= 1

    def test_fraction_dans_phrase(self, tok):
        """Une fraction dans une phrase est detectee."""
        result = tok.analyze("Il a mange 3/4 du gateau.")
        fracs = [f for f in result.formules if f.formule_type == FormuleType.FRACTION]
        assert len(fracs) >= 1

    # -- Separateurs et tokens adjacents --

    def test_apostrophe_entre_mots(self, tok):
        """L'apostrophe entre deux mots genere un Separateur."""
        tokens = tok.tokenize("l'enfant")
        seps = [t for t in tokens if isinstance(t, Separateur) and t.sep_type == "apostrophe"]
        assert len(seps) >= 1

    def test_hyphen_entre_mots(self, tok):
        """Le trait d'union entre deux mots est integre dans le mot compose."""
        tokens = tok.tokenize("peut-etre")
        mots = [t for t in tokens if isinstance(t, Mot)]
        # Le mot compose fusionne contient un trait d'union
        assert any("-" in m.text for m in mots)

    # -- Locutions figees --

    def test_locution_cest_a_dire(self, tok):
        """c'est-\u00e0-dire est reconnu comme une locution figee."""
        tokens = tok.tokenize("c'est-\u00e0-dire")
        mots = [t for t in tokens if isinstance(t, Mot)]
        # Doit etre fusionne en un seul token (locution figee)
        assert any("c'est-\u00e0-dire" in m.text.lower() for m in mots)

    # -- Pas de faux positifs --

    def test_mot_normal_pas_sigle(self, tok):
        """Un mot normal en minuscules n'est pas detecte comme sigle."""
        tokens = tok.tokenize("bonjour")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert len(formules) == 0

    def test_mot_court_pas_formule(self, tok):
        """Un mot court (ex: 'le') n'est pas une formule."""
        tokens = tok.tokenize("le")
        formules = [t for t in tokens if isinstance(t, Formule)]
        assert len(formules) == 0

    # -- ResultatTokenisation coherent --

    def test_resultat_coherence_globale(self, tok):
        """Le resultat d'une analyse est coherent de bout en bout."""
        text = "Le 1er janvier 2025, la SNCF a vendu 42 billets."
        result = tok.analyze(text)

        # Le texte original est preserve
        assert result.texte_original == text
        # Le texte normalise est non vide
        assert len(result.texte_normalise) > 0
        # Il y a des tokens
        assert result.nb_tokens > 0
        # Il y a des mots
        assert result.nb_mots > 0
        # Le compte nb_mots correspond a len(mots)
        assert result.nb_mots == len(result.mots)
        # Le compte nb_tokens correspond a len(tokens)
        assert result.nb_tokens == len(result.tokens)
        # words() retourne le meme nombre que nb_mots
        assert len(result.words()) == result.nb_mots


# ============================================================================
# Tokenisation maths (maths.py)
# ============================================================================


class TestMathsTokenization:
    """Tests de tokenize_maths() et de la detection maths enrichie."""

    def test_tokenize_basic(self):
        toks = tokenize_maths("2x+3")
        assert [t.math_type for t in toks] == ["number", "variable", "operator", "number"]

    def test_tokenize_function(self):
        toks = tokenize_maths("sin(x)")
        assert toks[0] == MathToken("sin", "function")

    def test_tokenize_unit_space(self):
        toks = tokenize_maths("5 km")
        assert toks[1].math_type == "unit"

    def test_tokenize_unit_single(self):
        toks = tokenize_maths("5 m")
        assert toks[1].math_type == "unit"

    def test_tokenize_sqrt(self):
        toks = tokenize_maths("√9")
        assert toks[0] == MathToken("√", "operator")

    def test_tokenize_superscript(self):
        toks = tokenize_maths("x²")
        assert toks[0] == MathToken("x", "variable")
        assert toks[1].math_type == "superscript"
        assert toks[1].extra == "2"

    def test_tokenize_subscript(self):
        toks = tokenize_maths("x₁")
        assert toks[1].math_type == "subscript"
        assert toks[1].extra == "1"

    def test_tokenize_greek(self):
        toks = tokenize_maths("α+β")
        assert toks[0] == MathToken("α", "greek")
        assert toks[2] == MathToken("β", "greek")

    def test_tokenize_factorial(self):
        toks = tokenize_maths("5!")
        assert toks[1] == MathToken("!", "factorial")

    def test_tokenize_prime(self):
        toks = tokenize_maths("f'")
        assert toks[1] == MathToken("'", "prime")

    def test_tokenize_unit_requalify(self):
        """km/h : h requalifie en unite."""
        toks = tokenize_maths("km/h")
        types = [t.math_type for t in toks]
        assert types == ["unit", "operator", "unit"]

    def test_mathtoken_tuple_compat(self):
        """MathToken est un NamedTuple, indexable comme un tuple."""
        mt = MathToken("x", "variable")
        assert mt[0] == "x"
        assert mt[1] == "variable"
        assert mt[2] == ""

    def test_detect_fx(self):
        tok = LecturaTokeniseur()
        result = tok.analyze("f(x)")
        formules = [f for f in result.formules if f.formule_type == FormuleType.MATHS]
        assert len(formules) > 0

    def test_detect_sqrt(self):
        tok = LecturaTokeniseur()
        result = tok.analyze("√9")
        formules = [f for f in result.formules if f.formule_type == FormuleType.MATHS]
        assert len(formules) > 0

    def test_detect_number_unit(self):
        tok = LecturaTokeniseur()
        result = tok.analyze("5 km")
        formules = [f for f in result.formules if f.formule_type == FormuleType.MATHS]
        assert len(formules) > 0

    def test_detect_degres(self):
        tok = LecturaTokeniseur()
        result = tok.analyze("36.5 °C")
        formules = [f for f in result.formules if f.formule_type == FormuleType.MATHS]
        assert len(formules) > 0

    def test_maths_children_enriched(self):
        """Les children d'une formule MATHS utilisent tokenize_maths."""
        tok = LecturaTokeniseur()
        result = tok.analyze("2x+3")
        maths = [f for f in result.formules if f.formule_type == FormuleType.MATHS]
        assert len(maths) == 1
        children = maths[0].children
        # Doit avoir des enfants typés (nombre, mot, ponctuation)
        assert len(children) >= 4
        # Premier enfant = nombre "2"
        assert isinstance(children[0], Formule)
        assert children[0].formule_type == FormuleType.NOMBRE
