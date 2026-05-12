"""Tests pour le schema v5 : sens/entites separes, lemme_synonymes, etc."""

import math
import sqlite3
import tempfile
from pathlib import Path

import pytest

from lectura_lexique import Lexique


@pytest.fixture(scope="module")
def db_v5_path():
    """Cree une mini BDD v5 de test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    # Schema v5 : tables inchangees (formes, lemmes, categories, categorie_hierarchie)
    conn.executescript("""
        CREATE TABLE formes (
            id INTEGER PRIMARY KEY,
            ortho TEXT NOT NULL,
            lemme_id INTEGER,
            multext TEXT,
            phone TEXT,
            phone_reversed TEXT,
            nb_syllabes INTEGER,
            syllabes TEXT,
            freq_opensubs REAL,
            freq_frantext REAL,
            freq_lm10 REAL,
            freq_frwac REAL,
            freq_composite REAL,
            source TEXT,
            orthocode TEXT DEFAULT '',
            consonne_latente TEXT DEFAULT ''
        );
        CREATE TABLE lemmes (
            id INTEGER PRIMARY KEY,
            lemme TEXT NOT NULL,
            cgram TEXT NOT NULL,
            genre TEXT,
            sous_type TEXT,
            etymologie TEXT,
            forme_developpee TEXT,
            freq_opensubs REAL,
            freq_frantext REAL,
            freq_lm10 REAL,
            freq_frwac REAL,
            freq_composite REAL,
            age REAL,
            source TEXT,
            UNIQUE(lemme, cgram)
        );

        -- v5 : sens (definitions)
        CREATE TABLE sens (
            id INTEGER PRIMARY KEY,
            lemme_id INTEGER NOT NULL REFERENCES lemmes(id),
            sens_num INTEGER NOT NULL,
            definition TEXT NOT NULL,
            registre TEXT,
            theme TEXT,
            source TEXT NOT NULL DEFAULT 'wiktionnaire'
        );
        CREATE TABLE sens_exemples (
            sens_id INTEGER NOT NULL REFERENCES sens(id),
            texte TEXT NOT NULL
        );

        -- v5 : entites
        CREATE TABLE entites (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL,
            description TEXT,
            qid TEXT,
            source TEXT NOT NULL DEFAULT 'wikidata',
            type_entite TEXT
        );
        CREATE TABLE entite_lemmes (
            entite_id INTEGER NOT NULL REFERENCES entites(id),
            lemme_id INTEGER NOT NULL REFERENCES lemmes(id),
            type TEXT NOT NULL,
            pertinence INTEGER DEFAULT 1,
            position INTEGER,
            PRIMARY KEY (entite_id, lemme_id)
        );
        CREATE TABLE entite_proprietes (
            entite_id INTEGER NOT NULL REFERENCES entites(id),
            cle TEXT NOT NULL,
            valeur TEXT NOT NULL,
            PRIMARY KEY (entite_id, cle)
        );
        CREATE TABLE entite_categories (
            entite_id INTEGER NOT NULL REFERENCES entites(id),
            categorie_id INTEGER NOT NULL REFERENCES categories(id),
            PRIMARY KEY (entite_id, categorie_id)
        );
        CREATE TABLE entite_synonymes (
            entite_a INTEGER NOT NULL REFERENCES entites(id),
            entite_b INTEGER NOT NULL REFERENCES entites(id),
            source TEXT,
            PRIMARY KEY (entite_a, entite_b)
        );
        CREATE TABLE entite_antonymes (
            entite_a INTEGER NOT NULL REFERENCES entites(id),
            entite_b INTEGER NOT NULL REFERENCES entites(id),
            source TEXT,
            PRIMARY KEY (entite_a, entite_b)
        );
        CREATE TABLE entite_exemples (
            entite_id INTEGER NOT NULL REFERENCES entites(id),
            texte TEXT NOT NULL
        );

        -- Synonymes / antonymes lemme
        CREATE TABLE lemme_synonymes (
            lemme_a INTEGER NOT NULL REFERENCES lemmes(id),
            lemme_b INTEGER NOT NULL REFERENCES lemmes(id),
            source TEXT,
            PRIMARY KEY (lemme_a, lemme_b)
        );
        CREATE TABLE lemme_antonymes (
            lemme_a INTEGER NOT NULL REFERENCES lemmes(id),
            lemme_b INTEGER NOT NULL REFERENCES lemmes(id),
            source TEXT,
            PRIMARY KEY (lemme_a, lemme_b)
        );

        -- Categories
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL UNIQUE,
            type TEXT,
            qid TEXT,
            description TEXT
        );
        CREATE TABLE categorie_hierarchie (
            ancestor_id INTEGER,
            descendant_id INTEGER,
            depth INTEGER,
            PRIMARY KEY (ancestor_id, descendant_id)
        );

        CREATE INDEX idx_formes_ortho ON formes(ortho COLLATE NOCASE);
        CREATE INDEX idx_formes_lemme_id ON formes(lemme_id);
        CREATE INDEX idx_lemmes_lemme ON lemmes(lemme COLLATE NOCASE);
        CREATE INDEX idx_sens_lemme_id ON sens(lemme_id);
        CREATE INDEX idx_entite_lemmes_lemme ON entite_lemmes(lemme_id);
        CREATE INDEX idx_entites_label ON entites(label COLLATE NOCASE);
    """)

    # ── Donnees de test ──────────────────────────────────────────────

    # Lemmes
    conn.executemany(
        "INSERT INTO lemmes (id, lemme, cgram, genre, sous_type, etymologie, forme_developpee, "
        "freq_opensubs, freq_frantext, freq_lm10, freq_frwac, age) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "chat", "NOM", "m", None, "Du latin cattus", None, 57.3, 12.5, 10.0, 20.0, 3.5),
            (2, "manger", "VER", None, None, "Du latin manducare", None, 35.4, 8.0, 6.0, 15.0, 4.0),
            (3, "grand", "ADJ", None, None, "Du latin grandis", None, 120.0, 45.0, 40.0, 80.0, 3.0),
            (4, "paris", "NOM PROPRE", None, "lieu", None, None, 5.0, None, None, None, None),
            (5, "petit", "ADJ", None, None, None, None, 125.0, 50.0, 42.0, 85.0, 3.0),
            (10, "jacques chirac", "NOM PROPRE", None, "personne", None, None, 2.0, None, None, None, None),
            (11, "jacques", "NOM PROPRE", None, "prénom", None, None, 1.0, None, None, None, None),
            (12, "chirac", "NOM PROPRE", None, "patronyme", None, None, 0.5, None, None, None, None),
            (13, "pomme", "NOM", "f", None, None, None, 10.0, 5.0, 4.0, 8.0, 4.0),
            (14, "terre", "NOM", "f", None, None, None, 30.0, 20.0, 15.0, 25.0, 3.0),
            (15, "pomme de terre", "NOM", "f", None, None, None, 3.0, 1.0, 1.0, 2.0, 4.0),
            # Maxime NP pour tester homonymes
            (20, "maxime", "NOM PROPRE", None, "prénom", None, None, 1.0, None, None, None, None),
        ],
    )

    # Formes
    conn.executemany(
        "INSERT INTO formes (id, ortho, lemme_id, multext, phone, phone_reversed, "
        "nb_syllabes, syllabes, freq_opensubs, freq_frantext, freq_lm10, freq_frwac, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "chat", 1, "Ncms", "ʃa", "aʃ", 1, "ʃa", 45.2, 12.5, 10.0, 20.0, "GLAFF"),
            (2, "chats", 1, "Ncmp", "ʃa", "aʃ", 1, "ʃa", 12.1, 3.0, 2.0, 5.0, "GLAFF"),
            (3, "mange", 2, "Vmip3s-", "mɑ̃ʒ", "ʒɑ̃m", 1, "mɑ̃ʒ", 15.4, 5.0, 4.0, 10.0, "GLAFF"),
            (4, "manger", 2, "Vmn----", "mɑ̃ʒe", "eʒɑ̃m", 2, "mɑ̃.ʒe", 20.0, 3.0, 2.0, 5.0, "GLAFF"),
            (5, "grand", 3, "Afpms", "ɡʁɑ̃", "ɑ̃ʁɡ", 1, "ɡʁɑ̃", 80.0, 30.0, 25.0, 50.0, "GLAFF"),
            (6, "grande", 3, "Afpfs", "ɡʁɑ̃d", "dɑ̃ʁɡ", 1, "ɡʁɑ̃d", 60.0, 15.0, 10.0, 20.0, "GLAFF"),
            (7, "paris", 4, "Np", "paʁi", "iʁap", 2, "pa.ʁi", 5.0, None, None, None, "NP"),
            (8, "petit", 5, "Afpms", "pəti", "itəp", 2, "pə.ti", 70.0, 35.0, 30.0, 60.0, "GLAFF"),
            (15, "jacques chirac", 10, "Np", None, None, None, None, 2.0, None, None, None, "NP"),
            (16, "jacques", 11, "Np", None, None, None, None, 1.0, None, None, None, "NP"),
            (17, "chirac", 12, "Np", None, None, None, None, 0.5, None, None, None, "NP"),
            (18, "pomme", 13, "Ncfs", "pɔm", "mɔp", 1, "pɔm", 10.0, 5.0, 4.0, 8.0, "GLAFF"),
            (19, "terre", 14, "Ncfs", "tɛʁ", "ʁɛt", 1, "tɛʁ", 30.0, 20.0, 15.0, 25.0, "GLAFF"),
            (20, "pomme de terre", 15, "Ncfs", "pɔm.də.tɛʁ", "ʁɛt.əd.mɔp", 3, "pɔm.də.tɛʁ", 3.0, 1.0, 1.0, 2.0, "GLAFF"),
            (21, "maxime", 20, "Np", None, None, None, None, 1.0, None, None, None, "NP"),
        ],
    )

    # freq_composite
    for table in ("formes", "lemmes"):
        cur = conn.execute(
            f"SELECT id, freq_opensubs, freq_frantext, freq_lm10, freq_frwac FROM {table}"
        )
        updates = []
        for row in cur.fetchall():
            freqs = [f for f in row[1:] if f and f > 0]
            if freqs:
                composite = math.exp(sum(math.log(f) for f in freqs) / len(freqs))
                updates.append((composite, row[0]))
        if updates:
            conn.executemany(
                f"UPDATE {table} SET freq_composite = ? WHERE id = ?", updates
            )

    # ── Sens (definitions wiktionnaire) ──────────────────────────────
    conn.executemany(
        "INSERT INTO sens (id, lemme_id, sens_num, definition, registre, theme, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, 1, "Petit felin domestique.", None, "zoologie", "wiktionnaire"),
            (2, 1, 2, "Jeu de poursuite entre enfants.", "familier", None, "wiktionnaire"),
            (3, 2, 1, "Macher et avaler un aliment.", None, "cuisine", "wiktionnaire"),
            (4, 3, 1, "De taille elevee.", None, None, "wiktionnaire"),
            (5, 4, 1, "Capitale de la France.", None, "géographie", "wiktionnaire"),
            (6, 4, 2, "Prenom masculin d'origine grecque.", None, None, "wiktionnaire"),
            (7, 5, 1, "De faible taille.", None, None, "wiktionnaire"),
            (9, 13, 1, "Fruit du pommier.", None, "botanique", "wiktionnaire"),
            (10, 14, 1, "Sol sur lequel on marche.", None, "géologie", "wiktionnaire"),
            (11, 15, 1, "Tubercule comestible de la plante Solanum tuberosum.", None, "alimentation", "wiktionnaire"),
            (20, 20, 1, "Prenom masculin.", None, None, "wiktionnaire"),
            (21, 20, 2, "Prenom feminin.", None, None, "wiktionnaire"),
        ],
    )

    # ── Sens exemples ────────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO sens_exemples (sens_id, texte) VALUES (?, ?)",
        [
            (1, "Le chat dort sur le canape."),
            (1, "Un chat de gouttiere."),
            (3, "Il mange une pomme."),
        ],
    )

    # ── Entites (wikidata) ───────────────────────────────────────────
    conn.executemany(
        "INSERT INTO entites (id, label, description, qid, source, type_entite) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (100, "Paris", "capitale de la France", "Q90", "wikidata", "lieu"),
            (101, "Jacques Chirac", "Président de la République française (1995-2007)", "Q2105", "wikidata", "personne"),
            (102, "Maxime", "humoriste français", "Q123", "wikidata", "personne"),
            (103, "Maxime", "film de Henri Verneuil, sorti en 1958", "Q456", "wikidata", "oeuvre"),
            (104, "Maxime Le Forestier N°5", "album de Maxime Le Forestier", "Q789", "wikidata", "oeuvre"),
        ],
    )

    # ── Entite-lemmes ────────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO entite_lemmes (entite_id, lemme_id, type, pertinence, position) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            # Paris : homonyme de lemme paris
            (100, 4, "homonyme", 1, None),
            # Jacques Chirac : composants
            (101, 12, "composant", 1, 1),   # chirac = principal
            (101, 11, "composant", 2, 0),   # jacques = secondaire
            # Maxime (humoriste) : homonyme de lemme maxime
            (102, 20, "homonyme", 1, None),
            # Maxime (film) : homonyme de lemme maxime
            (103, 20, "homonyme", 1, None),
            # Maxime Le Forestier N°5 : composant
            (104, 20, "composant", 2, 0),
        ],
    )

    # ── Entite proprietes ────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO entite_proprietes (entite_id, cle, valeur) VALUES (?, ?, ?)",
        [
            (100, "wikipedia_url", "https://fr.wikipedia.org/wiki/Paris"),
            (100, "image", "Paris_-_Eiffelturm.jpg"),
            (100, "coordonnees", "48.8566,2.3522"),
            (100, "population", "2161000"),
            (100, "pays", "France"),
            (101, "wikipedia_url", "https://fr.wikipedia.org/wiki/Jacques_Chirac"),
            (101, "date_naissance", "1932-11-29"),
            (101, "date_deces", "2019-09-26"),
            (102, "nationalite", "France"),
        ],
    )

    # ── Categories ───────────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO categories (id, label, type, qid) VALUES (?, ?, ?, ?)",
        [
            (1, "animal", "classe", "Q729"),
            (2, "lieu", "classe", "Q17334923"),
            (3, "capitale", "classe", "Q5119"),
            (4, "être vivant", "synthetique", "Q19088"),
            (5, "cinéma", "domaine", None),
        ],
    )
    conn.executemany(
        "INSERT INTO categorie_hierarchie (ancestor_id, descendant_id, depth) VALUES (?, ?, ?)",
        [
            (1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0), (5, 5, 0),
            (4, 1, 1),  # etre vivant → animal
            (2, 3, 1),  # lieu → capitale
        ],
    )

    # ── Entite categories ────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO entite_categories (entite_id, categorie_id) VALUES (?, ?)",
        [
            (100, 2),  # Paris → lieu
            (100, 3),  # Paris → capitale
            (103, 5),  # Maxime (film) → cinema
        ],
    )

    # ── Synonymes lemme ──────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO lemme_synonymes (lemme_a, lemme_b, source) VALUES (?, ?, ?)",
        [
            (3, 5, "wiktionnaire"),  # grand <-> petit (pas des vrais synonymes, juste pour tester)
        ],
    )

    # ── Antonymes lemme ──────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO lemme_antonymes (lemme_a, lemme_b, source) VALUES (?, ?, ?)",
        [
            (3, 5, "wiktionnaire"),  # grand <-> petit
        ],
    )

    # ── Entite exemples ──────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO entite_exemples (entite_id, texte) VALUES (?, ?)",
        [
            (100, "Paris est la plus grande ville de France."),
        ],
    )

    # ── Tables relations semantiques enrichies ──────────────────────
    conn.executescript("""
        CREATE TABLE lemme_hyperonymes (
            lemme_id INTEGER NOT NULL REFERENCES lemmes(id),
            hyperonyme_id INTEGER NOT NULL REFERENCES lemmes(id),
            source TEXT,
            PRIMARY KEY (lemme_id, hyperonyme_id)
        );
        CREATE TABLE lemme_derives (
            lemme_a INTEGER NOT NULL REFERENCES lemmes(id),
            lemme_b INTEGER NOT NULL REFERENCES lemmes(id),
            source TEXT,
            PRIMARY KEY (lemme_a, lemme_b)
        );
        CREATE TABLE lemme_apparentes_sem (
            lemme_a INTEGER NOT NULL REFERENCES lemmes(id),
            lemme_b INTEGER NOT NULL REFERENCES lemmes(id),
            source TEXT,
            PRIMARY KEY (lemme_a, lemme_b)
        );
        CREATE TABLE lemme_proverbes (
            lemme_id INTEGER NOT NULL REFERENCES lemmes(id),
            texte TEXT NOT NULL,
            source TEXT DEFAULT 'wiktionnaire',
            PRIMARY KEY (lemme_id, texte)
        );
    """)

    # Données de test : chat(1) IS-A animal→lemme manquant, on utilise "grand"(3) comme hyperonyme fictif
    # chat(1) hyperonyme → grand(3) (fictif, juste pour le test)
    conn.executemany(
        "INSERT INTO lemme_hyperonymes (lemme_id, hyperonyme_id, source) VALUES (?, ?, ?)",
        [
            (1, 3, "kaikki"),   # chat IS-A grand (fictif)
            (13, 1, "kaikki"),  # pomme IS-A chat (fictif)
        ],
    )

    # Dérivés : chat(1) <-> manger(2)
    conn.executemany(
        "INSERT INTO lemme_derives (lemme_a, lemme_b, source) VALUES (?, ?, ?)",
        [
            (1, 2, "kaikki"),  # chat <-> manger (fictif)
        ],
    )

    # Apparentés sémantiques : chat(1) <-> pomme(13)
    conn.executemany(
        "INSERT INTO lemme_apparentes_sem (lemme_a, lemme_b, source) VALUES (?, ?, ?)",
        [
            (1, 13, "kaikki"),  # chat <-> pomme (fictif)
        ],
    )

    # Proverbes
    conn.executemany(
        "INSERT INTO lemme_proverbes (lemme_id, texte, source) VALUES (?, ?, ?)",
        [
            (1, "Quand le chat n'est pas là, les souris dansent.", "kaikki"),
            (1, "Chat échaudé craint l'eau froide.", "kaikki"),
        ],
    )

    conn.commit()
    conn.close()

    yield db_path

    db_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def lexique_v5(db_v5_path):
    """Lexique charge depuis la BDD v5 de test."""
    with Lexique(db_v5_path) as lex:
        yield lex


# ═══ Detection de version ═══════════════════════════════════════════

def test_schema_version_v5(lexique_v5):
    assert lexique_v5.schema_version == 5


# ═══ Methodes de base (inchangees) ══════════════════════════════════

def test_existe_v5(lexique_v5):
    assert lexique_v5.existe("chat")
    assert lexique_v5.existe("paris")
    assert not lexique_v5.existe("inexistant")


def test_info_v5(lexique_v5):
    entries = lexique_v5.info("chat")
    assert len(entries) > 0
    assert entries[0]["ortho"] == "chat"
    assert entries[0]["cgram"] == "NOM"


def test_phone_de_v5(lexique_v5):
    assert lexique_v5.phone_de("chat") == "ʃa"


# ═══ sens_de() ══════════════════════════════════════════════════════

def test_sens_de(lexique_v5):
    sens = lexique_v5.sens_de("chat", "NOM")
    assert len(sens) == 2
    assert sens[0]["sens_num"] == 1
    assert "felin" in sens[0]["definition"].lower()
    assert sens[1]["sens_num"] == 2


def test_sens_de_alias_concepts_de(lexique_v5):
    """concepts_de() est un alias de sens_de() en v5."""
    sens = lexique_v5.concepts_de("chat", "NOM")
    assert len(sens) == 2
    assert "felin" in sens[0]["definition"].lower()


def test_sens_de_inexistant(lexique_v5):
    assert lexique_v5.sens_de("inexistant") == []


# ═══ definitions() (v5) ════════════════════════════════════════════

def test_definitions_v5(lexique_v5):
    defs = lexique_v5.definitions("chat", "NOM")
    assert len(defs) == 2
    assert defs[0]["domaine"] == "zoologie"
    assert defs[1]["domaine"] == ""


def test_definitions_v5_exemples(lexique_v5):
    defs = lexique_v5.definitions("chat", "NOM")
    assert len(defs[0]["exemples"]) == 2
    assert any("canape" in ex for ex in defs[0]["exemples"])


def test_definitions_v5_synonymes(lexique_v5):
    """En v5, synonymes sont au niveau lemme (partages entre tous les sens)."""
    defs = lexique_v5.definitions("grand", "ADJ")
    assert len(defs) == 1
    assert "petit" in defs[0]["synonymes"]


def test_definitions_v5_antonymes(lexique_v5):
    defs = lexique_v5.definitions("grand", "ADJ")
    assert "petit" in defs[0]["antonymes"]


# ═══ exemples_de() ═════════════════════════════════════════════════

def test_exemples_de_v5(lexique_v5):
    exemples = lexique_v5.exemples_de(1)  # sens_id=1 (chat sens 1)
    assert len(exemples) == 2
    assert any("canape" in ex for ex in exemples)


def test_exemples_de_v5_vide(lexique_v5):
    exemples = lexique_v5.exemples_de(99999)
    assert exemples == []


# ═══ synonymes_de() / antonymes_de() (v5 = lemme-level) ═══════════

def test_synonymes_de_v5(lexique_v5):
    """En v5, synonymes_de() prend un lemme_id."""
    syns = lexique_v5.synonymes_de(3)  # lemme grand (id=3)
    assert len(syns) == 1
    assert syns[0]["_lemme"] == "petit"


def test_antonymes_de_v5(lexique_v5):
    ants = lexique_v5.antonymes_de(3)  # lemme grand (id=3)
    assert len(ants) == 1
    assert ants[0]["_lemme"] == "petit"


# ═══ entites_associees() ═══════════════════════════════════════════

def test_entites_homonymes(lexique_v5):
    """entites_associees() avec type='homonyme' retourne les entites mono-mot."""
    ents = lexique_v5.entites_associees("maxime", type_lien="homonyme")
    assert len(ents) == 2  # humoriste + film
    labels = {e["label"] for e in ents}
    assert "Maxime" in labels


def test_entites_composants(lexique_v5):
    """entites_associees() avec type='composant' retourne les entites multi-mot."""
    ents = lexique_v5.entites_associees("maxime", type_lien="composant")
    assert len(ents) == 1  # Maxime Le Forestier N°5
    assert "Forestier" in ents[0]["label"]


def test_entites_associees_toutes(lexique_v5):
    """Sans filtre, retourne homonymes + composants."""
    ents = lexique_v5.entites_associees("maxime")
    assert len(ents) == 3  # 2 homonymes + 1 composant


def test_entites_associees_alias_concepts_associes(lexique_v5):
    """concepts_associes() est un alias en v5."""
    ents = lexique_v5.concepts_associes("maxime")
    assert len(ents) == 3


def test_entites_associees_inexistant(lexique_v5):
    assert lexique_v5.entites_associees("inexistant") == []


# ═══ categories_de() ═══════════════════════════════════════════════

def test_categories_de_entite_v5(lexique_v5):
    """categories_de() lit entite_categories en v5."""
    cats = lexique_v5.categories_de(100)  # Paris
    assert "lieu" in cats
    assert "capitale" in cats


def test_categories_de_avec_ancetres_v5(lexique_v5):
    """Propagation des ancetres via closure table."""
    cats = lexique_v5.categories_de(100, inclure_ancetres=True)
    assert "lieu" in cats
    assert "capitale" in cats


def test_categories_de_sans_ancetres_v5(lexique_v5):
    cats = lexique_v5.categories_de(100, inclure_ancetres=False)
    assert "lieu" in cats
    assert "capitale" in cats


# ═══ entites_par_categorie() ═══════════════════════════════════════

def test_entites_par_categorie_v5(lexique_v5):
    ents = lexique_v5.entites_par_categorie("lieu")
    assert len(ents) >= 1
    assert any(e["label"] == "Paris" for e in ents)


def test_entites_par_categorie_descendants_v5(lexique_v5):
    """Avec inclure_descendants, inclut les sous-categories."""
    ents = lexique_v5.entites_par_categorie("lieu", inclure_descendants=True)
    assert any(e["label"] == "Paris" for e in ents)


def test_entites_par_categorie_alias(lexique_v5):
    """concepts_par_categorie() est un alias."""
    ents = lexique_v5.concepts_par_categorie("lieu")
    assert len(ents) >= 1


# ═══ proprietes_entite() ═══════════════════════════════════════════

def test_proprietes_entite_v5(lexique_v5):
    props = lexique_v5.proprietes_entite(100)  # Paris
    assert props["wikipedia_url"] == "https://fr.wikipedia.org/wiki/Paris"
    assert props["image"] == "Paris_-_Eiffelturm.jpg"
    assert props["coordonnees"] == "48.8566,2.3522"


def test_proprietes_entite_alias(lexique_v5):
    """proprietes_concept() est un alias."""
    props = lexique_v5.proprietes_concept(100)
    assert props["wikipedia_url"] == "https://fr.wikipedia.org/wiki/Paris"


def test_proprietes_entite_vide(lexique_v5):
    assert lexique_v5.proprietes_entite(99999) == {}


# ═══ entite_detail() ═══════════════════════════════════════════════

def test_entite_detail_v5(lexique_v5):
    detail = lexique_v5.entite_detail(100)  # Paris
    assert detail is not None
    assert detail["label"] == "Paris"
    assert detail["description"] == "capitale de la France"
    assert detail["qid"] == "Q90"
    assert detail["type_entite"] == "lieu"


def test_entite_detail_proprietes(lexique_v5):
    detail = lexique_v5.entite_detail(100)
    props = detail["_proprietes"]
    assert props["pays"] == "France"
    assert props["population"] == "2161000"


def test_entite_detail_categories(lexique_v5):
    detail = lexique_v5.entite_detail(100)
    assert "lieu" in detail["_categories"]
    assert "capitale" in detail["_categories"]


def test_entite_detail_composants(lexique_v5):
    """entite_detail() inclut les lemmes lies."""
    detail = lexique_v5.entite_detail(101)  # Jacques Chirac
    assert detail is not None
    comps = detail["_composants"]
    assert len(comps) == 2
    lemmes = {c["comp_lemme"] for c in comps}
    assert "jacques" in lemmes
    assert "chirac" in lemmes


def test_entite_detail_exemples(lexique_v5):
    detail = lexique_v5.entite_detail(100)
    assert len(detail["_exemples"]) == 1
    assert "France" in detail["_exemples"][0]


def test_entite_detail_alias(lexique_v5):
    """concept_detail() est un alias."""
    detail = lexique_v5.concept_detail(100)
    assert detail is not None
    assert detail["label"] == "Paris"


def test_entite_detail_inexistant(lexique_v5):
    assert lexique_v5.entite_detail(99999) is None


# ═══ rechercher_entites() ══════════════════════════════════════════

def test_rechercher_entites_par_label(lexique_v5):
    results = lexique_v5.rechercher_entites("Paris")
    assert len(results) >= 1
    assert any(r["label"] == "Paris" for r in results)


def test_rechercher_entites_par_description(lexique_v5):
    results = lexique_v5.rechercher_entites("humoriste")
    assert len(results) >= 1
    assert any("humoriste" in (r.get("description") or "") for r in results)


def test_rechercher_entites_multi_mots(lexique_v5):
    results = lexique_v5.rechercher_entites("Jacques Chirac")
    assert len(results) >= 1


def test_rechercher_entites_via_composants(lexique_v5):
    """Recherche via entite_lemmes quand la recherche directe ne suffit pas."""
    results = lexique_v5.rechercher_entites("chirac")
    # Jacques Chirac doit etre trouve via entite_lemmes
    assert any("Chirac" in r["label"] for r in results)


def test_rechercher_entites_alias(lexique_v5):
    """rechercher_concepts() est un alias."""
    results = lexique_v5.rechercher_concepts("Paris")
    assert len(results) >= 1


def test_rechercher_entites_inexistant(lexique_v5):
    assert lexique_v5.rechercher_entites("xyztotoinexistant") == []


# ═══ lemmes_apparentes() ═══════════════════════════════════════════

def test_lemmes_apparentes(lexique_v5):
    app = lexique_v5.lemmes_apparentes("pomme")
    lemmes = [a["lemme"] for a in app]
    assert "pomme de terre" in lemmes


def test_lemmes_apparentes_inexistant(lexique_v5):
    assert lexique_v5.lemmes_apparentes("xyztotoinexistant") == []


# ═══ exemples_entite() ═════════════════════════════════════════════

def test_exemples_entite(lexique_v5):
    exemples = lexique_v5.exemples_entite(100)
    assert len(exemples) == 1
    assert "France" in exemples[0]


def test_exemples_entite_vide(lexique_v5):
    assert lexique_v5.exemples_entite(99999) == []


# ═══ Methodes categories (inchangees) ══════════════════════════════

def test_info_categorie_v5(lexique_v5):
    cat = lexique_v5.info_categorie("animal")
    assert cat is not None
    assert cat["type"] == "classe"


def test_lister_categories_v5(lexique_v5):
    cats = lexique_v5.lister_categories()
    assert len(cats) == 5


def test_ancetres_categorie_v5(lexique_v5):
    anc = lexique_v5.ancetres_categorie("animal")
    labels = [a["label"] for a in anc]
    assert "être vivant" in labels


def test_descendants_categorie_v5(lexique_v5):
    desc = lexique_v5.descendants_categorie("être vivant")
    labels = [d["label"] for d in desc]
    assert "animal" in labels


# ═══ hyperonymes_de() / hyponymes_de() ════════════════════════════════

def test_hyperonymes_de(lexique_v5):
    """hyperonymes_de() retourne les hyperonymes d'un lemme."""
    hypers = lexique_v5.hyperonymes_de(1)  # chat
    assert len(hypers) == 1
    assert hypers[0]["_lemme"] == "grand"


def test_hyponymes_de(lexique_v5):
    """hyponymes_de() retourne les hyponymes (lookup inverse)."""
    hypos = lexique_v5.hyponymes_de(1)  # chat est hyperonyme de pomme
    assert len(hypos) == 1
    assert hypos[0]["_lemme"] == "pomme"


# ═══ derives_de() ═════════════════════════════════════════════════════

def test_derives_de(lexique_v5):
    """derives_de() retourne les termes derives (bidirectionnel)."""
    derives = lexique_v5.derives_de(1)  # chat
    assert len(derives) == 1
    assert derives[0]["_lemme"] == "manger"
    # Test bidirectionnel
    derives2 = lexique_v5.derives_de(2)  # manger
    assert len(derives2) == 1
    assert derives2[0]["_lemme"] == "chat"


# ═══ apparentes_sem() ═════════════════════════════════════════════════

def test_apparentes_sem(lexique_v5):
    """apparentes_sem() retourne les termes apparentes (bidirectionnel)."""
    app = lexique_v5.apparentes_sem(1)  # chat
    assert len(app) == 1
    assert app[0]["_lemme"] == "pomme"
    # Bidirectionnel
    app2 = lexique_v5.apparentes_sem(13)  # pomme
    assert len(app2) == 1
    assert app2[0]["_lemme"] == "chat"


# ═══ proverbes_de() ═════════════════════════════════════���═════════════

def test_proverbes_de(lexique_v5):
    """proverbes_de() retourne les proverbes lies a un lemme."""
    prov = lexique_v5.proverbes_de(1)  # chat
    assert len(prov) == 2
    assert any("souris" in p for p in prov)
    assert any("eau froide" in p for p in prov)


# ═══ Graceful degradation (table absente) ═════════════════════════════

def test_relations_table_absente():
    """Les methodes retournent [] si les tables n'existent pas."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE formes (
            id INTEGER PRIMARY KEY,
            ortho TEXT NOT NULL,
            lemme_id INTEGER,
            multext TEXT,
            phone TEXT,
            phone_reversed TEXT,
            nb_syllabes INTEGER,
            syllabes TEXT,
            freq_opensubs REAL,
            freq_frantext REAL,
            freq_lm10 REAL,
            freq_frwac REAL,
            freq_composite REAL,
            source TEXT
        );
        CREATE TABLE lemmes (
            id INTEGER PRIMARY KEY,
            lemme TEXT NOT NULL,
            cgram TEXT NOT NULL,
            UNIQUE(lemme, cgram)
        );
        CREATE TABLE sens (
            id INTEGER PRIMARY KEY,
            lemme_id INTEGER NOT NULL,
            sens_num INTEGER NOT NULL,
            definition TEXT NOT NULL,
            registre TEXT,
            theme TEXT,
            source TEXT NOT NULL DEFAULT 'wiktionnaire',
            entite_liee_id INTEGER
        );
        CREATE INDEX idx_formes_ortho ON formes(ortho COLLATE NOCASE);
        INSERT INTO lemmes (id, lemme, cgram) VALUES (1, 'test', 'NOM');
        INSERT INTO formes (id, ortho, lemme_id, multext) VALUES (1, 'test', 1, 'Ncms');
    """)
    conn.commit()
    conn.close()

    try:
        with Lexique(db_path) as lex:
            assert lex.hyperonymes_de(1) == []
            assert lex.hyponymes_de(1) == []
            assert lex.derives_de(1) == []
            assert lex.apparentes_sem(1) == []
            assert lex.proverbes_de(1) == []
    finally:
        db_path.unlink(missing_ok=True)


# ═══ Idempotence INSERT OR IGNORE ═════════════════════════════════════

def test_insert_or_ignore_idempotent(db_v5_path):
    """INSERT OR IGNORE ne cree pas de doublon."""
    conn = sqlite3.connect(str(db_v5_path))
    # Re-inserer les memes donnees
    conn.execute(
        "INSERT OR IGNORE INTO lemme_hyperonymes (lemme_id, hyperonyme_id, source) "
        "VALUES (1, 3, 'kaikki')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO lemme_derives (lemme_a, lemme_b, source) "
        "VALUES (1, 2, 'kaikki')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO lemme_apparentes_sem (lemme_a, lemme_b, source) "
        "VALUES (1, 13, 'kaikki')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO lemme_proverbes (lemme_id, texte, source) "
        "VALUES (1, 'Quand le chat n''est pas là, les souris dansent.', 'kaikki')"
    )
    conn.commit()

    # Verifier pas de doublon
    assert conn.execute("SELECT COUNT(*) FROM lemme_hyperonymes WHERE lemme_id=1 AND hyperonyme_id=3").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM lemme_derives WHERE lemme_a=1 AND lemme_b=2").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM lemme_apparentes_sem WHERE lemme_a=1 AND lemme_b=13").fetchone()[0] == 1
    n_prov = conn.execute("SELECT COUNT(*) FROM lemme_proverbes WHERE lemme_id=1").fetchone()[0]
    assert n_prov == 2  # Toujours 2, pas de doublon
    conn.close()
