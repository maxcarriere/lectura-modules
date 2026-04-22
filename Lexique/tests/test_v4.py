"""Tests pour le schema v4 : detection, methodes v4, NP unifies."""

import math
import sqlite3
import tempfile
from pathlib import Path

import pytest

from lectura_lexique import Lexique


@pytest.fixture(scope="module")
def db_v4_path():
    """Cree une mini BDD v4 de test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    # Schema v4
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
            genre TEXT,
            contrainte_nombre TEXT,
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
        CREATE TABLE concepts (
            id INTEGER PRIMARY KEY,
            lemme_id INTEGER,
            sens_num INTEGER,
            definition TEXT,
            registre TEXT,
            theme TEXT,
            illustrable REAL,
            synset_id TEXT,
            qid TEXT,
            source TEXT
        );
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
        CREATE TABLE concept_categories (
            concept_id INTEGER,
            categorie_id INTEGER,
            PRIMARY KEY (concept_id, categorie_id)
        );
        CREATE TABLE concept_synonymes (
            concept_a INTEGER,
            concept_b INTEGER,
            source TEXT,
            PRIMARY KEY (concept_a, concept_b)
        );
        CREATE TABLE concept_antonymes (
            concept_a INTEGER,
            concept_b INTEGER,
            source TEXT,
            PRIMARY KEY (concept_a, concept_b)
        );
        CREATE TABLE concept_hyperonymes (
            concept_id INTEGER,
            hyperonyme_id INTEGER,
            source TEXT,
            PRIMARY KEY (concept_id, hyperonyme_id)
        );
        CREATE TABLE concept_exemples (
            id INTEGER PRIMARY KEY,
            concept_id INTEGER,
            exemple TEXT,
            source TEXT
        );
        CREATE TABLE concept_composants (
            concept_id INTEGER,
            lemme_id INTEGER,
            role TEXT,
            position INTEGER
        );

        CREATE INDEX idx_formes_ortho ON formes(ortho COLLATE NOCASE);
        CREATE INDEX idx_formes_lemme_id ON formes(lemme_id);
        CREATE INDEX idx_formes_phone ON formes(phone);
        CREATE INDEX idx_lemmes_lemme ON lemmes(lemme COLLATE NOCASE);
    """)

    # Donnees de test

    # Lemmes
    conn.executemany(
        "INSERT INTO lemmes (id, lemme, cgram, genre, sous_type, etymologie, forme_developpee, freq_opensubs, freq_frantext, freq_lm10, freq_frwac, age) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "chat", "NOM", "m", None, "Du latin cattus", None, 57.3, 12.5, 10.0, 20.0, 3.5),
            (2, "manger", "VER", None, None, "Du latin manducare", None, 35.4, 8.0, 6.0, 15.0, 4.0),
            (3, "grand", "ADJ", None, None, "Du latin grandis", None, 120.0, 45.0, 40.0, 80.0, 3.0),
            (4, "paris", "NOM PROPRE", None, "lieu", None, None, 5.0, None, None, None, None),
            (5, "petit", "ADJ", None, None, None, None, 125.0, 50.0, 42.0, 85.0, 3.0),
            (6, "sur", "ADJ", None, None, None, None, 2951.37, 0.6, 0.67, 8.49, None),
            (7, "sur", "PRE", None, None, None, None, 2951.37, 5300.0, 5525.0, 6610.0, None),
            (8, "sncf", "SIGLE", "f", None, None, "Société nationale des chemins de fer français", None, None, None, None, None),
            (9, "sida", "SIGLE", "m", None, None, None, None, None, None, None, None),
        ],
    )

    # Formes
    conn.executemany(
        "INSERT INTO formes (id, ortho, lemme_id, multext, phone, phone_reversed, nb_syllabes, syllabes, freq_opensubs, freq_frantext, freq_lm10, freq_frwac, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "chat", 1, "Ncms", "ʃa", "aʃ", 1, "ʃa", 45.2, 12.5, 10.0, 20.0, "GLAFF"),
            (2, "chats", 1, "Ncmp", "ʃa", "aʃ", 1, "ʃa", 12.1, 3.0, 2.0, 5.0, "GLAFF"),
            (3, "mange", 2, "Vmip3s-", "mɑ̃ʒ", "ʒɑ̃m", 1, "mɑ̃ʒ", 15.4, 5.0, 4.0, 10.0, "GLAFF"),
            (4, "manger", 2, "Vmn----", "mɑ̃ʒe", "eʒɑ̃m", 2, "mɑ̃.ʒe", 20.0, 3.0, 2.0, 5.0, "GLAFF"),
            (5, "grand", 3, "Afpms", "ɡʁɑ̃", "ɑ̃ʁɡ", 1, "ɡʁɑ̃", 80.0, 30.0, 25.0, 50.0, "GLAFF"),
            (6, "grande", 3, "Afpfs", "ɡʁɑ̃d", "dɑ̃ʁɡ", 1, "ɡʁɑ̃d", 60.0, 15.0, 10.0, 20.0, "GLAFF"),
            (7, "grands", 3, "Afpmp", "ɡʁɑ̃", "ɑ̃ʁɡ", 1, "ɡʁɑ̃", 40.0, 10.0, 5.0, 10.0, "GLAFF"),
            (8, "paris", 4, "Np", "paʁi", "iʁap", 2, "pa.ʁi", 5.0, None, None, None, "NP"),
            (9, "petit", 5, "Afpms", "pəti", "itəp", 2, "pə.ti", 70.0, 35.0, 30.0, 60.0, "GLAFF"),
            (10, "petite", 5, "Afpfs", "pətit", "titəp", 2, "pə.tit", 55.0, 15.0, 12.0, 25.0, "GLAFF"),
            (11, "sur", 6, "Afpms", "syʁ", "ʁys", 1, "syʁ", 2951.37, 0.6, 0.67, 8.49, "GLAFF"),
            (12, "sur", 7, "Sp", "syʁ", "ʁys", 1, "syʁ", 2951.37, 5300.0, 5525.0, 6610.0, "GLAFF"),
            (13, "SNCF", 8, "Ys", "ɛs.ɛn.se.ɛf", "ɛf.se.ɛn.ɛs", 4, "ɛs.ɛn.se.ɛf", None, None, None, None, "kaikki"),
            (14, "SIDA", 9, "Ya", "si.da", "ad.is", 2, "si.da", None, None, None, None, "kaikki"),
        ],
    )

    # Calculer freq_composite (moyenne geometrique des frequences disponibles)
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

    # Concepts
    conn.executemany(
        "INSERT INTO concepts (id, lemme_id, sens_num, definition, registre, theme, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, 1, "Petit felin domestique.", None, "zoologie", "wiktionnaire"),
            (2, 1, 2, "Jeu de poursuite entre enfants.", "familier", None, "wiktionnaire"),
            (3, 2, 1, "Macher et avaler un aliment.", None, "cuisine", "wiktionnaire"),
            (4, 3, 1, "De taille elevee.", None, None, "wiktionnaire"),
            (5, 4, 1, "Capitale de la France.", None, "géographie", "wiktionnaire"),
            (6, 4, 2, "Prenom masculin d'origine grecque.", None, None, "wiktionnaire"),
            (7, 5, 1, "De faible taille.", None, None, "wiktionnaire"),
        ],
    )

    # Synonymes : grand(4) <-> grand(4) pas utile, mais grand <-> vaste... simulons
    # On fait : petit(7) synonyme avec un concept hypothetique (on le skip)
    # Antonymes : grand(4) <-> petit(7)
    conn.execute(
        "INSERT INTO concept_antonymes (concept_a, concept_b, source) VALUES (4, 7, 'wiktionnaire')",
    )

    # Exemples
    conn.executemany(
        "INSERT INTO concept_exemples (concept_id, exemple, source) VALUES (?, ?, ?)",
        [
            (1, "Le chat dort sur le canape.", "wiktionnaire"),
            (1, "Un chat de gouttiere.", "wiktionnaire"),
            (3, "Il mange une pomme.", "wiktionnaire"),
        ],
    )

    # Categories avec type et qid
    conn.executemany(
        "INSERT INTO categories (id, label, type, qid) VALUES (?, ?, ?, ?)",
        [
            (1, "animal", "classe", "Q729"),
            (2, "lieu", "classe", "Q17334923"),
            (3, "capitale", "classe", "Q5119"),
            (4, "être vivant", "synthetique", "Q19088"),
        ],
    )
    conn.executemany(
        "INSERT INTO concept_categories (concept_id, categorie_id) VALUES (?, ?)",
        [(1, 1), (5, 2), (5, 3)],
    )

    # Hierarchie (closure table)
    conn.executemany(
        "INSERT INTO categorie_hierarchie (ancestor_id, descendant_id, depth) VALUES (?, ?, ?)",
        [
            (1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0),  # self
            (4, 1, 1),  # être vivant → animal
            (2, 3, 1),  # lieu → capitale
        ],
    )

    conn.commit()
    conn.close()

    yield db_path

    db_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def lexique_v4(db_v4_path):
    """Lexique charge depuis la BDD v4 de test."""
    with Lexique(db_v4_path) as lex:
        yield lex


# --- Detection de version ---

def test_schema_version_v4(lexique_v4):
    assert lexique_v4.schema_version == 4


def test_schema_version_v3(lexique_csv):
    """Le backend CSV n'a pas de version v4."""
    assert lexique_csv.schema_version == 3


# --- Methodes existantes avec v4 ---

def test_existe_v4(lexique_v4):
    assert lexique_v4.existe("chat")
    assert lexique_v4.existe("paris")
    assert not lexique_v4.existe("inexistant")


def test_info_v4(lexique_v4):
    entries = lexique_v4.info("chat")
    assert len(entries) > 0
    e = entries[0]
    assert e["ortho"] == "chat"
    assert e["lemme"] == "chat"
    assert e["cgram"] == "NOM"


def test_info_v4_freq_frantext(lexique_v4):
    """info() retourne freq_frantext distinct par POS."""
    entries = lexique_v4.info("sur")
    assert len(entries) == 2
    freqs = {e["cgram"]: e["freq_frantext"] for e in entries}
    assert freqs["ADJ"] == pytest.approx(0.6)
    assert freqs["PRE"] == pytest.approx(5300.0)


def test_info_v4_freq_frantext_chat(lexique_v4):
    """info() retourne freq_frantext pour un mot simple."""
    entries = lexique_v4.info("chat")
    assert len(entries) == 1
    assert entries[0]["freq_frantext"] == pytest.approx(12.5)


def test_info_v4_freq_lm10(lexique_v4):
    """info() retourne freq_lm10 distinct par POS."""
    entries = lexique_v4.info("sur")
    freqs = {e["cgram"]: e["freq_lm10"] for e in entries}
    assert freqs["ADJ"] == pytest.approx(0.67)
    assert freqs["PRE"] == pytest.approx(5525.0)


def test_info_v4_freq_frwac(lexique_v4):
    """info() retourne freq_frwac distinct par POS."""
    entries = lexique_v4.info("sur")
    freqs = {e["cgram"]: e["freq_frwac"] for e in entries}
    assert freqs["ADJ"] == pytest.approx(8.49)
    assert freqs["PRE"] == pytest.approx(6610.0)


def test_info_v4_decoded_multext(lexique_v4):
    """info() v4 decode le multext en traits lisibles."""
    entries = lexique_v4.info("mange")
    assert len(entries) > 0
    e = entries[0]
    assert e["mode"] == "indicatif"
    assert e["temps"] == "present"
    assert e["personne"] == "3"
    assert e["nombre"] == "singulier"


def test_phone_de_v4(lexique_v4):
    assert lexique_v4.phone_de("chat") == "ʃa"


def test_homophones_v4(lexique_v4):
    results = lexique_v4.homophones("ʃa")
    orthos = [r["ortho"] for r in results]
    assert "chat" in orthos
    assert "chats" in orthos


def test_formes_de_v4(lexique_v4):
    formes = lexique_v4.formes_de("chat")
    orthos = [f["ortho"] for f in formes]
    assert "chat" in orthos
    assert "chats" in orthos


def test_formes_de_adj_v4(lexique_v4):
    formes = lexique_v4.formes_de("grand", "ADJ")
    orthos = [f["ortho"] for f in formes]
    assert "grand" in orthos
    assert "grande" in orthos
    assert "grands" in orthos


# --- Methodes v4-only ---

def test_info_lemme(lexique_v4):
    result = lexique_v4.info_lemme("chat", "NOM")
    assert result is not None
    assert result["lemme"] == "chat"
    assert result["cgram"] == "NOM"
    assert result["genre"] == "m"
    assert result["etymologie"] == "Du latin cattus"


def test_info_lemme_sous_type(lexique_v4):
    result = lexique_v4.info_lemme("paris", "NOM PROPRE")
    assert result is not None
    assert result["sous_type"] == "lieu"


def test_info_lemme_inexistant(lexique_v4):
    result = lexique_v4.info_lemme("inexistant")
    assert result is None


def test_concepts_de(lexique_v4):
    concepts = lexique_v4.concepts_de("chat", "NOM")
    assert len(concepts) == 2
    assert concepts[0]["sens_num"] == 1
    assert "felin" in concepts[0]["definition"].lower()
    assert concepts[1]["sens_num"] == 2


def test_concepts_de_np(lexique_v4):
    concepts = lexique_v4.concepts_de("paris", "NOM PROPRE")
    assert len(concepts) == 2
    assert "capitale" in concepts[0]["definition"].lower()


def test_antonymes_de(lexique_v4):
    # grand(concept 4) <-> petit(concept 7)
    ants = lexique_v4.antonymes_de(4)
    assert len(ants) > 0
    # Le concept 7 (petit) doit etre dans les antonymes de 4 (grand)
    ant_ids = [a["id"] for a in ants]
    assert 7 in ant_ids


def test_exemples_de(lexique_v4):
    exemples = lexique_v4.exemples_de(1)
    assert len(exemples) == 2
    assert any("canape" in ex for ex in exemples)


def test_categories_de(lexique_v4):
    cats = lexique_v4.categories_de(5)  # paris sens 1 = lieu, capitale
    assert "lieu" in cats
    assert "capitale" in cats


def test_decoder_multext_method(lexique_v4):
    result = lexique_v4.decoder_multext("Vmip3s")
    assert result["pos"] == "VER"
    assert result["mode"] == "indicatif"


# --- NP unifies ---

def test_existe_nom_propre_v4(lexique_v4):
    assert lexique_v4.existe_nom_propre("paris")


def test_info_nom_propre_v4(lexique_v4):
    entries = lexique_v4.info_nom_propre("paris")
    assert len(entries) > 0
    assert entries[0]["cgram"] == "NOM PROPRE"


def test_phone_nom_propre_v4(lexique_v4):
    assert lexique_v4.phone_nom_propre("paris") == "paʁi"


# --- Theme et categories ---

def test_concept_theme(lexique_v4):
    """concepts_de() retourne le theme quand il est present."""
    concepts = lexique_v4.concepts_de("chat", "NOM")
    assert len(concepts) >= 2
    assert concepts[0]["theme"] == "zoologie"
    assert concepts[1]["theme"] is None  # sens 2 : pas de theme


def test_definitions_domaine_from_theme(lexique_v4):
    """definitions() mappe theme -> domaine dans le dict retourne."""
    defs = lexique_v4.definitions("chat", "NOM")
    assert len(defs) >= 2
    assert defs[0]["domaine"] == "zoologie"
    assert defs[1]["domaine"] == ""  # pas de theme -> chaine vide


# --- Categories : type, qid, hierarchie ---

def test_categorie_type(db_v4_path):
    """Les categories ont un type (classe, domaine, synthetique)."""
    conn = sqlite3.connect(str(db_v4_path))
    cur = conn.execute("SELECT label, type FROM categories WHERE type IS NOT NULL")
    rows = {row[0]: row[1] for row in cur}
    conn.close()
    assert "animal" in rows
    assert rows["animal"] == "classe"
    assert rows["être vivant"] == "synthetique"


def test_categorie_hierarchie_self(db_v4_path):
    """Chaque categorie est ancetre d'elle-meme a depth=0."""
    conn = sqlite3.connect(str(db_v4_path))
    cur = conn.execute(
        "SELECT COUNT(*) FROM categorie_hierarchie WHERE ancestor_id = descendant_id AND depth = 0"
    )
    nb_self = cur.fetchone()[0]
    cur2 = conn.execute("SELECT COUNT(*) FROM categories")
    nb_cats = cur2.fetchone()[0]
    conn.close()
    assert nb_self == nb_cats


def test_categorie_ancestors(db_v4_path):
    """'animal' a 'être vivant' comme ancetre."""
    conn = sqlite3.connect(str(db_v4_path))
    cur = conn.execute("""
        SELECT c.label, h.depth FROM categorie_hierarchie h
        JOIN categories c ON c.id = h.ancestor_id
        JOIN categories d ON d.id = h.descendant_id
        WHERE d.label = 'animal' AND h.depth > 0
    """)
    ancestors = {row[0]: row[1] for row in cur}
    conn.close()
    assert "être vivant" in ancestors
    assert ancestors["être vivant"] == 1


def test_categorie_descendants(db_v4_path):
    """'être vivant' a 'animal' comme descendant."""
    conn = sqlite3.connect(str(db_v4_path))
    cur = conn.execute("""
        SELECT c.label, h.depth FROM categorie_hierarchie h
        JOIN categories c ON c.id = h.descendant_id
        JOIN categories a ON a.id = h.ancestor_id
        WHERE a.label = 'être vivant' AND h.depth > 0
    """)
    descendants = {row[0]: row[1] for row in cur}
    conn.close()
    assert "animal" in descendants
    assert descendants["animal"] == 1


# --- API categories (methodes Lexique) ---

def test_info_categorie(lexique_v4):
    """info_categorie() retourne les champs d'une categorie."""
    cat = lexique_v4.info_categorie("animal")
    assert cat is not None
    assert cat["label"] == "animal"
    assert cat["type"] == "classe"
    assert cat["qid"] == "Q729"


def test_info_categorie_inexistante(lexique_v4):
    """info_categorie() retourne None pour un label inconnu."""
    assert lexique_v4.info_categorie("inexistant") is None


def test_lister_categories(lexique_v4):
    """lister_categories() retourne toutes les categories, filtrage par type."""
    toutes = lexique_v4.lister_categories()
    assert len(toutes) == 4
    # Filtrage par type
    classes = lexique_v4.lister_categories(type="classe")
    assert all(c["type"] == "classe" for c in classes)
    assert len(classes) == 3  # animal, lieu, capitale
    synth = lexique_v4.lister_categories(type="synthetique")
    assert len(synth) == 1
    assert synth[0]["label"] == "être vivant"


def test_ancetres_categorie(lexique_v4):
    """ancetres_categorie() remonte la hierarchie."""
    ancetres = lexique_v4.ancetres_categorie("animal")
    labels = {a["label"]: a["depth"] for a in ancetres}
    assert "être vivant" in labels
    assert labels["être vivant"] == 1


def test_descendants_categorie(lexique_v4):
    """descendants_categorie() descend la hierarchie."""
    desc = lexique_v4.descendants_categorie("être vivant")
    labels = {d["label"]: d["depth"] for d in desc}
    assert "animal" in labels
    assert labels["animal"] == 1


def test_concepts_par_categorie(lexique_v4):
    """concepts_par_categorie() retourne les concepts lies."""
    concepts = lexique_v4.concepts_par_categorie("animal")
    assert len(concepts) >= 1
    # Le concept chat (sens 1) est lie a la categorie animal
    assert any("felin" in c["definition"].lower() for c in concepts)
    # Sans descendants : "être vivant" n'a pas de concepts directs
    concepts_ev = lexique_v4.concepts_par_categorie("être vivant")
    assert len(concepts_ev) == 0
    # Avec descendants : "être vivant" inclut les concepts d'"animal"
    concepts_ev_desc = lexique_v4.concepts_par_categorie(
        "être vivant", inclure_descendants=True
    )
    assert len(concepts_ev_desc) >= 1


# --- freq_composite ---

def test_info_v4_freq_composite(lexique_v4):
    """info() retourne freq_composite et l'utilise comme entry['freq']."""
    entries = lexique_v4.info("chat")
    assert len(entries) == 1
    e = entries[0]
    # freq_composite = geomean(45.2, 12.5, 10.0, 20.0)
    expected = math.exp(
        (math.log(45.2) + math.log(12.5) + math.log(10.0) + math.log(20.0)) / 4
    )
    assert e["freq_composite"] == pytest.approx(expected, rel=1e-6)
    # entry["freq"] doit utiliser freq_composite en priorite
    assert e["freq"] == pytest.approx(expected, rel=1e-6)


def test_rechercher_tri_freq_composite(lexique_v4):
    """rechercher() trie les resultats par freq_composite decroissant."""
    # Pattern ".*a.*" matche chat, chats, mange, manger, grand, grande, grands, paris
    results = lexique_v4.rechercher("^.*a.*$", limite=50)
    assert len(results) >= 3
    # Verifier l'ordre decroissant de freq_composite
    freqs = [r.get("freq_composite") or r.get("freq_opensubs", 0.0) or 0.0
             for r in results]
    for i in range(len(freqs) - 1):
        assert freqs[i] >= freqs[i + 1], (
            f"Resultats non tries par freq_composite: "
            f"{results[i]['ortho']}({freqs[i]}) avant {results[i+1]['ortho']}({freqs[i+1]})"
        )


def test_info_v4_freq_composite_partial(lexique_v4):
    """freq_composite avec une seule source (NP : opensubs seul)."""
    entries = lexique_v4.info("paris")
    assert len(entries) == 1
    e = entries[0]
    # paris n'a que freq_opensubs=5.0, donc composite=5.0
    assert e["freq_composite"] == pytest.approx(5.0, rel=1e-6)


# --- Sigles ---

def test_existe_sigle(lexique_v4):
    """Un sigle en majuscules est trouve par existe()."""
    assert lexique_v4.existe("SNCF")


def test_info_sigle(lexique_v4):
    """info() retourne les traits corrects pour un sigle."""
    entries = lexique_v4.info("SNCF")
    assert len(entries) > 0
    e = entries[0]
    assert e["cgram"] == "SIGLE"
    assert e["multext"] == "Ys"


def test_decoder_multext_sigle():
    """decoder_multext('Ys') retourne sigle epele."""
    from lectura_lexique._multext import decoder_multext
    result = decoder_multext("Ys")
    assert result["pos"] == "SIGLE"
    assert result["sous_type"] == "sigle"


def test_decoder_multext_acronyme():
    """decoder_multext('Ya') retourne acronyme."""
    from lectura_lexique._multext import decoder_multext
    result = decoder_multext("Ya")
    assert result["pos"] == "SIGLE"
    assert result["sous_type"] == "acronyme"


def test_rechercher_sigle(lexique_v4):
    """rechercher() avec un pattern trouve les sigles."""
    results = lexique_v4.rechercher("^SN")
    orthos = [r["ortho"] for r in results]
    assert "SNCF" in orthos


# --- definitions() avec categories ---

def test_definitions_categories(lexique_v4):
    """definitions() inclut les categories de chaque sens."""
    defs = lexique_v4.definitions("chat", "NOM")
    assert len(defs) >= 2
    # Sens 1 (felin domestique) est lie a la categorie "animal"
    assert "animal" in defs[0]["categories"]
    # Sens 2 (jeu) n'a pas de categorie
    assert defs[1]["categories"] == []


# --- rechercher_concepts() ---

def test_rechercher_concepts(lexique_v4):
    """rechercher_concepts() par mot-cle dans la definition."""
    results = lexique_v4.rechercher_concepts("felin")
    assert len(results) >= 1
    assert any("felin" in c["definition"].lower() for c in results)


def test_rechercher_concepts_par_lemme(lexique_v4):
    """rechercher_concepts() par mot-cle dans le lemme."""
    results = lexique_v4.rechercher_concepts("chat")
    assert len(results) >= 2  # chat a 2 concepts


def test_rechercher_concepts_inexistant(lexique_v4):
    """rechercher_concepts() retourne [] pour un terme inexistant."""
    results = lexique_v4.rechercher_concepts("xyztotoinexistant")
    assert results == []


# --- Fixture CSV pour test v3 compat ---

@pytest.fixture(scope="module")
def lexique_csv():
    """Lexique CSV pour tester la compat v3."""
    from pathlib import Path
    test_csv = Path(__file__).parent / "donnees" / "test.csv"
    with Lexique(test_csv) as lex:
        yield lex
