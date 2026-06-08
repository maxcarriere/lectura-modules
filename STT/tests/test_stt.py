"""Tests unitaires pour lectura-stt.

Couvre :
- TestParseCTC : parsing de sorties CTC (mots, liaisons, elisions, ponctuation)
- TestAssembler : reconstruction de texte (majuscules, elisions, ponctuation)
- TestSTTEngine : integration avec mocks CTC + P2G
- TestFactory : cascade de creation (avec/sans P2G)
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import numpy as np

from lectura_stt._parse_ctc import parse_ctc_output, parse_ctc_v2, ParseResult
from lectura_stt._assembler import assembler_texte, rejoin_elisions, _commence_par_voyelle
from lectura_stt import STTEngine, ResultatSTT


# ── TestParseCTC ─────────────────────────────────────────────────────


class TestParseCTC(unittest.TestCase):
    """Tests du parsing de sortie CTC."""

    def test_phrase_simple(self):
        """Mots separes par |."""
        r = parse_ctc_output("b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d")
        self.assertEqual(r.mots_ipa, ["bɔ̃ʒuʁ", "lə", "mɔ̃d"])
        self.assertEqual(r.ponctuation_finale, "")
        self.assertEqual(r.liaisons, ["", "", ""])

    def test_phrase_avec_point(self):
        """Ponctuation finale."""
        r = parse_ctc_output("b ɔ̃ ʒ u ʁ .")
        self.assertEqual(r.mots_ipa, ["bɔ̃ʒuʁ"])
        self.assertEqual(r.ponctuation_finale, ".")

    def test_phrase_avec_question(self):
        r = parse_ctc_output("k ɔ m ɑ̃ ?")
        self.assertEqual(r.mots_ipa, ["kɔmɑ̃"])
        self.assertEqual(r.ponctuation_finale, "?")

    def test_phrase_avec_exclamation(self):
        r = parse_ctc_output("a t ɑ̃ s j ɔ̃ !")
        self.assertEqual(r.mots_ipa, ["atɑ̃sjɔ̃"])
        self.assertEqual(r.ponctuation_finale, "!")

    def test_liaison_z(self):
        """Liaison [z] entre mots."""
        r = parse_ctc_output("l e [z] ɑ̃ f ɑ̃")
        self.assertEqual(r.mots_ipa, ["le", "ɑ̃fɑ̃"])
        self.assertEqual(r.liaisons, ["", "z"])

    def test_liaison_t(self):
        """Liaison [t] entre mots."""
        r = parse_ctc_output("ɛ [t] i l")
        self.assertEqual(r.mots_ipa, ["ɛ", "il"])
        self.assertEqual(r.liaisons, ["", "t"])

    def test_liaison_n(self):
        """Liaison [n] entre mots."""
        r = parse_ctc_output("ɔ̃ [n] a")
        self.assertEqual(r.mots_ipa, ["ɔ̃", "a"])
        self.assertEqual(r.liaisons, ["", "n"])

    def test_elision(self):
        """Elision ['] entre clitique et mot."""
        r = parse_ctc_output("l ['] a m i")
        self.assertEqual(r.mots_ipa, ["l'ami"])
        self.assertEqual(len(r.liaisons), 1)

    def test_elision_je(self):
        """Elision j' + ai."""
        r = parse_ctc_output("ʒ ['] e")
        self.assertEqual(r.mots_ipa, ["ʒ'e"])

    def test_elision_multi_phones_split(self):
        """CTC fusionne pronom + clitique : 'ɛ m [']' → separe en ["ɛ", "m'..."]."""
        r = parse_ctc_output("ɛ m ['] a p ɛ l")
        self.assertEqual(r.mots_ipa, ["ɛ", "m'apɛl"])
        self.assertEqual(len(r.liaisons), 2)

    def test_elision_multi_phones_tu(self):
        """CTC fusionne tu + m' : 't y m [']' → ["ty", "m'..."]."""
        r = parse_ctc_output("t y m ['] a p ɛ l")
        self.assertEqual(r.mots_ipa, ["ty", "m'apɛl"])

    def test_elision_multi_phones_il(self):
        """CTC fusionne il + s' : 'i l s [']' → ["il", "s'..."]."""
        r = parse_ctc_output("i l s ['] a p ɛ l")
        self.assertEqual(r.mots_ipa, ["il", "s'apɛl"])

    def test_elision_single_phone_unchanged(self):
        """Un seul phone avant ['] reste identique (pas de split)."""
        r = parse_ctc_output("l ['] o m")
        self.assertEqual(r.mots_ipa, ["l'om"])

    def test_mot_compose(self):
        """Mot compose avec [-]."""
        r = parse_ctc_output("ɡ ʁ ɑ̃ [-] p ɛ ʁ")
        self.assertEqual(r.mots_ipa, ["ɡʁɑ̃", "pɛʁ"])

    def test_points_suspension(self):
        """… normalise en ."""
        r = parse_ctc_output("b ɔ̃ …")
        self.assertEqual(r.ponctuation_finale, ".")

    def test_chaine_vide(self):
        r = parse_ctc_output("")
        self.assertEqual(r.mots_ipa, [])
        self.assertEqual(r.ponctuation_finale, "")

    def test_espaces_seulement(self):
        r = parse_ctc_output("   ")
        self.assertEqual(r.mots_ipa, [])

    def test_phrase_complete(self):
        """Phrase complexe avec liaison et ponctuation."""
        r = parse_ctc_output("i l [z] ɔ̃ | m ɑ̃ ʒ e .")
        self.assertEqual(r.mots_ipa, ["il", "ɔ̃", "mɑ̃ʒe"])
        self.assertEqual(r.liaisons, ["", "z", ""])
        self.assertEqual(r.ponctuation_finale, ".")


# ── TestAssembler ────────────────────────────────────────────────────


class TestAssembler(unittest.TestCase):
    """Tests de la reconstruction de texte."""

    def test_phrase_simple(self):
        r = assembler_texte(["bonjour", "le", "monde"])
        self.assertEqual(r, "Bonjour le monde")

    def test_avec_ponctuation(self):
        r = assembler_texte(["bonjour", "le", "monde"], ".")
        self.assertEqual(r, "Bonjour le monde.")

    def test_majuscule_initiale(self):
        r = assembler_texte(["il", "mange"])
        self.assertEqual(r, "Il mange")

    def test_deja_majuscule(self):
        r = assembler_texte(["Paris", "est", "beau"])
        self.assertEqual(r, "Paris est beau")

    def test_elision_le(self):
        """le + voyelle → l'."""
        r = assembler_texte(["le", "ami"])
        self.assertEqual(r, "L'ami")

    def test_elision_la(self):
        r = assembler_texte(["la", "eau"])
        self.assertEqual(r, "L'eau")

    def test_elision_de(self):
        r = assembler_texte(["de", "abord"])
        self.assertEqual(r, "D'abord")

    def test_elision_je(self):
        r = assembler_texte(["je", "ai"])
        self.assertEqual(r, "J'ai")

    def test_elision_ne(self):
        r = assembler_texte(["ne", "est"])
        self.assertEqual(r, "N'est")

    def test_elision_se(self):
        r = assembler_texte(["il", "se", "est"])
        self.assertEqual(r, "Il s'est")

    def test_elision_que(self):
        r = assembler_texte(["que", "il"])
        self.assertEqual(r, "Qu'il")

    def test_pas_elision_consonne(self):
        """Pas d'elision devant consonne."""
        r = assembler_texte(["le", "chat"])
        self.assertEqual(r, "Le chat")

    def test_pas_elision_h_aspire(self):
        """Pas d'elision devant h aspire."""
        r = assembler_texte(["le", "hibou"])
        self.assertEqual(r, "Le hibou")

    def test_elision_h_muet(self):
        """Elision devant h muet."""
        r = assembler_texte(["le", "homme"])
        self.assertEqual(r, "L'homme")

    def test_liste_vide(self):
        self.assertEqual(assembler_texte([]), "")

    def test_mots_vides(self):
        self.assertEqual(assembler_texte(["", "", ""]), "")

    def test_question(self):
        r = assembler_texte(["comment"], "?")
        self.assertEqual(r, "Comment?")


class TestCommenceParVoyelle(unittest.TestCase):
    """Tests de la detection voyelle initiale."""

    def test_voyelle(self):
        self.assertTrue(_commence_par_voyelle("ami"))
        self.assertTrue(_commence_par_voyelle("eau"))
        self.assertTrue(_commence_par_voyelle("il"))
        self.assertTrue(_commence_par_voyelle("un"))

    def test_consonne(self):
        self.assertFalse(_commence_par_voyelle("chat"))
        self.assertFalse(_commence_par_voyelle("monde"))

    def test_h_muet(self):
        self.assertTrue(_commence_par_voyelle("homme"))
        self.assertTrue(_commence_par_voyelle("heure"))

    def test_h_aspire(self):
        self.assertFalse(_commence_par_voyelle("hibou"))
        self.assertFalse(_commence_par_voyelle("hache"))

    def test_vide(self):
        self.assertFalse(_commence_par_voyelle(""))


# ── TestSTTEngine ────────────────────────────────────────────────────


class TestSTTEngine(unittest.TestCase):
    """Tests d'integration avec mocks."""

    def _make_engine(
        self,
        ipa_output: str,
        p2g_output: list[str] | None = None,
        use_pipeline: bool = False,
    ):
        """Cree un STTEngine avec des mocks.

        Si use_pipeline=True, simule lectura_p2g.analyser (pipeline complet).
        Sinon, utilise le graphemiseur seul.
        """
        ctc = MagicMock()
        ctc.transcrire.return_value = ipa_output

        p2g = None
        p2g_analyser = None
        if p2g_output is not None:
            p2g = MagicMock()
            p2g.analyser.return_value = {"ortho": p2g_output}
            if use_pipeline:
                p2g_analyser = MagicMock(return_value={"ortho": p2g_output})

        return STTEngine(ctc, p2g, p2g_analyser)

    def test_transcription_sans_p2g(self):
        """Sans P2G, texte et mots sont None."""
        engine = self._make_engine("b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d .")
        audio = np.zeros(16000, dtype=np.float32)
        r = engine.transcrire(audio)

        self.assertEqual(r.ipa, "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d .")
        self.assertEqual(r.mots_ipa, ["bɔ̃ʒuʁ", "lə", "mɔ̃d"])
        self.assertIsNone(r.texte)
        self.assertIsNone(r.mots)
        self.assertEqual(r.ponctuation, ["."])

    def test_transcription_avec_p2g(self):
        """Avec P2G, texte et mots sont remplis."""
        engine = self._make_engine(
            "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d .",
            ["bonjour", "le", "monde"],
        )
        audio = np.zeros(16000, dtype=np.float32)
        r = engine.transcrire(audio)

        self.assertEqual(r.ipa, "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d .")
        self.assertEqual(r.mots_ipa, ["bɔ̃ʒuʁ", "lə", "mɔ̃d"])
        self.assertEqual(r.mots, ["bonjour", "le", "monde"])
        self.assertEqual(r.texte, "Bonjour le monde.")

    def test_transcription_vide(self):
        """Audio vide → resultat vide."""
        engine = self._make_engine("")
        audio = np.zeros(1600, dtype=np.float32)
        r = engine.transcrire(audio)

        self.assertEqual(r.ipa, "")
        self.assertEqual(r.mots_ipa, [])
        self.assertIsNone(r.texte)

    def test_batch(self):
        """transcrire_batch appelle transcrire pour chaque audio."""
        engine = self._make_engine("b ɔ̃ ʒ u ʁ")
        audios = [np.zeros(16000, dtype=np.float32) for _ in range(3)]
        results = engine.transcrire_batch(audios)

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.mots_ipa, ["bɔ̃ʒuʁ"])

    def test_repr(self):
        engine = self._make_engine("test")
        self.assertIn("STTEngine", repr(engine))
        self.assertIn("MagicMock", repr(engine))

    def test_p2g_avec_elision(self):
        """Elision CTC ['] → le P2G recoit les mots separes."""
        engine = self._make_engine(
            "l ['] a m i",
            ["le", "ami"],
        )
        audio = np.zeros(16000, dtype=np.float32)
        r = engine.transcrire(audio)

        # Le P2G recoit ["l", "ami"] (separes par _p2g_convertir)
        self.assertEqual(r.texte, "L'ami")

    def test_p2g_sans_ponctuation(self):
        """Sans ponctuation finale."""
        engine = self._make_engine(
            "b ɔ̃ ʒ u ʁ",
            ["bonjour"],
        )
        audio = np.zeros(16000, dtype=np.float32)
        r = engine.transcrire(audio)

        self.assertEqual(r.texte, "Bonjour")
        self.assertEqual(r.ponctuation, [])

    def test_pipeline_utilise_p2g_analyser(self):
        """Avec pipeline P2G, _p2g_analyser est appele (pas le graphemiseur)."""
        engine = self._make_engine(
            "b ɔ̃ ʒ u ʁ .",
            ["bonjour"],
            use_pipeline=True,
        )
        audio = np.zeros(16000, dtype=np.float32)
        r = engine.transcrire(audio)

        engine._p2g_analyser.assert_called_once()
        engine.p2g.analyser.assert_not_called()
        self.assertEqual(r.texte, "Bonjour.")

    def test_repr_avec_pipeline(self):
        """Le repr indique +formules si pipeline disponible."""
        engine = self._make_engine("test", ["test"], use_pipeline=True)
        self.assertIn("+formules", repr(engine))


# ── TestFactory ──────────────────────────────────────────────────────


class TestFactory(unittest.TestCase):
    """Tests de la cascade de creation."""

    @patch("lectura_stt.creer_engine.__module__", "lectura_stt")
    def test_factory_sans_p2g(self):
        """Si P2G non installe, engine fonctionne sans."""
        with patch("lectura_stt._creer_p2g", return_value=(None, None)):
            with patch("lectura_ctc.creer_engine") as mock_ctc:
                mock_ctc.return_value = MagicMock()
                from lectura_stt import creer_engine
                engine = creer_engine()
                self.assertIsInstance(engine, STTEngine)
                self.assertIsNone(engine.p2g)
                self.assertIsNone(engine._p2g_analyser)



# ── TestParseCTCv2 ────────────────────────────────────────────────


class TestParseCTCv2(unittest.TestCase):
    """Tests de parse_ctc_v2 (segments enrichis)."""

    def test_phrase_simple(self):
        segs = parse_ctc_v2("b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d")
        word_segs = [s for s in segs if s["type"] == "word"]
        self.assertEqual(len(word_segs), 3)
        self.assertEqual(word_segs[0]["ipa"], "bɔ̃ʒuʁ")
        self.assertEqual(word_segs[1]["ipa"], "lə")
        self.assertEqual(word_segs[2]["ipa"], "mɔ̃d")

    def test_liaison(self):
        segs = parse_ctc_v2("l e [z] ɑ̃ f ɑ̃")
        word_segs = [s for s in segs if s["type"] == "word"]
        self.assertEqual(len(word_segs), 2)
        self.assertNotIn("liaison_before", word_segs[0])
        self.assertEqual(word_segs[1].get("liaison_before"), "z")

    def test_elision(self):
        segs = parse_ctc_v2("l ['] a m i")
        word_segs = [s for s in segs if s["type"] == "word"]
        self.assertEqual(len(word_segs), 2)
        self.assertTrue(word_segs[0].get("is_clitic"))
        self.assertTrue(word_segs[1].get("elision_before"))

    def test_compound(self):
        segs = parse_ctc_v2("ɡ ʁ ɑ̃ [-] p ɛ ʁ")
        word_segs = [s for s in segs if s["type"] == "word"]
        self.assertEqual(len(word_segs), 2)
        self.assertTrue(word_segs[0].get("compound_after"))

    def test_ponctuation(self):
        segs = parse_ctc_v2("b ɔ̃ ʒ u ʁ .")
        punct_segs = [s for s in segs if s["type"] == "punct"]
        self.assertEqual(len(punct_segs), 1)
        self.assertEqual(punct_segs[0]["value"], ".")

    def test_elision_multi_phones_split(self):
        """CTC fusionne mot + clitique : 'ɛ m [']' → mot 'ɛ' + clitique 'm'."""
        segs = parse_ctc_v2("ɛ m ['] a p ɛ l")
        word_segs = [s for s in segs if s["type"] == "word"]
        self.assertEqual(len(word_segs), 3)
        # "ɛ" est un mot normal (pas clitique)
        self.assertEqual(word_segs[0]["ipa"], "ɛ")
        self.assertFalse(word_segs[0].get("is_clitic", False))
        # "m" est le clitique
        self.assertEqual(word_segs[1]["ipa"], "m")
        self.assertTrue(word_segs[1].get("is_clitic"))
        # "apɛl" recoit elision_before
        self.assertEqual(word_segs[2]["ipa"], "apɛl")
        self.assertTrue(word_segs[2].get("elision_before"))

    def test_elision_single_phone_v2(self):
        """Un seul phone avant ['] reste clitique sans split."""
        segs = parse_ctc_v2("l ['] a m i")
        word_segs = [s for s in segs if s["type"] == "word"]
        self.assertEqual(len(word_segs), 2)
        self.assertEqual(word_segs[0]["ipa"], "l")
        self.assertTrue(word_segs[0].get("is_clitic"))
        self.assertEqual(word_segs[1]["ipa"], "ami")

    def test_vide(self):
        self.assertEqual(parse_ctc_v2(""), [])
        self.assertEqual(parse_ctc_v2("   "), [])


# ── TestRejoinElisions ────────────────────────────────────────────


class TestRejoinElisions(unittest.TestCase):
    """Tests de rejoin_elisions."""

    def test_elision_devant_voyelle(self):
        """Clitique l + mot IPA commencant par voyelle → l'ami."""
        r = rejoin_elisions(["le", "ami"], ["l", "ami"])
        self.assertEqual(r, "L'ami")

    def test_clitique_devant_consonne(self):
        """Clitique l + mot commencant par consonne → le chat."""
        r = rejoin_elisions(["le", "chat"], ["l", "ʃa"])
        self.assertEqual(r, "Le chat")

    def test_pas_clitique(self):
        """Mot normal sans clitique IPA."""
        r = rejoin_elisions(["bonjour", "le", "monde"], ["bɔ̃ʒuʁ", "lə", "mɔ̃d"])
        self.assertEqual(r, "Bonjour le monde")

    def test_compose_tiret(self):
        """Mots composes joints par tiret."""
        r = rejoin_elisions(
            ["grand", "pere"], ["ɡʁɑ̃", "pɛʁ"],
            compound_joins={0},
        )
        self.assertEqual(r, "Grand-pere")

    def test_liste_vide(self):
        self.assertEqual(rejoin_elisions([], []), "")


if __name__ == "__main__":
    unittest.main()
