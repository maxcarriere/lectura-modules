"""Tagger hybride : G2P contextuel + overrides mots-outils + boost non-ambigus.

Combine le meilleur du G2P Unifie V2 (POS contextuel sur mots de contenu)
et du LexiqueTagger (POS fiable sur mots-outils via _FUNCTION_WORD_POS).
Les mots non-ambigus (1 seul POS, pas homophone, pas de voisin frequent)
recoivent une confiance elevee (0.98).
"""

from __future__ import annotations

import re
from typing import Any

from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS, _TOKEN_RE

# Alphabet francais pour la generation de voisins d=1 (substitutions)
_ALPHABET_FR = "abcdefghijklmnopqrstuvwxyzàâäéèêëïîôùûüÿçœæ"


class TaggerHybride:
    """Tagger hybride : G2P comme base + overrides mots-outils + boost confiance.

    Satisfait TaggerProtocol + G2PProtocol.
    """

    def __init__(
        self,
        g2p_tagger: Any,
        lexique: Any,
        lm_homophones: Any | None = None,
        seuil_freq_voisin: float = 50.0,
    ) -> None:
        self._g2p = g2p_tagger
        self._lexique = lexique
        self._lm_homophones = lm_homophones
        self._seuil_freq_voisin = seuil_freq_voisin
        self._cache_ambiguite: dict[str, bool] = {}

    # -- TaggerProtocol --

    def tokenize(self, text: str) -> list[tuple[str, bool]]:
        """Tokenise via regex (meme regex que LexiqueTagger / G2PUnifieAdapter)."""
        result: list[tuple[str, bool]] = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group()
            is_word = tok[0].isalpha() or tok[0] == "_"
            result.append((tok, is_word))
        return result

    def tag_words(self, words: list[str]) -> list[dict]:
        """Tague chaque mot (interface simple)."""
        tags = self._g2p.tag_words(words)
        self._appliquer_overrides(words, tags)
        return tags

    def tag_words_rich(self, words: list[str]) -> list[dict]:
        """Tague avec scores POS enrichis + overrides + boost confiance."""
        tags = self._g2p.tag_words_rich(words)
        self._appliquer_overrides(words, tags)
        self._boost_unambigus(words, tags)
        return tags

    def tag_words_dual(self, words: list[str]) -> list[dict]:
        """Double tagging (delegue au G2P) + overrides + boost."""
        if not hasattr(self._g2p, "tag_words_dual"):
            return self.tag_words_rich(words)
        tags = self._g2p.tag_words_dual(words)
        self._appliquer_overrides(words, tags)
        self._boost_unambigus(words, tags)
        return tags

    # -- G2PProtocol --

    def g2p(self, word: str) -> str:
        """Retourne l'IPA d'un mot (delegue au G2P)."""
        if hasattr(self._g2p, "g2p"):
            return self._g2p.g2p(word)
        return ""

    def prononcer(self, mot: str) -> str | None:
        """Alias de g2p() pour satisfaire l'interface _candidats.py."""
        if hasattr(self._g2p, "prononcer"):
            return self._g2p.prononcer(mot)
        r = self.g2p(mot)
        return r if r else None

    # -- Overrides mots-outils --

    def _appliquer_overrides(self, words: list[str], tags: list[dict]) -> None:
        """Force le POS des mots-outils avec confiance=1.0."""
        for i, word in enumerate(words):
            if i >= len(tags):
                break
            override = _FUNCTION_WORD_POS.get(word.lower())
            if override is not None:
                tags[i]["pos"] = override
                tags[i]["confiance_pos"] = 1.0
                # Mettre a jour pos_scores si present
                if "pos_scores" in tags[i]:
                    tags[i]["pos_scores"] = [(override, 1.0)]

    # -- Boost confiance non-ambigus --

    def _boost_unambigus(self, words: list[str], tags: list[dict]) -> None:
        """Boost confiance_pos a 0.98 pour les mots non-ambigus."""
        for i, word in enumerate(words):
            if i >= len(tags):
                break
            # Ne pas re-booster les overrides (deja a 1.0)
            if _FUNCTION_WORD_POS.get(word.lower()) is not None:
                continue
            conf = tags[i].get("confiance_pos", 0.0)
            if conf < 0.98 and self._est_unambigue(word):
                tags[i]["confiance_pos"] = 0.98

    def _est_unambigue(self, mot: str) -> bool:
        """True si 1 seul POS dans le lexique, pas homophone, pas de voisin frequent."""
        low = mot.lower()
        if low in self._cache_ambiguite:
            return self._cache_ambiguite[low]

        # 1. POS unique dans le lexique ?
        infos = self._lexique.info(low) if hasattr(self._lexique, "info") else []
        if not infos:
            self._cache_ambiguite[low] = False  # OOV = pas de certitude
            return False
        cgrams = {e.get("cgram") for e in infos if e.get("cgram")}
        if len(cgrams) != 1:
            self._cache_ambiguite[low] = False
            return False

        # 2. Pas homophone ?
        if self._lm_homophones and self._lm_homophones.est_homophone(low):
            self._cache_ambiguite[low] = False
            return False

        # 3. Pas de voisin tres frequent a distance 1 ?
        if self._a_voisin_frequent(low):
            self._cache_ambiguite[low] = False
            return False

        self._cache_ambiguite[low] = True
        return True

    def _a_voisin_frequent(self, mot: str) -> bool:
        """True si un voisin d=1 (substitution) existe avec freq > seuil."""
        seuil = self._seuil_freq_voisin
        n = len(mot)
        for i in range(n):
            prefix = mot[:i]
            suffix = mot[i + 1:]
            for c in _ALPHABET_FR:
                if c == mot[i]:
                    continue
                variante = prefix + c + suffix
                if (
                    hasattr(self._lexique, "frequence")
                    and self._lexique.existe(variante)
                    and self._lexique.frequence(variante) > seuil
                ):
                    return True
        return False
