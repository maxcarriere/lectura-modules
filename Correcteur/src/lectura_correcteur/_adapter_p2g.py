"""Adaptateur P2G (Phoneme-to-Grapheme) pour le correcteur V3.

Encapsule le moteur lectura_graphemiseur pour fournir une interface
simple au pipeline de correction par roundtrip phonetique.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class P2GAdapter:
    """Adaptateur pour le moteur P2G (lectura-graphemiseur).

    Fournit une interface simplifiee pour le roundtrip phonetique :
    IPA -> orthographe avec alternatives et confiances.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def transcrire(
        self,
        ipa_words: list[str],
        ortho_words: list[str] | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Transcrit des mots IPA en orthographe avec alternatives.

        Args:
            ipa_words: liste de mots en IPA (un par mot).
            ortho_words: formes orthographiques courantes (pour lex_features).
            k: nombre d'alternatives a retourner par mot.

        Returns:
            list[dict] par mot avec :
                - ortho: str (top-1)
                - confiance: float
                - alternatives: list[tuple[str, float]]
                - pos: str
        """
        if not ipa_words:
            return []

        try:
            result = self._engine.analyser_avec_alternatives(
                ipa_words, ortho_words=ortho_words, k=k,
            )
        except TypeError:
            # Ancienne API sans ortho_words
            result = self._engine.analyser_avec_alternatives(ipa_words, k=k)

        n = len(ipa_words)
        out: list[dict[str, Any]] = []

        ortho_list = result.get("ortho", [])
        confiance_list = result.get("confiance", [])
        alternatives_list = result.get("alternatives", [])
        pos_list = result.get("pos", [])

        for i in range(n):
            d: dict[str, Any] = {
                "ortho": ortho_list[i] if i < len(ortho_list) else "",
                "confiance": confiance_list[i] if i < len(confiance_list) else 0.0,
                "alternatives": alternatives_list[i] if i < len(alternatives_list) else [],
                "pos": pos_list[i] if i < len(pos_list) else "",
            }
            out.append(d)

        return out

    def transcrire_complet(
        self,
        ipa_words: list[str],
        ortho_words: list[str] | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        """Transcription complete avec morpho.

        Appelle le moteur P2G et retourne le dict brut incluant
        ortho, confiance, alternatives, pos, morpho, morpho_scores, pos_scores.

        Args:
            ipa_words: liste de mots en IPA.
            ortho_words: formes orthographiques (None pour mode sans hint).
            k: nombre d'alternatives par mot.

        Returns:
            dict avec cles: ortho, confiance, alternatives, pos,
            morpho (dict[str, list[str]]), morpho_scores, pos_scores
        """
        if not ipa_words:
            return {
                "ortho": [], "confiance": [], "alternatives": [],
                "pos": [], "morpho": {}, "morpho_scores": {}, "pos_scores": [],
            }

        try:
            result = self._engine.analyser_avec_alternatives(
                ipa_words, ortho_words=ortho_words, k=k,
            )
        except TypeError:
            # Ancienne API sans ortho_words
            result = self._engine.analyser_avec_alternatives(ipa_words, k=k)

        return result

    def transcrire_mot(
        self,
        ipa: str,
        ortho: str | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        """Transcrit un seul mot IPA (raccourci)."""
        results = self.transcrire(
            [ipa],
            ortho_words=[ortho] if ortho else None,
            k=k,
        )
        return results[0] if results else {
            "ortho": "", "confiance": 0.0, "alternatives": [], "pos": "",
        }


def creer_adapter_p2g(
    model_dir: str | Path | None = None,
    *,
    lex_select: bool | None = None,
    lex_threshold: float | None = None,
) -> P2GAdapter | None:
    """Factory : cree un P2GAdapter si le Graphemiseur est disponible.

    Cascade de resolution :
    1. lectura_graphemiseur installe → creer_engine(mode="onnx")
    2. None si indisponible (degradation gracieuse)

    Args:
        model_dir: repertoire des modeles P2G (None = auto-detect).
        lex_select: activer le lex_select P2G (None = ne pas modifier).
        lex_threshold: seuil softmax pour lex_select (None = ne pas modifier).

    Returns:
        P2GAdapter ou None
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        logger.info("onnxruntime non installe — P2G indisponible")
        return None

    # creer_engine() gere V6 + phone_lexicon automatiquement
    try:
        from lectura_graphemiseur import creer_engine
        kwargs: dict = {"mode": "onnx"}
        if model_dir is not None:
            kwargs["models_dir"] = model_dir
        engine = creer_engine(**kwargs)

        # Overrides lex_select si demande par le correcteur
        if lex_select is not None:
            engine.apply_lex_select = lex_select
        if lex_threshold is not None:
            engine.LEX_SELECT_THRESHOLD = lex_threshold

        logger.info(
            "P2G charge via creer_engine() (lex_select=%s, threshold=%s)",
            getattr(engine, "apply_lex_select", "?"),
            getattr(engine, "LEX_SELECT_THRESHOLD", "?"),
        )
        return P2GAdapter(engine)
    except Exception as e:
        logger.debug("lectura_graphemiseur indisponible: %s", e)

    logger.info("P2G indisponible")
    return None
