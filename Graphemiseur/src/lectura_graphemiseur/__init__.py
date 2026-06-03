"""Lectura P2G — Modele unifie P2G+POS+Morpho V6 pour le francais (IPA -> orthographe).

Architecture : BiLSTM char-level + word feedback multi-tete V6
  avec phone_lex_features (28d) + lex_select (2.56M params, ONNX INT8 = 2.6 Mo)

Pipeline complet : raw → lex_select → post-traitement (formules + coherence morpho + accents)
  Word accuracy : 90.95% (dev set, pipeline complet avec formules)

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md

Quatre backends d'inference au choix :
  - ONNX Runtime  (rapide, ~2ms/phrase)   — necessite modeles locaux
  - NumPy         (leger, ~50ms/phrase)    — necessite modeles locaux
  - Pure Python   (zero dependance)        — necessite modeles locaux
  - API           (serveur Lectura)        — mode par defaut (Niveau 1)

Exemple rapide (mode API, zero config)::

    from lectura_graphemiseur import creer_engine
    engine = creer_engine()
    result = engine.analyser(["b\u0254\u0303\u0292u\u0281"])
    print(result["ortho"])  # ['bonjour']

Exemple avec backend local::

    from lectura_graphemiseur import creer_engine
    engine = creer_engine(mode="local")
    result = engine.analyser(["l\u0259", "\u0283a", "\u025b", "b\u0254\u0303"])
    print(result["ortho"])  # ['le', 'chat', 'est', 'bon']
"""

import logging
import os
from pathlib import Path

__version__ = "4.1.1"

logger = logging.getLogger(__name__)

_MODELES_DIR = Path(__file__).parent / "modeles"


def get_model_path(filename: str) -> Path:
    """Retourne le chemin absolu vers un fichier modele embarque."""
    return _MODELES_DIR / filename


def _resoudre_modeles_dir(models_dir: str | Path | None = None) -> Path | None:
    """Cascade de resolution du dossier modeles.

    Ordre de priorite :
    1. Parametre explicite ``models_dir``
    2. Variable d'environnement ``LECTURA_MODELS_DIR`` / p2g
    3. Dossier utilisateur ``~/.lectura/models/p2g/``
    4. Site-packages (dossier ``modeles/`` du package installe)
    """
    candidats: list[Path] = []
    if models_dir is not None:
        candidats.append(Path(models_dir))
    env = os.environ.get("LECTURA_MODELS_DIR")
    if env:
        candidats.append(Path(env) / "p2g")
    candidats.append(Path.home() / ".lectura" / "models" / "p2g")
    candidats.append(_MODELES_DIR)

    for d in candidats:
        if (d / "unifie_p2g_v6_int8.onnx").exists() or \
           (d / "unifie_p2g_v5_int8.onnx").exists() or \
           (d / "unifie_p2g_v3_int8.onnx").exists():
            return d
    return None


def _resoudre_lexique(
    lexicon_path: str | Path | None,
    models_dir: Path | None,
) -> str | Path | dict | None:
    """Cascade de resolution du lexique pour les lex_features.

    1. Chemin explicite (si passe)
    2. Fichier JSON dans le dossier modeles resolu
    3. lectura_lexique installe -> generer le dict
    4. None (pas de lexique -> lex_features = zeros)
    """
    if lexicon_path is not None:
        return lexicon_path
    if models_dir is not None:
        json_path = models_dir / "lexique_pos_candidates.json"
        if json_path.exists():
            return json_path
    try:
        from lectura_lexique import Lexique
        lex = Lexique()
        result: dict[str, list[str]] = {}
        for mot, entrees in lex._index.items():
            pos_set: set[str] = set()
            for e in entrees:
                if hasattr(e, "pos") and e.pos:
                    pos_set.add(e.pos)
            if pos_set:
                result[mot] = sorted(pos_set)
        return result
    except (ImportError, Exception):
        pass
    return None


def _resoudre_phone_lexicon(models_dir: Path | None):
    """Cascade de resolution du phone_lexicon pour V6.

    1. phone_lexicon.db dans le dossier modeles
    2. lexique_correcteur.db dans le dossier modeles
    3. lectura_correcteur data dir (si installe)
    4. None (degradation gracieuse, features = zeros)
    """
    from lectura_graphemiseur._phone_lexicon import PhoneLexicon

    # 1. DB dediee dans le dossier modeles
    if models_dir is not None:
        db = models_dir / "phone_lexicon.db"
        if db.exists():
            try:
                return PhoneLexicon(db)
            except Exception as e:
                logger.warning("phone_lexicon.db present mais erreur: %s", e)

    # 2. DB correcteur dans le dossier modeles
    if models_dir is not None:
        db = models_dir / "lexique_correcteur.db"
        if db.exists():
            try:
                return PhoneLexicon(db)
            except Exception as e:
                logger.warning("lexique_correcteur.db present mais erreur: %s", e)

    # 3. DB correcteur installee (lectura_correcteur)
    try:
        from lectura_correcteur import _DATA_DIR
        db = Path(_DATA_DIR) / "lexique_correcteur.db"
        if db.exists():
            return PhoneLexicon(db)
    except (ImportError, Exception):
        pass

    # 4. Fallback : phone_lexicon.db embarque dans le package
    db = _MODELES_DIR / "phone_lexicon.db"
    if db.exists():
        try:
            return PhoneLexicon(db)
        except Exception as e:
            logger.warning("phone_lexicon.db embarque mais erreur: %s", e)

    logger.info("Aucun phone_lexicon trouve — V6 fonctionnera sans lex_features")
    return None


def creer_engine(
    mode: str = "auto",
    models_dir: str | Path | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    lexicon_path: str | Path | None = None,
):
    """Factory pour creer un engine d'inference P2G.

    Parameters
    ----------
    mode : str
        "auto" (defaut) : local si modeles presents, sinon API
        "local" : force le mode local (ONNX > NumPy > Pure)
        "api" : force le mode API
        "onnx", "numpy", "pure" : force un backend local specifique
    models_dir : str | Path | None
        Chemin vers le dossier contenant les modeles. Si None, cascade
        automatique : LECTURA_MODELS_DIR → ~/.lectura/models/p2g/ → site-packages.
    api_url : str | None
        URL du serveur Lectura (pour mode API)
    api_key : str | None
        Cle API (pour mode API)
    lexicon_path : str | Path | None
        Chemin vers le fichier lexique POS (JSON).
        Si None, resolution automatique (cascade).
    """
    if mode == "api":
        from lectura_graphemiseur.inference_api import ApiInferenceEngine
        return ApiInferenceEngine(api_url=api_url, api_key=api_key)

    resolved_dir = _resoudre_modeles_dir(models_dir)

    if mode == "auto" and resolved_dir is None:
        from lectura_graphemiseur.inference_api import ApiInferenceEngine
        return ApiInferenceEngine(api_url=api_url, api_key=api_key)

    # Mode local — essayer les backends dans l'ordre de preference
    # Preferer V6 (phone_lex_features) > V5 > V4 > V3
    base = resolved_dir or _MODELES_DIR
    if (base / "unifie_p2g_v6_int8.onnx").exists():
        model_onnx = base / "unifie_p2g_v6_int8.onnx"
        model_vocab = base / "unifie_p2g_v6_vocab.json"
    elif (base / "unifie_p2g_v5_int8.onnx").exists():
        model_onnx = base / "unifie_p2g_v5_int8.onnx"
        model_vocab = base / "unifie_p2g_v5_vocab.json"
    elif (base / "unifie_p2g_v4_int8.onnx").exists():
        model_onnx = base / "unifie_p2g_v4_int8.onnx"
        model_vocab = base / "unifie_p2g_v4_vocab.json"
    else:
        model_onnx = base / "unifie_p2g_v3_int8.onnx"
        model_vocab = base / "unifie_p2g_v3_vocab.json"
    resolved_lexicon = _resoudre_lexique(lexicon_path, resolved_dir)

    if mode in ("auto", "local", "onnx"):
        try:
            from lectura_graphemiseur.inference_onnx_v2 import OnnxInferenceEngineV2
            # Resoudre phone_lexicon pour V6
            phone_lex = None
            if "v6" in str(model_onnx):
                phone_lex = _resoudre_phone_lexicon(resolved_dir)
            if isinstance(resolved_lexicon, dict):
                return OnnxInferenceEngineV2(
                    str(model_onnx), str(model_vocab),
                    lexicon=resolved_lexicon,
                    phone_lexicon=phone_lex,
                )
            return OnnxInferenceEngineV2(
                str(model_onnx), str(model_vocab),
                lexicon_path=str(resolved_lexicon) if resolved_lexicon else None,
                phone_lexicon=phone_lex,
            )
        except (ImportError, FileNotFoundError, Exception):
            if mode == "onnx":
                raise

    if mode in ("auto", "local", "numpy"):
        try:
            from lectura_graphemiseur.inference_numpy import NumpyInferenceEngine
            weights_dir = Path(__file__).parent / "modeles_numpy"
            if not weights_dir.exists():
                weights_dir = resolved_dir if resolved_dir else _MODELES_DIR
            return NumpyInferenceEngine(
                str(weights_dir), str(model_vocab),
                lexicon_path=str(resolved_lexicon) if isinstance(resolved_lexicon, Path) else None,
                lexicon=resolved_lexicon if isinstance(resolved_lexicon, dict) else None,
            )
        except (ImportError, FileNotFoundError, Exception):
            if mode == "numpy":
                raise

    if mode in ("auto", "local", "pure"):
        try:
            from lectura_graphemiseur.inference_pure import PureInferenceEngine
            weights_dir = Path(__file__).parent / "modeles_numpy"
            if not weights_dir.exists():
                weights_dir = resolved_dir if resolved_dir else _MODELES_DIR
            return PureInferenceEngine(
                str(weights_dir), str(model_vocab),
                lexicon_path=str(resolved_lexicon) if isinstance(resolved_lexicon, Path) else None,
                lexicon=resolved_lexicon if isinstance(resolved_lexicon, dict) else None,
            )
        except (ImportError, FileNotFoundError, Exception):
            if mode == "pure":
                raise

    raise RuntimeError(
        f"Aucun backend d'inference disponible (mode={mode!r}). "
        "Verifiez que les modeles sont installes ou utilisez mode='api'."
    )


# API publique
from lectura_graphemiseur.tokeniseur import tokeniser_ipa, ipa_phrase_vers_chars
from lectura_graphemiseur.posttraitement import corriger_p2g, corriger_phrase_v2, corriger_phrase_v3
