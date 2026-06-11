"""lectura-correcteur : correcteur orthographique et grammatical Lectura.

Pipeline : Tokenisation -> Syntaxe -> Resegmentation -> Morpho CRF
           -> Orthographe -> Grammaire -> Reconstruction

Deux modes de fonctionnement :

1. **Local** (avec lexique) ::

    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur

    lex = Lexique("lexique.db")
    correcteur = Correcteur(lex)
    result = correcteur.corriger("Les enfant mange des pomme.")
    print(result.phrase_corrigee)  # "Les enfants mangent des pommes."

2. **API** (sans lexique local) ::

    from lectura_correcteur import creer_correcteur

    correcteur = creer_correcteur()  # auto-detect : API si pas de lexique
    result = correcteur.corriger("Les enfant mange des pomme.")
"""

__version__ = "1.1.0"

from lectura_correcteur._config import CorrecteurConfig, CorrecteurV2Config, CorrecteurV3Config, CorrecteurV4Config, CorrecteurV5Config, CorrecteurV6Config
from lectura_correcteur._types import (
    Correction,
    MotAnalyse,
    ResultatCorrection,
    TaggerProtocol,
    TokeniseurProtocol,
    TypeCorrection,
)
from lectura_correcteur.correcteur import Correcteur
from lectura_correcteur.correcteur_v2 import CorrecteurV2
from lectura_correcteur.correcteur_v3 import CorrecteurV3
from lectura_correcteur.correcteur_v4 import CorrecteurV4
from lectura_correcteur.correcteur_v5 import CorrecteurV5
from lectura_correcteur.correcteur_v6 import CorrecteurV6


def creer_correcteur(
    lexique_path: str | None = None,
    *,
    config: CorrecteurConfig | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
):
    """Factory : cree un Correcteur local ou un client API.

    Detection automatique :
    1. Si `lexique_path` est fourni → mode local
    2. Sinon, cherche le lexique dans les emplacements standards
    3. Si aucun lexique trouve → mode API

    Parameters
    ----------
    lexique_path : str | None
        Chemin vers le fichier lexique.db. Si None, auto-detection.
    config : CorrecteurConfig | None
        Configuration du correcteur (mode local uniquement).
    api_url : str | None
        URL du serveur Lectura (mode API, defaut: LECTURA_API_URL).
    api_key : str | None
        Cle API (mode API, defaut: LECTURA_API_KEY).

    Returns
    -------
    Correcteur | CorrecteurAPI
        Instance avec la meme interface `.corriger(phrase)`.
    """
    import os
    from pathlib import Path

    # 1. Chercher le lexique SQLite local
    if lexique_path is None:
        candidates = [
            os.environ.get("LECTURA_LEXIQUE_DB", ""),
            str(Path.home() / ".lectura" / "lexique_lectura.db"),
            "/opt/lectura/lexique_lectura.db",
        ]
        for p in candidates:
            if p and Path(p).exists():
                lexique_path = p
                break

    # Mode SQLite si lexique disponible
    if lexique_path is not None and Path(lexique_path).exists():
        try:
            from lectura_lexique import Lexique
            lex = Lexique(lexique_path)
            return Correcteur(lex, config=config)
        except Exception:
            pass  # fallback to CSV

    # 2. SQLite leger integre (wheel prive)
    lite_path = Path(__file__).parent / "data" / "lexique_correcteur.db"
    if lite_path.exists():
        from lectura_correcteur._lexique_lite import LexiqueLite
        lex = LexiqueLite(lite_path)
        return Correcteur(lex, config=config)

    # 3. CSV integre (fallback)
    csv_path = Path(__file__).parent / "data" / "lexique_correcteur.csv.gz"
    if csv_path.exists():
        from lectura_correcteur._lexique_csv import LexiqueCSV
        lex = LexiqueCSV(csv_path)
        return Correcteur(lex, config=config)

    # 4. Mode API
    from lectura_correcteur._api_client import CorrecteurAPI
    return CorrecteurAPI(api_url=api_url, api_key=api_key)


__all__ = [
    "Correcteur",
    "CorrecteurV2",
    "CorrecteurV3",
    "CorrecteurV4",
    "CorrecteurV5",
    "CorrecteurConfig",
    "CorrecteurV2Config",
    "CorrecteurV3Config",
    "CorrecteurV4Config",
    "CorrecteurV5Config",
    "CorrecteurV6",
    "CorrecteurV6Config",
    "CorrecteurAPI",
    "Correction",
    "MotAnalyse",
    "ResultatCorrection",
    "TaggerProtocol",
    "TokeniseurProtocol",
    "TypeCorrection",
    "creer_correcteur",
]
