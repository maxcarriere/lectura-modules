"""Types publics du module lectura-lexique."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable


class SensDefinition(TypedDict, total=False):
    """Un sens (definition) d'un mot, issu de la table ``definitions``."""

    sens_num: int
    definition: str
    exemples: list[str]
    synonymes: list[str]
    antonymes: list[str]
    domaine: str
    tags: list[str]


class EntreeLexicale(TypedDict, total=False):
    """Entree du lexique avec champs canoniques.

    Seul ``ortho`` est obligatoire ; tous les autres champs sont optionnels
    car chaque source de donnees ne fournit pas forcement tous les champs.
    """

    # Identification
    ortho: str  # required
    lemme: str
    cgram: str

    # Morphologie
    genre: str
    nombre: str
    mode: str
    temps: str
    personne: str

    # Phonetique
    phone: str

    # Frequences
    freq: float
    freq_opensubs: float
    freq_frwac: float
    freq_lm10: float
    freq_frantext: float
    freq_frwac_forme_pmw: float
    freqfilms2: float

    # Donnees educatives (issues de Manulex / Mini)
    age: float
    illustrable: float
    categorie: str
    criteres: str

    # Champs supplementaires (selon la source)
    infover: str
    syll: str
    nbsyll: str
    cvcv: str
    orthrenv: str
    phonrenv: str
    islem: str
    nblettres: str
    nbphons: str


@runtime_checkable
class LexiqueProtocol(Protocol):
    """Interface minimale que tout lexique doit satisfaire.

    Permet au correcteur (et a d'autres modules) de dependre d'une
    abstraction plutot que d'une implementation concrete.
    """

    def existe(self, mot: str) -> bool:
        """Test d'appartenance O(1)."""
        ...

    def info(self, mot: str) -> list[dict[str, Any]]:
        """Entrees lexicales completes pour un mot."""
        ...

    def frequence(self, mot: str) -> float:
        """Frequence maximale parmi toutes les entrees du mot."""
        ...

    def phone_de(self, mot: str) -> str | None:
        """Prononciation la plus frequente."""
        ...

    def homophones(self, phone: str) -> list[dict[str, Any]]:
        """Tous les mots ayant cette prononciation."""
        ...

    def close(self) -> None:
        """Libere les ressources."""
        ...
