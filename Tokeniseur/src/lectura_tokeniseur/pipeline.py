"""Pipeline complet de tokenisation et classe principale."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lectura_tokeniseur.models import (
    Token, TokenType, Mot, Ponctuation, Separateur, Formule,
)
from lectura_tokeniseur.normalisation import normalise
from lectura_tokeniseur.tokenisation import _scan_tokens, _merge_compounds, _merge_locutions
from lectura_tokeniseur.classification import _classify_and_merge

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def tokenise(text: str) -> list[Token]:
    """Découpe le texte normalisé en tokens avec détection de formules.

    Pipeline en 2 passes :
    1. Scan : MOT, PONCTUATION, SEPARATEUR, FORMULE brut
    2. Classification + fusion : détecte les sous-types de formules

    Args:
        text: Texte normalisé à tokeniser.

    Returns:
        Liste de Token (Mot, Ponctuation, Separateur, Formule).
    """
    logger.debug("tokenise() called, input length=%s", len(text) if text else 0)
    if not text:
        return []

    # Passe 1 : scan brut
    tokens = _scan_tokens(text)

    # Fusions de mots composés et locutions (sur les Mot)
    tokens = _merge_compounds(tokens)
    tokens = _merge_locutions(tokens)

    # Passe 2 : classification et fusion des formules
    tokens = _classify_and_merge(tokens)

    logger.debug("tokenise() produced %s tokens", len(tokens))
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Résultat structuré
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ResultatTokenisation:
    """Résultat complet de la normalisation + tokenisation."""

    texte_original: str
    texte_normalise: str
    tokens: list[Token]

    @property
    def mots(self) -> list[Mot]:
        """Retourne uniquement les tokens de type Mot."""
        return [t for t in self.tokens if isinstance(t, Mot)]

    @property
    def formules(self) -> list[Formule]:
        """Retourne uniquement les tokens de type Formule."""
        return [t for t in self.tokens if isinstance(t, Formule)]

    @property
    def nb_mots(self) -> int:
        return len(self.mots)

    @property
    def nb_tokens(self) -> int:
        return len(self.tokens)

    def words(self) -> list[str]:
        """Retourne la liste des formes orthographiques des mots."""
        return [t.ortho for t in self.mots]

    def format_table(self) -> str:
        """Retourne un affichage tabulaire des tokens."""
        lines = [f"{'Texte':15s}  {'Type':12s}  {'Span':10s}  Détail"]
        lines.append("-" * 65)
        for t in self.tokens:
            text_repr = repr(t.text)
            detail = ""
            if isinstance(t, Mot):
                detail = f"ortho={t.ortho!r}"
                if t.children:
                    detail += f" composé=[{'+'.join(c.text for c in t.children)}]"
            elif isinstance(t, Separateur):
                detail = f"sep={t.sep_type}"
            elif isinstance(t, Formule):
                detail = f"sous-type={t.formule_type.value}"
                if t.valeur:
                    detail += f" val={t.valeur!r}"
                if t.children:
                    detail += f" ({len(t.children)} enfants)"
            lines.append(
                f"{text_repr:15s}  {t.type.value:12s}  "
                f"{str(t.span):10s}  {detail}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale
# ══════════════════════════════════════════════════════════════════════════════


class LecturaTokeniseur:
    """Normalisateur et tokeniseur complet pour le français.

    Combine normalisation typographique (espaces, apostrophes, guillemets,
    nombres, ellipses, tirets) et tokenisation en types structurés
    (Mot, Ponctuation, Separateur, Formule) avec spans et détection
    de formules (nombres, sigles, dates, téléphones, etc.).
    """

    def normalize(self, text: str) -> str:
        """Normalise un texte brut (étape 1 seule).

        Transformations : espaces, apostrophes, ellipses,
        ponctuation, nombres, guillemets, parenthèses, tirets.
        """
        return normalise(text)

    def tokenize(self, text: str, normalize: bool = True) -> list[Token]:
        """Tokenise un texte en tokens typés avec spans.

        Args:
            text: Texte à tokeniser.
            normalize: Si True (défaut), normalise le texte avant tokenisation.

        Returns:
            Liste de Token (Mot, Ponctuation, Separateur, Formule).
        """
        if normalize:
            text = normalise(text)
        return tokenise(text)

    def analyze(self, text: str) -> ResultatTokenisation:
        """Normalisation + tokenisation avec résultat structuré.

        Args:
            text: Texte brut à analyser.

        Returns:
            ResultatTokenisation avec texte original, normalisé, et tokens.
        """
        text_norm = normalise(text)
        tokens = tokenise(text_norm)
        return ResultatTokenisation(
            texte_original=text,
            texte_normalise=text_norm,
            tokens=tokens,
        )

    def extract_words(self, text: str) -> list[str]:
        """Raccourci : retourne la liste des mots (formes normalisées).

        Args:
            text: Texte brut.

        Returns:
            Liste de chaînes (formes orthographiques en minuscules).
        """
        return self.analyze(text).words()

    def extract_formules(self, text: str) -> list[Formule]:
        """Raccourci : retourne la liste des formules détectées.

        Args:
            text: Texte brut.

        Returns:
            Liste de Formule avec sous-type, valeur, enfants.
        """
        return self.analyze(text).formules


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    tok = LecturaTokeniseur()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = tok.analyze(text)
        print(f"Original   : {result.texte_original}")
        print(f"Normalisé  : {result.texte_normalise}")
        print(f"Mots       : {result.nb_mots}")
        print(f"Formules   : {len(result.formules)}")
        print(f"Tokens     : {result.nb_tokens}")
        print()
        print(result.format_table())
    else:
        print("Lectura Tokeniseur Complet — Mode interactif (Ctrl+C pour quitter)")
        print()
        try:
            while True:
                text = input("Texte > ").strip()
                if not text:
                    continue
                result = tok.analyze(text)
                print(f"  Normalisé : {result.texte_normalise}")
                print(f"  Mots ({result.nb_mots}) : {result.words()}")
                if result.formules:
                    print(f"  Formules ({len(result.formules)}) :")
                    for f in result.formules:
                        print(f"    {f.text} → {f.formule_type.value} (val={f.valeur!r})")
                print()
                print(result.format_table())
                print()
        except (KeyboardInterrupt, EOFError):
            print()
