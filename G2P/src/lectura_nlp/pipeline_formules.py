"""Pipeline complet Tokeniseur → G2P pour phrases avec formules.

Orchestre le traitement : mots normaux via modèle neural,
formules via lecture algorithmique (lecture_formules.py).

Licence : CC-BY-SA-4.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from lectura_formules import (
    EventFormuleLecture,
    LectureFormuleResult,
    OptionsLecture,
    lire_formule,
)


# ── Protocoles pour les entrées externes ──────────────────────────────────

class EngineProtocol(Protocol):
    """Protocole minimal pour un moteur G2P neural."""
    def analyser(self, tokens: list[str]) -> dict[str, Any]: ...


class TokenProtocol(Protocol):
    """Protocole minimal pour un token du Tokeniseur."""
    @property
    def type(self) -> Any: ...
    @property
    def text(self) -> str: ...
    @property
    def span(self) -> tuple[int, int]: ...


class FormuleProtocol(TokenProtocol, Protocol):
    """Protocole minimal pour un token FORMULE du Tokeniseur."""
    @property
    def formule_type(self) -> Any: ...
    @property
    def children(self) -> list[Any]: ...


# ── Résultat unifié ──────────────────────────────────────────────────────

@dataclass
class MotAnalyseG2P:
    """Mot analysé par le G2P (neural ou algorithmique)."""
    text: str
    phone: str
    pos: str = ""
    liaison: str = ""
    morpho: dict[str, str] = field(default_factory=dict)
    est_formule: bool = False
    lecture: LectureFormuleResult | None = None


@dataclass
class ResultatPhraseG2P:
    """Résultat complet du pipeline G2P pour une phrase."""
    mots: list[MotAnalyseG2P] = field(default_factory=list)

    @property
    def tokens(self) -> list[str]:
        return [m.text for m in self.mots]

    @property
    def phones(self) -> list[str]:
        return [m.phone for m in self.mots]

    @property
    def pos_tags(self) -> list[str]:
        return [m.pos for m in self.mots]

    @property
    def liaisons(self) -> list[str]:
        return [m.liaison for m in self.mots]

    @property
    def formules(self) -> dict[int, LectureFormuleResult]:
        """Index → LectureFormuleResult pour les tokens FORMULE."""
        return {
            i: m.lecture
            for i, m in enumerate(self.mots)
            if m.lecture is not None
        }


# ── Placeholder ──────────────────────────────────────────────────────────

_PLACEHOLDER = "PLACEHOLDER"


def _is_formule(token: Any) -> bool:
    """Teste si un token est de type FORMULE."""
    t = getattr(token, "type", None)
    if t is None:
        return False
    # Supporte TokenType.FORMULE ou la string "formule"
    name = getattr(t, "value", str(t))
    return str(name).lower() == "formule"


def _get_formule_type(token: Any) -> str:
    """Extrait le type de formule (string) d'un token FORMULE."""
    ft = getattr(token, "formule_type", None)
    if ft is None:
        return "nombre"
    return getattr(ft, "value", str(ft)).lower()


# ── Pipeline principal ───────────────────────────────────────────────────

def analyser_phrase_complete(
    tokens: list[Any],
    engine: Any | None = None,
    options_lecture: OptionsLecture | None = None,
) -> ResultatPhraseG2P:
    """Pipeline complet : mots via neural, formules via algo.

    Parameters
    ----------
    tokens : list
        Tokens du Tokeniseur (Token, Formule, Mot, etc.).
        Seuls les tokens de type MOT et FORMULE sont traités.
        Les PONCTUATION et SEPARATEUR sont ignorés.
    engine : object | None
        Moteur G2P neural avec méthode ``analyser(tokens) -> dict``.
        Si None, les mots normaux n'auront pas de phonétisation.
    options_lecture : OptionsLecture | None
        Options pour la lecture des formules.

    Returns
    -------
    ResultatPhraseG2P
        Résultat unifié avec mots analysés et lectures de formules.
    """
    if options_lecture is None:
        options_lecture = OptionsLecture()

    # ── Étape 1 : séparer mots / formules ──

    # Index dans la liste filtrée (mots+formules) → token original
    items: list[tuple[int, Any, bool]] = []  # (idx_original, token, is_formule)
    for i, tok in enumerate(tokens):
        ttype = getattr(tok, "type", None)
        tname = str(getattr(ttype, "value", str(ttype))).lower() if ttype else ""
        if tname in ("mot", "formule"):
            items.append((i, tok, tname == "formule"))

    if not items:
        return ResultatPhraseG2P()

    # ── Étape 2 : préparer tokens pour le modèle neural ──

    neural_tokens: list[str] = []
    formule_indices: dict[int, int] = {}  # position dans neural_tokens → position dans items

    for item_idx, (orig_idx, tok, is_form) in enumerate(items):
        if is_form:
            neural_tokens.append(_PLACEHOLDER)
            formule_indices[len(neural_tokens) - 1] = item_idx
        else:
            neural_tokens.append(getattr(tok, "text", str(tok)))

    # ── Étape 3 : exécuter le modèle neural ──

    neural_result: dict[str, Any] = {}
    if engine is not None and neural_tokens:
        try:
            neural_result = engine.analyser(neural_tokens)
        except Exception:
            neural_result = {}

    g2p_list = neural_result.get("g2p", [""] * len(neural_tokens))
    pos_list = neural_result.get("pos", [""] * len(neural_tokens))
    liaison_list = neural_result.get("liaison", [""] * len(neural_tokens))
    morpho_dict = neural_result.get("morpho", {})

    # ── Étape 4 : assembler les résultats ──

    mots_resultat: list[MotAnalyseG2P] = []

    for neural_idx, (orig_idx, tok, is_form) in enumerate(items):
        if is_form:
            # Lecture algorithmique de la formule
            ftype = _get_formule_type(tok)
            text = getattr(tok, "text", str(tok))
            tok_span = getattr(tok, "span", (0, len(text)))
            children = getattr(tok, "children", None)

            lecture = lire_formule(
                formule_type=ftype,
                text=text,
                span=tok_span,
                children=children,
                options=options_lecture,
            )

            mots_resultat.append(MotAnalyseG2P(
                text=text,
                phone=lecture.phone,
                pos=pos_list[neural_idx] if neural_idx < len(pos_list) else "",
                liaison="",
                est_formule=True,
                lecture=lecture,
            ))
        else:
            # Résultat du modèle neural
            word_morpho: dict[str, str] = {}
            for feat, vals in morpho_dict.items():
                if neural_idx < len(vals):
                    word_morpho[feat] = vals[neural_idx]

            mots_resultat.append(MotAnalyseG2P(
                text=getattr(tok, "text", str(tok)),
                phone=g2p_list[neural_idx] if neural_idx < len(g2p_list) else "",
                pos=pos_list[neural_idx] if neural_idx < len(pos_list) else "",
                liaison=liaison_list[neural_idx] if neural_idx < len(liaison_list) else "",
                morpho=word_morpho,
            ))

    return ResultatPhraseG2P(mots=mots_resultat)
