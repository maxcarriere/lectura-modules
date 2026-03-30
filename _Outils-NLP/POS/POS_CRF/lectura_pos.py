"""Lectura POS Tagger — Étiqueteur grammatical CRF pour le français.

Fichier unique, autonome, zéro dépendance externe.
Utilise un modèle CRF avec décodage Viterbi pur Python.

Usage rapide :
    from lectura_pos import PosTagger

    tagger = PosTagger("modele/pos_model_crf.json",
                        lexicon_path="modele/mini_lexique.json")
    result = tagger.tag("Le chat mange la souris")
    # [("Le", "ART:def"), ("chat", "NOM"), ("mange", "VER"),
    #  ("la", "ART:def"), ("souris", "NOM")]

Copyright (c) 2025 Lectura — Licence CC BY-SA 4.0.
Voir LICENCE.txt et ATTRIBUTION.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__version__ = "1.0.0"

# ── Tagset ──────────────────────────────────────────────────────────────────

TAGSET: dict[str, str] = {
    "ART:def": "Article défini (le, la, les)",
    "ART:ind": "Article indéfini (un, une, des)",
    "PRO:per": "Pronom personnel (je, tu, il, nous…)",
    "PRO:rel": "Pronom relatif (qui, que, dont…)",
    "PRO:dem": "Pronom démonstratif (ce, ceci, cela…)",
    "PRO:ind": "Pronom indéfini (quelqu'un, rien…)",
    "PRO:int": "Pronom interrogatif (qui, quoi, lequel…)",
    "ADJ:pos": "Adjectif possessif (mon, ton, son…)",
    "ADJ:dem": "Adjectif démonstratif (ce, cette, ces)",
    "ADJ:int": "Adjectif interrogatif (quel, quelle…)",
    "NOM":     "Nom commun",
    "ADJ":     "Adjectif qualificatif",
    "VER":     "Verbe",
    "AUX":     "Auxiliaire (être, avoir)",
    "ADV":     "Adverbe",
    "PRE":     "Préposition (à, de, en, par…)",
    "CON":     "Conjonction (et, ou, mais, car…)",
    "INTJ":    "Interjection (oh, ah, hélas…)",
}


# ── Tokenisation simple ─────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"""
    (?P<elision>
        (?:[CcDdJjLlMmNnSsTt]|[Qq]u|[Jj]usqu|[Ll]orsqu|[Pp]uisqu|[Qq]uelqu)
        ['']
    )
    | (?P<word>   [A-ZÀ-Üa-zà-ÿœŒæÆ](?:[A-ZÀ-Üa-zà-ÿœŒæÆ''-]*[A-ZÀ-Üa-zà-ÿœŒæÆ])? )
    | (?P<other> \S )
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[tuple[str, bool]]:
    """Découpe un texte en tokens (mot, is_word).

    Gère les élisions françaises : « l'école » → [("l'", True), ("école", True)]
    tout en gardant « aujourd'hui » comme un seul token.
    """
    # Normalise les apostrophes typographiques → ASCII
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    tokens: list[tuple[str, bool]] = []
    for m in _TOKEN_RE.finditer(text):
        if m.group("elision"):
            tokens.append((m.group("elision"), True))
        elif m.group("word"):
            tokens.append((m.group("word"), True))
        else:
            tokens.append((m.group("other"), False))
    return tokens


# ── Extraction de features ──────────────────────────────────────────────────

_LEX_TAGS = ("VER", "AUX", "NOM", "ADJ", "ADV", "ART:def", "ART:ind",
             "PRO:per", "PRE", "CON")


def _extract_word_features(
    words: list[str],
    idx: int,
) -> dict[str, str | float]:
    """Extrait les features pour le mot à la position idx."""
    word = words[idx]
    low = word.lower()
    feats: dict[str, str | float] = {
        "bias": 1.0,
        "word": low,
        "len": float(len(low)),
        "suf3": low[-3:] if len(low) >= 3 else low,
        "suf2": low[-2:] if len(low) >= 2 else low,
        "pre2": low[:2] if len(low) >= 2 else low,
        "pre3": low[:3] if len(low) >= 3 else low,
        "is_upper": 1.0 if word.isupper() else 0.0,
        "is_title": 1.0 if word.istitle() else 0.0,
        "is_digit": 1.0 if word.isdigit() else 0.0,
    }

    # Pas de lexique → toutes les features lex.* à 0
    for tag in _LEX_TAGS:
        feats[f"lex.{tag}"] = 0.0

    # Position
    feats["BOS"] = 1.0 if idx == 0 else 0.0
    feats["EOS"] = 1.0 if idx == len(words) - 1 else 0.0

    # Contexte (bigramme)
    feats["w-1"] = words[idx - 1].lower() if idx > 0 else "__BOS__"
    feats["w+1"] = words[idx + 1].lower() if idx < len(words) - 1 else "__EOS__"

    return feats


def _extract_sequence_features(
    words: list[str],
) -> list[dict[str, str | float]]:
    """Extrait les features pour toute la séquence."""
    return [_extract_word_features(words, i) for i in range(len(words))]


# ── Modèle CRF + Viterbi ────────────────────────────────────────────────────

class CrfModel:
    """Modèle CRF avec décodage Viterbi pur Python.

    Le modèle est chargé depuis un fichier JSON contenant :
      - state_features : dict[feature, dict[tag, poids]]
      - transitions    : dict[tag_précédent, dict[tag_courant, poids]]
      - tags           : list[str]
    """

    __slots__ = ("state_features", "transitions", "tags", "_tag_idx")

    def __init__(
        self,
        state_features: dict[str, dict[str, float]],
        transitions: dict[str, dict[str, float]],
        tags: list[str],
    ) -> None:
        self.state_features = state_features
        self.transitions = transitions
        self.tags = tags
        self._tag_idx = {t: i for i, t in enumerate(tags)}

    @classmethod
    def load(cls, path: Path | str) -> CrfModel:
        """Charge un modèle depuis un fichier JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            state_features=data["state_features"],
            transitions=data["transitions"],
            tags=data["tags"],
        )

    def _score_state(self, feats: dict[str, str | float], tag: str) -> float:
        """Calcule le score d'état pour un tag donné les features."""
        score = 0.0
        for feat_key, feat_val in feats.items():
            lookup = f"{feat_key}:{feat_val}"
            weights = self.state_features.get(lookup)
            if weights and tag in weights:
                score += weights[tag]
        return score

    def _viterbi(
        self,
        observations: list[dict[str, str | float]],
    ) -> list[str]:
        """Décodage Viterbi sur la séquence d'observations."""
        n = len(observations)
        if n == 0:
            return []

        tags = self.tags
        n_tags = len(tags)
        NEG_INF = -1e30

        # Initialisation (t=0)
        scores_0: list[float] = []
        for tag in tags:
            s = self._score_state(observations[0], tag)
            s += self.transitions.get("__BOS__", {}).get(tag, 0.0)
            scores_0.append(s)

        viterbi: list[list[float]] = [scores_0]
        backptr: list[list[int]] = [[0] * n_tags]

        # Récurrence
        for t in range(1, n):
            scores_t: list[float] = []
            bp_t: list[int] = []
            for j, tag_j in enumerate(tags):
                state_score = self._score_state(observations[t], tag_j)
                best_score = NEG_INF
                best_prev = 0

                for i, tag_i in enumerate(tags):
                    prev = viterbi[t - 1][i]
                    if prev <= NEG_INF:
                        continue
                    trans = self.transitions.get(tag_i, {}).get(tag_j, 0.0)
                    score = prev + trans + state_score
                    if score > best_score:
                        best_score = score
                        best_prev = i

                scores_t.append(best_score)
                bp_t.append(best_prev)

            viterbi.append(scores_t)
            backptr.append(bp_t)

        # Backtrack
        best_last = max(range(n_tags), key=lambda j: viterbi[n - 1][j])
        result = [0] * n
        result[n - 1] = best_last
        for t in range(n - 2, -1, -1):
            result[t] = backptr[t + 1][result[t + 1]]

        return [tags[idx] for idx in result]

    def predict(self, words: list[str]) -> list[str]:
        """Prédit la séquence de POS pour une liste de mots."""
        if not words:
            return []
        features = _extract_sequence_features(words)
        return self._viterbi(features)


# ── Mini-lexique (correction post-CRF) ──────────────────────────────────────

def _load_lexicon(path: Path | str) -> dict[str, str]:
    """Charge le mini-lexique depuis un fichier JSON.

    Le JSON contient des catégories (contractions, prepositions, …),
    chacune étant un dict mot → tag. Les clés commençant par '_' sont ignorées.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    lexicon: dict[str, str] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            for word, tag in value.items():
                if not word.startswith("_"):
                    lexicon[word] = tag
        elif isinstance(value, str):
            lexicon[key] = value
    return lexicon


def _apply_lexicon(
    words: list[str],
    tags: list[str],
    lexicon: dict[str, str],
) -> list[str]:
    """Corrige les tags via le mini-lexique (post-traitement)."""
    corrected = list(tags)
    for i, word in enumerate(words):
        low = word.lower()
        if low in lexicon:
            corrected[i] = lexicon[low]
    return corrected


# ── API publique ─────────────────────────────────────────────────────────────

class PosTagger:
    """Étiqueteur grammatical CRF pour le français.

    Args:
        model_path: Chemin vers le modèle CRF (JSON).
        lexicon_path: Chemin vers le mini-lexique (JSON, optionnel).
            Si fourni, applique des corrections post-CRF pour les
            mots-outils non ambigus.

    Exemple :
        >>> tagger = PosTagger("modele/pos_model_crf.json",
        ...                     lexicon_path="modele/mini_lexique.json")
        >>> tagger.tag("Le chat mange la souris")
        [('Le', 'ART:def'), ('chat', 'NOM'), ('mange', 'VER'),
         ('la', 'ART:def'), ('souris', 'NOM')]
    """

    def __init__(
        self,
        model_path: str | Path,
        lexicon_path: str | Path | None = None,
    ) -> None:
        self.model = CrfModel.load(model_path)
        self.lexicon: dict[str, str] = {}
        if lexicon_path is not None:
            self.lexicon = _load_lexicon(lexicon_path)

    def _predict(self, words: list[str]) -> list[str]:
        """Prédit puis applique le mini-lexique si disponible."""
        tags = self.model.predict(words)
        if self.lexicon:
            tags = _apply_lexicon(words, tags, self.lexicon)
        return tags

    def tag_words(self, words: list[str]) -> list[tuple[str, str]]:
        """Étiquète une liste de mots déjà tokenisés.

        Args:
            words: Liste de mots (ex: ["Le", "chat", "mange"]).

        Returns:
            Liste de tuples (mot, tag).
        """
        tags = self._predict(words)
        return list(zip(words, tags))

    def tag(self, text: str) -> list[tuple[str, str]]:
        """Tokenise et étiquète un texte brut.

        Args:
            text: Texte en français (ex: "Le chat mange la souris").

        Returns:
            Liste de tuples (mot, tag) pour chaque mot détecté.
        """
        tokens = tokenize(text)
        words = [tok for tok, is_word in tokens if is_word]
        if not words:
            return []
        tags = self._predict(words)
        return list(zip(words, tags))

    def tag_detailed(self, text: str) -> list[dict[str, str]]:
        """Tokenise et étiquète avec descriptions des tags.

        Returns:
            Liste de dicts {"mot", "tag", "description"}.
        """
        pairs = self.tag(text)
        return [
            {"mot": mot, "tag": tag, "description": TAGSET.get(tag, tag)}
            for mot, tag in pairs
        ]

    def tag_formatted(self, text: str) -> str:
        """Retourne un résultat formaté lisible.

        Returns:
            Texte formaté avec un mot/tag par ligne.
        """
        pairs = self.tag(text)
        if not pairs:
            return "(aucun mot détecté)"
        max_word = max(len(mot) for mot, _ in pairs)
        max_tag = max(len(tag) for _, tag in pairs)
        lines = []
        for mot, tag in pairs:
            desc = TAGSET.get(tag, "")
            lines.append(f"  {mot:<{max_word}}  {tag:<{max_tag}}  {desc}")
        return "\n".join(lines)
