"""Lectura POS Tagger — Étiqueteur grammatical BiLSTM pour le français.

Fichier unique, autonome. Utilise un modèle BiLSTM via ONNX Runtime.
Dépendances : onnxruntime, numpy.

Usage rapide :
    from lectura_pos import PosTagger

    tagger = PosTagger("modele/pos_model_bilstm_int8.onnx",
                        vocab_path="modele/pos_vocab_bilstm.json",
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


# ── Modèle BiLSTM ONNX ──────────────────────────────────────────────────────

class OnnxPosModel:
    """Modèle POS BiLSTM via ONNX Runtime.

    Le modèle ONNX produit des émissions (logits par tag).
    Le décodage se fait par argmax simple sur les émissions.
    """

    __slots__ = (
        "_np", "_session", "_word2idx", "_char2idx",
        "_idx2tag", "tags", "_pad_idx", "_unk_idx", "_max_word_len",
    )

    def __init__(self, onnx_path: Path | str, vocab_path: Path | str) -> None:
        import numpy as np
        import onnxruntime as ort

        self._np = np
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)
        self._word2idx: dict[str, int] = vocab_data["word2idx"]
        self._char2idx: dict[str, int] = vocab_data.get("char2idx", {})
        self._idx2tag: dict[int, str] = {
            int(k): v for k, v in vocab_data["idx2tag"].items()
        }
        self.tags: list[str] = vocab_data["tags"]
        self._pad_idx: int = vocab_data.get("pad_idx", 0)
        self._unk_idx: int = vocab_data.get("unk_idx", 1)
        self._max_word_len: int = vocab_data.get("max_word_len", 20)

    def _encode_chars(self, word: str) -> list[int]:
        """Encode les caractères d'un mot en indices."""
        ids = []
        for ch in word[: self._max_word_len]:
            ids.append(self._char2idx.get(ch, self._unk_idx))
        while len(ids) < self._max_word_len:
            ids.append(self._pad_idx)
        return ids

    def predict(self, words: list[str]) -> list[str]:
        """Prédit la séquence de POS pour une liste de mots."""
        if not words:
            return []

        np = self._np
        n = len(words)

        # Encoder les mots
        word_ids = [
            self._word2idx.get(w.lower(), self._unk_idx) for w in words
        ]
        word_array = np.array([word_ids], dtype=np.int64)

        # Encoder les caractères
        char_ids = [self._encode_chars(w.lower()) for w in words]
        char_array = np.array([char_ids], dtype=np.int64)

        # Inférence ONNX
        inputs = {"word_ids": word_array, "char_ids": char_array}
        outputs = self._session.run(None, inputs)
        emissions = outputs[0]  # (1, seq_len, n_tags)

        # Argmax simple sur les émissions
        pred_indices = emissions[0, :n].argmax(axis=-1)
        return [self._idx2tag.get(int(idx), "NOM") for idx in pred_indices]


# ── Mini-lexique (correction post-modèle) ───────────────────────────────────

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
    """Étiqueteur grammatical BiLSTM pour le français.

    Args:
        model_path: Chemin vers le modèle ONNX (.onnx).
        vocab_path: Chemin vers le vocabulaire (.json).
        lexicon_path: Chemin vers le mini-lexique (JSON, optionnel).
            Si fourni, applique des corrections post-modèle pour les
            mots-outils non ambigus.

    Exemple :
        >>> tagger = PosTagger("modele/pos_model_bilstm_int8.onnx",
        ...                     vocab_path="modele/pos_vocab_bilstm.json",
        ...                     lexicon_path="modele/mini_lexique.json")
        >>> tagger.tag("Le chat mange la souris")
        [('Le', 'ART:def'), ('chat', 'NOM'), ('mange', 'VER'),
         ('la', 'ART:def'), ('souris', 'NOM')]
    """

    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        lexicon_path: str | Path | None = None,
    ) -> None:
        self.model = OnnxPosModel(model_path, vocab_path)
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
