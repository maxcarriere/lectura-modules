"""Backend d'inference NumPy pour le BiLSTM edit tagger.

Zero dependance PyTorch : charge les poids depuis un fichier JSON.gz
et effectue le forward pass en NumPy pur.

Architecture reproduite :
  8 embeddings (word, pos, genre, nombre, temps, mode, personne, suf3) -> 184d
  BiLSTM 3 couches, 192 hidden -> 384 bidirectionnel
  MLP : Linear(384,128) + ReLU + Linear(128, N_TAGS)
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from lectura_correcteur._tags import KEEP, N_TAGS, TAGS


# =========================================================================
# Constantes (doivent correspondre a modele_editeur.py)
# =========================================================================

SUF3_BUCKETS = 2000


def suf3_hash(mot: str) -> int:
    """Hash du suffixe 3 caracteres vers un bucket (identique a modele_editeur)."""
    suf = mot[-3:] if len(mot) >= 3 else mot
    return hash(suf) % SUF3_BUCKETS


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    pos = x >= 0
    z = np.zeros_like(x)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    z[~pos] = exp_x / (1.0 + exp_x)
    return z


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


# =========================================================================
# BiLSTM NumPy
# =========================================================================

class _LSTMLayer:
    """Une couche LSTM unidirectionnelle (forward pass seulement)."""

    def __init__(
        self,
        weight_ih: np.ndarray,   # (4*hidden, input)
        weight_hh: np.ndarray,   # (4*hidden, hidden)
        bias_ih: np.ndarray,     # (4*hidden,)
        bias_hh: np.ndarray,     # (4*hidden,)
    ) -> None:
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias = bias_ih + bias_hh  # PyTorch combine les deux biais
        self.hidden_size = weight_hh.shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass sur une sequence (seq_len, input_dim) -> (seq_len, hidden)."""
        seq_len = x.shape[0]
        H = self.hidden_size

        h = np.zeros(H, dtype=np.float32)
        c = np.zeros(H, dtype=np.float32)
        outputs = np.empty((seq_len, H), dtype=np.float32)

        for t in range(seq_len):
            gates = x[t] @ self.weight_ih.T + h @ self.weight_hh.T + self.bias
            i = _sigmoid(gates[0:H])
            f = _sigmoid(gates[H:2*H])
            g = np.tanh(gates[2*H:3*H])
            o = _sigmoid(gates[3*H:4*H])
            c = f * c + i * g
            h = o * np.tanh(c)
            outputs[t] = h

        return outputs


class _BiLSTM:
    """BiLSTM multi-couches (forward pass seulement)."""

    def __init__(self, layers_fwd: list[_LSTMLayer], layers_bwd: list[_LSTMLayer]) -> None:
        self.layers_fwd = layers_fwd
        self.layers_bwd = layers_bwd

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass (seq_len, input_dim) -> (seq_len, 2*hidden)."""
        h = x
        for fwd, bwd in zip(self.layers_fwd, self.layers_bwd):
            h_fwd = fwd.forward(h)
            h_bwd = bwd.forward(h[::-1])[::-1]
            h = np.concatenate([h_fwd, h_bwd], axis=-1)
        return h


# =========================================================================
# EditeurNumpy
# =========================================================================

class EditeurNumpy:
    """Backend d'inference NumPy pour le BiLSTM edit tagger.

    Usage :
        editeur = EditeurNumpy(weights_path, vocab_path)
        tags = editeur.predire_tags(tokens, morpho)
        tags_scores = editeur.predire_tags_avec_scores(tokens, morpho)
    """

    def __init__(self, weights_path: str | Path, vocab_path: str | Path) -> None:
        # Charger le vocabulaire
        with open(vocab_path) as f:
            vocab = json.load(f)

        self.word2idx: dict[str, int] = vocab["word2idx"]
        self.tag_list: list[str] = vocab["tag_list"]

        # Mappings morpho
        self.pos2idx = _build_label_map(vocab["pos_labels"])
        self.genre2idx = _build_label_map(vocab["genre_labels"])
        self.nombre2idx = _build_label_map(vocab["nombre_labels"])
        self.temps2idx = _build_label_map(vocab["temps_labels"])
        self.mode2idx = _build_label_map(vocab["mode_labels"])
        self.personne2idx = _build_label_map(vocab["personne_labels"])
        self.suf3_buckets = vocab.get("suf3_buckets", SUF3_BUCKETS)

        # Aliases pour accepter les formats court (LexiqueTagger) et long (lexique)
        _add_aliases(self.genre2idx, {
            "m": "Masc", "masculin": "Masc",
            "f": "Fem", "feminin": "Fem", "féminin": "Fem",
        })
        _add_aliases(self.nombre2idx, {
            "s": "Sing", "singulier": "Sing",
            "p": "Plur", "pluriel": "Plur",
        })
        _add_aliases(self.mode2idx, {
            "ind": "Ind", "indicatif": "Ind",
            "sub": "Sub", "subjonctif": "Sub",
            "con": "Cnd", "conditionnel": "Cnd",
            "imp": "Imp", "imperatif": "Imp", "impératif": "Imp",
            "par": "Part", "participe": "Part",
            "inf": "Inf", "infinitif": "Inf",
            "ger": "Ger", "gerondif": "Ger", "gérondif": "Ger",
        })
        _add_aliases(self.temps2idx, {
            "pre": "Pres", "present": "Pres", "présent": "Pres",
            "imp": "Imp", "imparfait": "Imp",
            "pas": "Past", "passe": "Past", "passé": "Past",
            "passe_simple": "Past", "passé_simple": "Past",
            "fut": "Fut", "futur": "Fut",
        })

        # Charger les poids
        weights_path = Path(weights_path)
        gz_path = weights_path if str(weights_path).endswith(".gz") else Path(str(weights_path) + ".gz")
        if gz_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                raw_weights = json.load(f)
        else:
            with open(weights_path) as f:
                raw_weights = json.load(f)

        W = {k: np.array(v, dtype=np.float32) for k, v in raw_weights.items()}

        # Embeddings
        self.word_embed = W["word_embed.weight"]
        self.pos_embed = W["pos_embed.weight"]
        self.genre_embed = W["genre_embed.weight"]
        self.nombre_embed = W["nombre_embed.weight"]
        self.temps_embed = W["temps_embed.weight"]
        self.mode_embed = W["mode_embed.weight"]
        self.personne_embed = W["personne_embed.weight"]
        self.suf3_embed = W["suf3_embed.weight"]

        # BiLSTM
        n_layers = 0
        while f"bilstm.weight_ih_l{n_layers}" in W:
            n_layers += 1

        layers_fwd = []
        layers_bwd = []
        for i in range(n_layers):
            layers_fwd.append(_LSTMLayer(
                W[f"bilstm.weight_ih_l{i}"],
                W[f"bilstm.weight_hh_l{i}"],
                W[f"bilstm.bias_ih_l{i}"],
                W[f"bilstm.bias_hh_l{i}"],
            ))
            layers_bwd.append(_LSTMLayer(
                W[f"bilstm.weight_ih_l{i}_reverse"],
                W[f"bilstm.weight_hh_l{i}_reverse"],
                W[f"bilstm.bias_ih_l{i}_reverse"],
                W[f"bilstm.bias_hh_l{i}_reverse"],
            ))

        self.bilstm = _BiLSTM(layers_fwd, layers_bwd)

        # MLP (Sequential: Dropout, Linear, ReLU, Dropout, Linear)
        self.mlp_w1 = W["mlp.1.weight"]   # (128, 384)
        self.mlp_b1 = W["mlp.1.bias"]     # (128,)
        self.mlp_w2 = W["mlp.4.weight"]   # (N_TAGS, 128)
        self.mlp_b2 = W["mlp.4.bias"]     # (N_TAGS,)

    # ------------------------------------------------------------------
    # Encodage des entrees
    # ------------------------------------------------------------------

    def _encode_tokens(
        self,
        tokens: list[str],
        morpho: list[dict],
    ) -> np.ndarray:
        """Encode tokens + morpho en features concat (seq_len, input_dim)."""
        n = len(tokens)
        emb_parts = []

        # Word embeddings
        word_ids = np.array(
            [self.word2idx.get(t.lower(), 1) for t in tokens],
            dtype=np.int64,
        )
        emb_parts.append(self.word_embed[word_ids])

        # POS embeddings
        pos_ids = np.array(
            [self.pos2idx.get(m.get("pos", ""), 1) for m in morpho],
            dtype=np.int64,
        )
        emb_parts.append(self.pos_embed[pos_ids])

        # Genre
        genre_ids = np.array(
            [self.genre2idx.get(m.get("genre", ""), 1) for m in morpho],
            dtype=np.int64,
        )
        emb_parts.append(self.genre_embed[genre_ids])

        # Nombre
        nombre_ids = np.array(
            [self.nombre2idx.get(m.get("nombre", ""), 1) for m in morpho],
            dtype=np.int64,
        )
        emb_parts.append(self.nombre_embed[nombre_ids])

        # Temps
        temps_ids = np.array(
            [self.temps2idx.get(m.get("temps", ""), 1) for m in morpho],
            dtype=np.int64,
        )
        emb_parts.append(self.temps_embed[temps_ids])

        # Mode
        mode_ids = np.array(
            [self.mode2idx.get(m.get("mode", ""), 1) for m in morpho],
            dtype=np.int64,
        )
        emb_parts.append(self.mode_embed[mode_ids])

        # Personne
        personne_ids = np.array(
            [self.personne2idx.get(m.get("personne", ""), 1) for m in morpho],
            dtype=np.int64,
        )
        emb_parts.append(self.personne_embed[personne_ids])

        # Suffixe 3 hash
        suf3_ids = np.array(
            [suf3_hash(t.lower()) for t in tokens],
            dtype=np.int64,
        )
        emb_parts.append(self.suf3_embed[suf3_ids])

        return np.concatenate(emb_parts, axis=-1)  # (seq_len, 184)

    def _encode_inputs_from_ids(
        self,
        word_ids: np.ndarray,
        pos_ids: np.ndarray,
        genre_ids: np.ndarray,
        nombre_ids: np.ndarray,
        temps_ids: np.ndarray,
        mode_ids: np.ndarray,
        personne_ids: np.ndarray,
        suf3_ids: np.ndarray,
    ) -> np.ndarray:
        """Encode directement depuis les indices (pour verification croisee)."""
        return np.concatenate([
            self.word_embed[word_ids],
            self.pos_embed[pos_ids],
            self.genre_embed[genre_ids],
            self.nombre_embed[nombre_ids],
            self.temps_embed[temps_ids],
            self.mode_embed[mode_ids],
            self.personne_embed[personne_ids],
            self.suf3_embed[suf3_ids],
        ], axis=-1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass complet : embeddings -> BiLSTM -> MLP -> logits.

        Args:
            x: (seq_len, input_dim) features concatenees.

        Returns:
            logits: (seq_len, N_TAGS)
        """
        # BiLSTM
        lstm_out = self.bilstm.forward(x)

        # MLP (sans dropout en inference)
        h = _relu(lstm_out @ self.mlp_w1.T + self.mlp_b1)
        logits = h @ self.mlp_w2.T + self.mlp_b2

        return logits

    def predire_tags(
        self,
        tokens: list[str],
        morpho: list[dict],
    ) -> list[str]:
        """Predit les tags d'edition pour une phrase tokenisee.

        Args:
            tokens: Liste de mots.
            morpho: Liste de dicts morpho (pos, genre, nombre, temps, mode, personne).

        Returns:
            Liste de tags (un par token).
        """
        if not tokens:
            return []

        x = self._encode_tokens(tokens, morpho)
        logits = self._forward(x)
        preds = np.argmax(logits, axis=-1)

        return [self.tag_list[i] for i in preds]

    def predire_tags_avec_scores(
        self,
        tokens: list[str],
        morpho: list[dict],
    ) -> list[tuple[str, float]]:
        """Predit les tags avec la confiance (softmax).

        Returns:
            Liste de (tag, confiance) pour chaque token.
        """
        if not tokens:
            return []

        x = self._encode_tokens(tokens, morpho)
        logits = self._forward(x)
        probs = _softmax(logits)
        preds = np.argmax(logits, axis=-1)

        return [
            (self.tag_list[preds[i]], float(probs[i, preds[i]]))
            for i in range(len(tokens))
        ]

    def predire_logits(
        self,
        tokens: list[str],
        morpho: list[dict],
    ) -> np.ndarray:
        """Retourne les logits bruts (seq_len, N_TAGS)."""
        if not tokens:
            return np.empty((0, N_TAGS), dtype=np.float32)

        x = self._encode_tokens(tokens, morpho)
        return self._forward(x)


# =========================================================================
# Utilitaires
# =========================================================================

def _build_label_map(labels: list[str]) -> dict[str, int]:
    return {lab: i for i, lab in enumerate(labels)}


def _add_aliases(label_map: dict[str, int], aliases: dict[str, str]) -> None:
    """Ajoute des aliases (format court/long) vers les indices existants."""
    for alias, canonical in aliases.items():
        if alias not in label_map and canonical in label_map:
            label_map[alias] = label_map[canonical]
