"""CTCDenoiser — Correcteur de phones et re-segmenteur de mots IPA.

Corrige les erreurs CTC (phones incorrects, mauvaises frontieres de mots)
au niveau de la phrase IPA, AVANT le P2G. Utilise du sequence labeling
(pas du seq2seq) pour eviter les regressions.

Architecture :
    char_embed(41, 64) || boundary_embed(2, 16)
    → input_proj(80, 64)
    → BiLSTM(64, 160, 2 layers) → 320d
    → TransformerEncoder(320d, 4 heads, 512 ff, 2 layers)
    → phone_head(320, 42) : 41 phones + DELETE
    → boundary_head(320, 1) : frontiere de mot (sigmoid)

Le char_embed et le BiLSTM sont initialises depuis les poids P2G v7.

Copyright (C) 2025-2026 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)

# Index du token DELETE dans la tete phone (= n_chars, dernier index)
DELETE_IDX = 41


class CTCDenoiser(nn.Module):
    """Correcteur seq-labeling pour la sortie CTC.

    Corrige les phones et re-segmente les mots au niveau de la phrase IPA.
    Utilise le vocabulaire char2idx du P2G (41 chars).

    Parameters
    ----------
    n_chars : int
        Taille du vocabulaire char (41 par defaut, inclut PAD/UNK/BOS/EOS/SEP).
    char_embed_dim : int
        Dimension de l'embedding char (64).
    boundary_embed_dim : int
        Dimension de l'embedding frontiere CTC (16).
    hidden_dim : int
        Dimension cachee du BiLSTM par direction (160 → 320 bidi).
    lstm_layers : int
        Nombre de couches LSTM (2).
    attn_layers : int
        Nombre de couches TransformerEncoder (2).
    attn_nhead : int
        Nombre de tetes d'attention (4).
    attn_ff_dim : int
        Dimension feedforward du Transformer (512).
    dropout : float
        Dropout general (0.1).
    """

    def __init__(
        self,
        n_chars: int = 41,
        char_embed_dim: int = 64,
        boundary_embed_dim: int = 16,
        hidden_dim: int = 160,
        lstm_layers: int = 2,
        attn_layers: int = 2,
        attn_nhead: int = 4,
        attn_ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_chars = n_chars
        self.hidden_dim = hidden_dim
        self.n_phone_classes = n_chars + 1  # +1 pour DELETE

        # ── Config pour serialisation ──
        self._config = {
            "n_chars": n_chars,
            "char_embed_dim": char_embed_dim,
            "boundary_embed_dim": boundary_embed_dim,
            "hidden_dim": hidden_dim,
            "lstm_layers": lstm_layers,
            "attn_layers": attn_layers,
            "attn_nhead": attn_nhead,
            "attn_ff_dim": attn_ff_dim,
            "dropout": dropout,
        }

        bidi_dim = hidden_dim * 2  # 320

        # ── 1. Char embedding (init depuis P2G) ──
        self.char_embed = nn.Embedding(n_chars, char_embed_dim, padding_idx=0)

        # ── 2. Boundary embedding (nouveau) ──
        self.boundary_embed = nn.Embedding(2, boundary_embed_dim)

        # ── 3. Projection concatenee → dim LSTM ──
        self.input_proj = nn.Linear(
            char_embed_dim + boundary_embed_dim, char_embed_dim,
        )

        # ── 4. BiLSTM encoder (init depuis P2G char_lstm) ──
        self.encoder_lstm = nn.LSTM(
            char_embed_dim, hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── 5. TransformerEncoder (nouveau, attention sur toute la phrase) ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bidi_dim,
            nhead=attn_nhead,
            dim_feedforward=attn_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.phrase_attn = nn.TransformerEncoder(
            encoder_layer,
            num_layers=attn_layers,
        )

        # ── 6. Phone head : 41 phones + DELETE ──
        self.phone_head = nn.Linear(bidi_dim, self.n_phone_classes)

        # ── 7. Boundary head : frontiere de mot (sigmoid) ──
        self.boundary_head = nn.Linear(bidi_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        char_ids: torch.Tensor,
        ctc_boundaries: torch.Tensor,
        char_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        char_ids : (B, L)
            Indices de chars P2G de la phrase CTC complete.
        ctc_boundaries : (B, L)
            Binaire, 1 la ou le CTC avait un |.
        char_lengths : (B,)
            Longueurs reelles (sans padding).

        Returns
        -------
        phone_logits : (B, L, n_phone_classes)
            Logits pour chaque position → phone predit ou DELETE.
        boundary_logits : (B, L)
            Logits pour la frontiere de mot (avant sigmoid).
        """
        # Embeddings
        emb = self.char_embed(char_ids)                     # (B, L, 64)
        bnd = self.boundary_embed(ctc_boundaries)           # (B, L, 16)
        x = self.input_proj(torch.cat([emb, bnd], dim=-1))  # (B, L, 64)
        x = self.dropout(x)

        # BiLSTM
        packed = pack_padded_sequence(
            x, char_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )
        x, _ = self.encoder_lstm(packed)
        x, _ = pad_packed_sequence(x, batch_first=True)     # (B, L, 320)
        x = self.dropout(x)

        # Padding mask pour le Transformer
        max_len = x.size(1)
        padding_mask = (
            torch.arange(max_len, device=x.device).unsqueeze(0)
            >= char_lengths.unsqueeze(1)
        )

        # Transformer attention
        x = self.phrase_attn(x, src_key_padding_mask=padding_mask)  # (B, L, 320)

        # Heads
        phone_logits = self.phone_head(x)                   # (B, L, 42)
        boundary_logits = self.boundary_head(x).squeeze(-1) # (B, L)

        return phone_logits, boundary_logits

    def decode(
        self,
        phone_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        char_length: int,
        idx2char: dict[int, str],
        boundary_threshold: float = 0.5,
    ) -> list[str]:
        """Decode les predictions en liste de mots IPA.

        Parameters
        ----------
        phone_logits : (L, n_phone_classes)
            Logits phone pour une seule phrase (sans batch).
        boundary_logits : (L,)
            Logits boundary pour une seule phrase.
        char_length : int
            Longueur reelle de la sequence.
        idx2char : dict[int, str]
            Mapping index → char IPA.
        boundary_threshold : float
            Seuil pour la frontiere de mot (sigmoid > seuil).

        Returns
        -------
        list[str]
            Mots IPA corriges.
        """
        # Predictions
        phone_preds = phone_logits[:char_length].argmax(dim=-1)  # (L,)
        boundary_probs = torch.sigmoid(
            boundary_logits[:char_length],
        )  # (L,)

        # Construire les mots
        current_chars: list[str] = []
        words: list[str] = []

        for i in range(char_length):
            phone_id = phone_preds[i].item()

            # Ignorer les tokens speciaux (PAD, UNK, BOS, EOS, SEP)
            if phone_id in (0, 1, 2, 3, 4):
                continue

            # Supprimer les positions DELETE
            if phone_id == DELETE_IDX:
                continue

            # Ajouter le char
            char = idx2char.get(phone_id, "")
            if char:
                current_chars.append(char)

            # Couper si frontiere de mot
            if boundary_probs[i].item() > boundary_threshold:
                if current_chars:
                    words.append("".join(current_chars))
                    current_chars = []

        # Flush le dernier mot
        if current_chars:
            words.append("".join(current_chars))

        return words

    def corriger(
        self,
        mots_ipa: list[str],
        char2idx: dict[str, int],
        idx2char: dict[int, str],
        boundary_threshold: float = 0.5,
    ) -> list[str]:
        """Corrige les phones et re-segmente les mots IPA.

        1. Concatene les mots en phrase char-level
        2. Encode les frontieres CTC originales
        3. Forward pass → phones corriges + nouvelles frontieres
        4. Re-segmente en mots
        5. Retourne la liste de mots corriges

        Parameters
        ----------
        mots_ipa : list[str]
            Mots IPA bruts du CTC (apres parse_ctc_v2).
        char2idx : dict[str, int]
            Vocabulaire char → index du P2G.
        idx2char : dict[int, str]
            Vocabulaire index → char du P2G.
        boundary_threshold : float
            Seuil pour la frontiere de mot.

        Returns
        -------
        list[str]
            Mots IPA corriges.
        """
        if not mots_ipa:
            return []

        device = next(self.parameters()).device

        # 1. Construire la sequence char + boundaries
        all_chars: list[int] = [char2idx.get("<BOS>", 2)]
        all_boundaries: list[int] = [0]

        for w_idx, mot in enumerate(mots_ipa):
            for ch in mot:
                all_chars.append(char2idx.get(ch, char2idx.get("<UNK>", 1)))
                all_boundaries.append(0)
            # Marquer la derniere position du mot comme frontiere CTC
            # (sauf le dernier mot)
            if w_idx < len(mots_ipa) - 1 and all_boundaries:
                all_boundaries[-1] = 1

        all_chars.append(char2idx.get("<EOS>", 3))
        all_boundaries.append(0)

        seq_len = len(all_chars)

        # 2. Tenseurs
        char_ids = torch.tensor([all_chars], dtype=torch.long, device=device)
        ctc_bnd = torch.tensor([all_boundaries], dtype=torch.long, device=device)
        lengths = torch.tensor([seq_len], dtype=torch.long, device=device)

        # 3. Forward
        with torch.no_grad():
            phone_logits, boundary_logits = self.forward(
                char_ids, ctc_bnd, lengths,
            )

        # 4. Decode
        return self.decode(
            phone_logits[0], boundary_logits[0],
            seq_len, idx2char,
            boundary_threshold=boundary_threshold,
        )

    def init_from_p2g(
        self,
        p2g_state_dict: dict[str, torch.Tensor],
    ) -> list[str]:
        """Initialise char_embed et encoder_lstm depuis les poids P2G.

        Parameters
        ----------
        p2g_state_dict : dict
            State dict du modele UnifiedP2G (ou checkpoint contenant les cles
            'char_embedding.*' et 'char_lstm.*').

        Returns
        -------
        list[str]
            Liste des cles initialisees avec succes.
        """
        initialized: list[str] = []

        # Mapping P2G → denoiser
        mapping = {
            "char_embedding.weight": "char_embed.weight",
            "char_lstm.weight_ih_l0": "encoder_lstm.weight_ih_l0",
            "char_lstm.weight_hh_l0": "encoder_lstm.weight_hh_l0",
            "char_lstm.bias_ih_l0": "encoder_lstm.bias_ih_l0",
            "char_lstm.bias_hh_l0": "encoder_lstm.bias_hh_l0",
            "char_lstm.weight_ih_l0_reverse": "encoder_lstm.weight_ih_l0_reverse",
            "char_lstm.weight_hh_l0_reverse": "encoder_lstm.weight_hh_l0_reverse",
            "char_lstm.bias_ih_l0_reverse": "encoder_lstm.bias_ih_l0_reverse",
            "char_lstm.bias_hh_l0_reverse": "encoder_lstm.bias_hh_l0_reverse",
            "char_lstm.weight_ih_l1": "encoder_lstm.weight_ih_l1",
            "char_lstm.weight_hh_l1": "encoder_lstm.weight_hh_l1",
            "char_lstm.bias_ih_l1": "encoder_lstm.bias_ih_l1",
            "char_lstm.bias_hh_l1": "encoder_lstm.bias_hh_l1",
            "char_lstm.weight_ih_l1_reverse": "encoder_lstm.weight_ih_l1_reverse",
            "char_lstm.weight_hh_l1_reverse": "encoder_lstm.weight_hh_l1_reverse",
            "char_lstm.bias_ih_l1_reverse": "encoder_lstm.bias_ih_l1_reverse",
            "char_lstm.bias_hh_l1_reverse": "encoder_lstm.bias_hh_l1_reverse",
        }

        own_state = self.state_dict()
        for p2g_key, denoiser_key in mapping.items():
            if p2g_key in p2g_state_dict and denoiser_key in own_state:
                src = p2g_state_dict[p2g_key]
                dst = own_state[denoiser_key]
                if src.shape == dst.shape:
                    own_state[denoiser_key].copy_(src)
                    initialized.append(f"{p2g_key} → {denoiser_key}")
                else:
                    logger.warning(
                        "Shape mismatch pour %s: P2G %s vs denoiser %s",
                        p2g_key, src.shape, dst.shape,
                    )

        self.load_state_dict(own_state)
        return initialized

    def get_config(self) -> dict[str, Any]:
        """Retourne la configuration du modele."""
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CTCDenoiser":
        """Cree un CTCDenoiser depuis un dict de configuration."""
        return cls(**config)

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modele (config + poids).

        Cree un fichier .pt contenant :
        - config : dict de configuration
        - state_dict : poids du modele
        """
        path = Path(path)
        torch.save({
            "config": self._config,
            "state_dict": self.state_dict(),
        }, path)
        logger.info("CTCDenoiser sauvegarde dans %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "CTCDenoiser":
        """Charge un CTCDenoiser depuis un fichier .pt.

        Parameters
        ----------
        path : str | Path
            Chemin vers le fichier .pt.
        device : str | torch.device
            Device cible.

        Returns
        -------
        CTCDenoiser
            Modele charge en mode eval.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        logger.info("CTCDenoiser charge depuis %s", path)
        return model
