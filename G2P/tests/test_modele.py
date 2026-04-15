"""Tests pour le modèle unifié (architecture PyTorch)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from lectura_nlp.modele import UnifiedFrenchNLP, MultiTaskLoss


@pytest.fixture
def small_model():
    """Crée un petit modèle pour les tests."""
    return UnifiedFrenchNLP(
        n_chars=50,
        n_g2p_labels=30,
        n_pos_labels=20,
        n_liaison_labels=7,
        morpho_label_sizes={"Number": 3, "Gender": 3},
        char_embed_dim=16,
        char_hidden_dim=32,
        char_num_layers=1,
        word_hidden_dim=16,
        word_num_layers=1,
        dropout=0.0,
    )


def test_model_creation(small_model):
    """Le modèle se crée correctement."""
    assert small_model is not None
    total = sum(p.numel() for p in small_model.parameters())
    assert total > 0


def test_model_forward_g2p_only(small_model):
    """Forward pass G2P seul (sans word boundaries)."""
    char_ids = torch.tensor([[2, 5, 6, 7, 3]], dtype=torch.long)
    lengths = torch.tensor([5], dtype=torch.long)

    outputs = small_model(char_ids, char_lengths=lengths)

    assert "g2p_logits" in outputs
    assert outputs["g2p_logits"].shape == (1, 5, 30)
    assert "pos_logits" not in outputs


def test_model_forward_full(small_model):
    """Forward pass complet avec word boundaries."""
    # Sentence: <BOS> abc <SEP> def <EOS>
    char_ids = torch.tensor([[2, 5, 6, 7, 4, 8, 9, 10, 3]], dtype=torch.long)
    char_lengths = torch.tensor([9], dtype=torch.long)
    word_starts = torch.tensor([[1, 5]], dtype=torch.long)
    word_ends = torch.tensor([[3, 7]], dtype=torch.long)
    word_lengths = torch.tensor([2], dtype=torch.long)

    outputs = small_model(
        char_ids, char_lengths,
        word_starts, word_ends, word_lengths,
    )

    assert "g2p_logits" in outputs
    assert "pos_logits" in outputs
    assert "liaison_logits" in outputs
    assert "morpho_Number_logits" in outputs
    assert "morpho_Gender_logits" in outputs

    assert outputs["g2p_logits"].shape == (1, 9, 30)
    assert outputs["pos_logits"].shape == (1, 2, 20)
    assert outputs["liaison_logits"].shape == (1, 2, 7)


def test_model_batch(small_model):
    """Forward pass avec batch > 1."""
    char_ids = torch.tensor([
        [2, 5, 6, 3, 0],
        [2, 7, 8, 9, 3],
    ], dtype=torch.long)
    char_lengths = torch.tensor([4, 5], dtype=torch.long)
    word_starts = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
    word_ends = torch.tensor([[2, 0], [3, 0]], dtype=torch.long)
    word_lengths = torch.tensor([1, 1], dtype=torch.long)

    outputs = small_model(
        char_ids, char_lengths,
        word_starts, word_ends, word_lengths,
    )

    assert outputs["g2p_logits"].shape[0] == 2
    assert outputs["pos_logits"].shape[0] == 2


def test_model_config_roundtrip(small_model):
    """Le modèle peut être recréé depuis sa config."""
    config = small_model.get_config()
    model2 = UnifiedFrenchNLP.from_config(config)

    p1 = sum(p.numel() for p in small_model.parameters())
    p2 = sum(p.numel() for p in model2.parameters())
    assert p1 == p2


def test_multitask_loss():
    """Le MultiTaskLoss fonctionne correctement."""
    criterion = MultiTaskLoss()

    outputs = {
        "g2p_logits": torch.randn(2, 10, 30),
        "pos_logits": torch.randn(2, 3, 20),
        "liaison_logits": torch.randn(2, 3, 7),
        "morpho_Number_logits": torch.randn(2, 3, 3),
    }

    targets = {
        "g2p_ids": torch.randint(0, 30, (2, 10)),
        "pos_ids": torch.randint(0, 20, (2, 3)),
        "liaison_ids": torch.randint(0, 7, (2, 3)),
        "morpho_ids": {"Number": torch.randint(0, 3, (2, 3))},
    }

    total_loss, losses = criterion(outputs, targets)

    assert total_loss.item() > 0
    assert "g2p" in losses
    assert "pos" in losses
    assert "liaison" in losses
    assert "total" in losses
