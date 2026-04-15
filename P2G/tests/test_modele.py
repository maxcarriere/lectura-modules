"""Tests pour le modèle unifié P2G (architecture PyTorch)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from lectura_p2g.modele import UnifiedP2G, MultiTaskLoss


@pytest.fixture
def small_model():
    """Crée un petit modèle pour les tests."""
    return UnifiedP2G(
        n_chars=50,
        n_p2g_labels=30,
        n_pos_labels=20,
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


def test_model_forward_p2g_only(small_model):
    """Forward pass P2G seul (sans word boundaries)."""
    char_ids = torch.tensor([[2, 5, 6, 7, 3]], dtype=torch.long)
    lengths = torch.tensor([5], dtype=torch.long)

    outputs = small_model(char_ids, char_lengths=lengths)

    assert "p2g_logits" in outputs
    assert outputs["p2g_logits"].shape == (1, 5, 30)
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

    assert "p2g_logits" in outputs
    assert "pos_logits" in outputs
    assert "morpho_Number_logits" in outputs
    assert "morpho_Gender_logits" in outputs

    assert outputs["p2g_logits"].shape == (1, 9, 30)
    assert outputs["pos_logits"].shape == (1, 2, 20)
    # No liaison head
    assert "liaison_logits" not in outputs


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

    assert outputs["p2g_logits"].shape[0] == 2
    assert outputs["pos_logits"].shape[0] == 2


def test_model_config_roundtrip(small_model):
    """Le modèle peut être recréé depuis sa config."""
    config = small_model.get_config()
    model2 = UnifiedP2G.from_config(config)

    p1 = sum(p.numel() for p in small_model.parameters())
    p2 = sum(p.numel() for p in model2.parameters())
    assert p1 == p2


def test_multitask_loss():
    """Le MultiTaskLoss fonctionne correctement (sans liaison)."""
    criterion = MultiTaskLoss()

    outputs = {
        "p2g_logits": torch.randn(2, 10, 30),
        "pos_logits": torch.randn(2, 3, 20),
        "morpho_Number_logits": torch.randn(2, 3, 3),
    }

    targets = {
        "p2g_ids": torch.randint(0, 30, (2, 10)),
        "pos_ids": torch.randint(0, 20, (2, 3)),
        "morpho_ids": {"Number": torch.randint(0, 3, (2, 3))},
    }

    total_loss, losses = criterion(outputs, targets)

    assert total_loss.item() > 0
    assert "p2g" in losses
    assert "pos" in losses
    assert "total" in losses
    # No liaison in P2G
    assert "liaison" not in losses


def test_p2g_labels():
    """Test the P2G label generation."""
    from lectura_p2g.utils.p2g_labels import labels_from_p2g_alignment, reconstruct_ortho, _CONT

    # Simple case: "ʃa" → ["ch", "a"]
    # dec_ph = ["ʃ", "a"], dec_gr = ["ch", "a"]
    labels = labels_from_p2g_alignment("ʃa", ["ʃ", "a"], ["ch", "a"])
    assert labels == ["ch", "a"]
    assert reconstruct_ortho(labels) == "cha"

    # Nasal vowel: "bɔ̃" (3 Unicode chars: b, ɔ, ̃) → ["b", "on", "_CONT"]
    phone = "bɔ̃"  # b + ɔ + combining tilde
    dec_ph = ["b", "ɔ̃"]
    dec_gr = ["b", "on"]
    labels = labels_from_p2g_alignment(phone, dec_ph, dec_gr)
    assert len(labels) == len(phone)
    assert labels[0] == "b"
    assert labels[1] == "on"
    assert labels[2] == _CONT
    assert reconstruct_ortho(labels) == "bon"
