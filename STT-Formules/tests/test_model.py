"""Tests pour training/model.py — architecture FormulaCTC."""

import sys
from pathlib import Path

import pytest
import torch

# Ajouter le dossier parent pour importer training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.model import FormulaCTC


class TestFormulaCTCArchitecture:
    """Tests de l'architecture du modele."""

    def test_param_count_range(self):
        """Le modele par defaut a ~600K params (entre 500K et 800K)."""
        model = FormulaCTC()
        n_params = model.count_parameters()
        assert 500_000 < n_params < 800_000, (
            f"Nombre de params {n_params:,} hors plage [500K, 800K]"
        )

    def test_default_vocab_size(self):
        """vocab_size par defaut = 87 (NUM_TOKENS)."""
        model = FormulaCTC()
        assert model.fc.out_features == 87

    def test_no_boundary_head(self):
        """FormulaCTC n'a pas de boundary_fc."""
        model = FormulaCTC()
        assert not hasattr(model, "boundary_fc")

    def test_default_cnn_channels(self):
        """Les canaux CNN par defaut sont [16, 32]."""
        model = FormulaCTC()
        # Conv1 : 1 -> 16
        conv1 = model.cnn[0]
        assert conv1.in_channels == 1
        assert conv1.out_channels == 16
        # Conv2 : 16 -> 32
        conv2 = model.cnn[3]
        assert conv2.in_channels == 16
        assert conv2.out_channels == 32

    def test_default_gru_config(self):
        """GRU par defaut : hidden=128, layers=2, bidirectionnel."""
        model = FormulaCTC()
        assert model.gru.hidden_size == 128
        assert model.gru.num_layers == 2
        assert model.gru.bidirectional is True

    def test_custom_architecture(self):
        """Architecture custom avec parametres differents."""
        model = FormulaCTC(
            cnn_channels=[8, 16],
            gru_hidden=64,
            gru_layers=1,
            vocab_size=50,
        )
        assert model.cnn[0].out_channels == 8
        assert model.cnn[3].out_channels == 16
        assert model.gru.hidden_size == 64
        assert model.gru.num_layers == 1
        assert model.fc.out_features == 50


class TestFormulaCTCForward:
    """Tests du forward pass."""

    def test_output_shape_basic(self):
        """Shape sortie (B, T//4, 87) pour un input standard."""
        model = FormulaCTC()
        model.eval()
        B, T = 2, 300
        mel = torch.randn(B, 1, 80, T)
        with torch.no_grad():
            logits = model(mel)
        assert logits.shape == (B, T // 4, 87)

    @pytest.mark.parametrize("T", [100, 200, 400, 800])
    def test_output_shape_various_lengths(self, T):
        """Shape sortie correcte pour differentes longueurs d'entree."""
        model = FormulaCTC()
        model.eval()
        B = 2
        mel = torch.randn(B, 1, 80, T)
        with torch.no_grad():
            logits = model(mel)
        assert logits.shape == (B, T // 4, 87)

    def test_forward_with_lengths(self):
        """Forward avec mel_lengths fonctionne (packing)."""
        model = FormulaCTC()
        model.eval()
        B, T = 3, 200
        mel = torch.randn(B, 1, 80, T)
        mel_lengths = torch.tensor([200, 150, 100])
        with torch.no_grad():
            logits = model(mel, mel_lengths)
        assert logits.shape[0] == B
        assert logits.shape[2] == 87

    def test_forward_without_lengths(self):
        """Forward sans mel_lengths (pour export ONNX)."""
        model = FormulaCTC()
        model.eval()
        B, T = 1, 200
        mel = torch.randn(B, 1, 80, T)
        with torch.no_grad():
            logits = model(mel)
        assert logits.shape == (B, T // 4, 87)

    def test_batch_size_one(self):
        """Fonctionne avec batch_size=1."""
        model = FormulaCTC()
        model.eval()
        mel = torch.randn(1, 1, 80, 100)
        with torch.no_grad():
            logits = model(mel)
        assert logits.shape == (1, 25, 87)

    def test_gradients_flow(self):
        """Les gradients remontent correctement."""
        model = FormulaCTC()
        model.train()
        mel = torch.randn(2, 1, 80, 100)
        logits = model(mel)
        loss = logits.sum()
        loss.backward()
        # Verifier que les gradients existent
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Pas de gradient pour {name}"
