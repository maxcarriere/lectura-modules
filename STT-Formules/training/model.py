#!/usr/bin/env python3
"""
FormulaCTC : CNN-BiGRU-CTC leger pour STT de formules (~600K params).

Architecture :
  Audio 16kHz -> Log-Mel 80 bins -> CNN frontend (x4 subsampling)
  -> BiGRU encoder -> Linear -> CTC

Adapte de PhoneCTC (stt/model.py) avec un modele plus petit et sans
boundary head.

Usage :
  python training/model.py   # smoke test : verifie shapes et params
"""

import torch
import torch.nn as nn


class FormulaCTC(nn.Module):
    """Modele CTC leger pour la reconnaissance de formules.

    Args:
        n_mels: nombre de bins mel (defaut 80)
        vocab_size: taille du vocabulaire (87 tokens, blank=0)
        cnn_channels: canaux des 2 couches CNN (defaut [16, 32])
        gru_hidden: taille hidden du GRU (defaut 128, bidirectionnel -> 256)
        gru_layers: nombre de couches BiGRU (defaut 2)
        dropout: dropout entre couches GRU (defaut 0.1)
    """

    def __init__(
        self,
        n_mels: int = 80,
        vocab_size: int = 87,
        cnn_channels: list[int] | None = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [16, 32]

        # --- CNN Frontend (x4 subsampling en temps ET frequence) ---
        self.cnn = nn.Sequential(
            # Conv1 : (B, 1, n_mels, T) -> (B, 16, n_mels/2, T/2)
            nn.Conv2d(1, cnn_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            # Conv2 : (B, 16, n_mels/2, T/2) -> (B, 32, n_mels/4, T/4)
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(inplace=True),
        )

        # Taille freq apres 2x stride-2 : n_mels // 4
        freq_out = n_mels // 4  # 80 // 4 = 20
        cnn_out_dim = cnn_channels[-1] * freq_out  # 32 x 20 = 640

        # --- Projection lineaire CNN -> GRU ---
        self.proj = nn.Linear(cnn_out_dim, gru_hidden)

        # --- BiGRU Encoder ---
        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # --- CTC Head ---
        self.fc = nn.Linear(gru_hidden * 2, vocab_size)  # bidirectionnel -> x2

    def forward(
        self,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mel: (B, 1, n_mels, T) log-mel spectrogramme
            mel_lengths: (B,) longueurs originales en frames (avant padding)

        Returns:
            logits: (B, T', vocab_size) ou T' = T // 4
        """
        # CNN : (B, 1, 80, T) -> (B, 32, 20, T//4)
        x = self.cnn(mel)

        B, C, F, T = x.shape
        # Reshape : (B, C, F, T) -> (B, T, C*F)
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)

        # Projection
        x = self.proj(x)  # (B, T, gru_hidden)

        # Pack si longueurs fournies (pour ignorer le padding)
        if mel_lengths is not None:
            lengths = (mel_lengths + 3) // 4  # ceil division
            lengths = lengths.clamp(min=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False,
            )

        # BiGRU
        x, _ = self.gru(x)

        if mel_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # CTC head
        logits = self.fc(x)  # (B, T', vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Nombre total de parametres entrainables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────

def _test():
    """Verifie les shapes et le nombre de parametres."""
    model = FormulaCTC()
    n_params = model.count_parameters()
    print(f"Parametres : {n_params:,} ({n_params / 1e6:.2f}M)")

    # Batch de test : 2 exemples, 80 mels, ~3 secondes (300 frames @ 100fps)
    B, T = 2, 300
    mel = torch.randn(B, 1, 80, T)
    mel_lengths = torch.tensor([300, 200])

    logits = model(mel, mel_lengths)
    print(f"Input  : mel {mel.shape}")
    print(f"Output : logits {logits.shape}")
    assert logits.shape[0] == B
    assert logits.shape[1] == T // 4  # subsampling x4
    assert logits.shape[2] == 87      # vocab_size

    # Test sans longueurs (pour export ONNX)
    logits2 = model(mel)
    print(f"Output (no lengths) : logits {logits2.shape}")
    assert logits2.shape == (B, T // 4, 87)

    # Test architecture custom
    model_custom = FormulaCTC(
        cnn_channels=[8, 16],
        gru_hidden=64,
        gru_layers=1,
    )
    n_custom = model_custom.count_parameters()
    print(f"\nCustom : {n_custom:,} params")
    logits3 = model_custom(mel)
    assert logits3.shape == (B, T // 4, 87)

    print("OK — tous les tests passent.")


if __name__ == "__main__":
    _test()
