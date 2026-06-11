#!/usr/bin/env python3
"""
Export ONNX + quantization INT8 du modele FormulaCTC.

- Export PyTorch -> ONNX (opset 17, dynamic axes batch+time)
- Quantization INT8 via onnxruntime.quantization
- Verification : comparaison sorties PyTorch vs ONNX sur un exemple

Adapte de stt/export_onnx.py (PhoneCTC).

Usage :
  python training/export_onnx.py training/checkpoints/best.pt
  python training/export_onnx.py training/checkpoints/best.pt --output-dir training/exports
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FormulaCTC


def export_onnx(
    model: FormulaCTC,
    output_path: Path,
    opset_version: int = 17,
) -> None:
    """Exporte le modele PyTorch en ONNX."""
    model.eval()
    model.cpu()

    # Dummy input : batch=1, 80 mels, ~2 secondes (200 frames)
    B, T = 1, 200
    dummy_mel = torch.randn(B, 1, 80, T)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_mel,),
        str(output_path),
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={
            "mel": {0: "batch", 3: "time"},
            "logits": {0: "batch", 1: "time_out"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX FP32 : {output_path} ({size_mb:.1f} Mo)")


def quantize_int8(onnx_path: Path) -> Path:
    """Quantifie le modele ONNX en INT8."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = onnx_path.parent / onnx_path.name.replace(".onnx", "_int8.onnx")
    quantize_dynamic(
        str(onnx_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )

    size_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"ONNX INT8 : {int8_path} ({size_mb:.1f} Mo)")
    return int8_path


def verify_onnx(
    model: FormulaCTC,
    onnx_path: Path,
    atol: float = 1e-4,
) -> bool:
    """Verifie que les sorties ONNX correspondent a PyTorch."""
    import onnxruntime as ort

    model.eval()
    model.cpu()

    # Creer un input de test
    B, T = 1, 150
    mel = torch.randn(B, 1, 80, T)

    # PyTorch
    with torch.no_grad():
        pt_logits = model(mel).numpy()

    # ONNX
    session = ort.InferenceSession(str(onnx_path))
    ort_logits = session.run(None, {"mel": mel.numpy()})[0]

    # Comparaison
    max_diff = np.abs(pt_logits - ort_logits).max()
    mean_diff = np.abs(pt_logits - ort_logits).mean()
    match = max_diff < atol

    print(f"\nVerification ONNX :")
    print(f"  Shape PyTorch : {pt_logits.shape}")
    print(f"  Shape ONNX    : {ort_logits.shape}")
    print(f"  Max diff      : {max_diff:.6f}")
    print(f"  Mean diff     : {mean_diff:.6f}")
    print(f"  Match (atol={atol}) : {'OK' if match else 'FAIL'}")

    return match


def main():
    parser = argparse.ArgumentParser(
        description="Export ONNX + INT8 du modele FormulaCTC",
    )
    parser.add_argument("checkpoint", type=Path,
                        help="Chemin vers le checkpoint (.pt)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Dossier de sortie (defaut: training/exports)")
    parser.add_argument("--vocab-size", type=int, default=87,
                        help="Taille du vocabulaire")
    parser.add_argument("--opset", type=int, default=17,
                        help="Version opset ONNX")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Ne pas verifier les sorties ONNX")
    parser.add_argument("--skip-int8", action="store_true",
                        help="Ne pas quantifier en INT8")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "exports"

    # Charger le modele
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Auto-detect architecture from model_config if available
    model_cfg = ckpt.get("model_config", {})
    model = FormulaCTC(
        vocab_size=model_cfg.get("vocab_size", args.vocab_size),
        cnn_channels=model_cfg.get("cnn_channels"),
        gru_hidden=model_cfg.get("gru_hidden", 128),
        gru_layers=model_cfg.get("gru_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = model.count_parameters()
    print(f"Modele : {n_params:,} params ({n_params / 1e6:.2f}M)")
    print(f"Checkpoint : {args.checkpoint} (epoch {ckpt.get('epoch', '?') + 1})")

    # Export ONNX FP32
    onnx_path = args.output_dir / "formula_ctc.onnx"
    export_onnx(model, onnx_path, opset_version=args.opset)

    # Verification
    if not args.skip_verify:
        ok = verify_onnx(model, onnx_path)
        if not ok:
            print("ATTENTION : les sorties ONNX divergent significativement !")

    # INT8
    if not args.skip_int8:
        int8_path = quantize_int8(onnx_path)

        # Verifier INT8 aussi (tolerance plus large)
        if not args.skip_verify:
            print("\nVerification INT8 :")
            verify_onnx(model, int8_path, atol=0.5)

    print("\nExport termine.")


if __name__ == "__main__":
    main()
