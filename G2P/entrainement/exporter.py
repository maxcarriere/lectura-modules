#!/usr/bin/env python3
"""Exporte le modèle unifié en ONNX, INT8 quantifié, et JSON weights.

Usage :
    python entrainement/exporter.py --modele modeles/unifie.pt
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
import numpy as np

from lectura_nlp.modele import UnifiedFrenchNLP


class _OnnxWrapper(torch.nn.Module):
    """Wrapper qui retourne un tuple au lieu d'un dict pour l'export ONNX."""

    def __init__(self, model: UnifiedFrenchNLP, morpho_keys: list[str]):
        super().__init__()
        self.model = model
        self.morpho_keys = morpho_keys

    def forward(
        self, char_ids: torch.Tensor,
        word_starts: torch.Tensor, word_ends: torch.Tensor,
    ) -> tuple:
        # Infer word_lengths from word_starts (count non-zero entries)
        word_lengths = (word_starts.sum(dim=-1) > 0).long()
        # Simple heuristic: count words where end > 0 or start > 0
        word_lengths = torch.tensor([word_starts.size(1)], dtype=torch.long)

        outputs = self.model(
            char_ids, None, word_starts, word_ends, word_lengths,
        )
        result = [outputs["g2p_logits"]]
        if "pos_logits" in outputs:
            result.append(outputs["pos_logits"])
        if "liaison_logits" in outputs:
            result.append(outputs["liaison_logits"])
        for key in self.morpho_keys:
            k = f"morpho_{key}_logits"
            if k in outputs:
                result.append(outputs[k])
        return tuple(result)


def export_onnx(
    model: UnifiedFrenchNLP,
    config: dict,
    output_path: Path,
) -> None:
    """Exporte le modèle en ONNX avec axes dynamiques."""
    import onnx

    model.eval()
    model.cpu()

    morpho_keys = sorted(config.get("morpho_label_sizes", {}).keys())
    wrapper = _OnnxWrapper(model, morpho_keys)
    wrapper.eval()

    # Create dummy inputs
    batch = 1
    max_chars = 20
    max_words = 5

    char_ids = torch.zeros(batch, max_chars, dtype=torch.long)
    word_starts = torch.zeros(batch, max_words, dtype=torch.long)
    word_ends = torch.zeros(batch, max_words, dtype=torch.long)

    # Fill with valid dummy data
    char_ids[0, :5] = torch.tensor([2, 5, 6, 7, 3])  # <BOS> a b c <EOS>
    word_starts[0, 0] = 1
    word_ends[0, 0] = 3

    output_names = ["g2p_logits", "pos_logits", "liaison_logits"] + [
        f"morpho_{f}_logits" for f in morpho_keys
    ]

    dynamic_axes = {
        "char_ids": {0: "batch", 1: "max_chars"},
        "word_starts": {0: "batch", 1: "max_words"},
        "word_ends": {0: "batch", 1: "max_words"},
        "g2p_logits": {0: "batch", 1: "max_chars"},
        "pos_logits": {0: "batch", 1: "max_words"},
        "liaison_logits": {0: "batch", 1: "max_words"},
    }

    torch.onnx.export(
        wrapper,
        (char_ids, word_starts, word_ends),
        str(output_path),
        input_names=["char_ids", "word_starts", "word_ends"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        dynamo=False,
    )

    size_kb = output_path.stat().st_size / 1024
    print(f"  ONNX : {output_path} ({size_kb:.0f} Ko)")


def quantize_int8(onnx_path: Path) -> Path:
    """Quantifie le modèle ONNX en INT8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quant_path = onnx_path.parent / onnx_path.name.replace(".onnx", "_int8.onnx")
    quantize_dynamic(
        str(onnx_path), str(quant_path),
        weight_type=QuantType.QInt8,
    )

    size_kb = quant_path.stat().st_size / 1024
    print(f"  INT8 : {quant_path} ({size_kb:.0f} Ko)")
    return quant_path


def export_json_weights(
    model: UnifiedFrenchNLP,
    output_path: Path,
) -> None:
    """Exporte les poids en JSON pour l'inférence pure Python.

    Format :
        {
            "param_name": {
                "shape": [dim1, dim2, ...],
                "data": [flat list of float values]
            }
        }

    Les poids sont arrondis à 6 décimales pour la taille.
    """
    weights = {}
    for name, param in model.state_dict().items():
        arr = param.cpu().numpy()
        weights[name] = {
            "shape": list(arr.shape),
            "data": [round(float(x), 6) for x in arr.flatten()],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f)

    size_kb = output_path.stat().st_size / 1024
    size_mb = size_kb / 1024
    print(f"  JSON weights : {output_path} ({size_mb:.1f} Mo)")


def verify_onnx(
    model: UnifiedFrenchNLP,
    onnx_path: Path,
    vocabs: dict,
) -> bool:
    """Vérifie que ONNX produit les mêmes résultats que PyTorch."""
    import onnxruntime as ort

    model.eval()
    model.cpu()

    # Test input
    test_word = "bonjour"
    char2idx = vocabs["char2idx"]
    char_ids_np = np.array(
        [[char2idx.get(ch, 1) for ch in test_word]], dtype=np.int64
    )
    word_starts_np = np.array([[0]], dtype=np.int64)
    word_ends_np = np.array([[len(test_word) - 1]], dtype=np.int64)

    # PyTorch
    with torch.no_grad():
        char_ids_pt = torch.tensor(char_ids_np)
        word_starts_pt = torch.tensor(word_starts_np)
        word_ends_pt = torch.tensor(word_ends_np)
        word_lengths_pt = torch.tensor([1])
        pt_out = model(char_ids_pt, None, word_starts_pt, word_ends_pt, word_lengths_pt)

    # ONNX
    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(
        None,
        {
            "char_ids": char_ids_np,
            "word_starts": word_starts_np,
            "word_ends": word_ends_np,
        },
    )

    # Compare G2P logits
    pt_g2p = pt_out["g2p_logits"].numpy()
    onnx_g2p = onnx_out[0]

    max_diff = np.abs(pt_g2p - onnx_g2p[:, :pt_g2p.shape[1], :]).max()
    match = max_diff < 0.01

    if match:
        print(f"  Vérification ONNX : OK (max diff = {max_diff:.6f})")
    else:
        print(f"  Vérification ONNX : ÉCHEC (max diff = {max_diff:.6f})")

    return match


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporte le modèle unifié")
    parser.add_argument(
        "--modele", type=Path, default=_ROOT / "modeles" / "unifie.pt",
    )
    parser.add_argument("--skip-json", action="store_true", help="Ne pas exporter JSON weights")
    args = parser.parse_args()

    # Charger le modèle
    print(f"Chargement modèle : {args.modele}")
    checkpoint = torch.load(args.modele, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    vocabs = checkpoint["vocabs"]

    model = UnifiedFrenchNLP.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {total_params:,} paramètres")

    output_dir = args.modele.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── ONNX ──
    print("\n── Export ONNX ──")
    onnx_path = output_dir / "unifie.onnx"
    try:
        export_onnx(model, config, onnx_path)
    except ImportError:
        print("  ERREUR : onnx requis. pip install onnx")
        return

    # ── INT8 ──
    print("\n── Quantification INT8 ──")
    try:
        quant_path = quantize_int8(onnx_path)
    except ImportError:
        print("  onnxruntime.quantization non disponible")
        quant_path = None

    # ── Vocab JSON ──
    print("\n── Vocabulaire ──")
    vocab_path = output_dir / "unifie_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "vocabs": vocabs,
        }, f, ensure_ascii=False, indent=1)
    size_kb = vocab_path.stat().st_size / 1024
    print(f"  Vocab : {vocab_path} ({size_kb:.0f} Ko)")

    # ── JSON weights ──
    if not args.skip_json:
        print("\n── JSON weights ──")
        json_path = output_dir / "unifie_weights.json"
        export_json_weights(model, json_path)

    # ── Vérification ──
    print("\n── Vérification cross-format ──")
    try:
        verify_onnx(model, onnx_path, vocabs)
    except Exception as e:
        print(f"  Vérification impossible : {e}")

    print("\nExport terminé.")


if __name__ == "__main__":
    main()
