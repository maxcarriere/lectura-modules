#!/usr/bin/env python3
"""Exporte le modèle unifié P2G en ONNX, INT8 quantifié, et JSON weights.

Usage :
    python entrainement/exporter.py --modele modeles/unifie_p2g.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
import numpy as np

from lectura_p2g.modele import UnifiedP2G


class _OnnxWrapper(torch.nn.Module):
    """Wrapper qui retourne un tuple au lieu d'un dict pour l'export ONNX."""

    def __init__(self, model: UnifiedP2G, morpho_keys: list[str]):
        super().__init__()
        self.model = model
        self.morpho_keys = morpho_keys

    def forward(
        self, char_ids: torch.Tensor,
        word_starts: torch.Tensor, word_ends: torch.Tensor,
    ) -> tuple:
        word_lengths = torch.tensor([word_starts.size(1)], dtype=torch.long)

        outputs = self.model(
            char_ids, None, word_starts, word_ends, word_lengths,
        )
        result = [outputs["p2g_logits"]]
        if "pos_logits" in outputs:
            result.append(outputs["pos_logits"])
        for key in self.morpho_keys:
            k = f"morpho_{key}_logits"
            if k in outputs:
                result.append(outputs[k])
        return tuple(result)


def export_onnx(
    model: UnifiedP2G,
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

    batch = 1
    max_chars = 20
    max_words = 5

    char_ids = torch.zeros(batch, max_chars, dtype=torch.long)
    word_starts = torch.zeros(batch, max_words, dtype=torch.long)
    word_ends = torch.zeros(batch, max_words, dtype=torch.long)

    char_ids[0, :5] = torch.tensor([2, 5, 6, 7, 3])
    word_starts[0, 0] = 1
    word_ends[0, 0] = 3

    output_names = ["p2g_logits", "pos_logits"] + [
        f"morpho_{f}_logits" for f in morpho_keys
    ]

    dynamic_axes = {
        "char_ids": {0: "batch", 1: "max_chars"},
        "word_starts": {0: "batch", 1: "max_words"},
        "word_ends": {0: "batch", 1: "max_words"},
        "p2g_logits": {0: "batch", 1: "max_chars"},
        "pos_logits": {0: "batch", 1: "max_words"},
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
    model: UnifiedP2G,
    output_path: Path,
) -> None:
    """Exporte les poids en JSON pour l'inférence pure Python."""
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
    model: UnifiedP2G,
    onnx_path: Path,
    vocabs: dict,
) -> bool:
    """Vérifie que ONNX produit les mêmes résultats que PyTorch."""
    import onnxruntime as ort

    model.eval()
    model.cpu()

    # Use a simple IPA input
    test_ipa = "bɔ̃ʒuʁ"
    char2idx = vocabs["char2idx"]
    char_ids_np = np.array(
        [[char2idx.get(ch, 1) for ch in test_ipa]], dtype=np.int64
    )
    word_starts_np = np.array([[0]], dtype=np.int64)
    word_ends_np = np.array([[len(test_ipa) - 1]], dtype=np.int64)

    with torch.no_grad():
        char_ids_pt = torch.tensor(char_ids_np)
        word_starts_pt = torch.tensor(word_starts_np)
        word_ends_pt = torch.tensor(word_ends_np)
        word_lengths_pt = torch.tensor([1])
        pt_out = model(char_ids_pt, None, word_starts_pt, word_ends_pt, word_lengths_pt)

    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(
        None,
        {
            "char_ids": char_ids_np,
            "word_starts": word_starts_np,
            "word_ends": word_ends_np,
        },
    )

    pt_p2g = pt_out["p2g_logits"].numpy()
    onnx_p2g = onnx_out[0]

    max_diff = np.abs(pt_p2g - onnx_p2g[:, :pt_p2g.shape[1], :]).max()
    match = max_diff < 0.01

    if match:
        print(f"  Vérification ONNX : OK (max diff = {max_diff:.6f})")
    else:
        print(f"  Vérification ONNX : ÉCHEC (max diff = {max_diff:.6f})")

    return match


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporte le modèle P2G unifié")
    parser.add_argument(
        "--modele", type=Path, default=_ROOT / "modeles" / "unifie_p2g.pt",
    )
    parser.add_argument("--skip-json", action="store_true", help="Ne pas exporter JSON weights")
    args = parser.parse_args()

    print(f"Chargement modèle : {args.modele}")
    checkpoint = torch.load(args.modele, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    vocabs = checkpoint["vocabs"]

    model = UnifiedP2G.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {total_params:,} paramètres")

    output_dir = args.modele.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine suffix from model filename (e.g. unifie_p2g_v2.pt → _v2)
    stem = args.modele.stem  # unifie_p2g or unifie_p2g_v2
    base = stem.replace("unifie_p2g", "unifie_p2g")  # keep as-is
    suffix = stem.replace("unifie_p2g", "")  # "" or "_v2"

    # ── ONNX ──
    print("\n── Export ONNX ──")
    onnx_path = output_dir / f"unifie_p2g{suffix}.onnx"
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
    vocab_path = output_dir / f"unifie_p2g{suffix}_vocab.json"
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
        json_path = output_dir / f"unifie_p2g{suffix}_weights.json"
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
