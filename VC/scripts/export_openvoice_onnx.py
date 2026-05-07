#!/usr/bin/env python3
"""Export OpenVoice v2 ToneColorConverter to ONNX.

Produces 2 ONNX models:
  - openvoice_se.onnx   : Reference Encoder (spec -> speaker embedding)
  - openvoice_vc.onnx   : Voice Converter (spec + SEs -> audio)

Usage:
    python export_openvoice_onnx.py \
        --config /path/to/converter/config.json \
        --checkpoint /path/to/converter/checkpoint.pth \
        --output_dir /path/to/output/

Requires PyTorch + the OpenVoice source code.
This is a one-time developer operation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_OPENVOICE_SRC = Path("/data/work/projets/lectura/workspace/Voix/vc/OpenVoice")
DEFAULT_CONFIG = DEFAULT_OPENVOICE_SRC / "checkpoints_v2" / "converter" / "config.json"
DEFAULT_CHECKPOINT = DEFAULT_OPENVOICE_SRC / "checkpoints_v2" / "converter" / "checkpoint.pth"


# ── Model wrappers for clean ONNX export ─────────────────────────────────────

class RefEncWrapper(nn.Module):
    """Wraps ref_enc for ONNX export.

    Input:  spec (1, T, 513) float32
    Output: se   (1, 256, 1) float32
    """

    def __init__(self, ref_enc):
        super().__init__()
        self.ref_enc = ref_enc

    def forward(self, spec):
        # ref_enc expects (N, T, n_freq)
        g = self.ref_enc(spec)  # (N, 256)
        return g.unsqueeze(-1)  # (N, 256, 1)


class VoiceConverterWrapper(nn.Module):
    """Wraps the voice_conversion pipeline for ONNX export.

    Uses tau=0 (deterministic) to avoid ONNX random ops.

    Input:  spec (1, 513, T), spec_lengths (1,), sid_src (1,256,1), sid_tgt (1,256,1)
    Output: audio (1, 1, T_audio)
    """

    def __init__(self, model, zero_g: bool = True):
        super().__init__()
        self.enc_q = model.enc_q
        self.flow = model.flow
        self.dec = model.dec
        self.zero_g = zero_g

    def forward(self, spec, spec_lengths, sid_src, sid_tgt):
        g_src = sid_src
        g_tgt = sid_tgt

        g_enc = torch.zeros_like(g_src) if self.zero_g else g_src
        g_dec = torch.zeros_like(g_tgt) if self.zero_g else g_tgt

        # tau=0 -> deterministic: z = m * mask
        z, m_q, logs_q, y_mask = self.enc_q(spec, spec_lengths, g=g_enc, tau=0.0)

        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)

        o_hat = self.dec(z_hat * y_mask, g=g_dec)
        return o_hat


# ── Export functions ──────────────────────────────────────────────────────────

def load_openvoice_model(config_path: Path, checkpoint_path: Path, device: str = "cpu"):
    """Load OpenVoice ToneColorConverter model."""
    # Add OpenVoice to path
    openvoice_root = config_path.parent.parent.parent
    sys.path.insert(0, str(openvoice_root))

    from openvoice.utils import get_hparams_from_file
    from openvoice.models import SynthesizerTrn

    hps = get_hparams_from_file(str(config_path))

    model = SynthesizerTrn(
        len(getattr(hps, "symbols", [])),
        hps.data.filter_length // 2 + 1,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info("Loaded checkpoint: missing=%s, unexpected=%s", missing, unexpected)
    model.eval()

    return model, hps


def export_se(model, output_path: Path):
    """Export the Reference Encoder to ONNX."""
    wrapper = RefEncWrapper(model.ref_enc)
    wrapper.eval()

    # Dummy input: (1, T=100, 513)
    dummy_spec = torch.randn(1, 100, 513)

    torch.onnx.export(
        wrapper,
        (dummy_spec,),
        str(output_path),
        input_names=["spec"],
        output_names=["se"],
        dynamic_axes={
            "spec": {1: "time"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Exported SE model: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


def export_vc(model, hps, output_path: Path):
    """Export the Voice Converter to ONNX."""
    zero_g = getattr(hps.model, "zero_g", True)
    wrapper = VoiceConverterWrapper(model, zero_g=zero_g)
    wrapper.eval()

    # Dummy inputs
    T = 100
    dummy_spec = torch.randn(1, hps.data.filter_length // 2 + 1, T)  # (1, 513, T)
    dummy_lengths = torch.LongTensor([T])
    dummy_src_se = torch.randn(1, 256, 1)
    dummy_tgt_se = torch.randn(1, 256, 1)

    torch.onnx.export(
        wrapper,
        (dummy_spec, dummy_lengths, dummy_src_se, dummy_tgt_se),
        str(output_path),
        input_names=["spec", "spec_lengths", "sid_src", "sid_tgt"],
        output_names=["audio"],
        dynamic_axes={
            "spec": {2: "time"},
            "audio": {2: "audio_time"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Exported VC model: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


def validate_export(model, hps, se_path: Path, vc_path: Path):
    """Validate ONNX export parity with PyTorch."""
    import onnxruntime as ort

    T = 80
    spec_channels = hps.data.filter_length // 2 + 1

    # Test SE
    dummy_spec_se = np.random.randn(1, T, spec_channels).astype(np.float32)
    with torch.no_grad():
        ref_se = model.ref_enc(torch.from_numpy(dummy_spec_se)).unsqueeze(-1).numpy()

    se_sess = ort.InferenceSession(str(se_path))
    onnx_se = se_sess.run(None, {"spec": dummy_spec_se})[0]
    se_diff = np.abs(ref_se - onnx_se).max()
    logger.info("SE parity: max diff = %.6f", se_diff)

    # Test VC
    dummy_spec = np.random.randn(1, spec_channels, T).astype(np.float32)
    dummy_lengths = np.array([T], dtype=np.int64)
    dummy_src = np.random.randn(1, 256, 1).astype(np.float32)
    dummy_tgt = np.random.randn(1, 256, 1).astype(np.float32)

    zero_g = getattr(hps.model, "zero_g", True)
    with torch.no_grad():
        g_enc = torch.zeros(1, 256, 1) if zero_g else torch.from_numpy(dummy_src)
        g_dec = torch.zeros(1, 256, 1) if zero_g else torch.from_numpy(dummy_tgt)
        z, _, _, y_mask = model.enc_q(
            torch.from_numpy(dummy_spec),
            torch.from_numpy(dummy_lengths),
            g=g_enc, tau=0.0,
        )
        z_p = model.flow(z, y_mask, g=torch.from_numpy(dummy_src))
        z_hat = model.flow(z_p, y_mask, g=torch.from_numpy(dummy_tgt), reverse=True)
        ref_audio = model.dec(z_hat * y_mask, g=g_dec).numpy()

    vc_sess = ort.InferenceSession(str(vc_path))
    onnx_audio = vc_sess.run(None, {
        "spec": dummy_spec,
        "spec_lengths": dummy_lengths,
        "sid_src": dummy_src,
        "sid_tgt": dummy_tgt,
    })[0]
    vc_diff = np.abs(ref_audio - onnx_audio).max()
    logger.info("VC parity: max diff = %.6f", vc_diff)

    if se_diff > 1e-4 or vc_diff > 1e-4:
        logger.warning("Parity check FAILED (threshold 1e-4)")
    else:
        logger.info("Parity check PASSED")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export OpenVoice v2 to ONNX")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validate", action="store_true", default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading OpenVoice model...")
    model, hps = load_openvoice_model(args.config, args.checkpoint)

    se_path = args.output_dir / "openvoice_se.onnx"
    vc_path = args.output_dir / "openvoice_vc.onnx"

    logger.info("Exporting SE model...")
    export_se(model, se_path)

    logger.info("Exporting VC model...")
    export_vc(model, hps, vc_path)

    if args.validate:
        logger.info("Validating export...")
        validate_export(model, hps, se_path, vc_path)

    logger.info("Done! Models saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
