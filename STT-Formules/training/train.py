#!/usr/bin/env python3
"""
Entrainement du modele FormulaCTC (CNN-BiGRU-CTC) pour STT de formules.

Usage :
  python training/train.py \
      --manifest /data/voix_ssd/formula_corpus/manifest.jsonl \
      --out-dir training/ \
      --epochs 100 --batch-size 64 --lr 3e-4 \
      --val-ratio 0.1 --patience 15 --seed 42

Pattern : AMP fp16, gradient accumulation, cosine LR avec warmup,
early stopping sur TER val, TensorBoard, checkpoints.

Adapte de stt/train.py (PhoneCTC) — pas de boundary head, TER au lieu
de PER, manifest JSONL + split stratifie.
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Imports locaux
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CTCCollator, FormulaDataset, load_manifest, stratified_split
from model import FormulaCTC

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# TER (Token Error Rate)
# ──────────────────────────────────────────────

def _levenshtein(s1: list, s2: list) -> int:
    """Distance de Levenshtein entre deux sequences."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _ctc_greedy_decode(logits: torch.Tensor, blank_id: int = 0) -> list[list[int]]:
    """Decodage CTC greedy sur un batch de logits.

    Args:
        logits: (B, T, V) log-probabilites
        blank_id: ID du token blank/PAD

    Returns:
        list de sequences decodees (sans blanks ni repetitions)
    """
    pred_ids = torch.argmax(logits, dim=-1)  # (B, T)
    results = []
    for seq in pred_ids:
        decoded = []
        prev = None
        for idx in seq.tolist():
            if idx != blank_id and idx != prev:
                decoded.append(idx)
            prev = idx
        results.append(decoded)
    return results


def compute_ter(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    blank_id: int = 0,
) -> float:
    """Calcule le Token Error Rate sur un batch.

    Args:
        logits: (B, T, V) sorties du modele
        labels: (B, L) labels paddes
        label_lengths: (B,) longueurs reelles des labels

    Returns:
        TER (float)
    """
    predictions = _ctc_greedy_decode(logits, blank_id=blank_id)

    total_errors = 0
    total_tokens = 0
    for pred, lab, lab_len in zip(predictions, labels, label_lengths):
        ref = lab[:lab_len].tolist()
        dist = _levenshtein(pred, ref)
        total_errors += dist
        total_tokens += len(ref)

    return total_errors / max(total_tokens, 1)


# ──────────────────────────────────────────────
# Warmup + Cosine scheduler
# ──────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup puis cosine decay jusqu'a eta_min."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-6,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        base_lr = optimizer.defaults["lr"]
        self._base_lr = base_lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(eta_min / base_lr, cosine)

        super().__init__(optimizer, lr_lambda)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum: int = 1,
) -> dict:
    """Entraine le modele pendant une epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        mel = batch["mel"].to(device, non_blocking=True)
        mel_lengths = batch["mel_lengths"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        label_lengths = batch["label_lengths"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(mel, mel_lengths)  # (B, T', V)

            # CTC loss attend (T, B, V) en log-probs
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T', B, V)
            input_lengths = (mel_lengths + 3) // 4  # subsampling x4

            loss = nn.functional.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                label_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )

            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * grad_accum
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evalue le modele sur le set de validation."""
    model.eval()
    total_loss = 0.0
    total_ter_errors = 0
    total_ter_tokens = 0
    n_batches = 0

    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        mel_lengths = batch["mel_lengths"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        label_lengths = batch["label_lengths"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(mel, mel_lengths)

            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            input_lengths = (mel_lengths + 3) // 4

            loss = nn.functional.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                label_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )

        total_loss += loss.item()

        # TER (en float32 pour argmax)
        logits_f32 = logits.float()
        predictions = _ctc_greedy_decode(logits_f32, blank_id=0)
        for pred, lab, lab_len in zip(predictions, labels, label_lengths):
            ref = lab[:lab_len].tolist()
            total_ter_errors += _levenshtein(pred, ref)
            total_ter_tokens += len(ref)

        n_batches += 1

    ter = total_ter_errors / max(total_ter_tokens, 1)
    return {
        "loss": total_loss / max(n_batches, 1),
        "ter": ter,
    }


# ──────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────

def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_ter: float,
    patience_counter: int,
    path: Path,
    model_config: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_ter": best_ter,
        "patience_counter": patience_counter,
    }
    if model_config is not None:
        data["model_config"] = model_config
    torch.save(data, path)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrainement FormulaCTC (CNN-BiGRU-CTC) pour STT de formules",
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Chemin vers le manifest JSONL du corpus")
    parser.add_argument("--corpus-dir", type=Path, default=None,
                        help="Dossier racine du corpus (pour resoudre les chemins relatifs)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Dossier de sortie (checkpoints, runs)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-audio-sec", type=float, default=10.0)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Proportion de validation (split stratifie)")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Validation toutes les N epochs")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Checkpoint toutes les N epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Reprendre depuis un checkpoint")
    parser.add_argument("--gru-hidden", type=int, default=128,
                        help="Taille hidden du GRU (defaut: 128)")
    parser.add_argument("--gru-layers", type=int, default=2,
                        help="Nombre de couches BiGRU (defaut: 2)")
    parser.add_argument("--cnn-channels", type=int, nargs=2, default=[16, 32],
                        help="Canaux des 2 couches CNN (defaut: 16 32)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout entre couches GRU (defaut: 0.1)")
    parser.add_argument("--finetune", type=Path, default=None,
                        help="Checkpoint pre-entraine a charger (reset optimizer/scheduler)")
    parser.add_argument("--freeze-encoder", type=int, default=0,
                        help="Freeze CNN+proj+GRU pendant N premieres epochs (defaut: 0)")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Chemins
    if args.out_dir is None:
        args.out_dir = Path(__file__).resolve().parent

    ckpt_dir = args.out_dir / "checkpoints"
    runs_dir = args.out_dir / "runs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device : %s", device)

    # Donnees : charger manifest + split stratifie
    all_entries = load_manifest(args.manifest, corpus_dir=args.corpus_dir)
    log.info("Manifest : %d entrees", len(all_entries))

    train_entries, val_entries = stratified_split(
        all_entries,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_ds = FormulaDataset(
        train_entries,
        max_audio_sec=args.max_audio_sec,
        augment=True,
    )
    val_ds = FormulaDataset(
        val_entries,
        max_audio_sec=args.max_audio_sec,
        augment=False,
    )

    collator = CTCCollator(pad_label_id=0)  # blank=0 pour CTC

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    log.info("Train : %d exemples, %d batches", len(train_ds), len(train_loader))
    log.info("Val   : %d exemples, %d batches", len(val_ds), len(val_loader))

    # Modele
    # vocab_size = 87 par defaut (NUM_TOKENS)
    model = FormulaCTC(
        cnn_channels=args.cnn_channels,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = model.count_parameters()
    log.info("Modele : %s params (%.2fM)", f"{n_params:,}", n_params / 1e6)
    log.info("  Config : gru_hidden=%d, gru_layers=%d, cnn=%s, dropout=%.2f",
             args.gru_hidden, args.gru_layers, args.cnn_channels, args.dropout)

    model_config = {
        "cnn_channels": args.cnn_channels,
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "dropout": args.dropout,
        "vocab_size": 87,
    }

    # Fine-tuning : charger poids pre-entraines (sauf tete CTC)
    if args.finetune is not None:
        ckpt = torch.load(args.finetune, map_location=device, weights_only=False)
        pretrained_sd = ckpt["model"]
        model_sd = model.state_dict()

        loaded = 0
        for key in pretrained_sd:
            if key.startswith("fc."):
                continue  # skip CTC head (vocab different)
            if key in model_sd and pretrained_sd[key].shape == model_sd[key].shape:
                model_sd[key] = pretrained_sd[key]
                loaded += 1

        model.load_state_dict(model_sd)
        log.info("Fine-tune : %d/%d poids charges depuis %s",
                 loaded, len(model_sd), args.finetune)

    # Freeze encoder pendant les premieres epochs
    if args.freeze_encoder > 0:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        log.info("Encoder freeze (CNN+proj+GRU), seule la tete CTC est entrainee")

    # Optimizer (filtre requires_grad pour supporter le freeze encoder)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )

    # Scheduler : warmup lineaire + cosine decay
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda")

    # Resume
    start_epoch = 0
    global_step = 0
    best_ter = float("inf")
    patience_counter = 0

    if args.resume is not None:
        log.info("Reprise depuis %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Verifier coherence config modele
        ckpt_cfg = ckpt.get("model_config", {})
        for key in ("gru_hidden", "gru_layers", "cnn_channels"):
            ckpt_val = ckpt_cfg.get(key)
            cli_val = getattr(args, key.replace("-", "_"))
            if ckpt_val is not None and ckpt_val != cli_val:
                raise ValueError(
                    f"{key} checkpoint={ckpt_val} != CLI={cli_val}. "
                    f"Utilisez les memes parametres pour --resume."
                )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_ter = ckpt.get("best_ter", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)

    # TensorBoard (optionnel)
    writer = SummaryWriter(log_dir=str(runs_dir)) if SummaryWriter else None

    log.info(
        "Demarrage : epochs=%d, batch=%d, grad_accum=%d, effective_batch=%d, lr=%.1e",
        args.epochs, args.batch_size, args.grad_accum,
        args.batch_size * args.grad_accum, args.lr,
    )

    # ──────────────────────────────────────────
    # Boucle d'entrainement
    # ──────────────────────────────────────────

    for epoch in range(start_epoch, args.epochs):
        # Unfreeze encoder apres la phase freeze
        if args.freeze_encoder > 0 and epoch == args.freeze_encoder:
            for param in model.parameters():
                param.requires_grad = True
            # Recreer optimizer et scheduler avec tous les parametres
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.98),
                weight_decay=0.01,
            )
            remaining_epochs = args.epochs - epoch
            remaining_steps = math.ceil(len(train_loader) / args.grad_accum) * remaining_epochs
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_steps=args.warmup_steps,
                total_steps=remaining_steps,
                eta_min=1e-6,
            )
            scaler = torch.amp.GradScaler("cuda")
            log.info("Epoch %d : unfreeze complet, fine-tuning global", epoch + 1)

        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, grad_accum=args.grad_accum,
        )
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        log.info(
            "Epoch %3d/%d  loss=%.4f  lr=%.2e  (%.1fs)",
            epoch + 1, args.epochs,
            train_metrics["loss"], lr, elapsed,
        )
        if writer:
            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/lr", lr, epoch)

        # Validation
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_loader, device)

            log.info(
                "  Val   loss=%.4f  TER=%.2f%%",
                val_metrics["loss"], val_metrics["ter"] * 100,
            )
            if writer:
                writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                writer.add_scalar("val/ter", val_metrics["ter"], epoch)

            # Best model
            if val_metrics["ter"] < best_ter:
                best_ter = val_metrics["ter"]
                patience_counter = 0
                _save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, best_ter, patience_counter,
                    ckpt_dir / "best.pt",
                    model_config=model_config,
                )
                log.info("  -> Nouveau meilleur TER : %.2f%%", best_ter * 100)
            else:
                patience_counter += 1
                log.info(
                    "  -> Patience %d/%d (best TER=%.2f%%)",
                    patience_counter, args.patience, best_ter * 100,
                )

            # Early stopping
            if patience_counter >= args.patience:
                log.info("Early stopping a l'epoch %d", epoch + 1)
                break

        # Checkpoint periodique
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, global_step, best_ter, patience_counter,
                ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
                model_config=model_config,
            )

    if writer:
        writer.close()
    log.info("Entrainement termine. Best TER : %.2f%%", best_ter * 100)
    log.info("Meilleur modele : %s", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
