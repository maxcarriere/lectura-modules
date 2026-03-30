#!/usr/bin/env python3
"""Entraîne le modèle unifié G2P+POS+Morpho+Liaison.

Phase 1 : Pré-entraînement G2P sur le lexique (mots isolés)
Phase 2 : Fine-tuning multi-tâche sur les phrases CoNLL-U enrichies

Usage :
    python entrainement/entrainer.py --donnees entrainement/donnees/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lectura_nlp.modele import (
    UnifiedFrenchNLP,
    LexiqueG2PDataset,
    UnifiedDataset,
    MultiTaskLoss,
    collate_lexique,
    collate_unified,
)
from lectura_nlp.utils.g2p_labels import _CONT, reconstruct_ipa
from lectura_nlp.utils.ipa import iter_phonemes


def _levenshtein(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def eval_g2p_lexique(
    model: UnifiedFrenchNLP,
    data: list[dict],
    char2idx: dict[str, int],
    idx2g2p: dict[int, str],
    device: torch.device,
    max_items: int = 5000,
) -> dict[str, float]:
    """Évalue le G2P sur des mots isolés."""
    model.eval()
    items = data[:max_items]
    total_words = 0
    correct_words = 0
    total_phonemes = 0
    edit_distance_sum = 0

    with torch.no_grad():
        for item in items:
            word = item["word"]
            gold_labels = item["labels"]

            char_ids = torch.tensor(
                [[char2idx.get(ch, 1) for ch in word]],
                dtype=torch.long,
            ).to(device)

            outputs = model(char_ids)
            pred_ids = outputs["g2p_logits"][0].argmax(dim=-1).cpu().tolist()
            pred_labels = [idx2g2p.get(idx, _CONT) for idx in pred_ids[:len(word)]]

            total_words += 1
            if pred_labels == gold_labels:
                correct_words += 1

            pred_ipa = reconstruct_ipa(pred_labels)
            gold_ipa = reconstruct_ipa(gold_labels)
            pred_ph = iter_phonemes(pred_ipa)
            gold_ph = iter_phonemes(gold_ipa)
            total_phonemes += len(gold_ph)
            edit_distance_sum += _levenshtein(pred_ph, gold_ph)

    word_acc = correct_words / total_words if total_words else 0
    per = edit_distance_sum / total_phonemes if total_phonemes else 0

    return {"word_acc": word_acc, "per": per, "n_words": total_words}


def compute_liaison_weights(
    sentences: list[dict],
    use_sqrt: bool = True,
    max_weight: float = 20.0,
) -> torch.Tensor:
    """Calcule les poids de classe pour liaison.

    Utilise sqrt(inverse_freq) pour éviter un biais trop fort vers les classes
    rares (qui causait beaucoup de faux positifs avec les poids bruts).
    Le poids de PAD (index 0) est mis à 0 car on utilise ignore_index=0
    et le label smoothing pourrait sinon inciter le modèle à prédire PAD.
    Les poids sont capés à max_weight pour éviter l'explosion sur les classes
    très rares (Lr: 70 exemples, Lp: 13 exemples).
    """
    counts = Counter()
    for sent in sentences:
        for tok in sent["tokens"]:
            counts[tok["liaison"]] += 1

    labels = ["<PAD>", "none", "Lz", "Lt", "Ln", "Lr", "Lp"]
    total = sum(counts.values())
    weights = []
    for lab in labels:
        if lab == "<PAD>":
            weights.append(0.0)  # PAD never appears as a real target
            continue
        c = counts.get(lab, 1)
        w = total / (len(labels) * c)
        if use_sqrt:
            w = w ** 0.5  # sqrt dampening
        weights.append(w)

    # Normaliser pour que le poids de "none" soit ~1.0
    none_w = weights[labels.index("none")] if "none" in labels else 1.0
    weights = [w / none_w if w > 0 else 0.0 for w in weights]

    # Cap max weight to avoid extreme overprediction of rare classes
    weights = [min(w, max_weight) if w > 0 else 0.0 for w in weights]

    return torch.tensor(weights, dtype=torch.float32)


def train_phase1(
    model: UnifiedFrenchNLP,
    train_data: list[dict],
    eval_data: list[dict] | None,
    vocabs: dict,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> None:
    """Phase 1 : Pré-entraînement G2P sur le lexique."""
    print("\n" + "=" * 60)
    print("PHASE 1 : Pré-entraînement G2P (lexique)")
    print("=" * 60)

    char2idx = vocabs["char2idx"]
    g2p_label2idx = vocabs["g2p_label2idx"]
    idx2g2p = {v: k for k, v in g2p_label2idx.items()}

    dataset = LexiqueG2PDataset(train_data, char2idx, g2p_label2idx)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_lexique,
    )

    # Freeze word-level heads during phase 1
    for name, param in model.named_parameters():
        if "word_lstm" in name or "pos_head" in name or "liaison_head" in name or "morpho" in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    n_labels = len(g2p_label2idx)

    print(f"  Dataset : {len(dataset)} mots")
    print(f"  Params G2P entraînés : "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for chars, labels, lengths in loader:
            chars = chars.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Phase 1: only char-level, pass only char_ids
            outputs = model(chars, char_lengths=lengths)
            logits = outputs["g2p_logits"]

            loss = criterion(
                logits.reshape(-1, n_labels),
                labels.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            msg = f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | {elapsed:.0f}s"

            if eval_data:
                metrics = eval_g2p_lexique(model, eval_data, char2idx, idx2g2p, device)
                msg += f" | word_acc={metrics['word_acc']:.1%} PER={metrics['per']:.1%}"

            print(msg)

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    print(f"  Phase 1 terminée ({time.time() - t0:.0f}s)")


def train_phase2(
    model: UnifiedFrenchNLP,
    train_sentences: list[dict],
    dev_sentences: list[dict],
    vocabs: dict,
    phone_to_graphs: dict[str, list[str]],
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 5e-4,
    patience: int = 20,
    label_smoothing: float = 0.1,
    save_path: Path | None = None,
) -> None:
    """Phase 2 : Fine-tuning multi-tâche sur les phrases."""
    print("\n" + "=" * 60)
    print("PHASE 2 : Fine-tuning multi-tâche (CoNLL-U)")
    print("=" * 60)

    print("  Préparation dataset train...")
    t0 = time.time()
    train_dataset = UnifiedDataset(train_sentences, vocabs, phone_to_graphs)
    print(f"  Train dataset : {len(train_dataset)} phrases ({time.time() - t0:.0f}s)")

    dev_dataset = None
    if dev_sentences:
        print("  Préparation dataset dev...")
        dev_dataset = UnifiedDataset(dev_sentences, vocabs, phone_to_graphs)
        print(f"  Dev dataset : {len(dev_dataset)} phrases")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_unified,
    )

    # Liaison class weights (sqrt dampened to reduce false positives)
    liaison_weights = compute_liaison_weights(train_sentences, use_sqrt=True).to(device)
    print(f"  Liaison weights (sqrt) : {[f'{w:.2f}' for w in liaison_weights.tolist()]}")

    criterion = MultiTaskLoss(
        w_g2p=1.0, w_pos=1.0, w_morpho=0.8, w_liaison=3.0,
        liaison_class_weights=liaison_weights,
        label_smoothing=label_smoothing,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Warmup + cosine annealing
    warmup_epochs = 5
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(1e-6 / lr, 0.5 * (1.0 + __import__('math').cos(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Loss weights : g2p=1.0 pos=1.0 morpho=0.8 liaison=3.0")
    print(f"  Label smoothing : {label_smoothing}")
    print(f"  LR warmup : {warmup_epochs} epochs, patience : {patience}")

    best_dev_loss = float("inf")
    best_state = None
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: dict[str, float] = {}
        n_batches = 0

        for batch_data in train_loader:
            # Move to device
            char_ids = batch_data["char_ids"].to(device)
            char_lengths = batch_data["char_lengths"].to(device)
            word_starts = batch_data["word_starts"].to(device)
            word_ends = batch_data["word_ends"].to(device)
            word_lengths = batch_data["word_lengths"].to(device)

            outputs = model(
                char_ids, char_lengths,
                word_starts, word_ends, word_lengths,
            )

            targets = {
                "g2p_ids": batch_data["g2p_ids"].to(device),
                "pos_ids": batch_data["pos_ids"].to(device),
                "liaison_ids": batch_data["liaison_ids"].to(device),
                "morpho_ids": {
                    k: v.to(device) for k, v in batch_data["morpho_ids"].items()
                },
            }

            total_loss, losses = criterion(outputs, targets)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_batches += 1

        scheduler.step()

        # Average losses
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

        # Dev evaluation every 2 epochs (more frequent for better checkpointing)
        dev_loss = None
        if dev_dataset and (epoch % 2 == 0 or epoch == 1 or epoch == epochs):
            dev_loss = eval_phase2(model, dev_dataset, criterion, device)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Log every 2 epochs
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"  Epoch {epoch:3d}/{epochs} | "
                f"total={avg_losses.get('total', 0):.4f} "
                f"g2p={avg_losses.get('g2p', 0):.4f} "
                f"pos={avg_losses.get('pos', 0):.4f} "
                f"liaison={avg_losses.get('liaison', 0):.4f} "
                f"lr={cur_lr:.2e} "
                f"| {elapsed:.0f}s"
            )
            if dev_loss is not None:
                msg += f" | dev={dev_loss:.4f}"
                if epoch == best_epoch:
                    msg += " *"
            print(msg)

        # Early stopping
        if patience > 0 and best_epoch > 0 and (epoch - best_epoch) >= patience:
            print(f"  Early stopping à epoch {epoch} "
                  f"(pas d'amélioration depuis epoch {best_epoch})")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        print(f"  Meilleur modèle restauré (epoch {best_epoch}, dev_loss={best_dev_loss:.4f})")

    # Save
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model.get_config(),
            "vocabs": vocabs,
        }, save_path)
        size_kb = save_path.stat().st_size / 1024
        print(f"  Modèle sauvegardé : {save_path} ({size_kb:.0f} Ko)")

    print(f"  Phase 2 terminée ({time.time() - t0:.0f}s)")


def eval_phase2(
    model: UnifiedFrenchNLP,
    dataset: "UnifiedDataset",
    criterion: MultiTaskLoss,
    device: torch.device,
) -> float:
    """Évalue le modèle sur un dataset (retourne la loss totale)."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_unified,
    )

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_data in loader:
            char_ids = batch_data["char_ids"].to(device)
            char_lengths = batch_data["char_lengths"].to(device)
            word_starts = batch_data["word_starts"].to(device)
            word_ends = batch_data["word_ends"].to(device)
            word_lengths = batch_data["word_lengths"].to(device)

            outputs = model(
                char_ids, char_lengths,
                word_starts, word_ends, word_lengths,
            )

            targets = {
                "g2p_ids": batch_data["g2p_ids"].to(device),
                "pos_ids": batch_data["pos_ids"].to(device),
                "liaison_ids": batch_data["liaison_ids"].to(device),
                "morpho_ids": {
                    k: v.to(device) for k, v in batch_data["morpho_ids"].items()
                },
            }

            loss, _ = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle unifié G2P+POS+Morpho+Liaison"
    )
    parser.add_argument(
        "--donnees", type=Path,
        default=Path(__file__).parent / "donnees",
    )
    parser.add_argument(
        "--sortie", type=Path,
        default=_ROOT / "modeles",
    )
    parser.add_argument("--seed", type=int, default=42)
    # Phase 1
    parser.add_argument("--phase1-epochs", type=int, default=30)
    parser.add_argument("--phase1-batch", type=int, default=128)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Charger un checkpoint existant (skip Phase 1)")
    # Phase 2
    parser.add_argument("--phase2-epochs", type=int, default=80)
    parser.add_argument("--phase2-batch", type=int, default=32)
    parser.add_argument("--phase2-lr", type=float, default=5e-4)
    parser.add_argument("--phase2-patience", type=int, default=20,
                        help="Early stopping patience (epochs without improvement)")
    # Architecture
    parser.add_argument("--char-embed-dim", type=int, default=64)
    parser.add_argument("--char-hidden-dim", type=int, default=160)
    parser.add_argument("--char-num-layers", type=int, default=2)
    parser.add_argument("--word-hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Charger les données préparées ──
    donnees = args.donnees
    print(f"Chargement des données depuis : {donnees}")

    with open(donnees / "vocabs.json", encoding="utf-8") as f:
        vocabs = json.load(f)

    with open(donnees / "phone_to_graphs.json", encoding="utf-8") as f:
        phone_to_graphs = json.load(f)

    with open(donnees / "sentences_train.json", encoding="utf-8") as f:
        train_sentences = json.load(f)
    with open(donnees / "sentences_dev.json", encoding="utf-8") as f:
        dev_sentences = json.load(f)

    lexique_train = []
    lex_path = donnees / "lexique_g2p_train.json"
    if lex_path.exists():
        with open(lex_path, encoding="utf-8") as f:
            lexique_train = json.load(f)

    lexique_eval = []
    lex_eval_path = donnees / "lexique_g2p_eval.json"
    if lex_eval_path.exists():
        with open(lex_eval_path, encoding="utf-8") as f:
            lexique_eval = json.load(f)

    print(f"  Vocabs chars: {len(vocabs['char2idx'])}, "
          f"g2p: {len(vocabs['g2p_label2idx'])}, "
          f"pos: {len(vocabs['pos2idx'])}")
    print(f"  Train: {len(train_sentences)} phrases, "
          f"Dev: {len(dev_sentences)} phrases")
    print(f"  Lexique: {len(lexique_train)} train, {len(lexique_eval)} eval")

    # ── Créer ou charger le modèle ──
    morpho_label_sizes = {
        feat: len(vocab)
        for feat, vocab in vocabs["morpho_vocabs"].items()
    }

    if args.checkpoint:
        print(f"\nChargement checkpoint : {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = UnifiedFrenchNLP.from_config(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Override vocabs from checkpoint if available
        if "vocabs" in checkpoint:
            vocabs = checkpoint["vocabs"]
        args.skip_phase1 = True  # Skip phase 1 when loading checkpoint
    else:
        model = UnifiedFrenchNLP(
            n_chars=len(vocabs["char2idx"]),
            n_g2p_labels=len(vocabs["g2p_label2idx"]),
            n_pos_labels=len(vocabs["pos2idx"]),
            n_liaison_labels=len(vocabs["liaison2idx"]),
            morpho_label_sizes=morpho_label_sizes,
            char_embed_dim=args.char_embed_dim,
            char_hidden_dim=args.char_hidden_dim,
            char_num_layers=args.char_num_layers,
            word_hidden_dim=args.word_hidden_dim,
            dropout=args.dropout,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModèle : {total_params:,} paramètres")

    # ── Phase 1 ──
    if not args.skip_phase1 and lexique_train:
        train_phase1(
            model, lexique_train,
            lexique_eval if lexique_eval else None,
            vocabs, device,
            epochs=args.phase1_epochs,
            batch_size=args.phase1_batch,
            lr=args.phase1_lr,
        )

    # ── Phase 2 ──
    save_path = args.sortie / "unifie.pt"
    train_phase2(
        model, train_sentences, dev_sentences,
        vocabs, phone_to_graphs, device,
        epochs=args.phase2_epochs,
        batch_size=args.phase2_batch,
        lr=args.phase2_lr,
        patience=args.phase2_patience,
        label_smoothing=args.label_smoothing,
        save_path=save_path,
    )

    print("\nEntraînement terminé.")


if __name__ == "__main__":
    main()
