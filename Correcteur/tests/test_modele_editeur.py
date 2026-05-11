"""Tests pour le modele BiLSTM edit tagger."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "entrainement"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modele_editeur import (
    INPUT_DIM,
    LSTM_BIDIR_DIM,
    EditTaggerDataset,
    EditTaggerModel,
    collate_fn,
    construire_vocabulaire_mots,
    suf3_hash,
)
from lectura_correcteur._tags import N_TAGS


# -- Test model creation and shapes --

def test_model_creation():
    """Le modele doit se creer sans erreur."""
    model = EditTaggerModel()
    assert model is not None


def test_input_dim():
    """La dimension d'entree doit etre 184."""
    assert INPUT_DIM == 184, f"INPUT_DIM = {INPUT_DIM}, attendu 184"


def test_parameter_count():
    """Le modele doit avoir environ 4.4M parametres."""
    model = EditTaggerModel()
    n_params = model.count_parameters()
    # Tolerance : entre 3M et 6M
    assert 3_000_000 < n_params < 6_000_000, (
        f"Nombre de parametres : {n_params:,} (attendu ~4.4M)"
    )
    print(f"\nParametres : {n_params:,}")


def test_forward_pass():
    """Forward pass avec des donnees factices."""
    model = EditTaggerModel()
    model.eval()

    batch_size = 4
    seq_len = 10

    word_ids = torch.randint(0, 100, (batch_size, seq_len))
    pos_ids = torch.randint(0, 18, (batch_size, seq_len))
    genre_ids = torch.randint(0, 4, (batch_size, seq_len))
    nombre_ids = torch.randint(0, 4, (batch_size, seq_len))
    temps_ids = torch.randint(0, 6, (batch_size, seq_len))
    mode_ids = torch.randint(0, 8, (batch_size, seq_len))
    personne_ids = torch.randint(0, 5, (batch_size, seq_len))
    suf3_ids = torch.randint(0, 2000, (batch_size, seq_len))
    lengths = torch.tensor([10, 8, 6, 4])

    with torch.no_grad():
        logits = model(
            word_ids, pos_ids, genre_ids, nombre_ids,
            temps_ids, mode_ids, personne_ids, suf3_ids, lengths,
        )

    assert logits.shape == (batch_size, seq_len, N_TAGS), (
        f"logits shape = {logits.shape}, attendu ({batch_size}, {seq_len}, {N_TAGS})"
    )


def test_forward_variable_lengths():
    """Le modele gere les longueurs variables correctement."""
    model = EditTaggerModel()
    model.eval()

    # Batch avec longueurs differentes (trie decroissant)
    batch_size = 3
    max_len = 15

    word_ids = torch.randint(0, 100, (batch_size, max_len))
    pos_ids = torch.randint(0, 18, (batch_size, max_len))
    genre_ids = torch.randint(0, 4, (batch_size, max_len))
    nombre_ids = torch.randint(0, 4, (batch_size, max_len))
    temps_ids = torch.randint(0, 6, (batch_size, max_len))
    mode_ids = torch.randint(0, 8, (batch_size, max_len))
    personne_ids = torch.randint(0, 5, (batch_size, max_len))
    suf3_ids = torch.randint(0, 2000, (batch_size, max_len))
    lengths = torch.tensor([15, 10, 5])

    with torch.no_grad():
        logits = model(
            word_ids, pos_ids, genre_ids, nombre_ids,
            temps_ids, mode_ids, personne_ids, suf3_ids, lengths,
        )

    assert logits.shape == (batch_size, max_len, N_TAGS)


def test_output_is_logits():
    """Les sorties sont des logits (pas de softmax applique)."""
    model = EditTaggerModel()
    model.eval()

    word_ids = torch.randint(0, 100, (2, 5))
    pos_ids = torch.randint(0, 18, (2, 5))
    genre_ids = torch.zeros(2, 5, dtype=torch.long)
    nombre_ids = torch.zeros(2, 5, dtype=torch.long)
    temps_ids = torch.zeros(2, 5, dtype=torch.long)
    mode_ids = torch.zeros(2, 5, dtype=torch.long)
    personne_ids = torch.zeros(2, 5, dtype=torch.long)
    suf3_ids = torch.randint(0, 2000, (2, 5))
    lengths = torch.tensor([5, 3])

    with torch.no_grad():
        logits = model(
            word_ids, pos_ids, genre_ids, nombre_ids,
            temps_ids, mode_ids, personne_ids, suf3_ids, lengths,
        )

    # Les logits peuvent etre negatifs (pas de softmax)
    assert logits.min() < 0 or logits.max() > 1


def test_suf3_hash_deterministic():
    """suf3_hash doit etre deterministe."""
    assert suf3_hash("manger") == suf3_hash("manger")
    assert suf3_hash("abc") == suf3_hash("abc")


def test_suf3_hash_range():
    """suf3_hash doit retourner un index dans [0, 2000)."""
    for mot in ["", "a", "ab", "abc", "manger", "constitution"]:
        h = suf3_hash(mot)
        assert 0 <= h < 2000


# -- Test collate function --

def test_collate_fn():
    """collate_fn doit produire des tenseurs paddes."""
    # Simuler 3 samples de tailles differentes
    samples = []
    for length in [5, 3, 8]:
        sample = {
            "word_ids": torch.randint(0, 100, (length,)),
            "pos_ids": torch.randint(0, 18, (length,)),
            "genre_ids": torch.randint(0, 4, (length,)),
            "nombre_ids": torch.randint(0, 4, (length,)),
            "temps_ids": torch.randint(0, 6, (length,)),
            "mode_ids": torch.randint(0, 8, (length,)),
            "personne_ids": torch.randint(0, 5, (length,)),
            "suf3_ids": torch.randint(0, 2000, (length,)),
            "tag_ids": torch.randint(0, N_TAGS, (length,)),
            "length": length,
        }
        samples.append(sample)

    batch = collate_fn(samples)

    # Doit etre trie par longueur decroissante
    assert batch["lengths"][0] == 8
    assert batch["lengths"][1] == 5
    assert batch["lengths"][2] == 3

    # Tous les tenseurs doivent avoir la meme shape (batch, max_len)
    for key in ["word_ids", "pos_ids", "tag_ids"]:
        assert batch[key].shape == (3, 8), f"{key} shape = {batch[key].shape}"


# -- Test dataset with real corpus (if available) --

def test_dataset_with_corpus():
    """Test du dataset avec le corpus edit (si disponible)."""
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "corpus" / "corpus_edit.jsonl"
    if not corpus_path.exists():
        pytest.skip("corpus_edit.jsonl not found (run convertir_corpus_edit.py first)")

    word2idx = construire_vocabulaire_mots(corpus_path, max_vocab=1000)
    ds = EditTaggerDataset(corpus_path, word2idx)

    assert len(ds) > 0

    sample = ds[0]
    assert "word_ids" in sample
    assert "tag_ids" in sample
    assert sample["word_ids"].shape == sample["tag_ids"].shape
