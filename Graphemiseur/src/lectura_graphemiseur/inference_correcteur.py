"""Inference ONNX du correcteur P2G (CorrecteurP2GInference).

Charge les 3 modeles ONNX (char_encoder + word_context + decoder_step)
et corrige les mots d'une phrase en decodage greedy avec copy mechanism.

Architecture en 3 modeles pour eviter les problemes de padding BiLSTM :
1. char_encoder : un mot a la fois (pas de padding)
2. word_context : word BiLSTM + decoder init (tous les mots)
3. decoder_step : un pas de decodage avec copy mechanism

Usage :
    from lectura_graphemiseur.inference_correcteur import CorrecteurP2GInference

    correcteur = CorrecteurP2GInference(models_dir)
    mots_corriges = correcteur.corriger(["le", "chat", "ai", "bo"])
    # ['le', 'chat', 'est', 'beau']
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Constantes
MAX_WORD_LEN = 30
MAX_DECODE_STEPS = MAX_WORD_LEN + 2


class CorrecteurP2GInference:
    """Inference ONNX du correcteur P2G.

    Charge les 3 sessions ONNX (char_encoder + word_context + decoder_step)
    et le vocabulaire. Corrige une phrase mot par mot avec copy mechanism.
    """

    def __init__(
        self,
        models_dir: str | Path,
        char_encoder_name: str = "correcteur_p2g_char_encoder_int8.onnx",
        word_context_name: str = "correcteur_p2g_word_context_int8.onnx",
        decoder_name: str = "correcteur_p2g_decoder_step_int8.onnx",
        vocab_name: str = "correcteur_p2g_vocab.json",
        copy_threshold: float = 0.0,
    ):
        import onnxruntime as ort

        models_dir = Path(models_dir)

        char_enc_path = models_dir / char_encoder_name
        word_ctx_path = models_dir / word_context_name
        decoder_path = models_dir / decoder_name
        vocab_path = models_dir / vocab_name

        for p, label in [
            (char_enc_path, "Char encoder"),
            (word_ctx_path, "Word context"),
            (decoder_path, "Decoder"),
            (vocab_path, "Vocab"),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"{label} introuvable : {p}")

        # Charger le vocabulaire
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.char2idx: dict[str, int] = vocab_data["char2idx"]
        self.idx2char: dict[int, str] = {
            int(k): v for k, v in vocab_data["idx2char"].items()
        }

        self.pad_idx = self.char2idx.get("<PAD>", 0)
        self.sos_idx = self.char2idx.get("<SOS>", 1)
        self.eos_idx = self.char2idx.get("<EOS>", 2)
        self.unk_idx = self.char2idx.get("<UNK>", 3)
        self.vocab_size = len(self.char2idx)

        # Charger les sessions ONNX
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3

        self._char_encoder = ort.InferenceSession(str(char_enc_path), opts)
        self._word_context = ort.InferenceSession(str(word_ctx_path), opts)
        self._decoder = ort.InferenceSession(str(decoder_path), opts)
        self.copy_threshold = copy_threshold

        # Detecter si le decoder expose copy_prob (4 sorties vs 3)
        self._has_copy_prob = len(self._decoder.get_outputs()) >= 4

        logger.info(
            "CorrecteurP2G charge : %s + %s + %s (%d chars, copy_threshold=%.2f)",
            char_encoder_name, word_context_name, decoder_name,
            self.vocab_size, self.copy_threshold,
        )

    def _encode_word(self, word: str) -> list[int]:
        """Encode un mot en indices de caracteres (avec SOS/EOS)."""
        chars = [self.sos_idx]
        for c in word[:MAX_WORD_LEN]:
            chars.append(self.char2idx.get(c, self.unk_idx))
        chars.append(self.eos_idx)
        return chars

    def _decode_chars(self, char_ids: list[int]) -> str:
        """Decode une liste d'indices en string (sans SOS/EOS/PAD)."""
        chars = []
        for idx in char_ids:
            if idx in (self.pad_idx, self.sos_idx, self.eos_idx):
                continue
            c = self.idx2char.get(idx, "")
            if c and not c.startswith("<"):
                chars.append(c)
        return "".join(chars)

    def corriger(self, words: list[str]) -> list[str]:
        """Corrige une phrase (liste de mots).

        Args:
            words: liste de mots predits par le P2G.

        Returns:
            liste de mots corriges.
        """
        if not words:
            return []

        n_words = len(words)

        # 1. Encoder chaque mot individuellement (pas de padding)
        encoded_words = [self._encode_word(w.lower()) for w in words]
        all_enc_outputs = []
        all_word_reprs = []

        for enc in encoded_words:
            src = np.array([enc], dtype=np.int64)  # (1, word_len)

            enc_result = self._char_encoder.run(None, {
                "src_char_ids": src,
            })
            all_enc_outputs.append(enc_result[0][0])  # (word_len, 256)
            all_word_reprs.append(enc_result[1][0])    # (256,)

        # Pad encoder_outputs to max_len for decoder attention
        max_char_len = max(len(enc) for enc in encoded_words)
        enc_dim = all_enc_outputs[0].shape[-1]
        enc_out_padded = np.zeros((n_words, max_char_len, enc_dim), dtype=np.float32)
        for i, eo in enumerate(all_enc_outputs):
            enc_out_padded[i, :eo.shape[0]] = eo

        word_reprs = np.stack(all_word_reprs, axis=0)  # (N, 256)

        # 2. Word context
        ctx_result = self._word_context.run(None, {
            "word_reprs": word_reprs,
        })
        full_context, hidden_h, hidden_c = ctx_result

        # 3. Decoder loop greedy
        src_char_ids = np.zeros((n_words, max_char_len), dtype=np.int64)
        src_char_lengths = np.zeros(n_words, dtype=np.int64)
        for i, enc in enumerate(encoded_words):
            src_char_ids[i, :len(enc)] = enc
            src_char_lengths[i] = len(enc)

        encoder_mask = np.arange(max_char_len)[None, :] < src_char_lengths[:, None]
        char_input = np.full(n_words, self.sos_idx, dtype=np.int64)
        decoded = [[] for _ in range(n_words)]
        finished = np.zeros(n_words, dtype=bool)
        # Accumuler copy_prob par mot pour le seuil
        copy_prob_sum = np.zeros(n_words, dtype=np.float64)
        copy_prob_count = np.zeros(n_words, dtype=np.int64)

        for t in range(MAX_DECODE_STEPS):
            dec_result = self._decoder.run(None, {
                "char_input": char_input,
                "hidden_h": hidden_h,
                "hidden_c": hidden_c,
                "encoder_outputs": enc_out_padded,
                "full_context": full_context,
                "encoder_mask": encoder_mask,
                "src_char_ids": src_char_ids,
            })
            next_char, hidden_h, hidden_c = dec_result[:3]
            step_copy_prob = dec_result[3] if self._has_copy_prob else None

            for w in range(n_words):
                if not finished[w]:
                    c = int(next_char[w])
                    if step_copy_prob is not None:
                        copy_prob_sum[w] += float(step_copy_prob[w])
                        copy_prob_count[w] += 1
                    if c == self.eos_idx:
                        finished[w] = True
                    else:
                        decoded[w].append(c)

            if finished.all():
                break
            char_input = next_char.astype(np.int64)

        # 4. Decoder les caracteres en strings, avec seuil copy_prob
        use_threshold = self.copy_threshold > 0 and self._has_copy_prob
        result = []
        for w in range(n_words):
            corrected = self._decode_chars(decoded[w])
            if not corrected:
                # Fallback : garder le mot original si le decodage echoue
                result.append(words[w].lower())
                continue

            if use_threshold and copy_prob_count[w] > 0:
                avg_copy_prob = copy_prob_sum[w] / copy_prob_count[w]
                if avg_copy_prob > self.copy_threshold:
                    # Le modele est confiant que ce mot doit etre copie
                    result.append(words[w].lower())
                    continue

            result.append(corrected)

        return result
