"""CTC prefix beam search avec scoring LM phone-level via KenLM.

Implemente l'algorithme de Graves & Jaitly (2014) avec un LM incremental
operant directement sur les phones individuels (pas les mots concatenes).

Compatible avec les tokens IPA multi-caracteres (nasales, affricatives)
contrairement a pyctcdecode qui concatene les labels en "mots".

Dependances : numpy, kenlm (pip install kenlm)

Copyright (C) 2025-2026 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import kenlm as _kenlm


# Tokens de ponctuation : le LM phone n'a pas ete entraine avec
_PUNCT_TOKENS = {",", ".", "?", "!", "…"}

# Tokens speciaux CTC v2 (liaisons, elision)
_SPECIAL_TOKENS = {"[z]", "[t]", "[n]", "[ʁ]", "[p]", "[']"}


def _logaddexp(a: float, b: float) -> float:
    """Log-sum-exp stable pour deux scalaires."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def _merge_hyp(
    hyps: dict[tuple, list],
    prefix: tuple,
    log_pb: float,
    log_pnb: float,
    lm_state: object,
    lm_score: float,
) -> None:
    """Fusionne une hypothese dans le dict (merge par prefix identique)."""
    if prefix in hyps:
        old_log_pb, old_log_pnb, old_lm_state, old_lm_score = hyps[prefix]
        new_log_pb = _logaddexp(old_log_pb, log_pb)
        new_log_pnb = _logaddexp(old_log_pnb, log_pnb)
        old_total = _logaddexp(old_log_pb, old_log_pnb) + old_lm_score
        new_total = _logaddexp(log_pb, log_pnb) + lm_score
        if new_total > old_total:
            hyps[prefix] = (new_log_pb, new_log_pnb, lm_state, lm_score)
        else:
            hyps[prefix] = (new_log_pb, new_log_pnb, old_lm_state, old_lm_score)
    else:
        hyps[prefix] = (log_pb, log_pnb, lm_state, lm_score)


class PhoneLMBeamDecoder:
    """CTC prefix beam search avec LM KenLM phone-level.

    Le LM opere sur les phones individuels (tokens du vocab CTC),
    pas sur les mots phonetiques concatenes.

    Parameters
    ----------
    vocab : dict[str, int]
        Mapping phone → CTC ID.
    lm_path : str
        Chemin vers le modele KenLM (.bin ou .arpa).
    alpha : float
        Poids du LM (defaut: 0.3). Valeur basse recommandee.
    beta : float
        Bonus par longueur de prefix (defaut: 0.5).
    beam_width : int
        Nombre d'hypotheses a garder (defaut: 10).
    top_k : int
        Nombre de tokens a considerer par frame (defaut: 10).
    """

    def __init__(
        self,
        vocab: dict[str, int],
        lm_path: str,
        alpha: float = 0.3,
        beta: float = 0.5,
        beam_width: int = 10,
        top_k: int = 10,
    ) -> None:
        import kenlm

        self.vocab = vocab
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        self.top_k = top_k

        self.lm = kenlm.Model(lm_path)
        self.blank_id = vocab.get("[PAD]", 0)

        # Mapping CTC ID → token LM et detection LM-invisibles
        self.id_to_lm_token: dict[int, str | None] = {}
        self.lm_invisible: set[int] = set()

        for phone, idx in vocab.items():
            if phone == "[PAD]":
                self.id_to_lm_token[idx] = None
            elif phone == "[UNK]":
                self.id_to_lm_token[idx] = "<unk>"
            elif phone == "|":
                self.id_to_lm_token[idx] = "|"
            elif phone in _PUNCT_TOKENS or phone in _SPECIAL_TOKENS:
                self.id_to_lm_token[idx] = None
                self.lm_invisible.add(idx)
            else:
                self.id_to_lm_token[idx] = phone

        self.vocab_size = max(vocab.values()) + 1

    def decode(
        self, log_probs: np.ndarray, length: int | None = None,
    ) -> list[int]:
        """Decode une sequence de log-probs (T, V) → liste de phone IDs.

        Parameters
        ----------
        log_probs : np.ndarray
            (T, V) log-probabilites (log_softmax).
        length : int | None
            Nombre de frames valides (si None, utilise T complet).

        Returns
        -------
        list[int]
            Phone IDs decodes (sans blank ni repetitions).
        """
        import kenlm

        T, V = log_probs.shape
        if length is not None:
            T = min(T, length)

        NEG_INF = float("-inf")

        init_state = kenlm.State()
        self.lm.BeginSentenceWrite(init_state)

        hyps: dict[tuple, tuple] = {
            (): (0.0, NEG_INF, init_state, 0.0),
        }

        for t in range(T):
            frame = log_probs[t]

            if self.top_k < V:
                top_indices = np.argpartition(frame, -self.top_k)[-self.top_k:]
                if self.blank_id not in top_indices:
                    top_indices = np.append(top_indices, self.blank_id)
            else:
                top_indices = np.arange(V)

            new_hyps: dict[tuple, list] = {}

            for prefix, (log_pb, log_pnb, lm_state, lm_score) in hyps.items():
                log_p_prefix = _logaddexp(log_pb, log_pnb)

                for c in top_indices:
                    c = int(c)
                    log_p_c = float(frame[c])

                    if c == self.blank_id:
                        _merge_hyp(new_hyps, prefix,
                                   log_p_prefix + log_p_c, NEG_INF,
                                   lm_state, lm_score)

                    elif prefix and c == prefix[-1]:
                        # Repetition : accumule dans log_pnb
                        _merge_hyp(new_hyps, prefix,
                                   NEG_INF, log_pnb + log_p_c,
                                   lm_state, lm_score)
                        # Extension apres blank (meme token, nouveau)
                        new_prefix = prefix + (c,)
                        new_log_pnb_ext = log_pb + log_p_c
                        if c in self.lm_invisible:
                            _merge_hyp(new_hyps, new_prefix,
                                       NEG_INF, new_log_pnb_ext,
                                       lm_state, lm_score)
                        else:
                            lm_token = self.id_to_lm_token.get(c)
                            if lm_token is not None:
                                out_state = kenlm.State()
                                lm_logprob = self.lm.BaseScore(
                                    lm_state, lm_token, out_state)
                                _merge_hyp(new_hyps, new_prefix,
                                           NEG_INF, new_log_pnb_ext,
                                           out_state, lm_score + lm_logprob)
                            else:
                                _merge_hyp(new_hyps, new_prefix,
                                           NEG_INF, new_log_pnb_ext,
                                           lm_state, lm_score)

                    else:
                        # Extension par un nouveau token
                        new_prefix = prefix + (c,)
                        new_log_pnb = log_p_prefix + log_p_c

                        if c in self.lm_invisible:
                            _merge_hyp(new_hyps, new_prefix,
                                       NEG_INF, new_log_pnb,
                                       lm_state, lm_score)
                        else:
                            lm_token = self.id_to_lm_token.get(c)
                            if lm_token is not None:
                                out_state = kenlm.State()
                                lm_logprob = self.lm.BaseScore(
                                    lm_state, lm_token, out_state)
                                _merge_hyp(new_hyps, new_prefix,
                                           NEG_INF, new_log_pnb,
                                           out_state, lm_score + lm_logprob)
                            else:
                                _merge_hyp(new_hyps, new_prefix,
                                           NEG_INF, new_log_pnb,
                                           lm_state, lm_score)

            # Pruning
            scored = []
            for prefix, (log_pb, log_pnb, lm_st, lm_sc) in new_hyps.items():
                log_ctc = _logaddexp(log_pb, log_pnb)
                combined = log_ctc + self.alpha * lm_sc + self.beta * len(prefix)
                scored.append((combined, prefix, log_pb, log_pnb, lm_st, lm_sc))

            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:self.beam_width]

            hyps = {}
            for combined, prefix, log_pb, log_pnb, lm_st, lm_sc in scored:
                hyps[prefix] = (log_pb, log_pnb, lm_st, lm_sc)

        # Selection finale avec bonus EOS
        best_prefix: tuple = ()
        best_score = NEG_INF
        for prefix, (log_pb, log_pnb, lm_st, lm_sc) in hyps.items():
            log_ctc = _logaddexp(log_pb, log_pnb)
            out_state = kenlm.State()
            eos_score = self.lm.BaseScore(lm_st, "</s>", out_state)
            combined = (log_ctc
                        + self.alpha * (lm_sc + eos_score)
                        + self.beta * len(prefix))
            if combined > best_score:
                best_score = combined
                best_prefix = prefix

        return list(best_prefix)

    def decode_logits(
        self, logits: np.ndarray, length: int | None = None,
    ) -> list[int]:
        """Decode des logits bruts (T, V) — applique log_softmax.

        Parameters
        ----------
        logits : np.ndarray
            (T, V) logits bruts (avant softmax).
        length : int | None
            Nombre de frames valides.

        Returns
        -------
        list[int]
            Phone IDs decodes.
        """
        # log_softmax numpy
        max_vals = logits.max(axis=-1, keepdims=True)
        shifted = logits - max_vals
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        return self.decode(log_probs, length=length)
