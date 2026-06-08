"""Inférence ONNX Runtime pour le modèle unifié P2G V2+ (avec features lexicales).

V2 par rapport à V1 :
  - Supporte les lex_features (24d) pour améliorer POS/Morpho
  - API V1-compatible (analyser) + API V2 (analyser_v2 avec top-K POS/Morpho)
  - Conserve analyser_avec_alternatives() pour les alternatives P2G
  - V4 : Supporte lex_select (sélection ortho parmi candidats lexique)
  - V6 : phone_lex_features (28d), NeighborContext, LexSelectHead V3

Usage :
    engine = OnnxInferenceEngineV2(
        "modeles/unifie_p2g_v6_int8.onnx", "modeles/unifie_p2g_v6_vocab.json",
        phone_lexicon=phone_lexicon,  # PhoneLexicon pour lex_select + phone_lex_features
    )
    result = engine.analyser(["le", "ʃa"])
    result = engine.analyser_v2(["le", "ʃa"], top_k=3)
    result = engine.analyser_avec_alternatives(["le", "ʃa"], k=5)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from lectura_graphemiseur.utils.p2g_labels import reconstruct_ortho

logger = logging.getLogger(__name__)

ALL_LEX_POS = [
    "ADJ", "ADJ:dem", "ADJ:ind", "ADJ:int", "ADJ:num", "ADJ:pos",
    "ADV", "ART:def", "ART:ind", "AUX", "CON", "INTJ",
    "NOM", "PRE",
    "PRO:dem", "PRO:ind", "PRO:int", "PRO:per", "PRO:pos", "PRO:rel",
    "VER",
]

LEX_FEATURE_DIM = len(ALL_LEX_POS) + 3

# V3/V6 phone_lex_features: 28d (lookup par IPA)
PHONE_LEX_FEATURE_DIM = 28

CAND_FEAT_DIM_FULL = 42   # V7 : 42d (POS+genre+nombre+freq+VerbForm+Tense+Person+is_lemme)
CAND_FEAT_DIM_V6 = 30     # V6 : 30d (POS+genre+nombre+freq)
K_MAX = 20

# ── Elision : decomposition des phones avec apostrophe ────────────────
# Correspondance IPA prefix → ortho prefix pour les elisions francaises.
# Ex: ʒ'ɑ̃kyl → prefix "ʒ" → ortho "j", base phone "ɑ̃kyl"
_ELISION_IPA_TO_ORTHO = {
    "ʒ": "j",      # je → j'ai
    "l": "l",      # le/la → l'homme
    "d": "d",      # de → d'accord
    "n": "n",      # ne → n'est
    "s": "s",      # se → s'en
    "m": "m",      # me → m'appelle
    "t": "t",      # te → t'en
    "k": "qu",     # que → qu'il
}


def _split_elision(phone: str) -> tuple[str, str] | None:
    """Decompose un phone avec apostrophe en (prefix_ortho, base_phone).

    Ex: "ʒ'ɑ̃kyl" → ("j", "ɑ̃kyl")
        "l'ɔm"   → ("l", "ɔm")
        "bonjour" → None (pas d'elision)
    """
    if "'" not in phone:
        return None
    idx = phone.index("'")
    ipa_prefix = phone[:idx]
    base_phone = phone[idx + 1:]
    if not base_phone:
        return None
    ortho_prefix = _ELISION_IPA_TO_ORTHO.get(ipa_prefix)
    if ortho_prefix is None:
        return None
    return ortho_prefix, base_phone


def _edit_distance(a: str, b: str) -> int:
    """Distance de Levenshtein entre deux chaines."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[lb]


def _build_lex_features(word: str, lexicon: dict[str, list[str]] | None) -> list[float]:
    feats = [0.0] * LEX_FEATURE_DIM
    if lexicon is None:
        return feats
    candidates = lexicon.get(word.lower())
    if candidates is None:
        return feats
    feats[len(ALL_LEX_POS)] = 1.0
    feats[len(ALL_LEX_POS) + 1] = min(len(candidates) / 5.0, 1.0)
    feats[len(ALL_LEX_POS) + 2] = 1.0 if len(candidates) == 1 else 0.0
    for pos in candidates:
        if pos in ALL_LEX_POS:
            feats[ALL_LEX_POS.index(pos)] = 1.0
        else:
            for i, lex_pos in enumerate(ALL_LEX_POS):
                if lex_pos.startswith(pos + ":") or lex_pos == pos:
                    feats[i] = 1.0
    return feats


def _build_phone_lex_features(phone: str, phone_lexicon) -> list[float]:
    """28d feature vector from IPA phone lookup in phone_lexicon (V6)."""
    feats = [0.0] * PHONE_LEX_FEATURE_DIM

    if phone_lexicon is None or not phone:
        return feats

    entries = phone_lexicon.all_entries(phone) if hasattr(phone_lexicon, 'all_entries') else []
    # Fallback elision : si le phone complet n'a pas d'entrees,
    # essayer la partie apres l'apostrophe (ex: ʒ'ɑ̃kyl → ɑ̃kyl)
    if not entries:
        elision = _split_elision(phone)
        if elision is not None:
            _, base_phone = elision
            entries = phone_lexicon.all_entries(base_phone) if hasattr(phone_lexicon, 'all_entries') else []
    if not entries:
        return feats

    by_ortho: dict[str, dict] = {}
    for e in entries:
        key = e["ortho"].lower()
        if key not in by_ortho or (e.get("freq", 0) or 0) > (by_ortho[key].get("freq", 0) or 0):
            by_ortho[key] = e
    unique = list(by_ortho.values())
    n_cands = len(unique)

    pos_counts: dict[str, float] = {}
    total_freq = 0.0
    max_freq = 0.0
    has_verb = False
    has_noun_adj = False

    for e in unique:
        cgram = e.get("cgram", "")
        freq = e.get("freq", 0) or 0
        total_freq += freq
        if freq > max_freq:
            max_freq = freq

        for i, pos in enumerate(ALL_LEX_POS):
            if cgram == pos or cgram.startswith(pos + ":") or pos.startswith(cgram + ":"):
                feats[i] = 1.0
                pos_counts[pos] = pos_counts.get(pos, 0) + freq

        base = cgram.split(":")[0] if cgram else ""
        if base in ("VER", "AUX"):
            has_verb = True
        if base in ("NOM", "ADJ"):
            has_noun_adj = True

    feats[21] = 1.0  # is_known
    feats[22] = min(n_cands / 10.0, 1.0)  # n_candidates normalise
    feats[23] = 1.0 if n_cands == 1 else 0.0  # is_unambiguous
    feats[24] = 1.0 if has_verb else 0.0
    feats[25] = 1.0 if has_noun_adj else 0.0

    if total_freq > 0:
        feats[26] = max_freq / total_freq  # max_freq_ratio
    else:
        feats[26] = 1.0 / n_cands if n_cands > 0 else 0.0

    if pos_counts and total_freq > 0:
        entropy = 0.0
        for freq_sum in pos_counts.values():
            p = freq_sum / total_freq
            if p > 0:
                entropy -= p * math.log2(p)
        feats[27] = min(entropy / 4.39, 1.0)  # pos_entropy

    return feats


def _encode_candidate(entry: dict, total_freq: float, max_freq: float) -> list[float]:
    """Encode une entree lexique en vecteur 42d.

    Dims 0-20:  POS one-hot (21d)
    Dims 21-23: genre m/f/autre (3d)
    Dims 24-26: nombre s/p/autre (3d)
    Dims 27-29: frequence (3d)
    Dims 30-34: VerbForm infinitif/participe/indicatif/subjonctif/imperatif (5d)
    Dims 35-37: Tense present/passe/futur (3d)
    Dims 38-40: Person 1/2/3 (3d)
    Dim 41:     is_lemme flag (1d)
    """
    feats = [0.0] * CAND_FEAT_DIM_FULL
    cgram = entry.get("cgram", "")
    for i, pos in enumerate(ALL_LEX_POS):
        if cgram == pos or cgram.startswith(pos + ":") or pos.startswith(cgram + ":"):
            feats[i] = 1.0
    genre = (entry.get("genre") or "").strip()
    if genre == "m":
        feats[21] = 1.0
    elif genre == "f":
        feats[22] = 1.0
    else:
        feats[23] = 1.0
    nombre = (entry.get("nombre") or "").strip()
    if nombre == "s":
        feats[24] = 1.0
    elif nombre == "p":
        feats[25] = 1.0
    else:
        feats[26] = 1.0
    freq = entry.get("freq", 0) or 0
    feats[27] = min(1.0, math.log10(freq + 1) / 5.0)
    feats[28] = 1.0 if freq >= max_freq and max_freq > 0 else 0.0
    feats[29] = freq / total_freq if total_freq > 0 else 0.0

    # ── Multext traits (12d) — verbes uniquement (V...) ──
    multext = (entry.get("multext") or "").strip()
    if len(multext) >= 3 and multext[0] == "V":
        mood = multext[2]
        if mood == "n":    feats[30] = 1.0  # infinitif
        elif mood == "p":  feats[31] = 1.0  # participe
        elif mood == "i":  feats[32] = 1.0  # indicatif
        elif mood == "c":  feats[32] = 1.0  # conditionnel -> indicatif
        elif mood == "s":  feats[33] = 1.0  # subjonctif
        elif mood == "m":  feats[34] = 1.0  # imperatif
        if len(multext) >= 4:
            tense = multext[3]
            if tense == "p":   feats[35] = 1.0  # present
            elif tense == "i": feats[36] = 1.0  # imparfait -> passe
            elif tense == "s": feats[36] = 1.0  # passe simple -> passe
            elif tense == "f": feats[37] = 1.0  # futur
        if len(multext) >= 5:
            person = multext[4]
            if person == "1":  feats[38] = 1.0
            elif person == "2": feats[39] = 1.0
            elif person == "3": feats[40] = 1.0

    feats[41] = 1.0 if entry.get("is_lemme") else 0.0
    return feats


def _deduplicate_by_ortho(entries: list[dict]) -> list[dict]:
    """1 entrée par ortho unique (la plus fréquente), triée par freq desc."""
    by_ortho: dict[str, dict] = {}
    for e in entries:
        key = e["ortho"].lower()
        if key not in by_ortho or (e.get("freq", 0) or 0) > (by_ortho[key].get("freq", 0) or 0):
            by_ortho[key] = dict(e)
    return sorted(by_ortho.values(), key=lambda e: -(e.get("freq", 0) or 0))


class OnnxInferenceEngineV2:
    """Inférence ONNX Runtime pour le modèle unifié P2G V2/V4."""

    def __init__(
        self,
        onnx_path: str | Path,
        vocab_path: str | Path,
        lexicon_path: str | Path | None = None,
        lexicon: dict[str, list[str]] | None = None,
        phone_lexicon=None,
    ):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        enc_path = onnx_path.with_suffix(onnx_path.suffix + ".enc")

        if enc_path.exists():
            logger.info("Loading P2G V2 ONNX model from %s (encrypted)", enc_path)
            from lectura_graphemiseur._crypto import load_encrypted_model
            model_bytes = load_encrypted_model(enc_path)
            self.session = ort.InferenceSession(model_bytes)
        else:
            logger.info("Loading P2G V2 ONNX model from %s", onnx_path)
            self.session = ort.InferenceSession(str(onnx_path))

        with open(vocab_path, encoding="utf-8") as f:
            data = json.load(f)

        self.config = data["config"]
        vocabs = data["vocabs"]
        self.char2idx = vocabs["char2idx"]
        self.idx2p2g = {int(v): k for k, v in vocabs["p2g_label2idx"].items()}
        self.idx2pos = {int(v): k for k, v in vocabs["pos2idx"].items()}
        self.idx2morpho = {}
        for feat, vocab in vocabs["morpho_vocabs"].items():
            self.idx2morpho[feat] = {int(v): k for k, v in vocab.items()}

        self.lex_feature_dim = self.config.get("lex_feature_dim", LEX_FEATURE_DIM)
        # V6 uses phone_lex_features (28d) instead of ortho_lex_features (24d)
        self.is_v6 = self.lex_feature_dim == PHONE_LEX_FEATURE_DIM
        if self.is_v6:
            logger.info("V6 model detected (phone_lex_features, %dd)", self.lex_feature_dim)

        if lexicon is not None:
            self.lexicon = lexicon
        elif lexicon_path is not None:
            with open(lexicon_path, encoding="utf-8") as f:
                self.lexicon = json.load(f)
            logger.info("Loaded lexicon: %d words", len(self.lexicon))
        else:
            self.lexicon = None

        # Lex select support (V4)
        self.phone_lexicon = phone_lexicon
        output_names = [o.name for o in self.session.get_outputs()]
        self.has_lex_select = "lex_select_logits" in output_names
        if self.has_lex_select:
            logger.info("Lex select head detected in ONNX model")
            if phone_lexicon is None:
                logger.warning("Lex select available but no phone_lexicon provided")

        # Detect word_mask support (V5 model)
        input_names = [i.name for i in self.session.get_inputs()]
        self.has_word_mask = "word_mask" in input_names

        # Detecter la dimension lex_cand_features du modele (30 pour V6, 42 pour V7)
        self.cand_feat_dim = CAND_FEAT_DIM_FULL
        for inp in self.session.get_inputs():
            if inp.name == "lex_cand_features" and inp.shape and len(inp.shape) == 4:
                dim = inp.shape[3]
                if isinstance(dim, int):
                    self.cand_feat_dim = dim
                break

        # UNK word embedding (pour prediction contexte seul)
        self.unk_word_embedding = data.get("unk_word_embedding")

        # Flag pour activer/desactiver l'application de lex_select
        self.apply_lex_select = True

        # Lex select gating params (V6)
        if data.get("lex_select_threshold") is not None:
            self.LEX_SELECT_THRESHOLD = data["lex_select_threshold"]
        if data.get("lex_select_max_edit") is not None:
            self.LEX_SELECT_MAX_EDIT = data["lex_select_max_edit"]

    def _encode_sentence(
        self, ipa_words: list[str], ortho_words: list[str] | None = None,
        *, use_lex: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Encode IPA words en inputs ONNX.

        Args:
            ipa_words: Mots IPA.
            ortho_words: Mots orthographiques pour lookup lexique.
                         Si None, pas de lex features.
            use_lex: Si True, utilise les features lexicales.
        """
        chars: list[str] = ["<BOS>"]
        word_starts: list[int] = []
        word_ends: list[int] = []

        for w_idx, word in enumerate(ipa_words):
            if w_idx > 0:
                chars.append("<SEP>")
            word_start = len(chars)
            for ch in word:
                chars.append(ch)
            word_end = len(chars) - 1
            word_starts.append(word_start)
            word_ends.append(word_end)

        chars.append("<EOS>")

        char_ids = np.array(
            [[self.char2idx.get(ch, 1) for ch in chars]], dtype=np.int64
        )
        ws = np.array([word_starts], dtype=np.int64)
        we = np.array([word_ends], dtype=np.int64)

        # Build lex features
        lex_feats = []
        for w_idx in range(len(ipa_words)):
            if not use_lex:
                lex_feats.append([0.0] * self.lex_feature_dim)
            elif self.is_v6:
                # V6: phone_lex_features from IPA lookup
                lex_feats.append(
                    _build_phone_lex_features(ipa_words[w_idx], self.phone_lexicon)
                )
            elif ortho_words and w_idx < len(ortho_words):
                # V2-V5: ortho_lex_features from word form
                lex_feats.append(_build_lex_features(ortho_words[w_idx], self.lexicon))
            else:
                lex_feats.append([0.0] * self.lex_feature_dim)

        lex_features = np.array([lex_feats], dtype=np.float32)

        return char_ids, ws, we, lex_features, chars

    def _build_lex_candidates(
        self, ipa_words: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[list[dict]], list[dict[str, str]]]:
        """Construit les features candidats pour lex_select.

        Returns:
            cand_features: (1, n_words, K_MAX, CAND_FEAT_DIM)
            cand_mask: (1, n_words, K_MAX)
            candidates_list: pour chaque mot, liste des dicts candidats (dédupliqués)
            resolved_maps: pour chaque mot, {ortho.lower(): source_phone}
        """
        n_words = len(ipa_words)
        cand_features = np.zeros((1, n_words, K_MAX, self.cand_feat_dim), dtype=np.float32)
        cand_mask = np.zeros((1, n_words, K_MAX), dtype=np.float32)
        candidates_list: list[list[dict]] = []
        resolved_maps: list[dict[str, str]] = []

        use_perturbations = (
            self.phone_lexicon is not None
            and hasattr(self.phone_lexicon, "all_entries_with_perturbations")
        )

        for w_idx, phone in enumerate(ipa_words):
            # D'abord essayer les entrées exactes
            raw = self.phone_lexicon.all_entries(phone) if self.phone_lexicon and phone else []
            unique = _deduplicate_by_ortho(raw) if raw else []
            resolved_map = {e["ortho"].lower(): "exact" for e in unique}

            # Fallback elision : si pas de resultats exacts et phone contient
            # une apostrophe, chercher la partie base et prefixer les candidats
            elision_prefix = None
            if not unique and phone:
                elision = _split_elision(phone)
                if elision is not None:
                    ortho_prefix, base_phone = elision
                    elision_prefix = ortho_prefix
                    raw_base = self.phone_lexicon.all_entries(base_phone) if self.phone_lexicon else []
                    if raw_base:
                        # Prefixer chaque candidat ortho avec le prefix d'elision
                        for e in raw_base:
                            e["ortho"] = ortho_prefix + "'" + e["ortho"]
                        unique = _deduplicate_by_ortho(raw_base)
                        resolved_map = {e["ortho"].lower(): "elision" for e in unique}

            # Enrichir avec perturbations phonétiques (tolérance o/ɔ, e/ɛ, ə, etc.)
            # Toujours enrichir (pas seulement en fallback) pour augmenter le pool
            if use_perturbations and phone:
                entries_perturbed, perturbed_map = (
                    self.phone_lexicon.all_entries_with_perturbations(
                        phone, k_max=K_MAX,
                    )
                )
                if not unique:
                    # Aucun exact : prendre tout
                    unique = entries_perturbed
                    resolved_map = perturbed_map
                else:
                    # Ajouter les perturbés non encore présents
                    existing = {e["ortho"].lower() for e in unique}
                    for e in entries_perturbed:
                        key = e["ortho"].lower()
                        if key not in existing:
                            unique.append(e)
                            existing.add(key)
                            resolved_map[key] = perturbed_map.get(key, "perturbed")
                    # Re-trier : exact d'abord par freq, puis perturbés par freq
                    unique.sort(
                        key=lambda e: (
                            0 if resolved_map.get(e["ortho"].lower()) == "exact" else 1,
                            -(e.get("freq", 0) or 0),
                        )
                    )
                    unique = unique[:K_MAX]

            total_freq = sum(e.get("freq", 0) or 0 for e in unique)
            max_freq = max((e.get("freq", 0) or 0 for e in unique), default=0)

            word_cands = []
            for k, entry in enumerate(unique[:K_MAX]):
                feats_full = _encode_candidate(entry, total_freq, max_freq)
                cand_features[0, w_idx, k] = feats_full[:self.cand_feat_dim]
                cand_mask[0, w_idx, k] = 1.0
                word_cands.append(entry)
            candidates_list.append(word_cands)
            resolved_maps.append(resolved_map)

        return cand_features, cand_mask, candidates_list, resolved_maps

    def _run_session(
        self, char_ids, word_starts, word_ends, lex_features,
        lex_cand_features=None, lex_cand_mask=None,
        word_mask: np.ndarray | None = None,
    ):
        """Exécute la session ONNX avec ou sans lex_select/word_mask inputs."""
        feed = {
            "char_ids": char_ids,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "lex_features": lex_features,
        }
        if self.has_lex_select:
            if lex_cand_features is not None:
                feed["lex_cand_features"] = lex_cand_features
                feed["lex_cand_mask"] = lex_cand_mask
            else:
                # Pas de phone_lexicon → zeros (lex_select inactif)
                n_words = word_starts.shape[1]
                feed["lex_cand_features"] = np.zeros(
                    (1, n_words, K_MAX, self.cand_feat_dim), dtype=np.float32,
                )
                feed["lex_cand_mask"] = np.zeros(
                    (1, n_words, K_MAX), dtype=np.float32,
                )
        if self.has_word_mask:
            if word_mask is not None:
                feed["word_mask"] = word_mask
            else:
                feed["word_mask"] = np.zeros(
                    (1, word_starts.shape[1]), dtype=np.float32,
                )

        outputs = self.session.run(None, feed)
        output_names = [o.name for o in self.session.get_outputs()]
        return dict(zip(output_names, outputs))

    @staticmethod
    def _pos_compatible(pred_pos: str, cand_cgram: str) -> bool:
        """Vérifie si le POS prédit est compatible avec le cgram du candidat."""
        if not pred_pos or not cand_cgram:
            return True
        if pred_pos == cand_cgram:
            return True
        # "VER" matche "VER:ind", "VER:sub", etc.
        if cand_cgram.startswith(pred_pos + ":"):
            return True
        if pred_pos.startswith(cand_cgram + ":"):
            return True
        # AUX et VER sont interchangeables (le POS head confond souvent les deux)
        _verb_tags = {"AUX", "VER"}
        pred_base = pred_pos.split(":")[0]
        cand_base = cand_cgram.split(":")[0]
        if pred_base in _verb_tags and cand_base in _verb_tags:
            return True
        return False

    def _morpho_score(self, cand: dict, morpho_preds: dict[str, str]) -> float:
        """Score de compatibilité morpho entre un candidat et les prédictions.

        Retourne un bonus (0.0 à 1.0) basé sur le nombre de traits morpho
        compatibles. Chaque trait compatible vaut +1, chaque incompatible -1.
        Les traits non renseignés (candidat ou prédiction = '_' ou vide) sont ignorés.
        """
        score = 0.0
        checks = 0

        # Number: Sing/Plur vs nombre s/p
        pred_num = morpho_preds.get("Number", "_")
        cand_nombre = (cand.get("nombre") or "").strip()
        if pred_num not in ("_", "") and cand_nombre:
            checks += 1
            if (pred_num == "Sing" and "singulier" in cand_nombre) or \
               (pred_num == "Plur" and "pluriel" in cand_nombre) or \
               (pred_num == "Sing" and cand_nombre == "s") or \
               (pred_num == "Plur" and cand_nombre == "p"):
                score += 1.0
            else:
                score -= 1.0

        # Gender: Masc/Fem vs genre m/f
        pred_gen = morpho_preds.get("Gender", "_")
        cand_genre = (cand.get("genre") or "").strip()
        if pred_gen not in ("_", "") and cand_genre:
            checks += 1
            if (pred_gen == "Masc" and ("masculin" in cand_genre or cand_genre == "m")) or \
               (pred_gen == "Fem" and ("feminin" in cand_genre or cand_genre == "f" or "féminin" in cand_genre)):
                score += 1.0
            else:
                score -= 1.0

        # Person: 1/2/3 — pour distinguer ai(1)/est(3), es(2)/est(3)
        pred_person = morpho_preds.get("Person", "_")
        cand_ortho = cand.get("ortho", "").lower()
        cand_cgram = cand.get("cgram", "")
        if pred_person not in ("_", "") and (
            cand_cgram in ("AUX", "VER") or
            (cand_cgram and cand_cgram.startswith(("AUX:", "VER:")))):

            # Heuristiques être/avoir pour les formes courantes
            person_map = {
                # être — présent
                "suis": "1", "es": "2", "est": "3",
                "sommes": "1", "êtes": "2", "sont": "3",
                # avoir — présent
                "ai": "1", "as": "2", "a": "3",
                "avons": "1", "avez": "2", "ont": "3",
                # être — imparfait
                "étais": "1", "était": "3", "étions": "1",
                "étiez": "2", "étaient": "3",
                # avoir — imparfait
                "avais": "1", "avait": "3", "avions": "1",
                "aviez": "2", "avaient": "3",
                # être — futur simple
                "serai": "1", "seras": "2", "sera": "3",
                "serons": "1", "serez": "2", "seront": "3",
                # avoir — futur simple
                "aurai": "1", "auras": "2", "aura": "3",
                "aurons": "1", "aurez": "2", "auront": "3",
                # être — conditionnel
                "serais": "1", "serait": "3",
                "serions": "1", "seriez": "2", "seraient": "3",
                # avoir — conditionnel
                "aurais": "1", "aurait": "3",
                "aurions": "1", "auriez": "2", "auraient": "3",
                # être — passé simple
                "fus": "1", "fut": "3", "fût": "3",
                "fûmes": "1", "fûtes": "2", "furent": "3",
                # avoir — passé simple
                "eus": "1", "eut": "3", "eût": "3",
                "eûmes": "1", "eûtes": "2", "eurent": "3",
                # être — subjonctif
                "sois": "1", "soit": "3", "soient": "3",
                "soyons": "1", "soyez": "2",
                # avoir — subjonctif
                "aie": "1", "ait": "3", "aient": "3",
                "ayons": "1", "ayez": "2",
                # aller — présent (auxiliaire fréquent)
                "vais": "1", "vas": "2", "va": "3",
                "allons": "1", "allez": "2", "vont": "3",
                # faire — présent
                "fais": "1", "fait": "3",
                "faisons": "1", "faites": "2", "font": "3",
                # pouvoir — présent
                "peux": "1", "peut": "3",
                "pouvons": "1", "pouvez": "2", "peuvent": "3",
                # vouloir — présent
                "veux": "1", "veut": "3",
                "voulons": "1", "voulez": "2", "veulent": "3",
                # devoir — présent
                "dois": "1", "doit": "3",
                "devons": "1", "devez": "2", "doivent": "3",
            }
            cand_person = person_map.get(cand_ortho)
            if cand_person:
                checks += 1
                if cand_person == pred_person:
                    score += 1.0
                else:
                    score -= 1.0

        # Mood: Ind/Sub/Cond — pour distinguer est(Ind) de ait(Sub)
        pred_mood = morpho_preds.get("Mood", "_")
        if pred_mood not in ("_", "") and (
            cand_cgram in ("AUX", "VER") or
            (cand_cgram and cand_cgram.startswith(("AUX:", "VER:")))):

            mood_map = {
                # Indicatif présent être/avoir
                "est": "Ind", "suis": "Ind", "es": "Ind",
                "sommes": "Ind", "êtes": "Ind", "sont": "Ind",
                "ai": "Ind", "as": "Ind", "a": "Ind",
                "avons": "Ind", "avez": "Ind", "ont": "Ind",
                # Indicatif imparfait
                "étais": "Ind", "était": "Ind", "étions": "Ind",
                "étiez": "Ind", "étaient": "Ind",
                "avais": "Ind", "avait": "Ind", "avions": "Ind",
                "aviez": "Ind", "avaient": "Ind",
                # Indicatif futur
                "serai": "Ind", "seras": "Ind", "sera": "Ind",
                "serons": "Ind", "serez": "Ind", "seront": "Ind",
                "aurai": "Ind", "auras": "Ind", "aura": "Ind",
                "aurons": "Ind", "aurez": "Ind", "auront": "Ind",
                # Conditionnel
                "serais": "Cnd", "serait": "Cnd",
                "serions": "Cnd", "seriez": "Cnd", "seraient": "Cnd",
                "aurais": "Cnd", "aurait": "Cnd",
                "aurions": "Cnd", "auriez": "Cnd", "auraient": "Cnd",
                # Subjonctif
                "sois": "Sub", "soit": "Sub", "soient": "Sub",
                "soyons": "Sub", "soyez": "Sub",
                "aie": "Sub", "ait": "Sub", "aient": "Sub",
                "ayons": "Sub", "ayez": "Sub",
                # Indicatif passé simple
                "fus": "Ind", "fut": "Ind", "fût": "Sub",
                "furent": "Ind",
                "eus": "Ind", "eut": "Ind", "eût": "Sub",
                "eurent": "Ind",
            }
            cand_mood = mood_map.get(cand_ortho)
            if cand_mood:
                checks += 1
                if cand_mood == pred_mood:
                    score += 1.0
                else:
                    score -= 1.0

        return score if checks > 0 else 0.0

    # Parametres lex_select (V7 smart mode)
    LEX_SELECT_THRESHOLD = 0.80   # seuil pour remplacement quand brut absent des candidats
    LEX_SELECT_THRESHOLD_MORPHO = 0.50  # seuil pour remplacement morpho-motive (brut in cands)
    LEX_SELECT_MAX_EDIT = 3       # edit distance max entre P2G et candidat
    LEX_SELECT_MORPHO_MIN_DELTA = 1.5  # delta morpho minimum pour overrider le P2G brut

    # Paires d'homophones grammaticaux que lex_select ne doit pas substituer
    # (c'est le role du POS, pas du lexique phonetique)
    _LEX_SELECT_BLACKLIST = frozenset({
        ("est", "ai"), ("ai", "est"), ("est", "et"), ("et", "est"),
        ("a", "à"), ("à", "a"),
        ("ou", "où"), ("où", "ou"),
        ("son", "sont"), ("sont", "son"),
        ("on", "ont"), ("ont", "on"),
        ("ses", "ces"), ("ces", "ses"),
        ("se", "ce"), ("ce", "se"),
        ("leur", "leurs"), ("leurs", "leur"),
    })

    def _apply_lex_select(
        self, output_dict, word_starts, n_words, candidates_list, ortho_results,
        ipa_words: list[str] | None = None,
    ) -> list[str]:
        """Remplace les résultats P2G par le candidat lex_select quand disponible.

        V6 (native POS/morpho): remplacement conditionné par confiance (softmax)
        et proximité (edit distance). Seuls les candidats proches du P2G brut
        avec une confiance élevée sont appliqués.

        Gardes supplementaires (V6) :
        - Blacklist d'homophones grammaticaux (est/et, a/à, etc.)
        - Garde de frequence : ne pas remplacer un mot frequent par un mot rare
        - Seuil adaptatif : confiance plus haute si le P2G brut est dans le lexique

        V2-V5 (post-hoc): filtre POS + re-rank morpho + safety check fréquence.
        """
        if "lex_select_logits" not in output_dict or not candidates_list:
            return ortho_results

        lex_logits = output_dict["lex_select_logits"][0]  # (W, K)

        # V6/V7: smart mode + morpho-aware lex_select
        if self.is_v6:
            # ── Extraire predictions morpho du modele ──
            morpho_preds_per_word: list[dict[str, str]] = []
            for w in range(n_words):
                mp: dict[str, str] = {}
                for feat_name, idx2label in self.idx2morpho.items():
                    key = f"morpho_{feat_name}_logits"
                    if key in output_dict:
                        pred_idx = int(output_dict[key][0][w].argmax())
                        mp[feat_name] = idx2label.get(pred_idx, "_")
                morpho_preds_per_word.append(mp)

            result = list(ortho_results)
            for w in range(n_words):
                cands = candidates_list[w]
                if not cands:
                    continue
                logits_w = lex_logits[w, :len(cands)]
                # Softmax pour confiance
                exp_l = np.exp(logits_w - logits_w.max())
                probs = exp_l / exp_l.sum()

                brut = result[w]
                brut_lower = brut.lower()
                morpho_w = morpho_preds_per_word[w]

                # ── Trouver si P2G brut est parmi les candidats ──
                brut_k = None
                for k, c in enumerate(cands):
                    if c.get("ortho", "").lower() == brut_lower:
                        brut_k = k
                        break

                # ── Calculer scores morpho pour chaque candidat ──
                morpho_scores = []
                for k, c in enumerate(cands):
                    ortho = c.get("ortho", "")
                    if not ortho or ortho.startswith("-"):
                        morpho_scores.append(-999.0)
                        continue
                    morpho_scores.append(self._morpho_score(c, morpho_w))

                # ── Choisir le meilleur candidat (lex_select score) ──
                best_k = int(np.argmax(probs))
                confidence = float(probs[best_k])
                cand_ortho = cands[best_k].get("ortho", "")

                if brut_k is not None:
                    # ═══ MODE SMART : P2G brut est un candidat valide ═══
                    # On ne remplace que si morpho penalise fortement le brut
                    # et qu'un autre candidat a un meilleur morpho
                    brut_morpho = morpho_scores[brut_k]

                    if brut_morpho >= 0:
                        # Brut a un morpho acceptable → on le garde
                        continue

                    # Brut a morpho negatif → chercher un candidat avec meilleur
                    # morpho ET bonne confiance lex_select
                    best_morpho_k = None
                    best_morpho_combined = -999.0
                    for k, c in enumerate(cands):
                        if morpho_scores[k] <= brut_morpho:
                            continue  # Pas mieux que le brut
                        if morpho_scores[k] < 0:
                            continue  # Pas assez bon
                        ortho = c.get("ortho", "")
                        if not ortho or ortho.startswith("-"):
                            continue
                        # Verifier edit distance par rapport au brut
                        ed = _edit_distance(brut_lower, ortho.lower())
                        if ed > self.LEX_SELECT_MAX_EDIT:
                            continue
                        # Blacklist
                        pair = (brut_lower, ortho.lower())
                        if pair in self._LEX_SELECT_BLACKLIST:
                            continue
                        # Score combine : confiance lex + bonus morpho
                        combined = float(probs[k]) + morpho_scores[k]
                        if combined > best_morpho_combined:
                            best_morpho_combined = combined
                            best_morpho_k = k

                    if best_morpho_k is None:
                        continue

                    # Delta morpho suffisant ?
                    delta = morpho_scores[best_morpho_k] - brut_morpho
                    if delta < self.LEX_SELECT_MORPHO_MIN_DELTA:
                        continue

                    # Confiance minimum pour override morpho
                    if float(probs[best_morpho_k]) < self.LEX_SELECT_THRESHOLD_MORPHO:
                        continue

                    result[w] = cands[best_morpho_k]["ortho"]

                else:
                    # ═══ MODE SMART : P2G brut n'est PAS un candidat ═══
                    # Remplacement avec seuil bas (le brut n'existe pas dans le lexique)

                    # Trouver le meilleur candidat avec morpho positif ou neutre
                    best_smart_k = None
                    best_smart_score = -999.0
                    for k, c in enumerate(cands):
                        ortho = c.get("ortho", "")
                        if not ortho or ortho.startswith("-"):
                            continue
                        # Blacklist
                        pair = (brut_lower, ortho.lower())
                        if pair in self._LEX_SELECT_BLACKLIST:
                            continue
                        # Edit distance
                        ed = _edit_distance(brut_lower, ortho.lower())
                        if ed > self.LEX_SELECT_MAX_EDIT:
                            continue
                        # Score: lex_select prob + morpho bonus leger
                        score = float(probs[k])
                        if morpho_scores[k] > 0:
                            score += 0.05  # petit bonus morpho
                        elif morpho_scores[k] < 0:
                            score -= 0.05  # petite penalite
                        if score > best_smart_score:
                            best_smart_score = score
                            best_smart_k = k

                    if best_smart_k is None:
                        continue

                    cand_ortho = cands[best_smart_k].get("ortho", "")
                    if cand_ortho.lower() == brut_lower:
                        continue

                    if float(probs[best_smart_k]) < self.LEX_SELECT_THRESHOLD:
                        continue

                    result[w] = cand_ortho

            return result

        # POS predictions with confidence
        pos_preds = None
        pos_conf = None  # confiance par mot (max softmax)
        if "pos_logits" in output_dict:
            pos_logits_raw = output_dict["pos_logits"][0]  # (W, n_pos)
            pos_preds = pos_logits_raw.argmax(axis=-1)
            # Softmax pour confiance
            exp_l = np.exp(pos_logits_raw - pos_logits_raw.max(axis=-1, keepdims=True))
            pos_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
            pos_conf = pos_probs.max(axis=-1)  # (W,)

        # Morpho predictions
        morpho_preds_per_word: list[dict[str, str]] = []
        for w in range(n_words):
            mp: dict[str, str] = {}
            for feat_name, idx2label in self.idx2morpho.items():
                key = f"morpho_{feat_name}_logits"
                if key in output_dict:
                    pred_idx = int(output_dict[key][0][w].argmax())
                    mp[feat_name] = idx2label.get(pred_idx, "_")
            morpho_preds_per_word.append(mp)

        MORPHO_WEIGHT = 10.0  # Poids du bonus morpho vs logits bruts
        FREQ_SAFETY_RATIO = 20  # Si le candidat filtré est N× moins fréquent que le meilleur, fallback

        result = list(ortho_results)
        for w in range(n_words):
            cands = candidates_list[w]
            if not cands:
                continue
            logits_w = lex_logits[w, :len(cands)]
            morpho_w = morpho_preds_per_word[w]

            # Meilleur candidat sans filtre (référence pour safety check)
            best_k_raw = int(np.argmax(logits_w))
            freq_best_raw = cands[best_k_raw].get("freq", 0) or 0

            # Étape 1 : filtre POS + re-rank morpho
            # N'appliquer le filtre POS que si confiance > 40%
            POS_CONF_THRESHOLD = 0.40
            if pos_preds is not None and (pos_conf is None or pos_conf[w] >= POS_CONF_THRESHOLD):
                pred_pos = self.idx2pos.get(int(pos_preds[w]), "")
                compatible = []
                for k, cand in enumerate(cands):
                    # Exclure les formes suffixées (-il, -ils, -on, etc.)
                    if cand.get("ortho", "").startswith("-"):
                        continue
                    if self._pos_compatible(pred_pos, cand.get("cgram", "")):
                        m_score = self._morpho_score(cand, morpho_w)
                        combined = float(logits_w[k]) + m_score * MORPHO_WEIGHT
                        compatible.append((k, combined))

                if compatible:
                    best_k = max(compatible, key=lambda x: x[1])[0]
                    freq_filtered = cands[best_k].get("freq", 0) or 0

                    # Safety : si le candidat filtré est beaucoup plus rare
                    # que le meilleur brut, le POS est probablement faux
                    if freq_best_raw > 0 and freq_filtered > 0 and \
                       freq_best_raw / freq_filtered > FREQ_SAFETY_RATIO:
                        result[w] = cands[best_k_raw]["ortho"]
                    else:
                        result[w] = cands[best_k]["ortho"]
                    continue

            # Fallback : argmax sans filtre
            result[w] = cands[best_k_raw]["ortho"]
        return result

    def analyser(
        self, ipa_words: list[str], ortho_words: list[str] | None = None,
        *, use_lex: bool = True,
        word_mask: list[bool] | None = None,
    ) -> dict[str, Any]:
        """API V1-compatible.

        Args:
            ipa_words: Mots IPA.
            ortho_words: Mots orthographiques pour lookup lexique (optionnel).
            use_lex: Si True (defaut), utilise les features lexicales.
            word_mask: Liste de booleans (un par mot). True = mot masque
                       (prediction contexte seul). None = pas de masque.
        """
        if not ipa_words:
            return {"ipa_words": [], "ortho": [], "pos": [], "morpho": {}}

        char_ids, word_starts, word_ends, lex_features, chars = self._encode_sentence(
            ipa_words, ortho_words, use_lex=use_lex,
        )

        wm = None
        if word_mask is not None:
            wm = np.array([[1.0 if m else 0.0 for m in word_mask]], dtype=np.float32)

        # Build lex candidates if lex_select available
        lex_cand_features = lex_cand_mask = None
        candidates_list = None
        resolved_maps = None
        if self.has_lex_select and self.phone_lexicon is not None:
            lex_cand_features, lex_cand_mask, candidates_list, resolved_maps = self._build_lex_candidates(ipa_words)

        output_dict = self._run_session(
            char_ids, word_starts, word_ends, lex_features,
            lex_cand_features, lex_cand_mask, word_mask=wm,
        )
        n_words = len(ipa_words)

        # P2G (char-level fallback)
        p2g_logits = output_dict["p2g_logits"]
        p2g_preds = p2g_logits[0].argmax(axis=-1)
        ortho_results = []
        for w in range(n_words):
            ws = word_starts[0, w]
            we = word_ends[0, w]
            word_labels = [self.idx2p2g.get(int(p2g_preds[i]), "_CONT") for i in range(ws, we + 1)]
            ortho = reconstruct_ortho(word_labels)
            ortho_results.append(ortho)

        # Apply lex_select (overrides P2G for words with candidates)
        if candidates_list is not None and self.apply_lex_select:
            ortho_results = self._apply_lex_select(
                output_dict, word_starts, n_words, candidates_list, ortho_results,
                ipa_words=ipa_words)

        # POS
        pos_results = []
        if "pos_logits" in output_dict:
            pos_preds = output_dict["pos_logits"][0].argmax(axis=-1)
            for w in range(n_words):
                pos_results.append(self.idx2pos.get(int(pos_preds[w]), "NOM"))

        # Morpho
        morpho_results: dict[str, list[str]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key in output_dict:
                feat_preds = output_dict[key][0].argmax(axis=-1)
                morpho_results[feat_name] = [
                    idx2label.get(int(feat_preds[w]), "_") for w in range(n_words)
                ]

        return {
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
        }

    def analyser_v2(
        self,
        ipa_words: list[str],
        ortho_words: list[str] | None = None,
        top_k: int = 3,
        *,
        use_lex: bool = True,
        word_mask: list[bool] | None = None,
    ) -> dict[str, Any]:
        """API V2 : comme analyser() + top-K POS/Morpho avec scores."""
        if not ipa_words:
            return {
                "ipa_words": [], "ortho": [], "pos": [], "morpho": {},
                "pos_scores": [], "morpho_scores": {}, "confiance_pos": [],
            }

        char_ids, word_starts, word_ends, lex_features, chars = self._encode_sentence(
            ipa_words, ortho_words, use_lex=use_lex,
        )

        wm = None
        if word_mask is not None:
            wm = np.array([[1.0 if m else 0.0 for m in word_mask]], dtype=np.float32)

        # Build lex candidates if lex_select available
        lex_cand_features = lex_cand_mask = None
        candidates_list = None
        resolved_maps = None
        if self.has_lex_select and self.phone_lexicon is not None:
            lex_cand_features, lex_cand_mask, candidates_list, resolved_maps = self._build_lex_candidates(ipa_words)

        output_dict = self._run_session(
            char_ids, word_starts, word_ends, lex_features,
            lex_cand_features, lex_cand_mask, word_mask=wm,
        )
        n_words = len(ipa_words)

        # P2G (standard)
        p2g_logits = output_dict["p2g_logits"]
        p2g_preds = p2g_logits[0].argmax(axis=-1)
        ortho_results = []
        for w in range(n_words):
            ws = word_starts[0, w]
            we = word_ends[0, w]
            word_labels = [self.idx2p2g.get(int(p2g_preds[i]), "_CONT") for i in range(ws, we + 1)]
            ortho_results.append(reconstruct_ortho(word_labels))

        # Apply lex_select
        if candidates_list is not None and self.apply_lex_select:
            ortho_results = self._apply_lex_select(
                output_dict, word_starts, n_words, candidates_list, ortho_results,
                ipa_words=ipa_words)

        # POS with top-K
        pos_results = []
        pos_scores = []
        confiance_pos = []
        if "pos_logits" in output_dict:
            pos_logits_2d = output_dict["pos_logits"][0]
            exp_l = np.exp(pos_logits_2d - pos_logits_2d.max(axis=-1, keepdims=True))
            pos_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

            for w in range(n_words):
                probs = pos_probs[w]
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_k_pos = [
                    (self.idx2pos.get(int(idx), "_"), float(probs[idx]))
                    for idx in top_indices
                    if self.idx2pos.get(int(idx), "<PAD>") != "<PAD>" and probs[idx] > 0.001
                ]
                pos_results.append(top_k_pos[0][0] if top_k_pos else "NOM")
                pos_scores.append(top_k_pos)
                confiance_pos.append(top_k_pos[0][1] if top_k_pos else 0.0)

        # Morpho with top-K
        morpho_results: dict[str, list[str]] = {}
        morpho_scores: dict[str, list[list[tuple[str, float]]]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key not in output_dict:
                continue
            feat_logits = output_dict[key][0]
            exp_l = np.exp(feat_logits - feat_logits.max(axis=-1, keepdims=True))
            feat_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

            feat_results = []
            feat_scores_list = []
            for w in range(n_words):
                probs = feat_probs[w]
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_k_feat = [
                    (idx2label.get(int(idx), "_"), float(probs[idx]))
                    for idx in top_indices
                    if idx2label.get(int(idx), "<PAD>") != "<PAD>" and probs[idx] > 0.001
                ]
                feat_results.append(top_k_feat[0][0] if top_k_feat else "_")
                feat_scores_list.append(top_k_feat)

            morpho_results[feat_name] = feat_results
            morpho_scores[feat_name] = feat_scores_list

        # Lex-select candidates avec confiance (pour post-traitement)
        lex_candidates: list[list[tuple[str, float]]] = []
        if candidates_list is not None and "lex_select_logits" in output_dict:
            lex_logits = output_dict["lex_select_logits"][0]
            for w in range(n_words):
                cands = candidates_list[w]
                if not cands:
                    lex_candidates.append([])
                    continue
                logits_w = lex_logits[w, :len(cands)]
                exp_l = np.exp(logits_w - logits_w.max())
                probs = exp_l / exp_l.sum()
                word_cands = [
                    (c.get("ortho", ""), float(probs[k]))
                    for k, c in enumerate(cands)
                    if c.get("ortho", "") and not c["ortho"].startswith("-")
                ]
                word_cands.sort(key=lambda x: -x[1])
                lex_candidates.append(word_cands)

        return {
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
            "pos_scores": pos_scores,
            "morpho_scores": morpho_scores,
            "confiance_pos": confiance_pos,
            "lex_candidates": lex_candidates,
        }

    def analyser_avec_alternatives(
        self,
        ipa_words: list[str],
        ortho_words: list[str] | None = None,
        k: int = 5,
        *,
        use_lex: bool = True,
        word_mask: list[bool] | None = None,
    ) -> dict[str, Any]:
        """Retourne alternatives P2G + top-K POS/Morpho.

        Combine les alternatives orthographiques (comme V1) avec les
        scores POS/Morpho V2.
        """
        if not ipa_words:
            return {
                "ipa_words": [], "ortho": [], "pos": [], "morpho": {},
                "alternatives": [], "confiance": [],
                "pos_scores": [], "morpho_scores": [],
            }

        char_ids, word_starts, word_ends, lex_features, chars = self._encode_sentence(
            ipa_words, ortho_words, use_lex=use_lex,
        )

        wm = None
        if word_mask is not None:
            wm = np.array([[1.0 if m else 0.0 for m in word_mask]], dtype=np.float32)

        # Build lex candidates if lex_select available
        lex_cand_features = lex_cand_mask = None
        candidates_list = None
        resolved_maps = None
        if self.has_lex_select and self.phone_lexicon is not None:
            lex_cand_features, lex_cand_mask, candidates_list, resolved_maps = self._build_lex_candidates(ipa_words)

        output_dict = self._run_session(
            char_ids, word_starts, word_ends, lex_features,
            lex_cand_features, lex_cand_mask, word_mask=wm,
        )
        n_words = len(ipa_words)
        p2g_logits = output_dict["p2g_logits"]

        # Softmax P2G
        logits_2d = p2g_logits[0]
        exp_l = np.exp(logits_2d - logits_2d.max(axis=-1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
        p2g_preds = logits_2d.argmax(axis=-1)

        ortho_results = []
        alternatives_results = []
        confiance_results = []

        for w in range(n_words):
            ws = word_starts[0, w]
            we = word_ends[0, w]
            positions = list(range(ws, we + 1))

            word_labels = [self.idx2p2g.get(int(p2g_preds[i]), "_CONT") for i in positions]
            ortho_top1 = reconstruct_ortho(word_labels)
            ortho_results.append(ortho_top1)

            word_probs = [float(probs[i, p2g_preds[i]]) for i in positions]
            confiance = 1.0
            for p in word_probs:
                confiance *= p
            n_pos = len(positions)
            if n_pos > 0:
                confiance = confiance ** (1.0 / n_pos)
            confiance_results.append(confiance)

            alternatives: list[tuple[str, float]] = [(ortho_top1, confiance)]
            for pos_idx, i in enumerate(positions):
                if probs[i, p2g_preds[i]] >= 0.8:
                    continue
                top_k_indices = np.argsort(probs[i])[-k:][::-1]
                for rank, alt_idx in enumerate(top_k_indices):
                    if rank == 0:
                        continue
                    alt_prob = float(probs[i, alt_idx])
                    if alt_prob < 0.01:
                        break
                    alt_labels = list(word_labels)
                    alt_labels[pos_idx] = self.idx2p2g.get(int(alt_idx), "_CONT")
                    alt_ortho = reconstruct_ortho(alt_labels)
                    if alt_ortho and alt_ortho != ortho_top1:
                        alt_probs = list(word_probs)
                        alt_probs[pos_idx] = alt_prob
                        alt_conf = 1.0
                        for p in alt_probs:
                            alt_conf *= p
                        if n_pos > 0:
                            alt_conf = alt_conf ** (1.0 / n_pos)
                        alternatives.append((alt_ortho, alt_conf))

            seen: set[str] = set()
            unique_alts: list[tuple[str, float]] = []
            for ortho, score in sorted(alternatives, key=lambda x: -x[1]):
                if ortho not in seen:
                    seen.add(ortho)
                    unique_alts.append((ortho, score))
            alternatives_results.append(unique_alts[:k])

        # Apply lex_select (overrides P2G top1 for words with candidates)
        if candidates_list is not None and self.apply_lex_select:
            ortho_results = self._apply_lex_select(
                output_dict, word_starts, n_words, candidates_list, ortho_results,
                ipa_words=ipa_words)

        # POS with top-K
        pos_results = []
        pos_scores = []
        if "pos_logits" in output_dict:
            pos_logits_2d = output_dict["pos_logits"][0]
            exp_l = np.exp(pos_logits_2d - pos_logits_2d.max(axis=-1, keepdims=True))
            pos_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
            for w in range(n_words):
                p = pos_probs[w]
                top_indices = np.argsort(p)[-k:][::-1]
                top_k_pos = [
                    (self.idx2pos.get(int(idx), "_"), float(p[idx]))
                    for idx in top_indices
                    if self.idx2pos.get(int(idx), "<PAD>") != "<PAD>" and p[idx] > 0.001
                ]
                pos_results.append(top_k_pos[0][0] if top_k_pos else "NOM")
                pos_scores.append(top_k_pos)

        # Morpho with top-K
        morpho_results: dict[str, list[str]] = {}
        morpho_scores: dict[str, list[list[tuple[str, float]]]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key not in output_dict:
                continue
            feat_logits = output_dict[key][0]
            exp_l = np.exp(feat_logits - feat_logits.max(axis=-1, keepdims=True))
            feat_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
            feat_results = []
            feat_scores_list = []
            for w in range(n_words):
                p = feat_probs[w]
                top_indices = np.argsort(p)[-k:][::-1]
                top_k_feat = [
                    (idx2label.get(int(idx), "_"), float(p[idx]))
                    for idx in top_indices
                    if idx2label.get(int(idx), "<PAD>") != "<PAD>" and p[idx] > 0.001
                ]
                feat_results.append(top_k_feat[0][0] if top_k_feat else "_")
                feat_scores_list.append(top_k_feat)
            morpho_results[feat_name] = feat_results
            morpho_scores[feat_name] = feat_scores_list

        return {
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
            "alternatives": alternatives_results,
            "confiance": confiance_results,
            "pos_scores": pos_scores,
            "morpho_scores": morpho_scores,
        }
