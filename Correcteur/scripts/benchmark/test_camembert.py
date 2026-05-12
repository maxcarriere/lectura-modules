#!/usr/bin/env python3
"""Test rapide : CamemBERT pseudo-log-likelihood vs trigram.

Approche PLL : pour chaque candidat, construire la phrase complete,
puis masquer chaque token un par un et sommer les log-probs.
Le candidat avec la PLL la plus haute gagne.

Usage :
    python scripts/benchmark/test_camembert.py
"""

from __future__ import annotations

import time
import torch
from transformers import CamembertTokenizer, CamembertForMaskedLM

# --- Cas de test ---
# (phrase_avec_erreur, position_mot, candidats, attendu, categorie)

CAS_TEST = [
    # Homophones
    ("il et content", 1, ["et", "est"], "est", "HOMO"),
    ("il a mange a la maison", 3, ["a", "à"], "à", "HOMO"),
    ("ils son contents", 1, ["son", "sont"], "sont", "HOMO"),
    ("on mange bien", 0, ["on", "ont"], "on", "HOMO"),
    ("ils ont faim", 1, ["on", "ont"], "ont", "HOMO"),
    ("il ce lave", 1, ["ce", "se"], "se", "HOMO"),
    ("ce livre est beau", 0, ["ce", "se"], "ce", "HOMO"),
    ("je mange ou je dors", 2, ["ou", "où"], "ou", "HOMO"),
    ("la maison ou je vis", 2, ["ou", "où"], "où", "HOMO"),
    ("son chat est la", 3, ["la", "là"], "là", "HOMO"),
    ("la maison est grande", 0, ["la", "là"], "la", "HOMO"),
    ("il peu manger", 1, ["peu", "peux", "peut"], "peut", "HOMO"),

    # Accords pluriel
    ("les chat dort", 1, ["chat", "chats"], "chats", "ACC_PLUR"),
    ("les enfant jouent", 1, ["enfant", "enfants"], "enfants", "ACC_PLUR"),
    ("des maison sont grandes", 1, ["maison", "maisons"], "maisons", "ACC_PLUR"),
    ("ses livre sont beaux", 1, ["livre", "livres"], "livres", "ACC_PLUR"),

    # Accord sujet-verbe
    ("les chats mange", 2, ["mange", "mangent"], "mangent", "ACC_SV"),
    ("les enfants joue", 2, ["joue", "jouent"], "jouent", "ACC_SV"),
    ("ils mange bien", 1, ["mange", "mangent"], "mangent", "ACC_SV"),
    ("il mangent bien", 1, ["mange", "mangent"], "mange", "ACC_SV"),

    # Participes passes
    ("il a mange une pomme", 2, ["mange", "mangé", "manger"], "mangé", "PP"),
    ("elle a chante une chanson", 2, ["chante", "chanté", "chanter"], "chanté", "PP"),
    ("il va manger une pomme", 2, ["mange", "mangé", "manger"], "manger", "PP"),

    # Genre
    ("la petit fille", 1, ["petit", "petite"], "petite", "GENRE"),
    ("le grande maison", 1, ["grand", "grande"], "grande", "GENRE"),

    # Phrases correctes (pas de faux positif)
    ("le chat mange", 1, ["chat", "chats"], "chat", "OK"),
    ("les enfants jouent", 1, ["enfant", "enfants"], "enfants", "OK"),
    ("il est content", 1, ["et", "est"], "est", "OK"),
    ("son chat dort", 0, ["son", "sont"], "son", "OK"),
]


def pll_score(
    model, tokenizer, phrase: str,
) -> float:
    """Pseudo-log-likelihood d'une phrase.

    Pour chaque token (hors <s> et </s>), masque ce token et
    somme le log-prob du token original a cette position.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(phrase, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    input_ids = inputs["input_ids"][0]  # (seq_len,)
    n = input_ids.size(0)

    # Positions a scorer : tout sauf <s> (0) et </s> (n-1)
    positions = list(range(1, n - 1))
    if not positions:
        return 0.0

    # Batch : creer n-2 copies avec un masque different a chaque fois
    batch_ids = input_ids.unsqueeze(0).repeat(len(positions), 1)  # (n-2, seq_len)
    labels = torch.full_like(batch_ids, -100)  # ignore tout

    for i, pos in enumerate(positions):
        labels[i, pos] = batch_ids[i, pos].item()
        batch_ids[i, pos] = tokenizer.mask_token_id

    # Inference en un seul forward pass
    with torch.no_grad():
        outputs = model(input_ids=batch_ids, attention_mask=inputs["attention_mask"].repeat(len(positions), 1))
        logits = outputs.logits  # (n-2, seq_len, vocab)

    # Extraire le log-prob de chaque token masque
    total = 0.0
    for i, pos in enumerate(positions):
        log_probs = torch.log_softmax(logits[i, pos], dim=-1)
        token_id = input_ids[pos].item()
        total += log_probs[token_id].item()

    return total


def score_candidats_pll(
    model, tokenizer, phrase_mots: list[str], position: int, candidats: list[str],
) -> list[tuple[str, float]]:
    """Score chaque candidat par PLL de la phrase complete."""
    scores = []
    for c in candidats:
        mots = list(phrase_mots)
        mots[position] = c
        phrase = " ".join(mots)
        s = pll_score(model, tokenizer, phrase)
        scores.append((c, s))
    scores.sort(key=lambda x: -x[1])
    return scores


def score_trigram(
    lm, phrase_mots: list[str], position: int, candidats: list[str],
) -> list[tuple[str, float]]:
    """Score les candidats avec le trigram."""
    ctx_g = [m.lower() for m in phrase_mots[:position]]
    ctx_d = [m.lower() for m in phrase_mots[position + 1:position + 2]]
    return lm.scorer_candidats(candidats, ctx_g, ctx_d)


def main():
    print("Chargement CamemBERT...", flush=True)
    t0 = time.perf_counter()
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertForMaskedLM.from_pretrained("camembert-base")
    model.eval()
    t_load = time.perf_counter() - t0
    print(f"  CamemBERT charge en {t_load:.1f}s")

    # GPU si disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}")

    # Monkey-patch tokenizer pour GPU
    orig_call = tokenizer.__class__.__call__
    def tokenizer_to_device(self, *args, **kwargs):
        kwargs.setdefault("return_tensors", "pt")
        result = orig_call(self, *args, **kwargs)
        return {k: v.to(device) if hasattr(v, 'to') else v for k, v in result.items()}

    # Charger trigram
    lm = None
    try:
        from lectura_correcteur._language_model import ScorerNgram
        lm = ScorerNgram("/data/work/projets/lectura/data/wikipedia/ngram_3.db")
        print("  Trigram charge")
    except Exception as e:
        print(f"  Trigram non disponible: {e}")

    # Evaluer
    n_ok_cb = 0
    n_ok_tri = 0
    n_total = len(CAS_TEST)
    t_cb_total = 0.0

    print(f"\n{'Phrase':<40} {'Att':>8} {'PLL':>8} {'Tri':>8} {'Cat':<10}")
    print("-" * 90)

    for phrase, pos, candidats, attendu, cat in CAS_TEST:
        mots = phrase.split()

        # CamemBERT PLL
        t1 = time.perf_counter()
        scores_cb = score_candidats_pll(model, tokenizer, mots, pos, candidats)
        t_cb_total += time.perf_counter() - t1
        top_cb = scores_cb[0][0]
        ok_cb = top_cb == attendu
        n_ok_cb += int(ok_cb)

        # Trigram
        top_tri = "—"
        ok_tri = False
        if lm:
            scores_tri = score_trigram(lm, mots, pos, candidats)
            top_tri = scores_tri[0][0]
            ok_tri = top_tri == attendu
            n_ok_tri += int(ok_tri)

        # Affichage
        mark_cb = "✓" if ok_cb else "✗"
        mark_tri = "✓" if ok_tri else "✗"
        phrase_short = phrase[:38]
        print(
            f"{phrase_short:<40} {attendu:>8} "
            f"{mark_cb}{top_cb:>7} "
            f"{mark_tri}{top_tri:>7} "
            f"{cat:<10}"
        )

        # Detail des scores si divergence
        if not ok_cb:
            for c, s in scores_cb:
                print(f"    PLL: {c:>10} = {s:+.3f}")

    # Resume
    ms_per = (t_cb_total / n_total) * 1000
    print(f"\n{'=' * 90}")
    print(f"CamemBERT PLL : {n_ok_cb}/{n_total} ({100*n_ok_cb/n_total:.1f}%)  "
          f"  {ms_per:.0f} ms/candidat")
    if lm:
        print(f"Trigram       : {n_ok_tri}/{n_total} ({100*n_ok_tri/n_total:.1f}%)")

    # Par categorie
    from collections import defaultdict
    par_cat = defaultdict(lambda: {"cb_ok": 0, "tri_ok": 0, "total": 0})
    for i, (phrase, pos, candidats, attendu, cat) in enumerate(CAS_TEST):
        mots = phrase.split()
        par_cat[cat]["total"] += 1
        scores_cb = score_candidats_pll(model, tokenizer, mots, pos, candidats)
        if scores_cb[0][0] == attendu:
            par_cat[cat]["cb_ok"] += 1
        if lm:
            scores_tri = score_trigram(lm, mots, pos, candidats)
            if scores_tri[0][0] == attendu:
                par_cat[cat]["tri_ok"] += 1

    print(f"\n{'Categorie':<12} {'CamemBERT PLL':>14} {'Trigram':>12}")
    print("-" * 42)
    for cat in sorted(par_cat):
        d = par_cat[cat]
        t = d["total"]
        pct_cb = 100 * d["cb_ok"] / t if t else 0
        pct_tri = 100 * d["tri_ok"] / t if t else 0
        print(f"{cat:<12} {d['cb_ok']}/{t} ({pct_cb:5.1f}%)  {d['tri_ok']}/{t} ({pct_tri:5.1f}%)")


if __name__ == "__main__":
    main()
