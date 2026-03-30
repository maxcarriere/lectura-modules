"""Lectura G2P Unifié — Démo Gradio.

Trois onglets :
  1. G2P : transcription phonémique IPA
  2. POS + Morphologie : étiquetage et traits morphologiques
  3. Pipeline complet : toutes les tâches + liaison
"""

from __future__ import annotations

import gradio as gr
from lectura_nlp.inference_numpy import NumpyInferenceEngine
from lectura_nlp.tokeniseur import tokeniser
from lectura_nlp.posttraitement import (
    appliquer_liaison,
    charger_corrections,
    corriger_g2p,
)

# Charger le modèle au démarrage
ENGINE = NumpyInferenceEngine("modeles/unifie_weights.json",
                              "modeles/unifie_vocab.json")
charger_corrections("modeles/g2p_corrections_unifie.json")


def analyser(phrase: str) -> dict:
    """Analyse commune à tous les onglets."""
    tokens = tokeniser(phrase)
    if not tokens:
        return {"tokens": [], "g2p": [], "pos": [], "liaison": [], "morpho": {}}
    result = ENGINE.analyser(tokens)
    # Appliquer corrections G2P
    for i, tok in enumerate(tokens):
        if i < len(result["g2p"]):
            result["g2p"][i] = corriger_g2p(tok, result["g2p"][i])
    return result


# ── Onglet 1 : G2P ──────────────────────────────────────────────────────

def tab_g2p(phrase: str) -> str:
    result = analyser(phrase)
    tokens = result["tokens"]
    g2p = result["g2p"]
    if not tokens:
        return "Aucun token trouvé."
    lines = [f"| Mot | IPA |", f"|-----|-----|"]
    for i, tok in enumerate(tokens):
        ipa = g2p[i] if i < len(g2p) else ""
        lines.append(f"| {tok} | /{ipa}/ |")
    lines.append("")
    lines.append(f"**IPA complet** : /{' '.join(g2p)}/")
    return "\n".join(lines)


# ── Onglet 2 : POS + Morphologie ────────────────────────────────────────

def tab_pos_morpho(phrase: str) -> str:
    result = analyser(phrase)
    tokens = result["tokens"]
    pos = result["pos"]
    morpho = result.get("morpho", {})
    if not tokens:
        return "Aucun token trouvé."

    feats = sorted(morpho.keys())
    header = "| Mot | POS |"
    sep = "|-----|-----|"
    for f in feats:
        header += f" {f} |"
        sep += "-----|"

    lines = [header, sep]
    for i, tok in enumerate(tokens):
        p = pos[i] if i < len(pos) else ""
        row = f"| {tok} | {p} |"
        for f in feats:
            vals = morpho.get(f, [])
            v = vals[i] if i < len(vals) else "_"
            row += f" {v} |"
        lines.append(row)
    return "\n".join(lines)


# ── Onglet 3 : Pipeline complet ─────────────────────────────────────────

def tab_pipeline(phrase: str) -> str:
    result = analyser(phrase)
    tokens = result["tokens"]
    g2p = result["g2p"]
    pos = result["pos"]
    liaison = result["liaison"]
    morpho = result.get("morpho", {})
    if not tokens:
        return "Aucun token trouvé."

    # Table principale
    feats = sorted(morpho.keys())
    header = "| Mot | IPA | POS | Liaison |"
    sep = "|-----|-----|-----|---------|"
    for f in feats:
        header += f" {f} |"
        sep += "-----|"

    lines = [header, sep]
    for i, tok in enumerate(tokens):
        ipa = g2p[i] if i < len(g2p) else ""
        p = pos[i] if i < len(pos) else ""
        lia = liaison[i] if i < len(liaison) else ""
        row = f"| {tok} | /{ipa}/ | {p} | {lia} |"
        for f in feats:
            vals = morpho.get(f, [])
            v = vals[i] if i < len(vals) else "_"
            row += f" {v} |"
        lines.append(row)

    # IPA avec liaisons
    ipa_final = appliquer_liaison(tokens, g2p, liaison)
    lines.append("")
    lines.append(f"**IPA avec liaisons** : /{' '.join(ipa_final)}/")
    return "\n".join(lines)


# ── Interface Gradio ─────────────────────────────────────────────────────

EXAMPLES = [
    "Les enfants sont arrivés à la maison.",
    "Un petit animal courait dans les bois.",
    "Ils ont été très heureux de vous revoir.",
    "Elle est allée acheter du pain.",
    "Les oiseaux chantent dans les arbres.",
]

with gr.Blocks(title="Lectura G2P Unifié") as demo:
    gr.Markdown(
        "# Lectura G2P Unifié\n"
        "Modèle unifié **G2P + POS + Morphologie + Liaison** pour le français "
        "(BiLSTM 1.75M params, ONNX INT8 = 1.8 Mo)"
    )

    with gr.Tabs():
        with gr.TabItem("G2P"):
            gr.Markdown("Transcription graphème → phonème (IPA)")
            inp1 = gr.Textbox(label="Phrase", placeholder="Entrez une phrase en français...")
            out1 = gr.Markdown(label="Résultat")
            btn1 = gr.Button("Transcrire")
            btn1.click(tab_g2p, inputs=inp1, outputs=out1)
            gr.Examples(EXAMPLES, inputs=inp1)

        with gr.TabItem("POS + Morphologie"):
            gr.Markdown("Étiquetage morpho-syntaxique et traits morphologiques")
            inp2 = gr.Textbox(label="Phrase", placeholder="Entrez une phrase en français...")
            out2 = gr.Markdown(label="Résultat")
            btn2 = gr.Button("Analyser")
            btn2.click(tab_pos_morpho, inputs=inp2, outputs=out2)
            gr.Examples(EXAMPLES, inputs=inp2)

        with gr.TabItem("Pipeline complet"):
            gr.Markdown("Toutes les tâches : G2P + POS + Morphologie + Liaison")
            inp3 = gr.Textbox(label="Phrase", placeholder="Entrez une phrase en français...")
            out3 = gr.Markdown(label="Résultat")
            btn3 = gr.Button("Analyser")
            btn3.click(tab_pipeline, inputs=inp3, outputs=out3)
            gr.Examples(EXAMPLES, inputs=inp3)

if __name__ == "__main__":
    demo.launch()
