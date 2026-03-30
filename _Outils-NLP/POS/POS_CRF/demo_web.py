#!/usr/bin/env python3
"""Interface web pour le POS Tagger Lectura.

Nécessite Gradio : pip install gradio

Usage :
    python demo_web.py
    # Ouvre http://localhost:7860 dans le navigateur
"""

from __future__ import annotations

from pathlib import Path

from lectura_pos import TAGSET, PosTagger

MODEL_PATH = Path(__file__).parent / "modele" / "pos_model_crf.json"
LEXICON_PATH = Path(__file__).parent / "modele" / "mini_lexique.json"

tagger = PosTagger(MODEL_PATH, lexicon_path=LEXICON_PATH)


def analyser(texte: str) -> str:
    """Analyse un texte et retourne le résultat formaté."""
    if not texte or not texte.strip():
        return ""
    return tagger.tag_formatted(texte)


def analyser_tableau(texte: str) -> list[list[str]]:
    """Analyse un texte et retourne les résultats en tableau."""
    if not texte or not texte.strip():
        return []
    details = tagger.tag_detailed(texte)
    return [[d["mot"], d["tag"], d["description"]] for d in details]


# ── Tagset de référence pour affichage ──

TAGSET_TABLE = [[tag, desc] for tag, desc in TAGSET.items()]

# ── Interface Gradio ──

try:
    import gradio as gr
except ImportError:
    print("Gradio n'est pas installé.")
    print("Installez-le avec : pip install gradio")
    print()
    print("En attendant, vous pouvez utiliser demo_cli.py")
    raise SystemExit(1)

EXEMPLES = [
    ["Le chat mange la souris dans le jardin."],
    ["Je suis allé au marché acheter des pommes."],
    ["Les enfants jouent dans la cour de l'école."],
    ["Il est parti sans dire un mot."],
    ["Cette belle maison rouge appartient à mon voisin."],
]

with gr.Blocks(title="Lectura POS Tagger") as demo:
    gr.Markdown("# Lectura POS Tagger")
    gr.Markdown("Étiqueteur grammatical CRF pour le français — 18 catégories, "
                "zéro dépendance, modèle 1.8 Mo.")

    with gr.Row():
        with gr.Column(scale=2):
            texte_input = gr.Textbox(
                label="Texte à analyser",
                placeholder="Entrez une phrase en français…",
                lines=3,
            )
            btn = gr.Button("Analyser", variant="primary")

        with gr.Column(scale=3):
            resultat = gr.Dataframe(
                headers=["Mot", "Tag", "Description"],
                label="Résultat",
                interactive=False,
            )

    gr.Examples(examples=EXEMPLES, inputs=texte_input)

    with gr.Accordion("Tagset (18 catégories)", open=False):
        gr.Dataframe(
            value=TAGSET_TABLE,
            headers=["Tag", "Description"],
            interactive=False,
        )

    btn.click(fn=analyser_tableau, inputs=texte_input, outputs=resultat)
    texte_input.submit(fn=analyser_tableau, inputs=texte_input, outputs=resultat)

if __name__ == "__main__":
    demo.launch()
