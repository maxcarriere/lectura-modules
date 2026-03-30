"""Lectura POS Tagger — Démo interactive.

Déployé sur Hugging Face Spaces (Gradio).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from lectura_pos import TAGSET, PosTagger

MODEL_PATH = Path(__file__).parent / "modele" / "pos_model_crf.json"
LEXICON_PATH = Path(__file__).parent / "modele" / "mini_lexique.json"

tagger = PosTagger(MODEL_PATH, lexicon_path=LEXICON_PATH)


def analyser(texte: str) -> list[list[str]]:
    if not texte or not texte.strip():
        return []
    details = tagger.tag_detailed(texte)
    return [[d["mot"], d["tag"], d["description"]] for d in details]


TAGSET_TABLE = [[tag, desc] for tag, desc in TAGSET.items()]

EXEMPLES = [
    ["Le chat mange la souris dans le jardin."],
    ["Je suis allé au marché acheter des pommes."],
    ["Les enfants jouent dans la cour de l'école."],
    ["Il est parti sans dire un mot."],
    ["Cette belle maison rouge appartient à mon voisin."],
    ["Nous avons toujours beaucoup aimé cette très belle maison."],
    ["Elle mange du pain avec du beurre."],
    ["Qui a dit que la vie était simple ?"],
]

DESCRIPTION = """\
# Lectura POS Tagger

**Étiqueteur grammatical CRF pour le français** — 97.5% d'accuracy, 1.8 Mo, zéro dépendance.

Tapez ou collez une phrase en français pour voir l'analyse grammaticale en temps réel.

| | |
|---|---|
| Architecture | CRF + Viterbi (Python pur) |
| Accuracy | 97.5% (Universal Dependencies) |
| Taille | 1.8 Mo |
| Tagset | 18 catégories |
| Dépendances | Aucune |
"""

FOOTER = """\
---

**Vous souhaitez intégrer ce modèle dans votre projet ?**

L'archive complète (modèle, code, exemples, données d'entraînement, documentation) :
**[Télécharger sur Gumroad — 29 €](https://TODO_LIEN_GUMROAD)** | Licence CC BY-SA 4.0

*Développé par Lectura.*
"""

with gr.Blocks(
    title="Lectura POS Tagger — Étiqueteur grammatical français",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            texte_input = gr.Textbox(
                label="Texte à analyser",
                placeholder="Entrez une phrase en français…",
                lines=3,
            )
            btn = gr.Button("Analyser", variant="primary", size="lg")

        with gr.Column(scale=3):
            resultat = gr.Dataframe(
                headers=["Mot", "Tag", "Description"],
                label="Analyse grammaticale",
                interactive=False,
                wrap=True,
            )

    gr.Examples(
        examples=EXEMPLES,
        inputs=texte_input,
        label="Exemples",
    )

    with gr.Accordion("Tagset complet (18 catégories)", open=False):
        gr.Dataframe(
            value=TAGSET_TABLE,
            headers=["Tag", "Description"],
            interactive=False,
        )

    gr.Markdown(FOOTER)

    btn.click(fn=analyser, inputs=texte_input, outputs=resultat)
    texte_input.submit(fn=analyser, inputs=texte_input, outputs=resultat)

if __name__ == "__main__":
    demo.launch()
