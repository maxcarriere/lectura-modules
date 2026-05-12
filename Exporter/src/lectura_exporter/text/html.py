"""Export HTML — template CSS + structure + annotations, pure Python."""

from __future__ import annotations

import re
from pathlib import Path


def export_html(
    enriched_html: str | None,
    plain_text: str,
    annotation_rows: list[tuple[str, list[tuple[str, str | None]]]],
    output_path: str,
) -> None:
    """Génère un fichier HTML complet avec texte enrichi et annotations.

    Parameters
    ----------
    enriched_html : str | None
        HTML pré-rendu par ColoredTextDisplay (contenu du ``<body>`` Qt),
        ou ``None`` pour utiliser le texte brut.
    plain_text : str
        Texte brut (fallback si enriched_html est None).
    annotation_rows : list[tuple[str, list[tuple[str, str | None]]]]
        Lignes d'annotation : ``(label, [(valeur, couleur_hex_ou_None), ...])``.
    output_path : str
        Chemin du fichier HTML de sortie.
    """
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="fr">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append("<title>Lectura — Texte exporté</title>")
    parts.append("<style>")
    parts.append("body { font-family: 'Noto Sans', sans-serif; background: #1e1e1e; color: #ddd; padding: 2em; }")
    parts.append(".enriched { font-size: 22px; line-height: 1.6; margin-bottom: 2em; }")
    parts.append("table.annotations { border-collapse: collapse; font-size: 14px; }")
    parts.append("table.annotations th, table.annotations td { border: 1px solid #555; padding: 4px 8px; text-align: center; }")
    parts.append("table.annotations th { background: #333; }")
    parts.append("</style>")
    parts.append("</head>")
    parts.append("<body>")

    # Section texte enrichi
    if enriched_html is not None:
        # Extraire le contenu du <body> Qt et nettoyer les propriétés -qt-*
        body_match = re.search(r"<body[^>]*>(.*)</body>", enriched_html, re.DOTALL)
        body_content = body_match.group(1) if body_match else enriched_html
        body_content = re.sub(r'\s*-qt-[a-z-]+:[^;]+;?', '', body_content)
        parts.append('<div class="enriched">')
        parts.append(body_content)
        parts.append("</div>")
    else:
        parts.append('<div class="enriched">')
        parts.append(f'<p>{plain_text}</p>')
        parts.append("</div>")

    # Tableau d'annotations
    if annotation_rows:
        parts.append('<table class="annotations">')
        for label, cells in annotation_rows:
            parts.append("<tr>")
            parts.append(f"<th>{label}</th>")
            for value, color in cells:
                style = ""
                if color:
                    style = f' style="background:{color}; color:white;"'
                parts.append(f"<td{style}>{value}</td>")
            parts.append("</tr>")
        parts.append("</table>")

    parts.append("</body>")
    parts.append("</html>")

    Path(output_path).write_text("\n".join(parts), encoding="utf-8")
