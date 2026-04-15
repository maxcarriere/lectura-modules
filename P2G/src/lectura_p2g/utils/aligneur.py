"""Aligneur graphème-phonème V2.

Port autonome depuis lectura-main : aligneur canonique + post-traitement spans.
"""

from __future__ import annotations

from lectura_p2g.utils.ipa import iter_phonemes

Span = tuple[int, int]

_LETTRES_MUETTES_POSSIBLES = {
    "e", "s", "t", "d", "p", "h", "x", "g", "c", "l",
    "m", "n", "b", "f", "r", "v", "z", "q",
}

_IGNORED_CHARS = {"-", " ", "'", "\u2019", "_"}


def _alignement_v2(
    ortho: str,
    phone: str,
    phone_to_graphs: dict[str, list[str]],
) -> tuple[list[str], list[str], bool]:
    """Alignement canonique V2 (DFS avec segmentation phonétique)."""
    ortho_orig = ortho
    ortho_norm = ortho.lower()

    phone_tokens = iter_phonemes(phone.lower())

    bloc_defs = [
        (b, iter_phonemes(b.lower()))
        for b in phone_to_graphs
    ]
    bloc_defs.sort(key=lambda x: len(x[1]), reverse=True)

    def generate_segmentations(max_segmentations: int = 32) -> list[list[str]]:
        segs: list[list[str]] = []

        def dfs_seg(pos: int, current: list[str]) -> None:
            if len(segs) >= max_segmentations:
                return
            if pos == len(phone_tokens):
                segs.append(current[:])
                return

            for bloc, toks in bloc_defs:
                ll = len(toks)
                if phone_tokens[pos : pos + ll] == toks:
                    current.append(bloc)
                    dfs_seg(pos + ll, current)
                    current.pop()

            current.append(phone_tokens[pos])
            dfs_seg(pos + 1, current)
            current.pop()

        dfs_seg(0, [])
        return segs

    segmentations = generate_segmentations()

    all_results: list[tuple[list[str], list[str], int]] = []
    M = len(ortho_orig)

    for elements0 in segmentations:

        def dfs(
            elements: list[str],
            i_el: int,
            pos_ortho: int,
            align_ph: list[str],
            align_gr: list[str],
            pending: str,
            muettes: int,
            allow_split: bool = True,
        ) -> None:
            N = len(elements)

            while pos_ortho < M and ortho_orig[pos_ortho] in _IGNORED_CHARS:
                if align_gr:
                    align_gr[-1] += ortho_orig[pos_ortho]
                pos_ortho += 1

            if i_el == N:
                if pending and align_gr:
                    align_gr[-1] += pending

                if pos_ortho < M and align_gr:
                    for k in range(pos_ortho, M):
                        ch = ortho_orig[k]
                        if ch.lower() in _LETTRES_MUETTES_POSSIBLES:
                            align_gr[-1] += ch + "°"
                            muettes += 1
                        else:
                            return

                all_results.append((align_ph[:], align_gr[:], muettes))
                return

            phon = elements[i_el]

            possible_here = False
            for g in phone_to_graphs.get(phon, []):
                g_norm = g.replace("²", "")
                if ortho_norm.startswith(g_norm, pos_ortho):
                    possible_here = True
                    break

            before = len(all_results)

            for g in phone_to_graphs.get(phon, []):
                g_norm = g.replace("²", "")
                ll = len(g_norm)

                if ortho_norm.startswith(g_norm, pos_ortho):
                    align_ph.append(phon)
                    align_gr.append(pending + g)

                    dfs(
                        elements, i_el + 1, pos_ortho + ll,
                        align_ph, align_gr, "", muettes,
                        allow_split=True,
                    )

                    align_ph.pop()
                    align_gr.pop()

            after = len(all_results)
            produced_solution = after > before

            if allow_split and not produced_solution and not possible_here:
                sub_phons = iter_phonemes(phon.lower())
                if len(sub_phons) > 1:
                    new_elements = (
                        elements[:i_el] + sub_phons + elements[i_el + 1:]
                    )
                    dfs(
                        new_elements, i_el, pos_ortho,
                        align_ph, align_gr, pending, muettes,
                        allow_split=False,
                    )

            if pos_ortho < M and not possible_here:
                ch = ortho_orig[pos_ortho]
                if ch.lower() in _LETTRES_MUETTES_POSSIBLES:
                    mu = ch + "°"

                    if ch.lower() == "h":
                        dfs(
                            elements, i_el, pos_ortho + 1,
                            align_ph, align_gr, pending + mu, muettes + 1,
                            allow_split=True,
                        )
                    else:
                        if align_gr:
                            align_gr[-1] += mu
                            dfs(
                                elements, i_el, pos_ortho + 1,
                                align_ph, align_gr, pending, muettes + 1,
                                allow_split=True,
                            )
                            align_gr[-1] = align_gr[-1][: -len(mu)]
                        else:
                            dfs(
                                elements, i_el, pos_ortho + 1,
                                align_ph, align_gr, pending + mu, muettes + 1,
                                allow_split=True,
                            )

        dfs(elements0, 0, 0, [], [], "", 0, allow_split=True)

    if not all_results:
        return [], [], False

    def _muette_penalty(gr: list[str]) -> int:
        penalty = 0
        pos = 0
        for g in gr:
            for ch in g:
                if ch == "°":
                    penalty += pos
                else:
                    pos += 1
        return penalty

    def _score(
        sol: tuple[list[str], list[str], int],
    ) -> tuple[int, int, int, int]:
        _, gr, muettes = sol
        pass2_count = sum(1 for g in gr if g not in phone_to_graphs)
        muette_pos_penalty = _muette_penalty(gr)
        return (muettes, muette_pos_penalty, pass2_count, len(gr))

    best = min(all_results, key=_score)
    align_ph, align_gr, _ = best

    new_gr: list[str] = []
    idx = 0
    lw = len(ortho_orig)

    for g in align_gr:
        rebuilt = ""
        for ch in g:
            if ch in {"°", "²"}:
                rebuilt += ch
            else:
                if idx < lw:
                    rebuilt += ortho_orig[idx]
                    idx += 1
                else:
                    rebuilt += ch
        new_gr.append(rebuilt)

    return align_ph, new_gr, True


def _build_spans(ortho: str, align_gr: list[str]) -> list[Span]:
    """Calcule les spans (start, end) depuis les graphèmes alignés."""
    spans: list[Span] = []
    pos = 0

    for gr in align_gr:
        g = gr.replace("²", "").replace("°", "")
        if not g:
            spans.append((pos, pos))
            continue
        end = pos + len(g)
        spans.append((pos, end))
        pos = end

    return spans


def _phonemise_alignment(
    align_ph: list[str],
    align_gr: list[str],
    spans: list[Span],
) -> tuple[list[str], list[str], list[Span]]:
    """Éclate les blocs multi-phonèmes en phonèmes atomiques."""
    new_ph: list[str] = []
    new_gr: list[str] = []
    new_spans: list[Span] = []

    for ph, gr, span in zip(align_ph, align_gr, spans):
        tokens = iter_phonemes(ph)

        if len(tokens) == 1:
            new_ph.append(ph)
            new_gr.append(gr)
            new_spans.append(span)
            continue

        first = tokens[0]
        rest = "".join(tokens[1:])

        new_ph.append(first)
        new_gr.append(gr[0] if gr else "")

        g0 = gr[0] if gr else ""
        if g0:
            new_spans.append((span[0], span[0] + len(g0)))
        else:
            new_spans.append((span[0], span[0]))

        if rest:
            new_ph.append(rest)
            new_gr.append(gr[1:] if len(gr) > 1 else "")
            g1 = gr[1:] if len(gr) > 1 else ""
            if g1:
                g1_clean = g1.replace("°", "").replace("²", "")
                new_spans.append((span[0] + len(g0), span[0] + len(g0) + len(g1_clean)))
            else:
                new_spans.append((span[1], span[1]))

    return new_ph, new_gr, new_spans


def aligner(
    ortho: str,
    phone: str,
    phone_to_graphs: dict[str, list[str]],
) -> tuple[list[str], list[str], list[Span], bool]:
    """Aligne graphèmes et phonèmes.

    Returns:
        (dec_phone, dec_ortho, dec_spans, success)
    """
    align_ph, align_gr, ok = _alignement_v2(ortho, phone, phone_to_graphs)

    if not ok:
        return [], [], [], False

    spans = _build_spans(ortho, align_gr)
    dec_ph, dec_gr, dec_spans = _phonemise_alignment(align_ph, align_gr, spans)

    return dec_ph, dec_gr, dec_spans, True
