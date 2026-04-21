"""Lectura Aligneur-Syllabeur — Aligneur grapheme-phoneme et syllabeur phonologique du francais.

Fichier unique, autonome, zero dependance Python.
Phonemiseur pluggable avec backend eSpeak-NG par defaut.

Architecture en 3 couches :
    Alignement : correspondance lettre-par-lettre orthographe <-> IPA (DFS)
    E1 : Groupes de lecture (elisions, liaisons, enchainements)
    E2 : Syllabation sur les groupes avec decomposition attaque/noyau/coda

Usage rapide :
    from lectura_aligneur import LecturaSyllabeur

    syl = LecturaSyllabeur()                        # eSpeak par défaut
    result = syl.analyze("chocolat")
    for s in result.syllabes:
        print(f"{s.ortho} -> /{s.phone}/")

Usage complet (avec groupes de lecture) :
    from lectura_aligneur import LecturaSyllabeur, MotAnalyse, OptionsGroupes

    mots = [
        MotAnalyse(token=..., phone="lez", liaison="Lz"),
        MotAnalyse(token=..., phone="ɑ̃fɑ̃", liaison="none"),
    ]
    r = syl.analyser_complet(mots)
    print(f"{r.nb_groupes} groupes, {r.nb_syllabes} syllabes")

IPA direct (sans phonémiseur) :
    sylls = syl.syllabify_ipa("ʃɔkɔla")            # -> ["ʃɔ", "kɔ", "la"]

Pré-requis système (mode eSpeak) :
    sudo apt install espeak-ng        # Linux
    brew install espeak               # macOS
    choco install espeak-ng           # Windows

Copyright (c) 2025 Lectura — Licence CC BY-SA 4.0.
Voir LICENCE.txt et ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from lectura_aligneur._chargeur import (
    alphabet_ipa as _get_alphabet_ipa,
    phone_to_graphemes as _get_phone_to_graphemes,
    espeak_to_ipa as _get_espeak_to_ipa,
    liaison_consonnes as _get_liaison_consonnes,
    lettres_muettes_possibles as _get_lettres_muettes_possibles,
    voyelles as _get_voyelles,
    consonnes as _get_consonnes,
    semi_voyelles as _get_semi_voyelles,
)

# Types et utilitaires partages (externalises pour le package public)
from lectura_aligneur._types import (  # noqa: F401
    Span,
    Phoneme,
    GroupePhonologique,
    Syllabe,
    ResultatAnalyse,
    MotAnalyse,
    EventFormule,
    LectureFormule,
    OptionsGroupes,
    GroupeLecture,
    ResultatGroupe,
    ResultatSyllabation,
)
from lectura_aligneur._utilitaires import (  # noqa: F401
    iter_phonemes,
    est_voyelle,
    est_consonne,
    est_semi_voyelle,
)

logger = logging.getLogger(__name__)

__version__ = "2.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# Donnees metier chargees depuis JSON (via _chargeur)
# ══════════════════════════════════════════════════════════════════════════════

# Acces paresseux — les fonctions _get_* retournent les donnees depuis le JSON.
# Les anciennes constantes sont remplacees par des proprietes calculees.
_ALPHABET_IPA = _get_alphabet_ipa
_PHONE_TO_GRAPHEMES = _get_phone_to_graphemes
_ESPEAK_TO_IPA = _get_espeak_to_ipa


# Aliases locaux pour le code interne de l'algo
_VOYELLES = _get_voyelles
_CONSONNES = _get_consonnes
_SEMI_VOYELLES = _get_semi_voyelles


# ══════════════════════════════════════════════════════════════════════════════
# Protocol Phonemizer
# ══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class Phonemizer(Protocol):
    """Interface pour brancher n'importe quel phonémiseur."""

    def phonemize(self, word: str) -> str: ...


# ══════════════════════════════════════════════════════════════════════════════
# Backend eSpeak-NG
# ══════════════════════════════════════════════════════════════════════════════


class EspeakPhonemizer:
    """Phonémiseur basé sur eSpeak-NG."""

    def __init__(self, lang: str = "fr") -> None:
        self._lang = lang
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                subprocess.run(
                    ["espeak-ng", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                self._available = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._available = False
                logger.warning("eSpeak-NG not available, phonemization will fail")
        return self._available

    def phonemize(self, word: str) -> str:
        if not self.is_available():
            raise RuntimeError(
                "eSpeak-NG n'est pas installe. "
                "Installez-le (sudo apt install espeak-ng) ou "
                "fournissez un phonemiseur custom."
            )

        proc = subprocess.run(
            ["espeak-ng", "-v", self._lang, "--ipa", "-q", word],
            capture_output=True,
            timeout=5,
        )
        raw = proc.stdout.decode("utf-8", errors="replace").strip()

        cleaned = raw.replace("ˈ", "").replace("ˌ", "").replace("-", "")
        cleaned = cleaned.replace(" ", "")

        return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# Adaptateur pour Lectura G2P (duck typing)
# ══════════════════════════════════════════════════════════════════════════════


class _G2PAdapter:
    """Adapte un objet LecturaG2P (méthode .predict) au Protocol Phonemizer."""

    def __init__(self, g2p_obj: object) -> None:
        if not hasattr(g2p_obj, "predict"):
            raise TypeError(
                "L'objet G2P doit avoir une méthode .predict(word) -> str"
            )
        self._g2p = g2p_obj

    def phonemize(self, word: str) -> str:
        return self._g2p.predict(word)  # type: ignore[union-attr]


# ══════════════════════════════════════════════════════════════════════════════
# Syllabeur (algorithme par sonorité)
# ══════════════════════════════════════════════════════════════════════════════

COMPOUND_PHONEMES: set[str] = {"tʃ", "dʒ", "ts", "ɡz"}


@dataclass(frozen=True)
class _SonorityClasses:
    O: set[str]  # Obstruantes
    N: set[str]  # Nasales
    L: set[str]  # Liquides
    Y: set[str]  # Semi-voyelles
    V: set[str]  # Voyelles


def _build_sonority_classes() -> _SonorityClasses:
    """Construit les 5 classes de sonorité depuis l'alphabet IPA."""
    O: set[str] = set()
    N: set[str] = set()
    L: set[str] = set()
    Y: set[str] = set()
    V: set[str] = set()

    for ph, meta in _ALPHABET_IPA().items():
        t = meta.get("type")
        if t == "voyelle":
            V.add(ph)
        elif t == "semi-voyelle":
            Y.add(ph)
        elif t == "consonne":
            st = meta.get("sous_type")
            if st == "nasale":
                N.add(ph)
            elif st == "liquide":
                L.add(ph)
            else:
                O.add(ph)

    O.add("s")
    L.add("s")
    O |= COMPOUND_PHONEMES

    return _SonorityClasses(O=O, N=N, L=L, Y=Y, V=V)


# Singleton — les classes ne changent pas
_SONORITY = _build_sonority_classes()


def _class_of(token: str, classes: _SonorityClasses) -> str:
    if token in classes.V or est_voyelle(token):
        return "V"
    if token in classes.Y or est_semi_voyelle(token):
        return "Y"
    if token in classes.L:
        return "L"
    if token in classes.N:
        return "N"
    return "O"


def _merge_compounds_ipa(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            pair = tokens[i] + tokens[i + 1]
            if pair in COMPOUND_PHONEMES:
                out.append(pair)
                i += 2
                continue
        out.append(tokens[i])
        i += 1
    return out


def _split_central_cluster(
    cluster: list[str],
    classes: _SonorityClasses,
    *,
    s_liquid_in_attack: bool = True,
) -> int:
    if len(cluster) <= 1:
        return 0

    left = 0
    right = len(cluster)

    def is_liquid_attack(c: str) -> bool:
        return c in classes.L and (s_liquid_in_attack or c != "s")

    layers = ["Y", "L", "N", "O"]

    for layer in layers:
        if right > left:
            c = cluster[right - 1]
            if layer == "Y" and c in classes.Y:
                right -= 1
            elif layer == "L" and is_liquid_attack(c):
                right -= 1
            elif layer == "N" and c in classes.N:
                right -= 1
            elif layer == "O" and c in classes.O:
                right -= 1

        if right > left:
            c = cluster[left]
            if layer == "Y" and c in classes.Y:
                left += 1
            elif layer == "L" and c in classes.L:
                left += 1
            elif layer == "N" and c in classes.N:
                left += 1
            elif layer == "O" and c in classes.O:
                left += 1

    if right > left:
        i = left
        while (
            i < right
            and _class_of(cluster[i], classes) in {"L", "Y"}
            and cluster[i] != "s"
        ):
            i += 1
        left = i

    return left


def _syllabify_ipa(phone: str) -> list[str]:
    """Découpe une chaîne IPA en syllabes par le modèle de sonorité."""
    tokens = _merge_compounds_ipa(iter_phonemes(phone))

    if not tokens:
        return [phone] if phone else []

    vowel_idx = [i for i, t in enumerate(tokens) if _class_of(t, _SONORITY) == "V"]
    if len(vowel_idx) <= 1:
        return ["".join(tokens)]

    boundaries: list[int] = []
    for vi, vj in zip(vowel_idx, vowel_idx[1:]):
        cluster = tokens[vi + 1 : vj]
        k = _split_central_cluster(cluster, _SONORITY)
        boundaries.append((vi + 1) + k)

    out: list[str] = []
    start = 0
    for b in boundaries:
        out.append("".join(tokens[start:b]))
        start = b
    out.append("".join(tokens[start:]))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Aligneur graphème-phonème
# ══════════════════════════════════════════════════════════════════════════════

_LETTRES_MUETTES_POSSIBLES = _get_lettres_muettes_possibles

_IGNORED_CHARS = {"-", " ", "'", "\u2019", "_"}


def _alignement_v2(
    ortho: str,
    phone: str,
    phone_to_graphs: dict[str, list[str]],
    word_boundaries: list[int] | None = None,
) -> tuple[list[str], list[str], bool]:
    ortho_orig = ortho
    ortho_norm = ortho.lower()
    phone_tokens = iter_phonemes(phone.lower())
    # Positions juste avant une frontière de mot (1 ou 2 chars avant)
    # pour autoriser l'exploration e-muet / s-muet / e°s° en fin de mot.
    _boundary_muet_positions: frozenset[int] = frozenset()
    if word_boundaries:
        pos_set: set[int] = set()
        for b in word_boundaries:
            if b - 1 >= 0:
                pos_set.add(b - 1)  # dernière lettre du mot (s, e)
            if b - 2 >= 0:
                pos_set.add(b - 2)  # avant-dernière (e dans e°s°)
        _boundary_muet_positions = frozenset(pos_set)

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
                        if ch.lower() in _LETTRES_MUETTES_POSSIBLES():
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
                        elements[:i_el] + sub_phons + elements[i_el + 1 :]
                    )
                    dfs(
                        new_elements, i_el, pos_ortho,
                        align_ph, align_gr, pending, muettes,
                        allow_split=False,
                    )

            if pos_ortho < M and not possible_here:
                ch = ortho_orig[pos_ortho]
                if ch.lower() in _LETTRES_MUETTES_POSSIBLES():
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

            # En mode groupe : explorer e/s muet juste avant une frontière
            # de mot, même quand un graphème matche à cette position.
            # Couvre : e° (Maxime+et), s° (cas liaison), e°s° (grandes+amis).
            if (pos_ortho < M and possible_here
                    and pos_ortho in _boundary_muet_positions
                    and ortho_orig[pos_ortho].lower() in ("e", "s")):
                ch = ortho_orig[pos_ortho]
                mu = ch + "°"
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
        pass2_count = sum(1 for g in gr if g not in phone_to_graphs and "²" not in g)
        muette_pos_penalty = _muette_penalty(gr)
        return (muettes, muette_pos_penalty, pass2_count, len(gr))

    best = min(all_results, key=_score)
    align_ph, align_gr, _ = best

    # Réinjection de la casse originale
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
        new_ph.append(first)
        new_gr.append(gr[0] if gr else "")
        g0 = gr[0] if gr else ""
        if g0:
            new_spans.append((span[0], span[0] + len(g0)))
        else:
            new_spans.append((span[0], span[0]))

        rest = "".join(tokens[1:])
        if rest:
            new_ph.append(rest)
            g1 = gr[1:] if len(gr) > 1 else ""
            if g1:
                g1_clean = g1.replace("°", "").replace("²", "")
                if g1_clean:
                    new_gr.append(g1)
                    new_spans.append((span[0] + len(g0), span[0] + len(g0) + len(g1_clean)))
                else:
                    # ² seul : la même lettre produit les deux sons
                    new_gr.append(g0 + "²")
                    new_spans.append((span[0], span[0] + len(g0)))
            else:
                new_gr.append("")
                new_spans.append((span[1], span[1]))

    return new_ph, new_gr, new_spans


def _aligner(
    ortho: str,
    phone: str,
    word_boundaries: list[int] | None = None,
) -> tuple[list[str], list[str], list[Span], bool]:
    """Aligne graphèmes et phonèmes."""
    align_ph, align_gr, ok = _alignement_v2(
        ortho, phone, _PHONE_TO_GRAPHEMES(), word_boundaries
    )
    if not ok:
        return [], [], [], False
    spans = _build_spans(ortho, align_gr)
    dec_ph, dec_gr, dec_spans = _phonemise_alignment(align_ph, align_gr, spans)
    return dec_ph, dec_gr, dec_spans, True


# ══════════════════════════════════════════════════════════════════════════════
# Construction des syllabes riches
# ══════════════════════════════════════════════════════════════════════════════


def _decouper_syllabe(syll_phone: str) -> tuple[str, str, str] | None:
    """Décompose une syllabe en (attaque, noyau, coda)."""
    phonemes = iter_phonemes(syll_phone)
    vowel_indices = [i for i, p in enumerate(phonemes) if est_voyelle(p)]
    if not vowel_indices:
        return None
    v = vowel_indices[0]
    return "".join(phonemes[:v]), phonemes[v], "".join(phonemes[v + 1 :])


def _build_syllabes(
    syll_phones: list[str],
    dec_ph: list[str],
    dec_gr: list[str],
    dec_spans: list[Span],
    word_offset: int,
    alignment_ok: bool,
) -> list[Syllabe]:
    """Construit les objets Syllabe depuis la syllabation et l'alignement."""
    syllabes: list[Syllabe] = []
    cursor = 0
    prev_last_grapheme = ""  # Pour la gestion ² inter-syllabe

    for syll_phone in syll_phones:
        syll_phonemes = iter_phonemes(syll_phone)
        decomp = _decouper_syllabe(syll_phone)

        if not alignment_ok or not dec_ph:
            # Mode dégradé : pas d'alignement
            if decomp is not None:
                att_str, noy_str, cod_str = decomp
                attaque = GroupePhonologique(
                    phonemes=[Phoneme(ipa=p) for p in iter_phonemes(att_str)]
                    if att_str else []
                )
                noyau = GroupePhonologique(phonemes=[Phoneme(ipa=noy_str)])
                coda = GroupePhonologique(
                    phonemes=[Phoneme(ipa=p) for p in iter_phonemes(cod_str)]
                    if cod_str else []
                )
            else:
                attaque = GroupePhonologique(
                    phonemes=[Phoneme(ipa=p) for p in syll_phonemes]
                )
                noyau = GroupePhonologique()
                coda = GroupePhonologique()

            syllabes.append(Syllabe(
                phone=syll_phone, ortho="", span=(0, 0),
                attaque=attaque, noyau=noyau, coda=coda,
            ))
            continue

        # Mode normal : on a l'alignement
        mapped: list[tuple[str, str, Span]] = []
        pos = cursor
        for sph in syll_phonemes:
            if pos < len(dec_ph):
                mapped.append((sph, dec_gr[pos], dec_spans[pos]))
                pos += 1
            else:
                mapped.append((sph, "", (0, 0)))
        cursor = pos

        # Calculer ortho et span de la syllabe (avec gestion ² lettres doublées)
        syll_ortho_parts: list[str] = []
        rel_start: float = float("inf")
        rel_end = 0
        bridge_letter = ""       # Lettre pont inter-syllabe (y dans ay²)
        doubled_to_prev = False  # ² seul en début → annoter syllabe précédente

        for j, (_, gr, sp) in enumerate(mapped):
            if sp[0] < sp[1]:
                rel_start = min(rel_start, sp[0])
                rel_end = max(rel_end, sp[1])

            if "²" not in gr:
                # Graphème normal (garder ° pour lettres muettes)
                syll_ortho_parts.append(gr)
            elif j == 0:
                # ² en début de syllabe → concerne la syllabe précédente
                clean = gr.replace("²", "")
                if clean:
                    # "y²" → lettre pont (parenthèses dans syllabe précédente)
                    bridge_letter = clean.replace("°", "")
                    syll_ortho_parts.append(clean)
                else:
                    # "²" seul → ajouter ² à la syllabe précédente
                    doubled_to_prev = True
                    syll_ortho_parts.append("")
            else:
                # ² intra-syllabe
                clean = gr.replace("²", "")
                if clean:
                    # "y²" en milieu de syllabe → garder avec marqueur
                    syll_ortho_parts.append(gr)
                else:
                    # "²" seul → annoter l'entrée précédente
                    syll_ortho_parts.append("")
                    for k in range(j - 1, -1, -1):
                        if syll_ortho_parts[k].replace("°", "").replace("²", ""):
                            syll_ortho_parts[k] += "²"
                            break

        if mapped:
            prev_last_grapheme = mapped[-1][1]

        if rel_start == float("inf") or rel_start >= rel_end:
            rel_start = 0
            rel_end = 0

        abs_start = word_offset + rel_start
        abs_end = word_offset + rel_end
        syll_ortho = "".join(syll_ortho_parts)

        # Lettres doublées inter-syllabe : modifier la syllabe précédente
        if syllabes:
            if bridge_letter:
                prev_ortho = syllabes[-1].ortho
                # Si la syllabe précédente finit déjà par cette lettre
                # (même graphème mappé dans les deux syllabes), remplacer
                # au lieu de doubler : tuy → tu(y), pas tuy(y)
                if prev_ortho.endswith(bridge_letter):
                    syllabes[-1].ortho = (
                        prev_ortho[:-len(bridge_letter)]
                        + f"({bridge_letter})"
                    )
                else:
                    syllabes[-1].ortho += f"({bridge_letter})"
                # Étendre le span pour inclure la lettre pont
                bridge_span = mapped[0][2] if mapped else None
                if bridge_span and bridge_span[0] < bridge_span[1]:
                    bridge_abs_end = word_offset + bridge_span[1]
                    prev_start, prev_end = syllabes[-1].span
                    if bridge_abs_end > prev_end:
                        syllabes[-1].span = (prev_start, bridge_abs_end)
            elif doubled_to_prev:
                syllabes[-1].ortho += "²"

        if decomp is not None:
            att_str, noy_str, cod_str = decomp
            att_phonemes_list = iter_phonemes(att_str) if att_str else []
            cod_phonemes_list = iter_phonemes(cod_str) if cod_str else []

            idx = 0
            att_ph: list[Phoneme] = []
            for p in att_phonemes_list:
                gr = mapped[idx][1].replace("°", "").replace("²", "") if idx < len(mapped) else ""
                att_ph.append(Phoneme(ipa=p, grapheme=gr))
                idx += 1

            noy_ph: list[Phoneme] = []
            gr = mapped[idx][1].replace("°", "").replace("²", "") if idx < len(mapped) else ""
            noy_ph.append(Phoneme(ipa=noy_str, grapheme=gr))
            idx += 1

            cod_ph: list[Phoneme] = []
            for p in cod_phonemes_list:
                gr = mapped[idx][1].replace("°", "").replace("²", "") if idx < len(mapped) else ""
                cod_ph.append(Phoneme(ipa=p, grapheme=gr))
                idx += 1

            attaque = GroupePhonologique(phonemes=att_ph)
            noyau = GroupePhonologique(phonemes=noy_ph)
            coda = GroupePhonologique(phonemes=cod_ph)
        else:
            attaque = GroupePhonologique(
                phonemes=[
                    Phoneme(ipa=m[0], grapheme=m[1].replace("°", "").replace("²", ""))
                    for m in mapped
                ]
            )
            noyau = GroupePhonologique()
            coda = GroupePhonologique()

        syllabes.append(Syllabe(
            phone=syll_phone,
            ortho=syll_ortho,
            span=(int(abs_start), int(abs_end)),
            attaque=attaque,
            noyau=noyau,
            coda=coda,
        ))

    return syllabes


# ══════════════════════════════════════════════════════════════════════════════
# E1 — Groupes de lecture
# ══════════════════════════════════════════════════════════════════════════════

_LIAISON_CONSONNES = _get_liaison_consonnes


def _phone_starts_with_vowel(phone: str) -> bool:
    """Vrai si la chaîne IPA commence par une voyelle."""
    if not phone:
        return False
    phonemes = iter_phonemes(phone)
    if not phonemes:
        return False
    return est_voyelle(phonemes[0])


def _phone_ends_with_consonne(phone: str) -> bool:
    """Vrai si la chaîne IPA finit par une consonne."""
    if not phone:
        return False
    phonemes = iter_phonemes(phone)
    if not phonemes:
        return False
    return est_consonne(phonemes[-1])


def _phone_ends_with_schwa(phone: str) -> bool:
    """Vrai si la chaîne IPA finit par un schwa (ə)."""
    if not phone:
        return False
    phonemes = iter_phonemes(phone)
    return phonemes[-1] == "ə" if phonemes else False


# ── Schwas pédagogiques ──────────────────────────────────────────────────────

# Ortho finit par e/es (sauf é/è/ê/ë)
_RE_E_MUET = re.compile(r"(?<![éèêë])es?$", re.IGNORECASE)
# Verbe finit par -ent (3e pers. pluriel)
_RE_VERB_ENT = re.compile(r"ent$", re.IGNORECASE)


def ajouter_schwa_final(ortho: str, pos: str, phone: str) -> str:
    """Ajoute un ə pédagogique final si le mot a un e-muet non prononcé.

    Conditions :
    - ortho finit par e/es (sauf é/è/ê/ë) OU ortho finit par -ent avec POS=VER
    - ET l'IPA ne finit pas déjà par une voyelle
    """
    if not phone or not ortho:
        return phone

    has_e_muet = bool(_RE_E_MUET.search(ortho))
    has_verb_ent = (
        bool(_RE_VERB_ENT.search(ortho))
        and isinstance(pos, str)
        and pos.upper().startswith("VER")
    )

    if not (has_e_muet or has_verb_ent):
        return phone

    phonemes = iter_phonemes(phone)
    if phonemes and est_voyelle(phonemes[-1]):
        return phone

    return phone + "ə"


def construire_groupes(
    mots: list[MotAnalyse],
    options: OptionsGroupes | None = None,
) -> list[GroupeLecture]:
    """E1 — Construit les groupes de lecture depuis une liste de mots analysés.

    Parcourt les mots séquentiellement et les regroupe selon :
    - Élisions (l'enfant → 1 groupe)
    - Liaisons (les‿enfants → 1 groupe)
    - Enchaînements (avec‿elle → 1 groupe)
    """
    if options is None:
        options = OptionsGroupes()

    if not mots:
        return []

    groupes: list[GroupeLecture] = []
    current_mots: list[MotAnalyse] = [mots[0]]
    current_phones: list[str] = [mots[0].phone]
    current_jonctions: list[str] = []

    for i in range(1, len(mots)):
        mot_courant = mots[i]
        mot_precedent = mots[i - 1]

        # La ponctuation ou une formule interdit toute fusion entre les mots
        if mot_courant.ponctuation_avant or mot_courant.est_formule or mot_precedent.est_formule:
            pass  # tombe dans le « pas de fusion » ci-dessous
        else:
            # Élision : apostrophe entre les deux mots (m'appelle, l'enfant)
            if options.gerer_elisions and (
                mot_precedent.text.endswith("'") or mot_courant.elision_avant
            ):
                current_mots.append(mot_courant)
                current_phones.append(mot_courant.phone)
                current_jonctions.append("elision")
                continue

            # Liaison : mot précédent a un label de liaison et mot courant commence par voyelle
            if options.gerer_liaisons and mot_precedent.liaison != "none":
                if _phone_starts_with_vowel(mot_courant.phone):
                    liaison_consonne = _LIAISON_CONSONNES().get(mot_precedent.liaison, "")
                    if liaison_consonne:
                        current_mots.append(mot_courant)
                        current_phones.append(mot_courant.phone)
                        current_jonctions.append(f"liaison_{liaison_consonne}")
                        continue

            # Enchaînement : consonne finale de mot1 + voyelle initiale de mot2
            if options.gerer_enchainement:
                if (_phone_ends_with_consonne(mot_precedent.phone) and
                        _phone_starts_with_vowel(mot_courant.phone)):
                    current_mots.append(mot_courant)
                    current_phones.append(mot_courant.phone)
                    current_jonctions.append("enchainement")
                    continue

        # Pas de fusion → fermer le groupe courant et en ouvrir un nouveau
        phone_groupe = "".join(current_phones)
        span_start = current_mots[0].span[0] if current_mots else 0
        span_end = current_mots[-1].span[1] if current_mots else 0
        is_formule = any(m.est_formule for m in current_mots)
        groupes.append(GroupeLecture(
            mots=current_mots,
            phone_groupe=phone_groupe,
            span=(span_start, span_end),
            jonctions=current_jonctions,
            est_formule=is_formule,
        ))
        current_mots = [mot_courant]
        current_phones = [mot_courant.phone]
        current_jonctions = []

    # Fermer le dernier groupe
    if current_mots:
        phone_groupe = "".join(current_phones)
        span_start = current_mots[0].span[0] if current_mots else 0
        span_end = current_mots[-1].span[1] if current_mots else 0

        is_formule = any(m.est_formule for m in current_mots)
        groupes.append(GroupeLecture(
            mots=current_mots,
            phone_groupe=phone_groupe,
            span=(span_start, span_end),
            jonctions=current_jonctions,
            est_formule=is_formule,
        ))

    return groupes


# ══════════════════════════════════════════════════════════════════════════════
# Conversion G2P → Syllabeur
# ══════════════════════════════════════════════════════════════════════════════


def lecture_depuis_g2p(result: object) -> LectureFormule:
    """Convertit un LectureFormuleResult (G2P) en LectureFormule (Syllabeur).

    Permet de rester indépendant du module G2P : accepte tout objet
    avec attributs display_fr et events (chaque event ayant ortho,
    phone, span_source).
    """
    events_syl: list[EventFormule] = []
    for evt in getattr(result, "events", []):
        events_syl.append(EventFormule(
            ortho=getattr(evt, "ortho", ""),
            phone=getattr(evt, "phone", ""),
            span_source=getattr(evt, "span_source", (0, 0)),
            span_lecture=(0, 0),
        ))
    return LectureFormule(
        display_fr=getattr(result, "display_fr", ""),
        events=events_syl,
    )


def _valider_spans_formule(lecture: LectureFormule) -> None:
    """Valide la cohérence des spans dans une LectureFormule.

    Vérifie que les span_source ne se chevauchent pas de façon incohérente
    et que start <= end pour chaque event.
    """
    for i, evt in enumerate(lecture.events):
        s, e = evt.span_source
        if s > e:
            raise ValueError(
                f"EventFormule #{i} ({evt.ortho!r}) : span_source "
                f"incohérent ({s}, {e}) — start > end"
            )


# ══════════════════════════════════════════════════════════════════════════════
# E2 — Syllabation sur les groupes
# ══════════════════════════════════════════════════════════════════════════════


def _syllabes_depuis_lecture(
    lecture: LectureFormule,
) -> list[Syllabe]:
    """Construit les syllabes à partir d'une LectureFormule.

    Chaque EventFormule = 1 syllabe (mode progressif).
    Pour un mode block, l'appelant pré-fusionne les events par composant
    avant de les passer au Syllabeur.
    """
    syllabes: list[Syllabe] = []
    for evt in lecture.events:
        syllabes.append(Syllabe(
            phone=evt.phone,
            ortho=evt.ortho,
            span=evt.span_source,
            attaque=GroupePhonologique(),
            noyau=GroupePhonologique(phonemes=[Phoneme(ipa=evt.phone)]),
            coda=GroupePhonologique(),
        ))
    return syllabes


def syllabifier_groupes(
    groupes: list[GroupeLecture],
    lectures_formules: dict[int, LectureFormule] | None = None,
) -> list[ResultatGroupe]:
    """E2 — Syllabifie chaque groupe de lecture.

    Pour les groupes de mots : syllabation IPA + alignement.
    Pour les groupes FORMULE avec lecture : événements pré-calculés.
    Chaque EventFormule = 1 syllabe (mode progressif).
    """
    if lectures_formules is None:
        lectures_formules = {}

    resultats: list[ResultatGroupe] = []

    for gi, groupe in enumerate(groupes):
        if groupe.est_formule and groupe.lecture is not None:
            # FORMULE avec lecture pré-calculée
            _valider_spans_formule(groupe.lecture)
            syllabes = _syllabes_depuis_lecture(groupe.lecture)
            resultats.append(ResultatGroupe(groupe=groupe, syllabes=syllabes))
            continue

        if groupe.est_formule and gi in lectures_formules:
            # FORMULE avec lecture fournie par dict
            lecture = lectures_formules[gi]
            _valider_spans_formule(lecture)
            groupe.lecture = lecture
            syllabes = _syllabes_depuis_lecture(lecture)
            resultats.append(ResultatGroupe(groupe=groupe, syllabes=syllabes))
            continue

        # Mot(s) normal(aux) → syllabation standard
        phone = groupe.phone_groupe
        if not phone:
            resultats.append(ResultatGroupe(groupe=groupe, syllabes=[]))
            continue

        # Syllabation IPA
        syll_phones = _syllabify_ipa(phone)

        # Alignement : on utilise le texte combiné du groupe
        ortho = " ".join(m.text for m in groupe.mots)
        # Pour l'alignement, retirer les espaces si le groupe est uni
        if len(groupe.mots) == 1:
            ortho = groupe.mots[0].text
            word_offset = groupe.mots[0].span[0]
            dec_ph, dec_gr, dec_spans, ok = _aligner(ortho, phone)
        else:
            # Multi-mots : concaténer sans espaces pour l'alignement IPA
            # et calculer les frontières de mots pour guider les muettes
            parts = [m.text for m in groupe.mots]
            ortho = "".join(parts)
            word_offset = 0  # on remappe en absolu ci-dessous
            boundaries: list[int] = []
            pos = 0
            for p in parts[:-1]:
                pos += len(p)
                boundaries.append(pos)
            dec_ph, dec_gr, dec_spans, ok = _aligner(
                ortho, phone, word_boundaries=boundaries
            )
            # Remapper les spans concat → positions absolues dans le texte
            # (le concat "dansunarbre" ignore les espaces inter-mots)
            _c2a: list[int] = []
            for m in groupe.mots:
                for ci in range(len(m.text)):
                    _c2a.append(m.span[0] + ci)
            n = len(_c2a)
            dec_spans = [
                (_c2a[s] if s < n else (_c2a[-1] + 1 if n else s),
                 _c2a[e - 1] + 1 if 0 < e <= n else (_c2a[-1] + 1 if n else e))
                for s, e in dec_spans
            ]
        # Séparer les phonèmes composés (ex: 'aj' → 'a'+'j') pour
        # que chaque phonème syllabique ait son entrée d'alignement
        if ok:
            dec_ph, dec_gr, dec_spans = _phonemise_alignment(
                dec_ph, dec_gr, dec_spans
            )
        syllabes = _build_syllabes(syll_phones, dec_ph, dec_gr, dec_spans, word_offset, ok)
        resultats.append(ResultatGroupe(groupe=groupe, syllabes=syllabes))

    return resultats


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale
# ══════════════════════════════════════════════════════════════════════════════


class LecturaSyllabeur:
    """Analyseur syllabique complet du français.

    Combine phonémisation, syllabation par sonorité, alignement
    graphème-phonème, et groupes de lecture (E1+E2).

    Parameters
    ----------
    phonemizer : Phonemizer | None
        Objet avec méthode ``phonemize(word) -> str``.
        Si None, utilise eSpeak-NG par défaut.
        Si l'objet a une méthode ``predict`` (comme LecturaG2P),
        il sera automatiquement adapté.
    """

    def __init__(self, phonemizer: Phonemizer | object | None = None) -> None:
        if phonemizer is None:
            self._phonemizer: Phonemizer = EspeakPhonemizer()
        elif isinstance(phonemizer, Phonemizer):
            self._phonemizer = phonemizer
        elif hasattr(phonemizer, "predict"):
            self._phonemizer = _G2PAdapter(phonemizer)
        elif hasattr(phonemizer, "phonemize"):
            self._phonemizer = phonemizer  # type: ignore[assignment]
        else:
            raise TypeError(
                "Le phonemiseur doit avoir une methode .phonemize(word) ou .predict(word)"
            )

    @classmethod
    def with_espeak(cls, lang: str = "fr") -> LecturaSyllabeur:
        """Crée un syllabeur avec le backend eSpeak-NG."""
        return cls(phonemizer=EspeakPhonemizer(lang=lang))

    # ── API rétrocompatible (identique au syllabeur simple) ──

    def analyze(self, word: str, phone: str | None = None) -> ResultatAnalyse:
        """Analyse syllabique complète d'un mot.

        Parameters
        ----------
        word : str
            Mot français à analyser.
        phone : str | None
            Transcription IPA manuelle.

        Returns
        -------
        ResultatAnalyse
        """
        logger.debug("analyze() word=%r phone=%r", word, phone)
        if phone is None:
            phone = self._phonemizer.phonemize(word)

        syll_phones = _syllabify_ipa(phone)
        dec_ph, dec_gr, dec_spans, ok = _aligner(word, phone)
        syllabes = _build_syllabes(syll_phones, dec_ph, dec_gr, dec_spans, 0, ok)

        return ResultatAnalyse(mot=word, phone=phone, syllabes=syllabes)

    def analyze_text(self, text: str) -> list[ResultatAnalyse]:
        """Analyse syllabique de chaque mot d'un texte."""
        words = re.findall(r"[a-zA-ZÀ-ÿ\u0100-\u024F]+(?:['-][a-zA-ZÀ-ÿ]+)*", text)
        return [self.analyze(w) for w in words]

    def syllabify_ipa(self, phone: str) -> list[str]:
        """Découpage syllabique bas-niveau sur de l'IPA brut.

        >>> syl.syllabify_ipa("ʃɔkɔla")
        ['ʃɔ', 'kɔ', 'la']
        """
        return _syllabify_ipa(phone)

    # ── API complète avec groupes de lecture ──

    def analyser_complet(
        self,
        mots: list[MotAnalyse],
        lectures_formules: dict[int, LectureFormule] | None = None,
        options: OptionsGroupes | None = None,
    ) -> ResultatSyllabation:
        """Analyse complète E1 + E2 : groupes de lecture puis syllabation.

        Parameters
        ----------
        mots : list[MotAnalyse]
            Liste de mots avec annotations G2P (phone, liaison, etc.)
        lectures_formules : dict[int, LectureFormule] | None
            Lectures pré-calculées pour les formules (index groupe -> lecture)
        options : OptionsGroupes | None
            Options de regroupement (élisions, liaisons, enchaînements, schwas)

        Returns
        -------
        ResultatSyllabation
        """
        if options is None:
            options = OptionsGroupes()

        logger.debug("analyser_complet() called with %s mots", len(mots))

        # E1 : construire les groupes de lecture
        groupes = construire_groupes(mots, options)

        # Remap lectures_formules : mot index → group index
        # (construire_groupes regroupe les mots, décalant les indices)
        group_lectures: dict[int, LectureFormule] = {}
        if lectures_formules:
            mot_idx = 0
            for gi, groupe in enumerate(groupes):
                for _m in groupe.mots:
                    if mot_idx in lectures_formules:
                        group_lectures[gi] = lectures_formules[mot_idx]
                        groupe.est_formule = True
                    mot_idx += 1

        # E2 : syllabifier chaque groupe
        resultats_groupes = syllabifier_groupes(
            groupes, group_lectures,
        )

        # Reconstituer le texte original
        texte = " ".join(m.text for m in mots)

        result = ResultatSyllabation(
            texte_original=texte,
            groupes=resultats_groupes,
            options=options,
        )
        logger.info("analyser_complet() produced %s groupes, %s syllabes",
                     result.nb_groupes, result.nb_syllabes)
        return result

    def construire_groupes(
        self,
        mots: list[MotAnalyse],
        options: OptionsGroupes | None = None,
    ) -> list[GroupeLecture]:
        """E1 seul : construit les groupes de lecture.

        Parameters
        ----------
        mots : list[MotAnalyse]
            Liste de mots avec annotations G2P.
        options : OptionsGroupes | None
            Options de regroupement.

        Returns
        -------
        list[GroupeLecture]
        """
        return construire_groupes(mots, options)

    def syllabifier_groupes(
        self,
        groupes: list[GroupeLecture],
        lectures_formules: dict[int, LectureFormule] | None = None,
    ) -> list[ResultatGroupe]:
        """E2 seul : syllabifie des groupes de lecture.

        Parameters
        ----------
        groupes : list[GroupeLecture]
            Groupes de lecture (sortie de E1).
        lectures_formules : dict[int, LectureFormule] | None
            Lectures pré-calculées pour les formules.

        Returns
        -------
        list[ResultatGroupe]
        """
        return syllabifier_groupes(groupes, lectures_formules)


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    syl = LecturaSyllabeur()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        results = syl.analyze_text(text)
        for r in results:
            print(r.format_detail())
            print()
    else:
        print("Lectura Syllabeur Complet — Mode interactif (Ctrl+C pour quitter)")
        print()
        try:
            while True:
                word = input("Mot > ").strip()
                if not word:
                    continue
                if word.startswith("/"):
                    # Mode IPA direct
                    phone = word[1:].strip()
                    sylls = syl.syllabify_ipa(phone)
                    print(f"  /{phone}/ -> {'.'.join(sylls)} ({len(sylls)} syll.)")
                else:
                    result = syl.analyze(word)
                    print(result.format_detail())
                print()
        except (KeyboardInterrupt, EOFError):
            print()
