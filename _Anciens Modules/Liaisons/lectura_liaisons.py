"""Lectura Liaisons — Moteur de liaisons et jonctions pour le français.

Fichier unique, autonome, zéro dépendance externe.
Gère 4 types de jonctions entre mots français :
- Liaisons grammaticales (les‿enfants → /lezɑ̃fɑ̃/)
- Enchaînements phonétiques (avec‿elle → /avɛkɛl/)
- Élisions (l'enfant → /lɑ̃fɑ̃/)
- Mots composés (peut-être → /pøɛtʁ/)

Deux niveaux d'API :
1. Par paires : classify(w1, w2) + merge() pour analyser deux mots
2. Par tokens : apply_jonctions(tokens) pour traiter une phrase entière

Usage rapide :
    from lectura_liaisons import LecturaLiaisons, MotInfo

    lia = LecturaLiaisons()
    decision = lia.classify(
        MotInfo("les", "le", ["ART:def"]),
        MotInfo("enfants", "ɑ̃fɑ̃", ["NOM"]),
    )
    print(decision)
    # LiaisonDecision(kind='grammaticale', typ='obligatoire', latent='z')

    phone = lia.merge("le", "ɑ̃fɑ̃", decision)
    # → "lezɑ̃fɑ̃"

Pipeline complet :
    from lectura_liaisons import (
        LecturaLiaisons, TokenMot, TokenSep, JonctionOptions,
    )

    lia = LecturaLiaisons()
    tokens = [
        TokenMot("L'", "l", ["ART:def"], (0, 2)),
        TokenSep("'", "apostrophe", (1, 2)),
        TokenMot("enfant", "ɑ̃fɑ̃", ["NOM"], (2, 8)),
    ]
    groups = lia.apply_jonctions(tokens)

Pré-requis :
    - Python 3.10+
    - Aucune bibliothèque externe requise

Copyright (c) 2025 Lectura — Licence CC BY-SA 4.0.
Voir LICENCE.txt et ATTRIBUTION.md.
"""

from __future__ import annotations

import base64
import re
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path

__version__ = "1.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires IPA (sous-ensemble minimal)
# ══════════════════════════════════════════════════════════════════════════════

_VOYELLES: set[str] = {
    "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə",
}

_CONSONNES: set[str] = {
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
    "ʃ", "ʒ", "m", "n", "ɲ", "ŋ", "l", "ʁ",
}


def _iter_phonemes(ipa: str) -> list[str]:
    """Découpe une chaîne IPA en phonèmes (regroupe les combining marks)."""
    if not ipa:
        return []
    phonemes: list[str] = []
    current = ""
    for ch in ipa:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            current += ch
        else:
            if current:
                phonemes.append(current)
            current = ch
    if current:
        phonemes.append(current)
    return phonemes


def _est_voyelle(phoneme: str) -> bool:
    if not phoneme:
        return False
    if phoneme in _VOYELLES:
        return True
    return bool(phoneme[0] in _VOYELLES)


def _est_consonne(phoneme: str) -> bool:
    return bool(phoneme and phoneme in _CONSONNES)


def _premier_phoneme(ipa: str) -> str | None:
    phonemes = _iter_phonemes(ipa)
    return phonemes[0] if phonemes else None


def _dernier_phoneme(ipa: str) -> str | None:
    phonemes = _iter_phonemes(ipa)
    return phonemes[-1] if phonemes else None


# ══════════════════════════════════════════════════════════════════════════════
# Liste des mots à h aspiré (compressée zlib + base85)
# ══════════════════════════════════════════════════════════════════════════════

_H_ASPIRE_BLOB = (
    "c-m!IOJeId4&2Wv<^nmC93u`plIS7D%u9~bt9EbRA$+kRstTmivk0p2A;k{}(aQVsulMD$"
    ">{gt-zDwSWm~219gwpA^m-~{|RZ6&uU`d0>QwhgmgUHlMpbX!DR0Xf^AUTyldRm!dY~w+l"
    "4Vwi<gXd!qn{0Z?o`03-R}CV31S|BTi|aKL|A6p`-<EO}SYQ=##8M}NaJ{c%-UD}Q)lSN<"
    ";E3FfQi?)BHV&Ff_|Y1Q`_=^lTkKGQTE#VDsL6;eUF}!|5$v=E2}`)55teVohOR4wQC_q(L"
    "F?p+jfWx^A)LbwnIS-tUaJPc6k9(r_OYv-G0Jf}*__u<rkSP(B^zS*x8;?R%GIB;U0S)};"
    "bQRF(geO8+qP^1csDD3dZ2fsgi<*Rpc_0x_T^3q1JB~F)3skOT;=_mUKt0%?IE|x%84$6*T"
    "FF3pmyfx81SCg0;szkNM{n5>hrN2=|2tqAeZ)$O1m;?S0e4oqy1oXL~+E2Mti8pkVHg|Sayi"
    "a+mBQ&kao`P(7cYA)4}=N?+axTa0GX)WJl>nxrN{MwBIJHqH-2N^nyX$wYc|K*t-_BuhmXh"
    "XNSNYWqJ8XC4ud?whW?SN||(8ZICh<7oW;x-@!0SF`JZ;l65}2^E0y(t&;~0cPt_JWJN?{a"
    "AK3yLBU=P)eV>iQE%WQEDhW6@>x#WWE`K<Z;>?EC?`-JgV<zsP_pM=s|^w}9?Fb4Im&XP>C"
    "AsbX;k1maYI2AYHgRH6#@dUiP8i$u-arY#-OiUOWP=^AoU$W^(6vDo417lH@w0es!;2EJHN"
    "%GjSw%KCB5b&5T0~3zEZ?!>?!8>ktku@b-^Kt)uS>4(y)$k3}r-d^_(di_Vb;A+G8?1(K@-"
    "P9-fJ)D21h>@HJ5G+?jg{RU3RymB;IjV(YLkgWSTkPr~tmmdQpNWxi+)&bg4;q~QmmW2{i+)"
    "yH88OoMO;<aMxN8(s^XmNLjP+2}+$6m&Z^R2Gr{5(^6^M}qc!PjTI3Z)n&qc=I|?HB7fL%N"
    "$Z5p9)%oge6=i*hcWkt`^0wsrCU!7Zx$!#1t_}2dz`?M<>re)oLVWQypzlM%$>OEsAJE;PG;"
    "`m)AMkvtEbcEtgI1x4d7;)*hJ4T~YU?sGd$muHv=y4ZPn|18tBRjO4-t(hf$9)jJeuG1Pe6c"
    "#=AgjwJtYM)l{ED(2*_&qar{6B9s%Vmt44B)`B4Bx01IpV*$yBBNE@97){tg%kY6&3pcaFiK"
    "iJD5W$3)k3)lQXp3dv<3-_49KL8DM*Y!?~Uhgsy%%#tDWdDkEcY=lMUPOM(Y_rI?=;Dp$ChB"
    "L`FY<_h#mM*~@&t?%THFSC{R(|11QcYiDCzp-j4;sgHn)^68{ZS2zfMgsk7Kt?)%)Wlu};{S"
    "8)0Wf`g`N598&Fr^x0As`6H69Dz-i($?GyxzE61SybX1Fb>A5-t-Ay#y|Uq{&Wkqc1Dm5Ab"
    "i|wBUESNSMMWKrsSQF#>-C+8`0{E6@j(Ufv+_+b1oDXMDQkM%foMz53rrE?*t5IQxJ~^i@u$"
    "7eb1+LE@_b6*S%6U)J<yVEsmTm66ngS8i}xAu^TS?@j6Me=qMDh&MIBTjBkX_<EQrnRHMhFC"
    "!@vj!5KN;tK$<1hOCBfTUxQyv}cWTC8niolh$BDXM%@k;hX}ju(H=4hk2<kq!%V!}C%4eexZ"
    "%8~p6R@4B<5<vv-?;dSx!k>NPCKK{}6IJfM--Zlm1+xm5X|EbioHv61<59oy>UoIGX$1j)t?("
    "79zq~}D&J<(+nU_4nD#jIOm*7vZC-{5&%81uX>W_=fPyl0UjnBD4<p7|@@L0a_j4d8*x^QM`"
    "3&Fp2+z63Y`!94o<htFlb{=40QFGBoGC3@KExz6bYwKE?MTEG6AbN^Ye|4?@OhgyH=u5Gv*0-"
    "zd%)BV~CJKS#{{f}eWZ+_AMQgfQbCWCQLltw57xpdm{d<*1R-7~9u#?RHv_ku&KHi&kYO87lp"
    "@|td5o<!WmAMAi7Fbz@%8$o+8q=%C{Its+toM6*&Mb7*QmkUl)R+a8|7(Aow{YA|!LoSs1a>>"
    "|$&i?Q>i<5PE=@pH<*N1}T<HHUPhaYM?pqzN|&FzBkl|*+oc;w_)VPXk_DB7jq^9cYde^Yi$l"
    "vKR>0jPd1ayx;k^7sOOf!zD!mdSL^LOb!3-6lI*%b+T=p~VrD`Z!TL^HY-tc{Kr4AMuw@z(3="
    "}7aV@P;dh&cBhz?de7O&St0+B<b)&Iv_(tICP8Xhh_>j|XJvaSyiFo*}*^bUgY%q7?`AHY@PW"
    "lgw?ZiA}Ws(qKXuj$IO#?JYSWGbVLd3$g&HomNM9IJn$HsvgjcB0&^wZ^nILvzEd1#v;cCE7qr"
    "4ihVnWAX59oGT?R2&1QYvWhDrmuD_zS=c?wZp$0`2U7hupR{;qX2w(;fE-oblHt^i@11?I1N(~"
    "EP-i|850lC6%Gt5QsRq{GsUzLC<6nvc4fug%`sN5Am0HNs8B9}mj;(u(o)jva%LV<KfYY@TLIC"
    "6z+Z6#A-aLN#)8z8aAQCi4hN&5U?9&wd;LQ_3l>J7vItD8Q9|)k<1z{G3^2KqblHtEv?5pn({"
    "x2sJ;+zOLwXa?%7u6GX@Ohxkv`Qnf>f8l7!u3k^dT@IBH1lF!UsuyJ#gtJZ}affb(sY0@xhdJ"
    "3>__ymJZ*5F@ndc-uUExhigY54ZI|a#8XPVtwpER2H8%raZaS29A&Z}z1xWPTn#piTuRb0m!f-"
    "hFgwxNQWI6T(PtJsfA2P#8~;8E{t+l5lMG$`A7nALiU"
)


def _load_h_aspire_embedded() -> set[str]:
    """Décompresse la liste h aspiré embarquée."""
    raw = zlib.decompress(base64.b85decode(_H_ASPIRE_BLOB))
    return {w for line in raw.decode("utf-8").splitlines() if (w := line.strip().lower())}


def _load_h_aspire_file(path: str | Path) -> set[str]:
    """Charge la liste h aspiré depuis un fichier texte (un mot par ligne)."""
    words: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.add(w)
    return words


# ══════════════════════════════════════════════════════════════════════════════
# Modèles de données
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MotInfo:
    """Information sur un mot pour la classification de liaison.

    Attributs :
        ortho : Forme orthographique (ex. "les", "enfants")
        phone : Transcription IPA (ex. "le", "ɑ̃fɑ̃")
        pos   : Liste de POS tags possibles (ex. ["ART:def"], ["NOM"])
    """
    ortho: str
    phone: str
    pos: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LiaisonDecision:
    """Résultat de la classification d'une liaison.

    Attributs :
        kind             : "grammaticale" | "enchainement" | "none"
        typ              : "obligatoire" | "facultative" | "interdite" | "none"
        latent_phoneme   : Phonème latent réalisé (ex. "z", "t", "n")
        latent_ortho     : Lettre orthographique source (ex. "s", "t", "n")
        phone_patch      : Modification du phone du mot 1 (old, new) — ex. dénasalisation
        realized_phoneme : Phonème réalisé pour l'enchaînement
    """
    kind: str = "none"
    typ: str = "none"
    latent_phoneme: str | None = None
    latent_ortho: str | None = None
    phone_patch: tuple[str, str] | None = None
    realized_phoneme: str | None = None


@dataclass(frozen=True)
class JonctionDecision:
    """Résultat de la classification de jonction entre deux mots/groupes.

    Attributs :
        kind     : "liaison_gram" | "enchainement" | "elision" | "compose" | "none"
        liaison  : Détail de la liaison (si applicable)
    """
    kind: str = "none"
    liaison: LiaisonDecision | None = None


# ── Types tokens (pour apply_jonctions) ──


@dataclass
class TokenMot:
    """Un mot avec ses informations linguistiques.

    Attributs :
        ortho : Forme orthographique (ex. "les")
        phone : Transcription IPA (ex. "le")
        pos   : Liste de POS tags possibles (ex. ["ART:def"])
        span  : Position (début, fin) dans le texte source
    """
    ortho: str
    phone: str = ""
    pos: list[str] = field(default_factory=list)
    span: tuple[int, int] = (0, 0)


@dataclass
class TokenSep:
    """Un séparateur (espace, apostrophe, trait d'union).

    Attributs :
        text     : Caractère séparateur (" ", "'", "-")
        sep_type : "space" | "apostrophe" | "hyphen"
        span     : Position (début, fin) dans le texte source
    """
    text: str
    sep_type: str
    span: tuple[int, int] = (0, 0)


@dataclass
class TokenPonct:
    """Ponctuation (ignorée par les jonctions).

    Attributs :
        text : Texte de la ponctuation
        span : Position (début, fin) dans le texte source
    """
    text: str
    span: tuple[int, int] = (0, 0)


@dataclass
class GroupeJonction:
    """Résultat d'une jonction : groupe de tokens fusionnés.

    Attributs :
        components     : Tokens composant le groupe
        phone          : Phonétique IPA combinée
        span           : Position (début, fin) dans le texte source
        jonction_type  : "compose" | "elision" | "liaison_gram" | "enchainement" | ""
    """
    components: list = field(default_factory=list)
    phone: str = ""
    span: tuple[int, int] = (0, 0)
    jonction_type: str = ""


@dataclass
class JonctionOptions:
    """Options pour apply_jonctions.

    Attributs :
        elisions       : Fusionner les élisions (l'enfant)
        mots_composes  : Fusionner les composés (peut-être)
        liaisons_gram  : Appliquer les liaisons grammaticales
        enchainements  : Appliquer les enchaînements phonétiques
    """
    elisions: bool = True
    mots_composes: bool = True
    liaisons_gram: bool = True
    enchainements: bool = False


# Type union pour les tokens
TokenItem = TokenMot | TokenSep | TokenPonct | GroupeJonction


# ══════════════════════════════════════════════════════════════════════════════
# Constantes lexicales
# ══════════════════════════════════════════════════════════════════════════════

_LEX_STOP_MOT1 = {"et"}
_LEX_STOP_MOT2 = {"onze"}

_LEX_LATENT_N = {"on", "un", "en", "bon", "mon", "ton", "son", "aucun", "certain"}
_LEX_N_DENASALIZE: dict[str, tuple[str, str]] = {"bon": ("ɔ̃", "ɔ")}
_LEX_LATENT_R = {"premier", "dernier"}

# Mots dont le phone finit par consonne mais qui ont un latent /z/
_LEX_LATENT_CONSONANT: dict[str, str] = {"ils": "z", "elles": "z", "quelques": "z"}

# Mots à liaison systématique — court-circuite les règles POS
# Valeurs = phonème latent (pas la lettre orthographique)
_LEX_ALWAYS_LIAISON: dict[str, str] = {
    # Déterminants
    "les": "z", "des": "z", "mes": "z", "tes": "z", "ses": "z",
    "ces": "z", "aux": "z", "nos": "z", "vos": "z",
    "un": "n", "mon": "n", "ton": "n", "son": "n",
    # Pronoms clitiques
    "nous": "z", "vous": "z", "ils": "z", "elles": "z",
    "on": "n", "en": "n",
    # Prépositions monosyllabiques
    "dans": "z", "sans": "z", "chez": "z", "sous": "z",
    # Adverbes courts catégoriques
    "très": "z", "plus": "z", "moins": "z", "bien": "n",
    "tout": "t", "trop": "p",
    # Conjonction catégorique
    "quand": "t",
    # Auxiliaires / verbes fréquents (0% de non-liaison PFC)
    "est": "t", "suis": "z",
    "ont": "t", "sont": "t", "font": "t", "vont": "t",
}

_CLEAN_RE = re.compile(r"[^\w''\-]", flags=re.UNICODE)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers internes
# ══════════════════════════════════════════════════════════════════════════════


def _clean(w: str) -> str:
    return _CLEAN_RE.sub("", w.strip()).lower()


def _last_letter(w: str) -> str | None:
    w = _clean(w)
    return w[-1] if w else None


def _starts_with_vowel_or_h_muet(word: str, h_aspire: set[str]) -> bool:
    w = _clean(word)
    if not w:
        return False
    if w in h_aspire:
        return False
    return w[0] in "aeiouyàâäéèêëîïôöùûüœæh"


def _latent_segment(
    ortho: str, phone: str
) -> tuple[str | None, str | None, tuple[str, str] | None]:
    """Détermine un segment latent grammatical."""
    last_ortho = _last_letter(ortho)
    if not last_ortho:
        return None, None, None

    last_phon = _dernier_phoneme(phone)
    if not last_phon or not _est_voyelle(last_phon):
        # Fallback : mots avec latent malgré phone consonantique
        m1 = _clean(ortho)
        if m1 in _LEX_LATENT_CONSONANT:
            return _LEX_LATENT_CONSONANT[m1], "s", None
        return None, None, None

    m1 = _clean(ortho)

    # n morphologique
    if m1 in _LEX_LATENT_N:
        patch = _LEX_N_DENASALIZE.get(m1)
        return "n", "n", patch

    # r exceptionnel
    if m1 in _LEX_LATENT_R:
        return "r", "r", None

    # productif classique
    if last_ortho in {"s", "x", "z"}:
        return "z", "s", None

    if last_ortho in {"t", "d"}:
        return "t", "t", None

    return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# Classification liaison grammaticale
# ══════════════════════════════════════════════════════════════════════════════


def _classify_liaison_gram(
    w1: MotInfo, w2: MotInfo, h_aspire: set[str]
) -> LiaisonDecision:
    """Décide si une liaison grammaticale est possible entre w1 et w2."""
    m1c = _clean(w1.ortho)
    m2c = _clean(w2.ortho)

    # Condition phonologique côté mot2
    if not _starts_with_vowel_or_h_muet(m2c, h_aspire):
        return LiaisonDecision()

    # Blocages lexicaux
    if m1c in _LEX_STOP_MOT1:
        return LiaisonDecision(kind="grammaticale", typ="interdite")
    if m2c in _LEX_STOP_MOT2:
        return LiaisonDecision(kind="grammaticale", typ="interdite")

    # Court-circuit : mots à liaison systématique
    if m1c in _LEX_ALWAYS_LIAISON:
        latent_phon = _LEX_ALWAYS_LIAISON[m1c]
        latent_ortho = _last_letter(m1c) or ""
        m1_patch = _LEX_N_DENASALIZE.get(m1c)
        return LiaisonDecision(
            kind="grammaticale", typ="obligatoire",
            latent_phoneme=latent_phon, latent_ortho=latent_ortho,
            phone_patch=m1_patch,
        )

    c1 = w1.pos
    c2 = w2.pos

    # Blocage NUM en mot2
    if "NUM" in c2:
        if not any(x in {"ART:def", "ART:ind", "NUM"} for x in c1):
            return LiaisonDecision()

    # Détection latente
    latent_phon, latent_ortho, m1_patch = _latent_segment(m1c, w1.phone)
    if not latent_phon:
        return LiaisonDecision()

    # OBLIGATOIRE
    obligatoire_rules = [
        # ART + NOM/ADJ
        (lambda: any(x in {"ART:def", "ART:ind"} for x in c1)
         and any(y in {"NOM", "ADJ"} for y in c2)),
        # PRO:pos + NOM/ADJ
        (lambda: "PRO:pos" in c1 and any(y in {"NOM", "ADJ"} for y in c2)),
        # PRO:per + VER/AUX
        (lambda: "PRO:per" in c1 and any(y in {"VER", "AUX"} for y in c2)),
        # VER/AUX + ADJ
        (lambda: any(x in {"VER", "AUX"} for x in c1) and "ADJ" in c2),
        # "est" + ADJ/VER/AUX/ADV/PRO:per
        (lambda: m1c == "est"
         and any(y in {"ADJ", "VER", "AUX", "ADV", "PRO:per"} for y in c2)),
        # "quand" + voyelle
        (lambda: m1c == "quand"),
        # NUM + NOM/ADJ
        (lambda: "NUM" in c1 and any(y in {"NOM", "ADJ"} for y in c2)),
        # ADV monosyllabique + ADJ/VER (1 seul noyau vocalique)
        (lambda: "ADV" in c1 and any(y in {"ADJ", "VER"} for y in c2)
         and sum(1 for p in _iter_phonemes(w1.phone) if _est_voyelle(p)) <= 1),
        # ADJ + NOM
        (lambda: "ADJ" in c1 and "NOM" in c2),
        # PRE
        (lambda: "PRE" in c1),
    ]

    for rule in obligatoire_rules:
        if rule():
            return LiaisonDecision(
                kind="grammaticale", typ="obligatoire",
                latent_phoneme=latent_phon, latent_ortho=latent_ortho,
                phone_patch=m1_patch,
            )

    # FACULTATIVE
    if any(x in {"NOM", "ADJ"} for x in c1) and any(
        y in {"NOM", "ADJ"} for y in c2
    ):
        return LiaisonDecision(
            kind="grammaticale", typ="facultative",
            latent_phoneme=latent_phon, latent_ortho=latent_ortho,
            phone_patch=m1_patch,
        )

    return LiaisonDecision()


# ══════════════════════════════════════════════════════════════════════════════
# Classification enchaînement
# ══════════════════════════════════════════════════════════════════════════════


def _classify_enchainement(
    w1: MotInfo, w2: MotInfo, h_aspire: set[str],
    liaison_gram_active: bool = False,
) -> LiaisonDecision:
    """Décide si un enchaînement phonétique est possible."""
    if liaison_gram_active:
        return LiaisonDecision()

    if _clean(w2.ortho) in _LEX_STOP_MOT2:
        return LiaisonDecision()

    if "ADJ" in w1.pos and "NUM" in w2.pos:
        return LiaisonDecision()

    if _clean(w2.ortho) in h_aspire:
        return LiaisonDecision()

    last_p1 = _dernier_phoneme(w1.phone)
    first_p2 = _premier_phoneme(w2.phone)

    if not last_p1 or not first_p2:
        return LiaisonDecision()

    # Cas spécial : neuf → v
    if _clean(w1.ortho) == "neuf" and _est_voyelle(first_p2):
        return LiaisonDecision(
            kind="enchainement", typ="enchainement",
            realized_phoneme="v",
        )

    if not _est_consonne(last_p1):
        return LiaisonDecision()

    if not _est_voyelle(first_p2):
        return LiaisonDecision()

    return LiaisonDecision(
        kind="enchainement", typ="enchainement",
        realized_phoneme=last_p1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fusion phonétique
# ══════════════════════════════════════════════════════════════════════════════


def _apply_suffix_patch(phone: str, patch: tuple[str, str] | None) -> str:
    if not patch:
        return phone
    old, new = patch
    if old and phone.endswith(old):
        return phone[: -len(old)] + new
    return phone


def _replace_last_phoneme(phone: str, new_phon: str) -> str:
    last = _dernier_phoneme(phone)
    if not last:
        return phone
    return phone[: -len(last)] + new_phon


def merge_phones(
    phone1: str, phone2: str, decision: LiaisonDecision
) -> str:
    """Calcule le phone combiné de deux mots selon la décision de liaison.

    Args:
        phone1: Phonétique IPA du mot 1
        phone2: Phonétique IPA du mot 2
        decision: Résultat de classify()

    Returns:
        Phonétique IPA combinée
    """
    if decision.kind == "grammaticale" and decision.latent_phoneme:
        p1 = _apply_suffix_patch(phone1, decision.phone_patch)
        return p1 + decision.latent_phoneme + phone2

    if decision.kind == "enchainement" and decision.realized_phoneme:
        p1 = _replace_last_phoneme(phone1, decision.realized_phoneme)
        return p1 + phone2

    return phone1 + phone2


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline jonctions (composés, élisions, liaisons)
# ══════════════════════════════════════════════════════════════════════════════


def _motinfo_from_token(mot: TokenMot) -> MotInfo:
    """Construit un MotInfo depuis un TokenMot."""
    return MotInfo(ortho=mot.ortho, phone=mot.phone, pos=list(mot.pos))


def _last_mot_in_group(group: GroupeJonction) -> TokenMot | None:
    """Retourne le dernier TokenMot dans les components d'un groupe."""
    for comp in reversed(group.components):
        if isinstance(comp, TokenMot):
            return comp
        if isinstance(comp, GroupeJonction):
            sub = _last_mot_in_group(comp)
            if sub is not None:
                return sub
    return None


def _first_mot_in_group(group: GroupeJonction) -> TokenMot | None:
    """Retourne le premier TokenMot dans les components d'un groupe."""
    for comp in group.components:
        if isinstance(comp, TokenMot):
            return comp
        if isinstance(comp, GroupeJonction):
            sub = _first_mot_in_group(comp)
            if sub is not None:
                return sub
    return None


def _apply_composes(tokens: list[TokenItem]) -> list[TokenItem]:
    """Fusionne Mot + trait-d'union + Mot → GroupeJonction(compose)."""
    out: list[TokenItem] = []
    i = 0

    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and isinstance(tokens[i], TokenMot)
            and isinstance(tokens[i + 1], TokenSep)
            and tokens[i + 1].sep_type == "hyphen"
            and isinstance(tokens[i + 2], TokenMot)
        ):
            w1 = tokens[i]
            sep = tokens[i + 1]
            w2 = tokens[i + 2]

            group = GroupeJonction(
                components=[w1, sep, w2],
                phone=(w1.phone or "") + (w2.phone or ""),
                span=(w1.span[0], w2.span[1]),
                jonction_type="compose",
            )
            out.append(group)
            i += 3
        else:
            out.append(tokens[i])
            i += 1

    return out


def _apply_elisions(tokens: list[TokenItem]) -> list[TokenItem]:
    """Fusionne Mot + apostrophe + Mot → GroupeJonction(elision)."""
    out: list[TokenItem] = []
    i = 0

    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and isinstance(tokens[i], TokenMot)
            and isinstance(tokens[i + 1], TokenSep)
            and tokens[i + 1].sep_type == "apostrophe"
            and isinstance(tokens[i + 2], TokenMot)
        ):
            w1 = tokens[i]
            sep = tokens[i + 1]
            w2 = tokens[i + 2]

            group = GroupeJonction(
                components=[w1, sep, w2],
                phone=(w1.phone or "") + (w2.phone or ""),
                span=(w1.span[0], w2.span[1]),
                jonction_type="elision",
            )
            out.append(group)
            i += 3
        else:
            out.append(tokens[i])
            i += 1

    return out


def _apply_liaisons_pass(
    tokens: list[TokenItem],
    h_aspire: set[str],
    use_gram: bool,
    use_ench: bool,
) -> tuple[list[TokenItem], bool]:
    """Un passage de fusion par liaisons. Retourne (tokens, changed)."""
    out: list[TokenItem] = []
    merged_any = False
    i = 0

    while i < len(tokens):
        if i + 2 < len(tokens):
            t1 = tokens[i]
            t_sep = tokens[i + 1]
            t2 = tokens[i + 2]

            is_unit1 = isinstance(t1, (GroupeJonction, TokenMot))
            is_space = isinstance(t_sep, TokenSep) and t_sep.sep_type == "space"
            is_unit2 = isinstance(t2, (GroupeJonction, TokenMot))

            if is_unit1 and is_space and is_unit2:
                # Extraire les MotInfo
                if isinstance(t1, GroupeJonction):
                    last_mot = _last_mot_in_group(t1)
                    phone1 = t1.phone or ""
                else:
                    last_mot = t1
                    phone1 = t1.phone or ""

                if isinstance(t2, GroupeJonction):
                    first_mot = _first_mot_in_group(t2)
                    phone2 = t2.phone or ""
                else:
                    first_mot = t2
                    phone2 = t2.phone or ""

                if last_mot is not None and first_mot is not None:
                    w1_info = _motinfo_from_token(last_mot)
                    w2_info = _motinfo_from_token(first_mot)
                    decision = _classify_liaison_gram(w1_info, w2_info, h_aspire)

                    should_merge = False
                    jonction_type = ""

                    if use_gram and decision.typ in {"obligatoire", "facultative"}:
                        should_merge = True
                        jonction_type = "liaison_gram"

                    if not should_merge and use_ench:
                        ench = _classify_enchainement(
                            w1_info, w2_info, h_aspire,
                            liaison_gram_active=(decision.typ in {"obligatoire", "facultative"}),
                        )
                        if ench.kind == "enchainement":
                            decision = ench
                            should_merge = True
                            jonction_type = "enchainement"

                    if should_merge:
                        merged_phone = merge_phones(phone1, phone2, decision)

                        # Construire les components fusionnés
                        components: list = []
                        if isinstance(t1, GroupeJonction):
                            components.extend(t1.components)
                        else:
                            components.append(t1)
                        components.append(t_sep)
                        if isinstance(t2, GroupeJonction):
                            components.extend(t2.components)
                        else:
                            components.append(t2)

                        new_group = GroupeJonction(
                            components=components,
                            phone=merged_phone,
                            span=(
                                t1.span[0] if isinstance(t1, GroupeJonction) else t1.span[0],
                                t2.span[1] if isinstance(t2, GroupeJonction) else t2.span[1],
                            ),
                            jonction_type=jonction_type,
                        )
                        out.append(new_group)
                        merged_any = True
                        i += 3
                        continue

        out.append(tokens[i])
        i += 1

    return out, merged_any


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale
# ══════════════════════════════════════════════════════════════════════════════


class LecturaLiaisons:
    """Moteur de liaisons et jonctions pour le français.

    Classifie les liaisons grammaticales (obligatoires/facultatives/interdites),
    les enchaînements phonétiques, et calcule la phonétique résultante.

    Parameters
    ----------
    h_aspire_path : str | Path | None
        Chemin vers un fichier h_aspire.txt custom. Si None, utilise
        la liste embarquée (863 mots).
    """

    def __init__(self, h_aspire_path: str | Path | None = None) -> None:
        if h_aspire_path is not None:
            self._h_aspire = _load_h_aspire_file(h_aspire_path)
        else:
            self._h_aspire = _load_h_aspire_embedded()

    @property
    def h_aspire(self) -> set[str]:
        """Retourne le set des mots à h aspiré."""
        return self._h_aspire

    def is_h_aspire(self, word: str) -> bool:
        """Vrai si le mot a un h aspiré."""
        return _clean(word) in self._h_aspire

    def classify(self, w1: MotInfo, w2: MotInfo) -> LiaisonDecision:
        """Classifie la liaison entre deux mots adjacents.

        Teste d'abord la liaison grammaticale (basée sur les POS tags),
        puis l'enchaînement phonétique.

        Args:
            w1: Mot de gauche (avec ortho, phone, POS)
            w2: Mot de droite (avec ortho, phone, POS)

        Returns:
            LiaisonDecision avec kind, typ, phonème latent/réalisé, etc.
        """
        # 1) Liaison grammaticale
        gram = _classify_liaison_gram(w1, w2, self._h_aspire)
        if gram.typ in {"obligatoire", "facultative"}:
            return gram

        # 2) Enchaînement phonétique
        ench = _classify_enchainement(w1, w2, self._h_aspire)
        if ench.kind == "enchainement":
            return ench

        # 3) Interdit explicitement ?
        if gram.typ == "interdite":
            return gram

        return LiaisonDecision()

    def merge(
        self, phone1: str, phone2: str, decision: LiaisonDecision
    ) -> str:
        """Calcule la phonétique combinée de deux mots.

        Args:
            phone1: IPA du mot 1
            phone2: IPA du mot 2
            decision: Résultat de classify()

        Returns:
            IPA combiné (ex. "lezɑ̃fɑ̃" pour "les enfants")
        """
        return merge_phones(phone1, phone2, decision)

    def analyze_pair(
        self, w1: MotInfo, w2: MotInfo
    ) -> tuple[LiaisonDecision, str]:
        """Classifie et fusionne en une seule opération.

        Returns:
            (decision, merged_phone)
        """
        decision = self.classify(w1, w2)
        merged = self.merge(w1.phone, w2.phone, decision)
        return decision, merged

    def format_decision(self, w1: MotInfo, w2: MotInfo) -> str:
        """Retourne un affichage lisible de la décision de liaison."""
        decision, merged = self.analyze_pair(w1, w2)

        parts = [f"{w1.ortho} + {w2.ortho}"]

        if decision.kind == "grammaticale":
            parts.append(f"  → liaison {decision.typ}")
            if decision.latent_phoneme:
                parts.append(f"    latent: /{decision.latent_phoneme}/ (← «{decision.latent_ortho}»)")
            if decision.phone_patch:
                parts.append(f"    patch: {decision.phone_patch[0]} → {decision.phone_patch[1]}")
        elif decision.kind == "enchainement":
            parts.append(f"  → enchaînement")
            if decision.realized_phoneme:
                parts.append(f"    réalisé: /{decision.realized_phoneme}/")
        else:
            parts.append(f"  → pas de liaison")

        parts.append(f"  phone: /{w1.phone}/ + /{w2.phone}/ → /{merged}/")

        return "\n".join(parts)

    def apply_jonctions(
        self,
        tokens: list[TokenItem],
        options: JonctionOptions | None = None,
    ) -> list[GroupeJonction]:
        """Applique les jonctions sur une liste de tokens.

        Traite dans l'ordre : composés, élisions, liaisons.
        Les tokens non-prononcés (ponctuation, séparateurs restants)
        sont ignorés en sortie.

        Args:
            tokens: Liste de TokenMot, TokenSep, TokenPonct
            options: Options (activer/désactiver chaque type de jonction)

        Returns:
            Liste de GroupeJonction (chaque mot ou groupe fusionné)
        """
        if options is None:
            options = JonctionOptions()

        working: list[TokenItem] = list(tokens)

        # 1. Composés
        if options.mots_composes:
            working = _apply_composes(working)

        # 2. Élisions
        if options.elisions:
            working = _apply_elisions(working)

        # 3. Liaisons (itérer jusqu'à stabilisation)
        if options.liaisons_gram or options.enchainements:
            changed = True
            while changed:
                working, changed = _apply_liaisons_pass(
                    working, self._h_aspire,
                    options.liaisons_gram, options.enchainements,
                )

        # 4. Emballer les tokens restants dans des GroupeJonction
        groups: list[GroupeJonction] = []
        for token in working:
            if isinstance(token, GroupeJonction):
                groups.append(token)
            elif isinstance(token, TokenMot):
                groups.append(GroupeJonction(
                    components=[token],
                    phone=token.phone or "",
                    span=token.span,
                ))
            # TokenPonct et TokenSep ignorés

        return groups


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    lia = LecturaLiaisons()

    # ── 1. Classification par paires ──
    print("Lectura Liaisons — Exemples de classification")
    print()

    exemples = [
        (MotInfo("les", "le", ["ART:def"]), MotInfo("enfants", "ɑ̃fɑ̃", ["NOM"])),
        (MotInfo("un", "œ̃", ["ART:ind"]), MotInfo("ami", "ami", ["NOM"])),
        (MotInfo("petit", "pəti", ["ADJ"]), MotInfo("oiseau", "wazo", ["NOM"])),
        (MotInfo("ils", "il", ["PRO:per"]), MotInfo("ont", "ɔ̃", ["AUX"])),
        (MotInfo("est", "ɛ", ["AUX"]), MotInfo("arrivé", "aʁive", ["VER"])),
        (MotInfo("et", "e", ["CON"]), MotInfo("alors", "alɔʁ", ["ADV"])),
        (MotInfo("les", "le", ["ART:def"]), MotInfo("héros", "eʁo", ["NOM"])),
        (MotInfo("neuf", "nœf", ["NUM"]), MotInfo("heures", "œʁ", ["NOM"])),
    ]

    for w1, w2 in exemples:
        print(lia.format_decision(w1, w2))
        print()

    # ── 2. Pipeline apply_jonctions ──
    print("=" * 60)
    print("Pipeline apply_jonctions")
    print("Phrase : L'enfant est peut-être arrivé")
    print()

    tokens: list[TokenItem] = [
        TokenMot("L'", "l", ["ART:def"], (0, 2)),
        TokenSep("'", "apostrophe", (1, 2)),
        TokenMot("enfant", "ɑ̃fɑ̃", ["NOM"], (2, 8)),
        TokenSep(" ", "space", (8, 9)),
        TokenMot("est", "ɛ", ["AUX"], (9, 12)),
        TokenSep(" ", "space", (12, 13)),
        TokenMot("peut", "pø", ["VER"], (13, 17)),
        TokenSep("-", "hyphen", (17, 18)),
        TokenMot("être", "ɛtʁ", ["VER"], (18, 22)),
        TokenSep(" ", "space", (22, 23)),
        TokenMot("arrivé", "aʁive", ["VER"], (23, 29)),
    ]

    groups = lia.apply_jonctions(tokens)

    for g in groups:
        orthos = []
        for c in g.components:
            if isinstance(c, TokenMot):
                orthos.append(c.ortho)
            elif isinstance(c, TokenSep):
                orthos.append(c.text)
        label = "".join(orthos)
        print(f"  {label:20s}  /{g.phone}/  ({g.jonction_type or 'simple'})")
