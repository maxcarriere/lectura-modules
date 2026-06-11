"""Correcteur V6 — Pipeline dual G2P/P2G zero-FP.

Architecture en 3 etapes :
  1. Preprocessing orthographique conservateur (OOV -> candidat proche)
  2. Analyse duale G2P + P2G (enrichissement sans correction)
  3. Corrections ciblees (homophones, accords, participe passe)

Contrainte primaire : zero faux positif sur texte propre.
Haute precision sur corrections faciles (typos, accents).
Corrections grammaticales via coherence morphologique G2P/P2G.

Ce que V6 ne fait PAS (vs V5) :
- Pas de regles heuristiques de grammaire (accord sujet-verbe par pattern)
- Pas de detection d'homophones par dictionnaire statique
- Pas d'insertion de negation
- Pas de correction de casse (noms propres)
- Pas de resegmentation (mots colles/coupes)
- Pas de correction agressive d'orthographe (distance > 1)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from lectura_correcteur._config import CorrecteurV6Config
from lectura_correcteur._types import (
    Correction,
    MotAnalyse,
    ResultatCorrection,
    TypeCorrection,
)
from lectura_correcteur._utils import (
    PUNCT_RE,
    LexiqueNormalise,
    normaliser_morpho,
    reconstruire_phrase,
    transferer_casse,
)

logger = logging.getLogger(__name__)

# Regex tokenisation francaise (identique V2/V3/V4)
_TOKEN_RE = re.compile(
    r"(?:[dlnmtscjqDLNMTSCJQ]|[Qq]u|[Ll]orsqu|[Pp]uisqu|[Jj]usqu|[Qq]uelqu)"
    r"['\u2019]"
    r"|[\w]+(?:-[\w]+)*"
    r"|[^\s\w]+",
)

# Mots-outils a ne jamais corriger en orthographe
_MOTS_PROTEGES = frozenset({
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "le", "la", "les", "un", "une", "des", "du", "au", "aux",
    "de", "a", "\u00e0", "en", "y", "ne", "pas", "plus", "que", "qui",
    "dont", "ou", "\u00f2", "et", "si", "ni", "me", "te", "se",
    "ce", "ces", "ses", "sa", "son", "mon", "ma", "ton", "ta",
    "leur", "leurs", "notre", "votre", "nos", "vos",
    "est", "es", "ai", "as", "ont", "sont", "suis",
    "dans", "sur", "sous", "par", "pour", "avec", "sans", "chez",
    "mais", "car", "donc", "or", "puis",
    "l\u00e0", "o\u00f9", "d\u00e8s", "vers",
    "an", "ans", "net", "marc",
    "cent", "cents", "vent", "vents", "sais", "sait",
    "rang", "rangs", "suspens",
    "crawler", "crawlers",
    "acre",
    "fils", "mont", "ceux", "celle", "celles",
    # Mots modernes / anglicismes courants a proteger
    "data", "datas",
})

# Homographes accentuels : mots existants sans accent qui ont un sens
# DIFFERENT de leur variante accentuee. Ne pas corriger via accent_lexique.
# Ex: foret (outil) vs foret (bois), soul (musique) vs soul (ivre)
_HOMOGRAPHES_ACCENT = frozenset({
    "foret",     # foret (outil de percage) vs foret (bois)
    "soul",      # soul (genre musical) vs soul (ivre)
    "cote",      # cote (valeur/classement) vs cote (rivage) vs cote (notation)
    "mur",       # mur (paroi) vs mur (fruit)
    "jeune",     # jeune (age) vs jeune (abstinence)
    "tache",     # tache (salissure) vs tache (travail)
    "peche",     # peche (fruit) vs peche (activite)
    "cru",       # cru (participe croire, vignoble) vs cru (participe croitre)
    "des",       # des (article) vs des (pluriel de de)
    "hale",      # hale (bronzage) vs hale (tirer)
    "male",      # male (genre) vs male (malle: coffre)
    "pale",      # pale (helice) vs pale (couleur)
    "sale",      # sale (adjectif) vs sale (piece)
    "mat",       # mat (echecs, surface) vs mat (poteau)
    "acre",      # acre (surface) vs acre (odeur)
    "faite",     # faite (participe faire) vs faîte (sommet toit)
    "sur",       # sur (preposition) vs sûr (certain)
    "du",        # du (article) vs dû (participe devoir)
    "criste",    # criste (plante : criste-marine) vs cristé (ornement)
})

# Passe simple / subjonctif imparfait — formes a proteger
_PASSE_SIMPLE_SUBJ = frozenset({
    # Passe simple courant (3eme personne sg/pl)
    "fut", "eut", "dit", "fit", "vit", "prit", "mit",
    "dut", "put", "sut", "lut", "vint", "tint",
    "crut", "parut", "connut", "vecut",
    "alla", "donna", "trouva", "arriva",
    # Formes derivees courantes du passe simple
    "apparut", "disparut", "resolut", "r\u00e9solut",
    "sortit", "produisit", "introduisit", "conduisit",
    "eteignit", "\u00e9teignit", "atteignit",
    "apercut", "aper\u00e7ut", "recut", "re\u00e7ut",
    "mourut", "courut", "parcourut",
    "naquit", "vainquit", "rompit",
    "reprit", "apprit", "comprit", "surprit",
    "suivit", "poursuivit", "servit",
    "revint", "devint", "parvint", "survint",
    "obtint", "retint", "maintint", "soutint",
    "rendit", "perdit", "entendit", "repondit", "r\u00e9pondit",
    "descendit", "defendit", "d\u00e9fendit", "attendit",
    "ecrivit", "\u00e9crivit",
    "sentit", "mentit", "consentit",
    "ouvrit", "offrit", "souffrit", "couvrit", "decouvrit", "d\u00e9couvrit",
    # Subjonctif imparfait (accent circonflexe)
    "f\u00fbt", "e\u00fbt", "d\u00fbt", "p\u00fbt", "s\u00fbt",
})

# Suffixes du passe simple 3sg (pour heuristique)
_PASSE_SIMPLE_SUFFIXES = ("ut", "it", "int", "\u00fbt", "\u00eet", "\u00e2t")

# Noms invariables en "s" (ne pas singulariser)
_INVARIABLES_S = frozenset({
    "fils", "bras", "corps", "temps", "bois", "mois", "fois",
    "poids", "voix", "noix", "croix", "pays", "repas",
    "tapis", "souris", "avis", "propos",
})

# Nombres romains (pour garde ordinal : "XII e" != "XII et")
_ROMAINS = frozenset({
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC", "C",
})

# Auxiliaires etre/avoir — formes conjuguees
_AUXILIAIRES = frozenset({
    "a", "ai", "as", "avons", "avez", "ont",
    "avait", "avais", "avaient", "avions", "aviez",
    "aura", "auras", "auront", "aurons", "aurez",
    "aurait", "aurais", "auraient", "aurions", "auriez",
    "est", "suis", "es", "sommes", "etes", "sont",
    "etait", "etais", "etaient", "etions", "etiez",
    "sera", "seras", "seront", "serons", "serez",
    "serait", "serais", "seraient", "serions", "seriez",
    "fut", "fus", "furent",
    "soit", "soient", "sois", "soyons", "soyez",
    "ait", "aie", "aies", "aient", "ayons", "ayez",
    "ayant", "etant",
    "etre", "avoir",
    # Avec accents
    "\u00eates", "\u00e9tait", "\u00e9tais", "\u00e9taient",
    "\u00e9tions", "\u00e9tiez", "\u00eatre", "\u00e9tant",
    "f\u00fbt", "f\u00fbmes", "f\u00fbtes",
})

# Mots de moins de 2 caracteres autorises en correction homophone
_HOMOPHONES_COURTS_WHITELIST = frozenset({"a", "\u00e0"})

# Homophones "structurels" ou la divergence de POS EST le signal de correction.
# Pour ces paires, div_pos=True est attendu et ne doit pas bloquer.
_HOMOPHONES_POS_DIVERGENT = frozenset({
    ("a", "\u00e0"), ("\u00e0", "a"),           # VER vs PRE
    ("et", "est"), ("est", "et"),               # CON vs AUX
    ("ou", "o\u00f9"), ("o\u00f9", "ou"),       # CON vs ADV
    ("son", "sont"), ("sont", "son"),           # DET vs AUX
    ("on", "ont"), ("ont", "on"),               # PRO vs AUX
    ("ce", "se"), ("se", "ce"),                 # DET vs PRO
})

# Numeraux > 1 (impliquent un pluriel)
_NUMERAUX_PLURIEL = frozenset({
    "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
    "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "vingt", "trente", "quarante", "cinquante", "soixante",
    "cent", "mille",
})

# Verbes d'etat (pour accord attribut a travers le verbe)
_VERBES_ETAT = frozenset({
    "est", "sont", "suis", "es", "sommes", "\u00eates",
    "\u00e9tait", "\u00e9tais", "\u00e9taient", "\u00e9tions", "\u00e9tiez",
    "etait", "etais", "etaient", "etions", "etiez",
    "sera", "seras", "seront", "serons", "serez",
    "serait", "serais", "seraient", "serions", "seriez",
    "semble", "semblent", "semblait", "semblaient",
    "parait", "paraissent", "paraissait", "paraissaient",
    "para\u00eet", "para\u00eetrait",
    "devient", "deviennent", "devenait", "devenaient",
    "reste", "restent", "restait", "restaient",
    "demeure", "demeurent",
})

# Auxiliaire etre — formes conjuguees (pour accord PP+etre)
_AUXILIAIRES_ETRE = frozenset({
    "suis", "es", "est", "sommes", "\u00eates", "etes", "sont",
    "\u00e9tais", "etais", "\u00e9tait", "etait",
    "\u00e9tions", "etions", "\u00e9tiez", "etiez", "\u00e9taient", "etaient",
    "sera", "seras", "seront", "serons", "serez",
    "serait", "serais", "seraient", "serions", "seriez",
    "fut", "fus", "furent", "f\u00fbt", "f\u00fbmes", "f\u00fbtes",
    "soit", "soient", "sois", "soyons", "soyez",
    "\u00e9tant", "etant",
})

# Auxiliaire "avoir" conjugue (pour regle INF->PP)
_AVOIR_CONJUGUE = frozenset({
    "a", "ai", "as", "avons", "avez", "ont",
    "avait", "avais", "avaient", "avions", "aviez",
    "aura", "auras", "auront", "aurons", "aurez",
    "aurait", "aurais", "auraient", "aurions", "auriez",
    "ait", "aie", "aies", "aient", "ayons", "ayez",
    "ayant",
})

# Pronoms clitiques objets (peuvent s'intercaler entre auxiliaire et PP)
_CLITIQUES_OBJETS = frozenset({
    "le", "la", "les", "l'", "l\u2019",
    "me", "m'", "m\u2019", "te", "t'", "t\u2019",
    "se", "s'", "s\u2019",
    "nous", "vous",
    "lui", "leur",
    "y", "en",
})

# Pronoms sujets qui precedent "est" (pour regle et->est structurelle)
_PRO_SUJET_EST = frozenset({
    "il", "elle", "on", "c'", "c\u2019", "ce", "qui", "tout",
})

# Determinants singuliers (pour regle DET+NOM+et->est)
_DET_SING_EST = frozenset({
    "le", "la", "l'", "l\u2019", "un", "une",
    "sa", "son", "ma", "mon", "ta", "ton",
    "ce", "cet", "cette", "notre", "votre", "leur",
})

# Verbes modaux (apres lesquels on attend un infinitif, pas un PP)
_MODAUX = frozenset({
    "doit", "doivent", "devait", "devaient", "devons", "devez",
    "dois", "devra", "devrait", "devraient",
    "peut", "peuvent", "pouvait", "pouvaient", "pouvons", "pouvez",
    "peux", "pourra", "pourrait", "pourraient",
    "veut", "veulent", "voulait", "voulaient", "voulons", "voulez",
    "veux", "voudra", "voudrait", "voudraient",
    "sait", "savent", "savait", "savaient", "savons", "savez",
    "sais", "saura", "saurait", "sauraient",
    "fait", "font", "faisait", "faisaient", "faisons", "faites",
    "fais", "fera", "ferait", "feraient",
    "va", "vont", "allait", "allaient", "allons", "allez",
    "vais", "ira", "irait", "iraient",
    "faut",
})

# _MODAUX_ELARGI : utilise uniquement comme TRIGGER dans les regles PP→INF.
# Inclut infinitifs, PP de modaux et gerondifs en plus des formes conjuguees.
# NE PAS utiliser comme guard de blocage (sinon on bloque les homophones).
_MODAUX_ELARGI = _MODAUX | frozenset({
    # Infinitifs de semi-modaux / causatifs / perception
    "faire", "laisser", "voir",
    "devoir", "pouvoir", "vouloir", "savoir",
    "aller",
    # PP de modaux (apres auxiliaire: "a pu manger", "a du partir")
    "pu", "d\u00fb", "voulu", "su", "vu", "laiss\u00e9",
    # Participes presents / gerondifs
    "faisant", "laissant", "voulant", "pouvant", "devant", "sachant", "allant",
})


# Particules de negation (deuxieme partie : pas, plus, jamais, etc.)
_NEGATION_PARTICULES = frozenset({
    "pas", "jamais", "rien", "personne", "aucun", "aucune",
    "guere", "gu\u00e8re", "point", "nullement",
    # NB: "plus" retire — trop ambigu (comparatif "plus grand" vs negation
    # "ne mange plus"). Les cas ne...plus sont rares a l'ecrit sans "ne".
})

# Voyelles pour contraction ne -> n'
_VOYELLES = frozenset("aeiouyàâäéèêëïîôùûüÿæœ")

# Adjacence clavier AZERTY (chaque touche → ensemble des touches voisines)
# Utilisee pour scorer les substitutions typographiques.
_AZERTY_VOISINS: dict[str, frozenset[str]] = {}
_AZERTY_LIGNES = [
    "azertyuiop",
    "qsdfghjklm",
    "wxcvbn",
]


def _init_azerty() -> dict[str, frozenset[str]]:
    """Construit la map d'adjacence AZERTY (horizontal + inter-lignes)."""
    adj: dict[str, set[str]] = {}
    rows = _AZERTY_LIGNES
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch not in adj:
                adj[ch] = set()
            # Voisins sur la meme ligne
            if c > 0:
                adj[ch].add(row[c - 1])
            if c < len(row) - 1:
                adj[ch].add(row[c + 1])
            # Voisins sur la ligne au-dessus
            if r > 0:
                above = rows[r - 1]
                for dc in (c - 1, c, c + 1):
                    if 0 <= dc < len(above):
                        adj[ch].add(above[dc])
                        if above[dc] not in adj:
                            adj[above[dc]] = set()
                        adj[above[dc]].add(ch)
            # Voisins sur la ligne en-dessous
            if r < len(rows) - 1:
                below = rows[r + 1]
                for dc in (c - 1, c, c + 1):
                    if 0 <= dc < len(below):
                        adj[ch].add(below[dc])
                        if below[dc] not in adj:
                            adj[below[dc]] = set()
                        adj[below[dc]].add(ch)
    return {k: frozenset(v) for k, v in adj.items()}


_AZERTY_VOISINS = _init_azerty()


def _est_adjacent_azerty(c1: str, c2: str) -> bool:
    """Verifie si deux caracteres sont des touches adjacentes sur AZERTY."""
    return c2 in _AZERTY_VOISINS.get(c1, frozenset())


def _est_changement_genre(forme: str, p2g: str) -> bool:
    """True si la difference entre forme et p2g est un pattern de genre.

    Detecte les transformations fem<->masc courantes :
    -ees <-> -es (participiales), -ee <-> -e, -ues <-> -us, -ive <-> -if, etc.
    Les formes sont attendues en minuscules.
    """
    f, p = forme, p2g
    # -ees <-> -es (participiales : transformees <-> transformes)
    if (f.endswith("\u00e9es") and p == f[:-3] + "\u00e9s") or \
       (p.endswith("\u00e9es") and f == p[:-3] + "\u00e9s"):
        return True
    # -ee <-> -e (singulier : transformee <-> transforme)
    if (f.endswith("\u00e9e") and p == f[:-2] + "\u00e9") or \
       (p.endswith("\u00e9e") and f == p[:-2] + "\u00e9"):
        return True
    # -ues <-> -us (connues <-> connus)
    if (f.endswith("ues") and p == f[:-3] + "us") or \
       (p.endswith("ues") and f == p[:-3] + "us"):
        return True
    # -ue <-> -u (mais pas -que)
    if (f.endswith("ue") and not f.endswith("que") and p == f[:-2] + "u") or \
       (p.endswith("ue") and not p.endswith("que") and f == p[:-2] + "u"):
        return True
    # -ive <-> -if (active <-> actif)
    if (f.endswith("ive") and p == f[:-3] + "if") or \
       (p.endswith("ive") and f == p[:-3] + "if"):
        return True
    # -ives <-> -ifs (actives <-> actifs)
    if (f.endswith("ives") and p == f[:-4] + "ifs") or \
       (p.endswith("ives") and f == p[:-4] + "ifs"):
        return True
    # -ique <-> -ic (publique <-> public)
    if (f.endswith("ique") and p == f[:-4] + "ic") or \
       (p.endswith("ique") and f == p[:-4] + "ic"):
        return True
    return False


# ---------------------------------------------------------------------------
# MotV6 — structure par mot
# ---------------------------------------------------------------------------

@dataclass
class MotV6:
    """Donnees par mot pour le pipeline V6 dual G2P/P2G."""

    forme: str            # forme originale
    index: int            # position dans la phrase

    # G2P (grapheme -> phoneme) — fiable pour POS
    g2p_phone: str = ""
    g2p_pos: str = ""
    g2p_nombre: str = ""
    g2p_genre: str = ""
    g2p_personne: str = ""

    # P2G (phoneme -> grapheme) — fiable pour homophones
    p2g_ortho: str = ""
    p2g_pos: str = ""
    p2g_nombre: str = ""
    p2g_genre: str = ""
    p2g_personne: str = ""
    p2g_confiance: float = 0.0
    p2g_alternatives: list[dict] = field(default_factory=list)

    # Divergences detectees
    div_pos: bool = False       # G2P.pos != P2G.pos
    div_ortho: bool = False     # forme != P2G.ortho
    div_nombre: bool = False    # G2P.nombre != P2G.nombre
    div_genre: bool = False     # G2P.genre != P2G.genre

    # Contexte
    preceded_by_punct: bool = False  # ponctuation entre ce mot et le precedent

    # Resultat
    correction: str = ""        # forme corrigee (= forme si pas de correction)
    regle: str = ""             # nom de la regle appliquee ("" si aucune)


# ---------------------------------------------------------------------------
# CorrecteurV6
# ---------------------------------------------------------------------------

class CorrecteurV6:
    """Pipeline V6 : dual G2P/P2G zero-FP.

    3 etapes :
      1. Preprocessing orthographique conservateur
      2. Analyse duale G2P + P2G (enrichissement)
      3. Corrections ciblees (homophones, accords, participe passe)
    """

    def __init__(
        self,
        lexique: Any,
        *,
        config: CorrecteurV6Config | None = None,
        tokeniseur: Any | None = None,
    ) -> None:
        self._lexique = LexiqueNormalise(lexique)
        self._config = config or CorrecteurV6Config()
        self._tokeniseur = tokeniseur

        # G2P tagger (neural, pour phonemisation + POS + morpho)
        self._g2p_tagger = self._init_g2p_tagger()

        # P2G adapter (neural, pour transcription phoneme -> grapheme)
        self._p2g_adapter = self._init_p2g_adapter()

    def _init_g2p_tagger(self) -> Any | None:
        """Charge le tagger G2P unifie."""
        try:
            from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
            g2p_adapter = creer_adapter_g2p_unifie()
            if g2p_adapter is not None:
                from lectura_correcteur._tagger_hybride import TaggerHybride
                return TaggerHybride(
                    g2p_adapter, self._lexique,
                    lm_homophones=None,
                )
        except Exception:
            logger.debug("G2P Unifie V2 indisponible", exc_info=True)
        return None

    def _init_p2g_adapter(self) -> Any | None:
        """Charge l'adaptateur P2G."""
        try:
            from lectura_correcteur._adapter_p2g import creer_adapter_p2g
            adapter = creer_adapter_p2g(
                lex_select=self._config.p2g_lex_select,
                lex_threshold=self._config.p2g_lex_threshold,
            )
            if adapter is None:
                logger.warning("P2G indisponible — V6 sans corrections P2G")
            return adapter
        except Exception:
            logger.debug("P2G indisponible", exc_info=True)
            return None

    @property
    def lexique(self) -> Any:
        return self._lexique

    @property
    def p2g_disponible(self) -> bool:
        """True si le pipeline G2P + P2G est disponible."""
        return self._g2p_tagger is not None and self._p2g_adapter is not None

    # ==================================================================
    # API publique
    # ==================================================================

    # Regex pour detecter les phrases contenant du markdown ou du LaTeX
    _MARKDOWN_LATEX_RE = re.compile(
        r"\*\*"           # gras markdown
        r"|```"           # bloc code markdown
        r"|`[^`]+`"       # inline code markdown
        r"|(?<!\w)\$[^$]+\$"  # inline LaTeX $...$
        r"|^\s*[-*+]\s"   # listes markdown
        r"|^\s*#{1,6}\s"  # titres markdown
        r"|\!\[.*\]\("    # images markdown
        r"|\\(?:frac|sqrt|sum|int|begin|end)\b"  # commandes LaTeX courantes
    )

    # Ligatures typographiques Unicode -> caracteres normaux
    _LIGATURES_TYPO: dict[str, str] = {
        "\ufb01": "fi",   # ﬁ
        "\ufb02": "fl",   # ﬂ
        "\ufb00": "ff",   # ﬀ
        "\ufb03": "ffi",  # ﬃ
        "\ufb04": "ffl",  # ﬄ
    }

    def corriger(self, phrase: str) -> ResultatCorrection:
        """Corrige une phrase. Retourne le resultat complet."""
        if not phrase.strip():
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        # Normaliser les ligatures typographiques (ﬁ→fi, ﬂ→fl, etc.)
        for lig, repl in self._LIGATURES_TYPO.items():
            if lig in phrase:
                phrase = phrase.replace(lig, repl)

        # Bypass markdown/LaTeX : la tokenisation detruit le formatage
        if self._config.bypass_markdown and self._MARKDOWN_LATEX_RE.search(phrase):
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        # Tokeniser
        tokens = self._tokenize(phrase)
        if not tokens:
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        # Pre-traitement : fusion + elision sur les tokens
        tokens, corrections_fusion = self._pretraitement_fusion_elision(tokens)

        # Separer ponctuation des mots
        is_punct = [bool(PUNCT_RE.match(t)) for t in tokens]
        word_tokens = [t for t, p in zip(tokens, is_punct) if not p]
        word_indices = [i for i, p in enumerate(is_punct) if not p]

        if not word_tokens:
            return ResultatCorrection(
                phrase_originale=phrase,
                phrase_corrigee=reconstruire_phrase(tokens),
            )

        # Detecter les mots adjacents a une apostrophe ou un trait d'union
        # (partie de mot elide ou compose : quoiqu', criste-marine, etc.)
        # On ne corrige pas ces mots en etape 1 car ils forment un tout.
        _skip_ortho: set[int] = set()
        for j, wi in enumerate(word_indices):
            if wi + 1 < len(tokens) and tokens[wi + 1] in ("'", "\u2019"):
                _skip_ortho.add(j)
            # Mot-compose avec trait d'union : proteger les deux parties
            if wi + 1 < len(tokens) and tokens[wi + 1] == "-":
                _skip_ortho.add(j)
            if wi > 0 and tokens[wi - 1] == "-":
                _skip_ortho.add(j)

        # Etape 1 : Orthographe conservatrice
        formes = list(word_tokens)
        corrections_ortho = self._v6_etape1_ortho(formes, followed_by_apo=_skip_ortho)

        # Etape 2 : Analyse duale G2P + P2G
        mots_v6 = self._v6_etape2_analyse(formes, word_tokens)

        # Marquer les mots precedes de ponctuation (ex: "trois, feu")
        for j in range(1, len(mots_v6)):
            wi_cur = word_indices[j]
            wi_prev = word_indices[j - 1]
            if wi_cur - wi_prev > 1:
                mots_v6[j].preceded_by_punct = True

        # Etape 3 : Corrections ciblees
        if not self._config.mode_analyse:
            corrections_gram = self._v6_etape3_corrections(mots_v6)
        else:
            corrections_gram = []

        # Assembler les corrections et le resultat
        all_corrections = corrections_fusion + corrections_ortho + corrections_gram

        # Si aucune correction, retourner la phrase originale telle quelle
        # (evite les artefacts de reconstruction sur la ponctuation)
        if not all_corrections:
            has_any_change = any(
                mv.correction != (word_tokens[j] if j < len(word_tokens) else mv.forme)
                for j, mv in enumerate(mots_v6)
            )
            if not has_any_change:
                phrase_out = phrase
                # Capitalisation debut de phrase
                if phrase_out and phrase_out[0].isalpha() and phrase_out[0].islower():
                    phrase_out = phrase_out[0].upper() + phrase_out[1:]
                return ResultatCorrection(
                    phrase_originale=phrase,
                    phrase_corrigee=phrase_out,
                )

        analyses: list[MotAnalyse] = []

        # Lookup des regles etape 1 par index mot
        _regle_ortho: dict[int, str] = {}
        for c in corrections_ortho:
            _regle_ortho[c.index] = c.regle

        # Confiance par regle etape 1 (OOV)
        _CONF_REGLE: dict[str, float] = {
            "ortho.accent": 0.92,
            "ortho.distance": 0.75,
            "ortho.fusion": 0.80,
            "ortho.elision": 0.85,
            "ortho.resegmentation": 0.55,
            "ortho.phonetique": 0.55,
        }

        for i, mv in enumerate(mots_v6):
            corrige = mv.correction
            original = word_tokens[i] if i < len(word_tokens) else mv.forme
            # Transferer la casse de l'original
            if original != corrige and corrige:
                corrige = transferer_casse(original, corrige)

            # Calcul de la confiance
            if original.lower() == corrige.lower():
                conf = 1.0  # pas de correction
            elif mv.regle:
                # Etape 3 (P2G / structurel) : utiliser p2g_confiance
                if mv.regle.startswith("p2."):
                    conf = max(0.50, min(0.70, mv.p2g_confiance))
                else:
                    conf = max(0.50, min(0.99, mv.p2g_confiance))
            elif i in _regle_ortho:
                # Etape 1 (OOV) : confiance par type de regle
                conf = _CONF_REGLE.get(_regle_ortho[i], 0.65)
            else:
                conf = 0.65  # correction de source inconnue

            analyses.append(MotAnalyse(
                original=original,
                corrige=corrige,
                pos=mv.g2p_pos,
                morpho={
                    k: v for k, v in {
                        "genre": mv.g2p_genre,
                        "nombre": mv.g2p_nombre,
                    }.items() if v
                },
                dans_lexique=self._lexique.existe(mv.forme.lower()),
                type_correction=(
                    TypeCorrection.GRAMMAIRE if mv.regle and "homophone" in mv.regle else
                    TypeCorrection.HORS_LEXIQUE if mv.regle and "ortho" in mv.regle else
                    TypeCorrection.GRAMMAIRE if mv.regle else
                    TypeCorrection.HORS_LEXIQUE if i in _regle_ortho else
                    TypeCorrection.AUCUNE
                ),
                confiance=conf,
            ))

        # Reconstruction
        final_tokens = list(tokens)
        for idx, wi in enumerate(word_indices):
            if idx < len(analyses):
                final_tokens[wi] = analyses[idx].corrige

        phrase_corrigee = reconstruire_phrase(final_tokens)

        # Capitalisation debut de phrase
        if phrase_corrigee and phrase_corrigee[0].isalpha() and phrase_corrigee[0].islower():
            phrase_corrigee = phrase_corrigee[0].upper() + phrase_corrigee[1:]

        return ResultatCorrection(
            phrase_originale=phrase,
            phrase_corrigee=phrase_corrigee,
            mots=analyses,
            corrections=all_corrections,
        )

    def analyser(self, phrase: str) -> list[MotV6]:
        """Analyse une phrase sans corriger. Retourne les MotV6 enrichis.

        Utile pour le debug : inspecter les divergences G2P/P2G.
        """
        if not phrase.strip():
            return []

        tokens = self._tokenize(phrase)
        is_punct = [bool(PUNCT_RE.match(t)) for t in tokens]
        word_tokens = [t for t, p in zip(tokens, is_punct) if not p]

        if not word_tokens:
            return []

        formes = list(word_tokens)
        self._v6_etape1_ortho(formes)
        mots_v6 = self._v6_etape2_analyse(formes, word_tokens)
        return mots_v6

    # ==================================================================
    # Tokenisation
    # ==================================================================

    def _tokenize(self, phrase: str) -> list[str]:
        """Tokenise la phrase."""
        if self._tokeniseur is not None:
            raw = self._tokeniseur.tokeniser(phrase)
            if raw and isinstance(raw[0], tuple):
                return [tok for tok, _iw in raw]
            return raw
        return [m.group() for m in _TOKEN_RE.finditer(phrase)]

    # ==================================================================
    # Pre-traitement : fusion + elision
    # ==================================================================

    # Fusions connues : paires de mots separes -> mot fusionne
    _FUSIONS_CONNUES: dict[tuple[str, str], str] = {
        ("beau", "coup"): "beaucoup",
        ("en", "semble"): "ensemble",
        ("peut", "etre"): "peut-être",
        # NB: ("peut", "être") omis — ambigu avec le passif (peut être comparé)
        ("long", "temps"): "longtemps",
        ("par", "fois"): "parfois",
        ("sur", "tout"): "surtout",  # garde contextuelle dans _pretraitement_fusion_elision
        ("mal", "gré"): "malgré",
        ("mal", "gre"): "malgré",
        ("mal", "heureusement"): "malheureusement",
        ("des", "ormais"): "désormais",
        ("dés", "ormais"): "désormais",
        ("auto", "bus"): "autobus",
    }

    # Mots qui suivent typiquement "surtout" (adverbe) et non "sur tout" (prep+det)
    _ADV_APRES_SURTOUT: frozenset[str] = frozenset({
        "très", "tres", "plus", "pas", "bien", "quand", "lorsque",
        "si", "que", "pour", "ne", "en", "le", "la", "les", "un", "une",
        "dans", "avec", "par", "au", "aux", "du", "des",
    })

    # Mots eligibles a l'elision (mot + voyelle -> mot' + voyelle)
    # NB: "la" omis car ambigu (article "la" vs adverbe "là")
    _ELISION_MOTS: dict[str, str] = {
        "que": "qu'",
        "je": "j'",
        "le": "l'",
        "de": "d'",
        "ne": "n'",
        "se": "s'",
        "me": "m'",
        "te": "t'",
        "ce": "c'",
        "lorsque": "lorsqu'",
        "puisque": "puisqu'",
        "jusque": "jusqu'",
        # NB: "quelque" omis — "quelque autre" est valide en francais moderne
    }

    def _pretraitement_fusion_elision(
        self, tokens: list[str],
    ) -> tuple[list[str], list[Correction]]:
        """Fusionne les mots separes et applique les elisions manquantes.

        Opere sur la liste de tokens bruts (mots + ponctuation).
        Retourne la nouvelle liste de tokens et les corrections.
        """
        corrections: list[Correction] = []
        result: list[str] = []
        lex = self._lexique
        i = 0
        n = len(tokens)

        while i < n:
            # Cas special : aujourd + ' + hui -> aujourd'hui
            if (i < n - 2
                    and tokens[i].lower() == "aujourd"
                    and tokens[i + 1] in ("'", "\u2019")
                    and tokens[i + 2].lower() == "hui"):
                result.append("aujourd'hui")
                i += 3
                continue

            # Cas special : aujourdhui (sans apostrophe) -> aujourd'hui
            if tokens[i].lower() == "aujourdhui":
                corrections.append(Correction(
                    index=len(result),
                    original=tokens[i],
                    corrige="aujourd'hui",
                    type_correction=TypeCorrection.HORS_LEXIQUE,
                    regle="ortho.resegmentation",
                    explication=f"Resegmentation: '{tokens[i]}' -> 'aujourd'hui'",
                ))
                result.append("aujourd'hui")
                i += 1
                continue

            # Fusionner les apostrophes isolees : "qu" + "'" + "il" -> "qu'" + "il"
            if tokens[i] in ("'", "\u2019") and result:
                result[-1] = result[-1] + "'"
                i += 1
                continue

            # Ne pas re-elider un mot deja elide (finit par apostrophe)
            if tokens[i].endswith(("'", "\u2019")) and not PUNCT_RE.match(tokens[i]):
                result.append(tokens[i])
                i += 1
                continue

            # Tenter fusion de 2 tokens consecutifs (mots seulement)
            if i < n - 1 and not PUNCT_RE.match(tokens[i]) and not PUNCT_RE.match(tokens[i + 1]):
                low_pair = (tokens[i].lower(), tokens[i + 1].lower())
                fused = self._FUSIONS_CONNUES.get(low_pair)
                if fused is not None:
                    # Garde contextuelle "sur tout" : ne fusionner que si
                    # suivi de ponctuation, fin de phrase, ou adverbe connu
                    # (evite "sur tout ceci" -> "surtout ceci")
                    if low_pair == ("sur", "tout"):
                        next_after = tokens[i + 2].lower() if i + 2 < n else ""
                        if (next_after not in self._ADV_APRES_SURTOUT
                                and next_after not in {"", ",", ".", "!", "?", ";", ":"}):
                            i += 1
                            result.append(tokens[i - 1])
                            continue
                    # Fusionner : les paires sont choisies manuellement
                    original = tokens[i] + " " + tokens[i + 1]
                    result.append(transferer_casse(tokens[i], fused))
                    corrections.append(Correction(
                        index=len(result) - 1,
                        original=original,
                        corrige=result[-1],
                        type_correction=TypeCorrection.HORS_LEXIQUE,
                        regle="ortho.fusion",
                        explication=f"Fusion: '{original}' -> '{result[-1]}'",
                    ))
                    i += 2
                    continue

            # Tenter elision : "que" + "il" -> "qu'" + "il"
            if i < n - 1 and not PUNCT_RE.match(tokens[i]) and not PUNCT_RE.match(tokens[i + 1]):
                low_mot = tokens[i].lower()
                next_tok = tokens[i + 1]
                next_low = next_tok.lower()

                elided_form = self._ELISION_MOTS.get(low_mot)
                # Garde mot etranger/verbe : pour le/la, ne pas elider
                # si le mot suivant n'a pas d'entree NOM/ADJ dans le lexique
                # (le user ID → user est VER uniquement → pas d'elision)
                _skip_elision = False
                if low_mot in ("le", "la") and self._lexique.existe(next_low):
                    _infos = self._lexique.info(next_low)
                    _has_nom_adj = any(
                        e.get("cgram", "").startswith(("NOM", "ADJ"))
                        for e in _infos
                    )
                    if not _has_nom_adj:
                        _skip_elision = True
                # "ce" ne s'elide que devant etre/avoir (c'est, c'etait, c'a)
                # Pas devant preposition/pronom (ce a quoi, ce en quoi)
                _CE_ELISION_SUIVANTS = frozenset({
                    "est", "était", "etait", "a", "avait", "eut", "eût",
                    "aurait", "ait", "aura", "sera",
                })
                if low_mot == "ce" and next_low not in _CE_ELISION_SUIVANTS:
                    _skip_elision = True
                # Pas d'elision devant onze, huit (h aspire numerique)
                if next_low in ("onze", "huit"):
                    _skip_elision = True
                if (elided_form is not None
                        and next_low and next_low[0] in _VOYELLES
                        # Pas d'elision forcee devant nom propre (Yéddo, Ouistreham)
                        and not next_tok[0].isupper()
                        and not _skip_elision):
                    original = tokens[i] + " " + next_tok
                    elided_display = elided_form + next_tok
                    # Produire 2 tokens : prefixe elide + mot suivant
                    result.append(transferer_casse(tokens[i], elided_form))
                    result.append(next_tok)
                    corrections.append(Correction(
                        index=len(result) - 2,
                        original=original,
                        corrige=transferer_casse(tokens[i], elided_display),
                        type_correction=TypeCorrection.HORS_LEXIQUE,
                        regle="ortho.elision",
                        explication=f"Elision: '{original}' -> '{elided_display}'",
                    ))
                    i += 2
                    continue

            result.append(tokens[i])
            i += 1

        return result, corrections

    # ==================================================================
    # Etape 1 : Orthographe conservatrice
    # ==================================================================

    def _v6_etape1_ortho(
        self,
        formes: list[str],
        *,
        followed_by_apo: set[int] | None = None,
    ) -> list[Correction]:
        """Corrige uniquement les fautes d'orthographe evidentes.

        Methode : pour chaque mot OOV, chercher un candidat a distance
        d'edition <= 1 qui est dans le lexique et a le meme phoneme G2P.
        Ne corriger que si le candidat est unique ou tres frequent.

        Modifie `formes` in-place.
        """
        corrections: list[Correction] = []
        lex = self._lexique

        for i, forme in enumerate(formes):
            low = forme.lower()

            # Ne pas corriger les mots proteges
            if low in _MOTS_PROTEGES:
                continue

            # Ne pas corriger les tokens suivis d'une apostrophe
            # = partie gauche d'un mot elide (quoiqu', entr', etc.)
            if followed_by_apo and i in followed_by_apo:
                continue

            # Ne pas corriger les tokens elides
            if forme.endswith(("'", "\u2019")):
                continue

            # Tokens avec trait d'union : corriger les accents partie par partie
            # mais pas de resegmentation/edit-distance/phonetique
            if "-" in forme:
                parties = low.split("-")
                corrigees = []
                modifie = False
                _ACC = frozenset("àâäéèêëïîôùûüÿçœæ")
                for p in parties:
                    if p in _HOMOGRAPHES_ACCENT:
                        corrigees.append(p)
                        continue
                    if not lex.existe(p):
                        candidat_acc = self._trouver_variante_accent_oov(p)
                        if candidat_acc is not None:
                            # Ne pas retirer d'accents (entre-bâillée → entre-baillée)
                            nb_orig = sum(1 for c in p if c in _ACC)
                            nb_cand = sum(1 for c in candidat_acc if c in _ACC)
                            if nb_cand >= nb_orig:
                                corrigees.append(candidat_acc)
                                modifie = True
                            else:
                                corrigees.append(p)
                        else:
                            corrigees.append(p)
                    else:
                        corrigees.append(p)
                if modifie:
                    new_forme = "-".join(corrigees)
                    formes[i] = transferer_casse(forme, new_forme)
                    corrections.append(Correction(
                        index=i, original=forme, corrige=formes[i],
                        type_correction=TypeCorrection.HORS_LEXIQUE,
                        regle="ortho.accent",
                        explication=f"Accent compose: '{forme}' -> '{formes[i]}'",
                    ))
                continue

            # Ne pas corriger les mots capitalises (noms propres)
            if forme[0].isupper():
                continue

            # Ne pas corriger les mots tres courts (1-2 chars)
            if len(low) <= 2:
                continue

            # Ne pas corriger les suffixes ordinaux apres un chiffre
            # (2 ème, 1 er, 3 ère, 20 ème, etc.)
            if i > 0 and formes[i - 1].isdigit():
                continue

            # Si le mot est dans le lexique, rien a faire
            if lex.existe(low):
                continue

            # OOV : priorite 1 — variante accent pure (zero-FP sur texte propre)
            candidat_accent = self._trouver_variante_accent_oov(low)
            if candidat_accent is not None:
                # Ne pas retirer trop d'accents (ème->eme interdit)
                # Mais permettre de corriger des accents mal places
                # (nécéssite -> nécessite : 2 acc -> 1 acc, OK)
                _ACC_OOV = frozenset("àâäéèêëïîôùûüÿçœæ")
                nb_acc_o = sum(1 for c in low if c in _ACC_OOV)
                nb_acc_c = sum(1 for c in candidat_accent if c in _ACC_OOV)
                if nb_acc_c >= nb_acc_o - 1 and nb_acc_c > 0:
                    formes[i] = transferer_casse(forme, candidat_accent)
                    corrections.append(Correction(
                        index=i,
                        original=forme,
                        corrige=formes[i],
                        type_correction=TypeCorrection.HORS_LEXIQUE,
                        regle="ortho.accent",
                        explication=f"Accent OOV: '{forme}' -> '{formes[i]}'",
                    ))
                    continue
                # Sinon : accent retire, on tombe sur les priorites suivantes

            # OOV : priorite 1b — conjugaisons fautives irrégulières
            # Ex: "allent" -> "vont", "faisent" -> "font"
            candidat_irreg = self._corriger_oov_irregulier(low, formes, i)
            if candidat_irreg is not None:
                formes[i] = transferer_casse(forme, candidat_irreg)
                corrections.append(Correction(
                    index=i, original=forme, corrige=formes[i],
                    type_correction=TypeCorrection.HORS_LEXIQUE,
                    regle="ortho.irregulier",
                    explication=(
                        f"Conjugaison irreguliere: "
                        f"'{forme}' -> '{formes[i]}'"
                    ),
                ))
                continue

            # OOV : priorite 2 — candidat edit-distance (existant)
            candidat = self._trouver_candidat_ortho(low)
            if candidat is not None:
                formes[i] = transferer_casse(forme, candidat)
                corrections.append(Correction(
                    index=i,
                    original=forme,
                    corrige=formes[i],
                    type_correction=TypeCorrection.HORS_LEXIQUE,
                    regle="ortho.distance",
                    explication=f"OOV: '{forme}' -> '{formes[i]}'",
                ))
                continue

            # OOV : priorite 3 — resegmentation (mots colles)
            reseg = self._resegmenter_oov(low)
            if reseg is not None:
                formes[i] = transferer_casse(forme, reseg)
                corrections.append(Correction(
                    index=i, original=forme, corrige=formes[i],
                    type_correction=TypeCorrection.HORS_LEXIQUE,
                    regle="ortho.resegmentation",
                    explication=f"Resegmentation: '{forme}' -> '{formes[i]}'",
                ))
                continue

            # OOV : priorite 4 — correction phonetique
            phonetique = self._corriger_oov_phonetique(low)
            if phonetique is not None:
                formes[i] = transferer_casse(forme, phonetique)
                corrections.append(Correction(
                    index=i, original=forme, corrige=formes[i],
                    type_correction=TypeCorrection.HORS_LEXIQUE,
                    regle="ortho.phonetique",
                    explication=f"Phonetique: '{forme}' -> '{formes[i]}'",
                ))

        return corrections

    def _trouver_candidat_ortho(self, mot: str) -> str | None:
        """Cherche un candidat orthographique conservateur pour un mot OOV.

        Criteres :
        - Distance d'edition <= 1
        - Candidat dans le lexique
        - Meme phoneme G2P (si G2P disponible)
        - Frequence >= ortho_frequence_min
        """
        from lectura_correcteur.orthographe._suggestions import (
            _variantes_accents,
            _est_variante_accent,
            _est_doublement_consonne,
            _edits_distance_1,
        )

        lex = self._lexique
        cfg = self._config
        best_forme: str | None = None
        best_freq: float = 0.0

        def _freq(c: str) -> float:
            return lex.frequence(c) if hasattr(lex, "frequence") else 0.0

        # Priorite 1 : variantes d'accent (toujours fiable)
        _ACC = frozenset("àâäéèêëïîôùûüÿçœæ")
        accents = _variantes_accents(mot, lex)
        for forme, freq in accents:
            if freq > best_freq:
                # Ne pas retirer d'accents (appellé -> appelle serait faux)
                nb_acc_o = sum(1 for c in mot if c in _ACC)
                nb_acc_c = sum(1 for c in forme if c in _ACC)
                if nb_acc_c >= nb_acc_o:
                    best_forme, best_freq = forme, freq

        # Priorite 2 : edit distance 1, avec garde phonetique
        if best_forme is None and len(mot) >= 3:
            phone_orig = self._g2p_phone(mot) if self._g2p_tagger else None

            d1_candidats = _edits_distance_1(mot)
            for c in d1_candidats:
                if not lex.existe(c):
                    continue
                freq = _freq(c)
                if freq < cfg.ortho_frequence_min:
                    continue
                if freq <= best_freq:
                    continue

                # Variante accent : accepter si pas de retrait d'accent
                if _est_variante_accent(mot, c):
                    nb_acc_o = sum(1 for ch in mot if ch in _ACC)
                    nb_acc_c = sum(1 for ch in c if ch in _ACC)
                    if nb_acc_c >= nb_acc_o:
                        best_forme, best_freq = c, freq
                    continue

                # Doublement/dedoublement consonne : morphologiquement fiable
                if _est_doublement_consonne(mot, c):
                    best_forme, best_freq = c, freq
                    continue

                # Sinon : exiger meme phoneme G2P + freq elevee
                if phone_orig and abs(len(c) - len(mot)) <= 1:
                    phone_c = self._g2p_phone(c)
                    if phone_c and phone_c == phone_orig and freq >= 50.0:
                        best_forme, best_freq = c, freq
                        continue

                # Transposition adjacente (chein->chien) : accepter
                # meme si phoneme differe, si swap exact de 2 chars
                if freq >= 5.0 and len(c) == len(mot):
                    diffs = [k for k in range(len(mot)) if mot[k] != c[k]]
                    if (len(diffs) == 2 and diffs[1] == diffs[0] + 1
                            and mot[diffs[0]] == c[diffs[1]]
                            and mot[diffs[1]] == c[diffs[0]]):
                        best_forme, best_freq = c, freq
                        continue

                # Substitution clavier AZERTY (quk->qui) :
                # le char fautif est adjacent au char correct sur AZERTY
                if freq >= 5.0 and len(c) == len(mot):
                    diffs = [k for k in range(len(mot)) if mot[k] != c[k]]
                    if len(diffs) == 1:
                        ch_orig = mot[diffs[0]].lower()
                        ch_cand = c[diffs[0]].lower()
                        if _est_adjacent_azerty(ch_orig, ch_cand):
                            best_forme, best_freq = c, freq
                            continue

                # Substitution/deletion/insertion generale : accepter
                # sans phoneme si freq haute et mot long (>=5)
                # Garde stricte car pas de signal clavier/transposition.
                if freq >= 50.0 and len(mot) >= 5:
                    best_forme, best_freq = c, freq

        if best_forme is not None and best_freq >= cfg.ortho_frequence_min:
            return best_forme
        return None

    def _trouver_variante_accent_oov(self, mot: str) -> str | None:
        """Variante accent pour un mot OOV. Pas de seuil de frequence.

        Sur texte propre accentue, le mot est dans le lexique et on ne
        passe jamais par ce code => zero FP garanti.

        Prefere la variante qui preserve la terminaison du mot original
        (ex: nécéssite -> nécessite et non nécessité).
        """
        from lectura_correcteur.orthographe._suggestions import _variantes_accents

        variantes = _variantes_accents(mot, self._lexique)
        if not variantes:
            return None
        if len(variantes) == 1:
            return variantes[0][0]
        # Preferer la variante dont le dernier caractere correspond
        # a l'original (preserve morpho : -e verb vs -é PP/nom)
        fin = mot[-1]
        matching = [(v, f) for v, f in variantes if v and v[-1] == fin]
        if matching:
            return max(matching, key=lambda x: x[1])[0]
        return variantes[0][0]

    def _g2p_phone(self, mot: str) -> str | None:
        """Obtient le phoneme G2P d'un mot."""
        if self._g2p_tagger is None:
            return None
        if hasattr(self._g2p_tagger, "prononcer"):
            return self._g2p_tagger.prononcer(mot)
        return None

    def _resegmenter_oov(self, mot: str) -> str | None:
        """Decoupe un mot OOV en 2 parties valides du lexique."""
        from lectura_correcteur.orthographe._suggestions import _variantes_accents
        lex = self._lexique
        if len(mot) < 5:
            return None
        # Ignorer les mots contenant des caracteres non-alphabetiques
        if not all(c.isalpha() for c in mot):
            return None

        best_split: str | None = None
        best_freq: float = 0.0

        def _freq(w: str) -> float:
            return lex.frequence(w) if hasattr(lex, "frequence") else 0.0

        # Seuils de frequence pour eviter les splits en mots rares
        # (pressoire -> pres soire, bourbes -> bour bes)
        _RESEG_FREQ_MIN_PART = 5.0    # freq minimum pour chaque partie
        _RESEG_FREQ_MIN_TOTAL = 50.0  # freq combinee minimum

        for i in range(2, len(mot) - 3):
            p1, p2 = mot[:i], mot[i:]

            # Strategie 1 : split direct (parce+que, pardessus -> par-dessus)
            if lex.existe(p1) and lex.existe(p2):
                freq_p1 = _freq(p1)
                freq_p2 = _freq(p2)
                # Garde frequence : les deux parties doivent etre des mots courants
                if freq_p1 < _RESEG_FREQ_MIN_PART or freq_p2 < _RESEG_FREQ_MIN_PART:
                    continue
                freq = freq_p1 + freq_p2
                if freq < _RESEG_FREQ_MIN_TOTAL:
                    continue
                if freq > best_freq:
                    # Preferer trait d'union si le compose existe dans le lexique
                    compose_tiret = p1 + "-" + p2
                    if lex.existe(compose_tiret):
                        best_split = compose_tiret
                    else:
                        best_split = p1 + " " + p2
                    best_freq = freq

            # Strategie 2 : apostrophe/elision (l'+ecole, d'+accord)
            if i <= 3:
                p1_apo = mot[:i] + "'"
                _elisions = {"l'", "d'", "n'", "s'", "j'", "c'", "m'", "t'", "qu'"}
                if p1_apo.lower() in _elisions:
                    # Direct
                    if lex.existe(p2):
                        freq = _freq(p2) + 100.0  # bonus elision
                        if freq > best_freq:
                            best_split = p1_apo + p2
                            best_freq = freq
                    # Avec variante accent sur p2
                    variantes = _variantes_accents(p2, lex)
                    if variantes:
                        p2_acc, freq_acc = variantes[0]
                        freq = freq_acc + 100.0
                        if freq > best_freq:
                            best_split = p1_apo + p2_acc
                            best_freq = freq

            # Strategie 3 : split + variante accent sur p2 (grandpere -> grand-père)
            if lex.existe(p1) and not lex.existe(p2):
                variantes = _variantes_accents(p2, lex)
                if variantes:
                    p2_acc, freq_acc = variantes[0]
                    freq_p1 = _freq(p1)
                    if freq_p1 >= _RESEG_FREQ_MIN_PART and freq_acc >= _RESEG_FREQ_MIN_PART:
                        freq = freq_p1 + freq_acc
                        if freq > best_freq and freq >= _RESEG_FREQ_MIN_TOTAL:
                            # Preferer trait d'union si le compose existe
                            compose_tiret = p1 + "-" + p2_acc
                            if lex.existe(compose_tiret):
                                best_split = compose_tiret
                            else:
                                best_split = p1 + " " + p2_acc
                            best_freq = freq

        return best_split

    def _corriger_oov_phonetique(self, mot: str) -> str | None:
        """Corrige un mot OOV via equivalence phonetique G2P."""
        lex = self._lexique
        if not hasattr(lex, "homophones"):
            return None

        # Ignorer les mots contenant des caracteres non-alphabetiques
        # (chiffres, symboles speciaux, diacritiques exotiques comme æ)
        if not all(c.isalpha() for c in mot):
            return None

        # Mots trop courts : risque eleve de faux positif
        if len(mot) < 3:
            return None

        # Mots contenant des ligatures latines (æ, œ) : texte latin/specialise
        if "æ" in mot or "œ" in mot:
            return None

        phone = self._g2p_phone(mot)
        if not phone:
            return None

        best_forme: str | None = None
        best_freq: float = 0.0

        # Match phonetique exact
        # A frequence comparable (ecart < 20%), preferer la forme dont la
        # longueur est la plus proche de l'original (pommes vs pomme pour "pome")
        for entry in lex.homophones(phone):
            ortho = entry.get("ortho", "")
            if not ortho or ortho.lower() == mot:
                continue
            freq = entry.get("freq") or 0.0
            if freq > best_freq * 1.2:
                best_forme = ortho.lower()
                best_freq = freq
            elif freq >= best_freq * 0.8 and best_forme:
                if abs(len(ortho) - len(mot)) < abs(len(best_forme) - len(mot)):
                    best_forme = ortho.lower()
                    best_freq = freq
            elif not best_forme and freq > 0:
                best_forme = ortho.lower()
                best_freq = freq

        # Garde match exact : candidat ne doit pas etre trop court vs l'original
        # (tailes->tel exclu car 3 < 6-2=4)
        # + distance d'edition max 2 (ou 3 si candidat plus long = lettres muettes)
        if best_forme and best_freq >= 10.0:
            if len(best_forme) >= max(3, len(mot) - 2):
                from lectura_correcteur.orthographe._suggestions import _edit_distance_rapide
                max_dist = 2
                # Autoriser d=3 si le candidat est plus long (lettres muettes :
                # bato->bateau, phone match exact garanti en amont)
                if len(best_forme) > len(mot):
                    max_dist = 3
                if _edit_distance_rapide(mot, best_forme) <= max_dist:
                    return best_forme

        # Match phonetique d=1 si pas de match exact retenu
        # Desactive pour les mots courts (<5 chars) : trop de FP
        # (else->elle, fort->fore, etc.)
        if len(mot) < 5:
            return None
        best_d1: str | None = None
        best_d1_freq: float = 0.0
        from lectura_correcteur._phones import generer_phones_d1
        for phone_var in generer_phones_d1(phone):
            for entry in lex.homophones(phone_var):
                ortho = entry.get("ortho", "")
                if not ortho or ortho.lower() == mot:
                    continue
                freq = entry.get("freq") or 0.0
                if freq > best_d1_freq:
                    best_d1 = ortho.lower()
                    best_d1_freq = freq

        # Garde d=1 plus stricte : candidat doit etre au moins aussi long
        # que l'original et frequence elevee (evite livers->livre, etc.)
        if best_d1 and best_d1_freq >= 10.0:
            if len(best_d1) >= len(mot):
                return best_d1
        return None

    # ==================================================================
    def _corriger_oov_irregulier(
        self, mot: str, formes: list[str], idx: int,
    ) -> str | None:
        """Corrige les conjugaisons fautives de verbes irreguliers OOV.

        Ex: "allent" -> "vont", "faisent" -> "font", "voivent" -> "voient"

        Strategie : si le mot finit par une terminaison verbale courante
        (-ent, -ons, -ez, -e, -es, -t) et que le radical tronque mene
        a un lemme irregulier, proposer la bonne conjugaison.
        """
        lex = self._lexique

        # Terminaisons verbales + personne/nombre correspondants
        _TERMINAISONS = [
            ("ent", "3", "p"),
            ("ons", "1", "p"),
            ("ez", "2", "p"),
            ("es", "2", "s"),
            ("e", "1", "s"),   # aussi P3s pour -er
        ]

        # Lemmes irreguliers courants : radical fautif -> lemme
        _RADICAL_LEMME: dict[str, str] = {
            "all": "aller",
            "fais": "faire",
            "voiv": "voir",
            "sav": "savoir",
            "pouv": "pouvoir",
            "voul": "vouloir",
            "prenn": "prendre",
            "vienn": "venir",
            "tienn": "tenir",
            "meur": "mourir",
            "cour": "courir",
            "recev": "recevoir",
            "boiv": "boire",
            "ecrit": "\u00e9crire",
            "connais": "conna\u00eetre",
        }

        for term, pers, nombre in _TERMINAISONS:
            if not mot.endswith(term):
                continue
            radical = mot[: -len(term)]
            if not radical:
                continue

            lemme = _RADICAL_LEMME.get(radical)
            if not lemme:
                continue

            # Chercher la forme conjuguee au present indicatif
            formes_lex = lex.formes_de(lemme)
            candidat = None
            candidat_freq = -1.0
            for f in formes_lex:
                f_cgram = f.get("cgram", "")
                if not (f_cgram.startswith("VER") or f_cgram == "AUX"):
                    continue
                f_mode = normaliser_morpho(f.get("mode", ""))
                f_temps = normaliser_morpho(f.get("temps", ""))
                if f_mode != "ind" or f_temps != "pre":
                    continue
                f_pers = f.get("personne", "")
                f_nombre = normaliser_morpho(f.get("nombre", ""))
                if f_pers == pers and f_nombre == nombre:
                    f_freq = float(f.get("freq", 0))
                    if f_freq > candidat_freq:
                        candidat = f.get("ortho", "")
                        candidat_freq = f_freq

            if candidat and candidat.lower() != mot:
                return candidat

        return None

    # Etape 2 : Analyse duale G2P + P2G
    # ==================================================================

    def _v6_etape2_analyse(
        self,
        formes: list[str],
        originaux: list[str],
    ) -> list[MotV6]:
        """Enrichit chaque mot avec les predictions G2P et P2G.

        Ne corrige rien — remplit les champs MotV6 et calcule les divergences.
        """
        n = len(formes)
        formes_low = [f.lower() for f in formes]

        # Initialiser les MotV6
        mots = [
            MotV6(
                forme=formes_low[i],
                index=i,
                correction=formes[i],  # par defaut = forme originale
            )
            for i in range(n)
        ]

        if not self.p2g_disponible:
            return mots

        # G2P : phonemes + POS + morpho
        g2p_tags = self._g2p_tagger.tag_words_rich(formes_low)

        phones: list[str] = []
        for i in range(n):
            tag = g2p_tags[i] if i < len(g2p_tags) else {}
            phone = tag.get("g2p", "")
            if not phone and hasattr(self._g2p_tagger, "prononcer"):
                phone = self._g2p_tagger.prononcer(formes_low[i]) or ""
            phones.append(phone if phone else formes_low[i])

            mots[i].g2p_phone = phone
            mots[i].g2p_pos = tag.get("pos", "")

            # Morpho G2P — les traits sont directement sur le tag dict
            mots[i].g2p_nombre = tag.get("nombre", "")
            mots[i].g2p_genre = tag.get("genre", "")
            mots[i].g2p_personne = tag.get("personne", "")

        # P2G : ortho + POS + morpho + confiance + alternatives
        try:
            p2g_result = self._p2g_adapter.transcrire_complet(
                phones, ortho_words=None, k=self._config.homophone_top_k,
            )
        except Exception:
            logger.warning("P2G transcrire_complet echoue", exc_info=True)
            return mots

        p2g_ortho_list = p2g_result.get("ortho", [])
        p2g_confiance_list = p2g_result.get("confiance", [])
        p2g_alternatives_list = p2g_result.get("alternatives", [])
        p2g_pos_list = p2g_result.get("pos", [])
        p2g_morpho = p2g_result.get("morpho", {})

        # Extraire morpho P2G
        p2g_gender = p2g_morpho.get("Gender", [])
        p2g_number = p2g_morpho.get("Number", [])
        p2g_person = p2g_morpho.get("Person", [])

        # Mapping UD -> short-form
        _gender_map = {"Masc": "m", "Fem": "f"}
        _number_map = {"Sing": "s", "Plur": "p"}
        _person_map = {"1": "1", "2": "2", "3": "3"}

        for i in range(n):
            # P2G ortho
            mots[i].p2g_ortho = (
                p2g_ortho_list[i] if i < len(p2g_ortho_list) else ""
            )
            mots[i].p2g_confiance = (
                p2g_confiance_list[i] if i < len(p2g_confiance_list) else 0.0
            )
            mots[i].p2g_alternatives = (
                p2g_alternatives_list[i] if i < len(p2g_alternatives_list) else []
            )
            mots[i].p2g_pos = (
                p2g_pos_list[i] if i < len(p2g_pos_list) else ""
            )

            # P2G morpho
            g_raw = p2g_gender[i] if i < len(p2g_gender) else "_"
            n_raw = p2g_number[i] if i < len(p2g_number) else "_"
            mots[i].p2g_nombre = _number_map.get(n_raw, "")
            mots[i].p2g_genre = _gender_map.get(g_raw, "")

            p_raw = p2g_person[i] if i < len(p2g_person) else "_"
            mots[i].p2g_personne = _person_map.get(p_raw, "")

            # Calculer les divergences
            mots[i].div_ortho = (
                mots[i].forme != mots[i].p2g_ortho
                and mots[i].p2g_ortho != ""
            )
            mots[i].div_pos = (
                mots[i].g2p_pos != mots[i].p2g_pos
                and mots[i].g2p_pos != ""
                and mots[i].p2g_pos != ""
            )
            mots[i].div_nombre = (
                mots[i].g2p_nombre != mots[i].p2g_nombre
                and mots[i].g2p_nombre != ""
                and mots[i].p2g_nombre != ""
            )
            mots[i].div_genre = (
                mots[i].g2p_genre != mots[i].p2g_genre
                and mots[i].g2p_genre != ""
                and mots[i].p2g_genre != ""
            )

        return mots

    # ==================================================================
    # Etape 3 : Corrections ciblees
    # ==================================================================

    def _v6_etape3_corrections(self, mots: list[MotV6]) -> list[Correction]:
        """Applique les corrections ciblees avec gardes strictes."""
        corrections: list[Correction] = []

        # P2G global — source de verite principale
        if self._config.activer_p2g_global:
            corrections.extend(self._corriger_p2g_global(mots))

        # 3a. Homophones via divergence P2G (meme POS)
        if self._config.activer_homophones_p2g:
            corrections.extend(self._corriger_homophones_v6(mots))

        # 3b. Accord morphologique (ADJ/PART nombre/genre)
        if self._config.activer_accords:
            corrections.extend(self._corriger_accords_v6(mots))

        # 3c. Participe passe (INF -> PP apres auxiliaire)
        corrections.extend(self._corriger_participe_v6(mots))

        # 3c-bis. PP accent (mange -> mange apres auxiliaire)
        corrections.extend(self._corriger_pp_accent(mots))

        # 3c-ter. INF -> PP apres auxiliaire avoir (manger -> mange)
        corrections.extend(self._corriger_inf_pp_avoir(mots))

        # 3c-quater. INF -> PP structurel (couche 2, sans P2G)
        corrections.extend(self._corriger_inf_pp_structurel(mots))

        # 3d. Correction verbe via P2G (div_pos acceptable quand homophones)
        if self._config.activer_verbe_p2g:
            corrections.extend(self._corriger_verbe_p2g(mots))

        # 3d-bis. Accord sujet-verbe (conjugaison via lexique)
        if self._config.activer_accord_sujet_verbe:
            corrections.extend(self._corriger_accord_sujet_verbe(mots))

        # 3d-ter. Infinitif -> conjugue apres pronom sujet
        if self._config.activer_accord_sujet_verbe:
            corrections.extend(self._corriger_infinitif_sujet(mots))

        # 3e. PP -> INF apres modal (travaillé -> travailler)
        corrections.extend(self._corriger_pp_inf_modal(mots))

        # 3e-bis. PP -> INF structurel (couche 2, sans P2G)
        corrections.extend(self._corriger_pp_inf_structurel(mots))

        # 3f. Accord DET-NOM (les chien -> les chiens)
        if self._config.activer_accord_det_nom:
            corrections.extend(self._corriger_accord_det_nom(mots))

        # 3f-bis. Accord NOM+ADJ en nombre
        if self._config.activer_accord_nom_adj:
            corrections.extend(self._corriger_accord_nom_adj(mots))

        # 3f-ter. Phase 2 — NOM generique via P2G
        if self._config.activer_phase2:
            corrections.extend(self._corriger_nom_p2g(mots))

        # 3f-quater. Phase 2 — ADJ generique via P2G
        if self._config.activer_phase2:
            corrections.extend(self._corriger_adj_p2g(mots))

        # 3g. Accord attribut a travers verbe d'etat
        if self._config.activer_accord_attribut:
            corrections.extend(self._corriger_accord_attribut(mots))

        # 3h. Accord PP + etre (sont arrive -> arrivees)
        if self._config.activer_pp_etre:
            corrections.extend(self._corriger_pp_etre(mots))

        # 3h-bis. PP+avoir genre invariable (a emergee -> emerge)
        if self._config.activer_pp_avoir_genre:
            corrections.extend(self._corriger_pp_avoir_genre(mots))

        # 3i. Accent via P2G (foret -> foret)
        if self._config.activer_accent_p2g:
            corrections.extend(self._corriger_accent_p2g(mots))

        # 3i-bis. Accent lexique fallback (pour mots in-lexique a basse freq
        # quand P2G ne predit pas la bonne forme)
        if self._config.activer_accent_lexique:
            corrections.extend(self._corriger_accent_lexique(mots))

        # 3j. Negation (il veut pas -> il ne veut pas)
        if self._config.activer_negation:
            corrections.extend(self._corriger_negation(mots))

        # 3k. Homophones structurels (regles sans P2G)
        if self._config.activer_homophones_struct:
            corrections.extend(self._corriger_homophones_structurels(mots))

        # 3l. Locutions figees plurielles
        corrections.extend(self._corriger_locutions_pluriel(mots))

        return corrections

    # ------------------------------------------------------------------
    # P2G global — source de verite principale
    # ------------------------------------------------------------------

    def _corriger_p2g_global(self, mots: list[MotV6]) -> list[Correction]:
        """Applique les predictions P2G comme source de verite.

        Corrige quand P2G predit une forme differente, dans le lexique,
        plausible par rapport a l'original, et avec confiance suffisante.
        """
        from lectura_correcteur.orthographe._suggestions import (
            _est_variante_accent,
            _est_doublement_consonne,
            _edit_distance_rapide,
        )

        corrections: list[Correction] = []
        lex = self._lexique

        for i, mv in enumerate(mots):
            # GARDE 0 — Exclusions de base
            if mv.regle:
                continue
            if not mv.div_ortho:
                continue
            p2g = mv.p2g_ortho
            if not p2g or p2g == mv.forme:
                continue
            forme_low = mv.forme.lower()
            if forme_low in _MOTS_PROTEGES or forme_low in _AUXILIAIRES:
                continue
            if mv.forme.endswith(("\u0027", "\u2019")):
                continue
            # Nom propre : forme originale commence par majuscule
            if mv.correction and mv.correction[0].isupper():
                continue
            # Garde ordinal : ne pas corriger "e" apres un chiffre ou nombre romain
            if forme_low == "e" and i > 0:
                prev_forme = mots[i - 1].forme
                if prev_forme[-1].isdigit() or prev_forme.upper() in _ROMAINS:
                    continue
            # Garde passe simple / subjonctif imparfait
            if forme_low in _PASSE_SIMPLE_SUBJ:
                continue
            # Garde passe simple heuristique : si la forme finit par un
            # suffixe de passe simple et existe comme VER dans le lexique
            if (forme_low.endswith(_PASSE_SIMPLE_SUFFIXES)
                    and lex.existe(forme_low)):
                _infos_ps = lex.info(forme_low)
                if any(e.get("cgram", "").startswith("VER") for e in _infos_ps):
                    continue
            # Garde contexte nom propre / mot etranger : mot court entre
            # mots capitalises (ex: "dei Carabinieri", "Gwenn ha du")
            if len(forme_low) <= 3 and not mv.forme[0].isupper():
                _prev_upper = (
                    i > 0 and mots[i - 1].correction
                    and mots[i - 1].correction[0].isupper()
                )
                _next_upper = (
                    i < len(mots) - 1 and mots[i + 1].correction
                    and mots[i + 1].correction[0].isupper()
                )
                if _prev_upper or _next_upper:
                    continue

            # Garde mot tres court (1-2 chars) : le P2G n'est pas fiable
            # (s->c, hi->i, mm->m, e->et). Les corrections d'accent
            # (a->à) sont traitees par d'autres regles.
            if len(forme_low) <= 2:
                continue

            # Garde trait d'union : ne pas supprimer les traits d'union
            # (in-folio->infolio, physico-chimiques->physicochimiques)
            if "-" in forme_low:
                _sans_tiret = forme_low.replace("-", "")
                _p2g_sans = p2g.lower().replace("-", "")
                if _sans_tiret == _p2g_sans:
                    continue
                # Compose avec "-" : si l'original a un tiret et le P2G
                # non (in-8->in, contre-amiraux->contramiraux), bloquer
                # quand la forme originale n'est pas dans le lexique
                if "-" not in p2g and not lex.existe(forme_low):
                    continue

            # GARDE 1 — La forme P2G doit etre dans le lexique
            if not lex.existe(p2g):
                continue

            # GARDE 2 — Correction plausible
            est_variante_acc = _est_variante_accent(mv.forme, p2g)
            est_ml = self._meme_lemme(mv.forme, p2g)
            dist = _edit_distance_rapide(mv.forme.lower(), p2g.lower())
            est_doublement = _est_doublement_consonne(mv.forme, p2g)

            if not (est_variante_acc or est_ml or dist <= 2 or est_doublement):
                continue

            # GARDE 2b — Bloquer les corrections entre lemmes differents
            # quand l'original est dans le lexique et plus frequent
            if lex.existe(mv.forme):
                freq_orig = lex.frequence(mv.forme) if hasattr(lex, "frequence") else 0.0
                freq_p2g = lex.frequence(p2g) if hasattr(lex, "frequence") else 0.0
                if not est_variante_acc and not est_ml:
                    # Lemmes differents : bloquer si original plus frequent
                    if freq_orig > 0 and freq_p2g <= freq_orig:
                        continue
                    # Mots courts (<=4) avec dist=2 : exiger ratio 10x
                    # (evite faim->fin, etc.)
                    if len(mv.forme) <= 4 and dist >= 2 and freq_orig > 0:
                        if freq_p2g < freq_orig * 10:
                            continue
                elif est_variante_acc and freq_orig >= 0.1:
                    # Variante accent, original en lexique avec freq mesurable :
                    # ratio adaptatif (5x si confiance >= 0.90, sinon 20x)
                    ratio_min = 5 if mv.p2g_confiance >= 0.90 else 20
                    if freq_p2g < freq_orig * ratio_min:
                        continue

            # GARDE 2d — Homophones de lemmes differents : seuil plus eleve
            # (peu->peut, pin->pain, cors->corps, ancre->encre)
            # Les homophones requierent une confiance P2G forte car le
            # contexte est ambigu par definition. On ne bloque PAS les
            # homophones de meme lemme (conjugaisons silencieuses).
            if not est_ml and lex.existe(mv.forme) and mv.g2p_phone:
                _phone_p2g = self._g2p_phone(p2g.lower())
                if _phone_p2g and mv.g2p_phone == _phone_p2g:
                    if mv.p2g_confiance < 0.95:
                        continue

            # GARDE 2e — Changement de genre seul sur mot existant
            # (transformees->transformes, connues->connus, publique->public)
            # p2g.global n'a pas le contexte pour decider le genre.
            if lex.existe(mv.forme) and _est_changement_genre(forme_low, p2g.lower()):
                continue

            # GARDE 2c — Meme lemme, meme nombre, personne differente
            # (ex: viens -> vient) : bloquer si original plus frequent
            if est_ml and lex.existe(mv.forme) and lex.existe(p2g):
                freq_orig = lex.frequence(mv.forme) if hasattr(lex, "frequence") else 0.0
                freq_p2g = lex.frequence(p2g) if hasattr(lex, "frequence") else 0.0
                if freq_orig > 0 and freq_p2g <= freq_orig:
                    # Meme lemme, original plus frequent : verifier si
                    # la correction est confirmee par le contexte
                    if not self._contexte_confirme_nombre_p2g(mots, i, mv, p2g):
                        continue

            # GARDE 3 — Filtrer nombre silencieux
            if self._est_changement_nombre_seul(mv.forme, p2g):
                if not self._contexte_confirme_nombre_p2g(mots, i, mv, p2g):
                    continue

            # GARDE 4 — Confiance
            seuil = 0.70 if est_variante_acc else 0.85
            if mv.p2g_confiance < seuil:
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = "p2g.global"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"P2G global: '{mv.forme}' -> '{p2g}' "
                    f"(confiance={mv.p2g_confiance:.2f})"
                ),
            ))

        return corrections

    def _meme_lemme(self, forme_a: str, forme_b: str) -> bool:
        """Verifie si deux formes partagent un lemme dans le lexique."""
        lex = self._lexique
        infos_a = lex.info(forme_a)
        infos_b = lex.info(forme_b)
        if not infos_a or not infos_b:
            return False
        lemmes_a = {e.get("lemme") for e in infos_a if e.get("lemme")}
        lemmes_b = {e.get("lemme") for e in infos_b if e.get("lemme")}
        return bool(lemmes_a & lemmes_b)

    def _est_changement_nombre_seul(self, forme: str, p2g: str) -> bool:
        """True si le seul changement est l'ajout/retrait de 's' ou 'x' final.

        Les deux formes doivent exister dans le lexique pour que ce soit
        un vrai changement de nombre silencieux (pas une faute d'ortho).
        """
        lex = self._lexique
        # Les deux formes doivent exister
        if not lex.existe(forme) or not lex.existe(p2g):
            return False

        f = forme.lower()
        p = p2g.lower()

        # Ajout/retrait de 's' final
        if p == f + "s" or f == p + "s":
            return True
        # Ajout/retrait de 'x' final (pour -eau/-aux, -eu/-eux, etc.)
        if p == f + "x" or f == p + "x":
            return True
        # Variante -al/-aux
        if (f.endswith("al") and p == f[:-2] + "aux") or \
           (p.endswith("al") and f == p[:-2] + "aux"):
            return True

        # Meme lemme avec changement de nombre seulement
        infos_f = lex.info(forme)
        infos_p = lex.info(p2g)
        if infos_f and infos_p:
            lemmes_f = {e.get("lemme") for e in infos_f if e.get("lemme")}
            lemmes_p = {e.get("lemme") for e in infos_p if e.get("lemme")}
            if lemmes_f & lemmes_p:
                # Meme lemme — verifier que seul le nombre change
                nombres_f = {e.get("nombre") for e in infos_f if e.get("nombre")}
                nombres_p = {e.get("nombre") for e in infos_p if e.get("nombre")}
                if nombres_f != nombres_p:
                    return True

        return False

    def _contexte_confirme_nombre_p2g(
        self,
        mots: list[MotV6],
        idx: int,
        mv: MotV6,
        p2g: str,
    ) -> bool:
        """Verifie qu'un DET ou le contexte confirme le nombre predit par P2G."""
        # Determiner le nombre de la FORME P2G (pas mv.p2g_nombre qui peut
        # etre incoherent avec la forme proposee)
        lex = self._lexique
        nombre_p2g = ""
        infos_p2g = lex.info(p2g)
        if infos_p2g:
            nombres = {e.get("nombre") for e in infos_p2g if e.get("nombre")}
            if len(nombres) == 1:
                nombre_p2g = nombres.pop()
        # Fallback sur mv.p2g_nombre si lexique ne donne rien
        if not nombre_p2g:
            nombre_p2g = mv.p2g_nombre
        if not nombre_p2g:
            return False

        # Chercher un determinant dans les 2 mots precedents
        for j in range(max(0, idx - 2), idx):
            det = mots[j].forme.lower()
            det_corr = mots[j].correction.lower() if mots[j].correction else det
            for d in (det, det_corr):
                nb_det = self._DET_NOMBRE.get(d, "")
                if nb_det and nb_det == nombre_p2g:
                    return True
                if nombre_p2g == "p" and d in _NUMERAUX_PLURIEL:
                    return True

        # Extension PREP : si mot precedent est une preposition et confiance
        # P2G >= 0.90, accepter le changement de nombre. Les constructions
        # PREP + NOM_nombre sont rarement ambigues a haute confiance.
        _PREPS_NOMBRE = frozenset({
            "en", "par", "pour", "sans", "avec", "sous", "sur",
            "entre", "vers", "contre", "depuis", "durant",
        })
        if idx >= 1 and mv.p2g_confiance >= 0.90:
            prev_low = mots[idx - 1].forme.lower()
            if prev_low in _PREPS_NOMBRE:
                # Exclure preps partitives "de/d'" quand nombre_cible = "s"
                # (le singulier partitif apres "de" est souvent correct)
                # Note: "de/d'" ne sont pas dans _PREPS_NOMBRE
                # Verifier que la forme P2G a le bon nombre dans le lexique
                if infos_p2g:
                    for e in infos_p2g:
                        e_n = e.get("nombre", "")
                        if e_n == nombre_p2g:
                            return True

        return False

    # ------------------------------------------------------------------
    # 3a. Homophones
    # ------------------------------------------------------------------

    def _corriger_homophones_v6(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige les homophones via divergence P2G.

        Declencheur : div_ortho=True ET forme et p2g_ortho sont homophones.

        6 gardes obligatoires :
        1. Le mot P2G doit exister dans le lexique
        2. p2g_confiance >= homophone_confiance_min
        3. G2P et P2G s'accordent sur le POS de la correction (pas de div_pos)
        4. Pas de correction sur mots < 2 chars sauf whitelist
        5. La correction doit etre dans les p2g_alternatives top-k
        6. Si le mot original existe avec le meme POS que G2P -> ne pas corriger
        """
        corrections: list[Correction] = []
        lex = self._lexique
        cfg = self._config

        for i, mv in enumerate(mots):
            # Declencheur : divergence orthographique
            if not mv.div_ortho:
                continue

            forme = mv.forme
            p2g = mv.p2g_ortho

            if not p2g or forme == p2g:
                continue

            # Garde 0a : pas de correction sur tokens elides (d', l', qu', etc.)
            if forme.endswith("'") or forme.endswith("\u2019"):
                continue

            # Garde 0b : pas de correction sur noms propres (majuscule initiale)
            if mv.correction and mv.correction[0].isupper():
                continue

            # Garde ordinal : ne pas corriger "e" apres un chiffre ou nombre romain
            if forme.lower() == "e" and i > 0:
                prev_forme = mots[i - 1].forme
                if prev_forme[-1].isdigit() or prev_forme.upper() in _ROMAINS:
                    continue

            # Garde passe simple / subjonctif imparfait
            forme_low_h = forme.lower()
            if forme_low_h in _PASSE_SIMPLE_SUBJ:
                continue
            # Garde passe simple heuristique : si la forme finit par un
            # suffixe de passe simple et existe comme VER dans le lexique
            if (forme_low_h.endswith(_PASSE_SIMPLE_SUFFIXES)
                    and lex.existe(forme_low_h)):
                _infos_ps = lex.info(forme_low_h)
                if any(e.get("cgram", "").startswith("VER") for e in _infos_ps):
                    continue
            # Garde contexte nom propre / mot etranger : mot court entre
            # mots capitalises (ex: "dei Carabinieri", "Gwenn ha du")
            if len(forme_low_h) <= 3 and not forme[0].isupper():
                _prev_upper = (
                    i > 0 and mots[i - 1].correction
                    and mots[i - 1].correction[0].isupper()
                )
                _next_upper = (
                    i < len(mots) - 1 and mots[i + 1].correction
                    and mots[i + 1].correction[0].isupper()
                )
                if _prev_upper or _next_upper:
                    continue

            # Garde 0c : pas de correction sur auxiliaires/modaux (trop risque)
            # Exception : homophones structurels (est→et, sont→son, ont→on)
            if forme.lower() in _AUXILIAIRES or forme.lower() in _MODAUX:
                paire_0c = (forme.lower(), p2g.lower())
                if paire_0c not in _HOMOPHONES_POS_DIVERGENT:
                    continue

            # Garde 0d : pas de correction sur mots proteges
            # Exception : homophones structurels
            if forme.lower() in _MOTS_PROTEGES:
                paire_0d = (forme.lower(), p2g.lower())
                if paire_0d not in _HOMOPHONES_POS_DIVERGENT:
                    continue

            # Verifier que forme et p2g_ortho sont homophones (meme phone)
            if mv.g2p_phone:
                phone_p2g = self._g2p_phone(p2g)
                if phone_p2g and phone_p2g != mv.g2p_phone:
                    continue  # Pas homophones

            # Garde 1 : le mot P2G doit exister dans le lexique
            if not lex.existe(p2g):
                continue

            # Garde 2 : confiance P2G suffisante
            if mv.p2g_confiance < cfg.homophone_confiance_min:
                continue

            # Garde 3 : G2P et P2G s'accordent sur le POS de la correction
            # On verifie que le P2G POS est coherent avec le G2P POS
            # (pas de div_pos signifie accord)
            # Exception : les homophones structurels ou div_pos EST le signal
            is_structural_homo = False
            if mv.div_pos:
                paire = (forme.lower(), p2g.lower())
                if paire not in _HOMOPHONES_POS_DIVERGENT:
                    continue
                # Pour les homophones structurels, verifier le contexte
                if not self._contexte_confirme_homo_structurel(mots, mv, p2g):
                    continue
                is_structural_homo = True

            # Garde 4 : pas de correction sur mots < 2 chars sauf whitelist
            if len(forme) < 2 and forme not in _HOMOPHONES_COURTS_WHITELIST:
                continue

            # Garde 5 : la correction doit etre dans les alternatives top-k
            alts = mv.p2g_alternatives
            if alts:
                alt_formes = set()
                for alt in alts[:cfg.homophone_top_k]:
                    if isinstance(alt, dict):
                        alt_formes.add(alt.get("ortho", ""))
                    elif isinstance(alt, (list, tuple)) and len(alt) >= 1:
                        alt_formes.add(alt[0])
                    elif isinstance(alt, str):
                        alt_formes.add(alt)
                if p2g not in alt_formes:
                    continue

            # Garde 6a : le mot original doit exister dans le lexique
            # (les OOV sont corriges par l'etape 1 ortho, pas par homophones)
            if not lex.existe(forme):
                continue

            # Garde 6b : pas de correction sur tokens avec tiret (composes)
            if "-" in forme:
                continue

            # Garde 6c : verifier meme lemme (independant du POS,
            # car G2P peut tagger un PP comme ADJ alors que le lexique dit VER)
            # Skip pour homophones structurels (lemmes intentionnellement differents)
            if not is_structural_homo and lex.existe(p2g):
                infos_forme = lex.info(forme)
                infos_p2g = lex.info(p2g)
                lemmes_forme = {
                    e.get("lemme") for e in infos_forme if e.get("lemme")
                }
                lemmes_p2g = {
                    e.get("lemme") for e in infos_p2g if e.get("lemme")
                }
                if lemmes_forme & lemmes_p2g:
                    # Meme lemme = variante morphologique
                    # Autoriser SEULEMENT les changements de nombre
                    # confirmes par le contexte ET contredisant l'original
                    p2g_nombre = mv.p2g_nombre
                    g2p_nombre = mv.g2p_nombre
                    if p2g_nombre and g2p_nombre and p2g_nombre != g2p_nombre:
                        # Le contexte doit contredire l'original
                        if self._contexte_confirme_nombre(
                            mots, mv.index, g2p_nombre,
                        ):
                            continue  # contexte confirme l'original
                        # Le contexte doit confirmer la correction
                        if not self._contexte_confirme_nombre(
                            mots, mv.index, p2g_nombre,
                        ):
                            continue  # contexte ne confirme pas P2G
                    else:
                        # Pas de changement de nombre (genre seul, etc.)
                        continue

            # Garde 6d : si le mot original existe dans le lexique avec le
            # meme POS que G2P, bloquer (lemmes differents, meme POS)
            # Skip pour homophones structurels (POS intentionnellement differents)
            if not is_structural_homo and mv.g2p_pos:
                infos_forme = lex.info(forme) if lex.existe(forme) else []
                cgrams_forme = {e.get("cgram", "") for e in infos_forme}
                g2p_prefix = mv.g2p_pos.split(":")[0]
                pos_match = any(
                    c == mv.g2p_pos
                    or c.startswith(g2p_prefix)
                    or g2p_prefix.startswith(c.split(":")[0])
                    for c in cgrams_forme if c
                )
                if pos_match:
                    continue

            # Garde 6e : ratio de frequence — bloquer si le candidat P2G
            # est nettement plus rare que l'original (consulta 0.20 > consultat 0.01)
            if not is_structural_homo:
                def _max_freq(mot: str) -> float:
                    if not lex.existe(mot):
                        return 0.0
                    return max(
                        (e.get("freq") or 0.0 for e in lex.info(mot)),
                        default=0.0,
                    )
                freq_orig = _max_freq(forme)
                freq_p2g = _max_freq(p2g)
                if freq_orig > 0 and freq_p2g < freq_orig / 5:
                    continue

            # Garde 6f : POS lexique disjoints — si l'original et le P2G
            # existent dans le lexique avec des POS completement differents,
            # le P2G propose un mot d'une autre categorie (prit VER → prix NOM)
            if not is_structural_homo and lex.existe(forme) and lex.existe(p2g):
                pos_orig = {
                    e.get("cgram", "").split(":")[0]
                    for e in lex.info(forme) if e.get("cgram")
                }
                pos_p2g = {
                    e.get("cgram", "").split(":")[0]
                    for e in lex.info(p2g) if e.get("cgram")
                }
                if pos_orig and pos_p2g and not (pos_orig & pos_p2g):
                    continue  # POS completement disjoints

            # Garde 6g : meme POS, formes proches dans le lexique —
            # si les deux existent avec le meme POS et des frequences
            # proches, ne pas corriger (coralliens ≠ coraliens, ratio 2.7×)
            # Seuil : le P2G doit etre au moins 3× plus frequent pour justifier
            # le changement d'une forme valide en lexique.
            if not is_structural_homo and lex.existe(forme) and lex.existe(p2g):
                pos_orig_g = {
                    e.get("cgram", "").split(":")[0]
                    for e in lex.info(forme) if e.get("cgram")
                }
                pos_p2g_g = {
                    e.get("cgram", "").split(":")[0]
                    for e in lex.info(p2g) if e.get("cgram")
                }
                if pos_orig_g & pos_p2g_g:
                    # Meme categorie POS : bloquer sauf si ratio >= 3×
                    freq_o = _max_freq(forme)
                    freq_p = _max_freq(p2g)
                    if freq_o > 0 and freq_p < freq_o * 3:
                        continue

            # Toutes les gardes passees : appliquer la correction
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = f"homophone.p2g.{forme}_{p2g}"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"Homophone P2G: '{forme}' -> '{p2g}' "
                    f"(confiance={mv.p2g_confiance:.2f})"
                ),
            ))

        return corrections

    def _contexte_confirme_nombre(
        self,
        mots: list[MotV6],
        idx: int,
        nombre_attendu: str,
    ) -> bool:
        """Verifie que le contexte (determinant/quantifieur) confirme le nombre.

        Pour eviter les FP sur variantes morphologiques (marche/marches),
        on exige que le determinant ou article precedant le nom soit coherent
        avec le nombre predit par le P2G.

        Args:
            mots: liste complete des MotV6
            idx: index du mot a verifier
            nombre_attendu: "s" (singulier) ou "p" (pluriel) du P2G

        Returns:
            True si le contexte confirme le nombre attendu.
        """
        if not nombre_attendu:
            return False

        # Determinants/articles singuliers et pluriels
        _DET_SING = frozenset({
            "le", "la", "l'", "un", "une", "du", "au",
            "ce", "cet", "cette", "mon", "ma", "ton", "ta", "son", "sa",
            "notre", "votre", "leur",
            "quel", "quelle", "chaque", "tout", "toute",
        })
        _DET_PLUR = frozenset({
            "les", "des", "aux",
            "ces", "mes", "tes", "ses", "nos", "vos", "leurs",
            "quelques", "plusieurs", "tous", "toutes",
            "quels", "quelles",
        })

        # Chercher un determinant ou numeral dans les 2 mots precedents
        for j in range(max(0, idx - 2), idx):
            det = mots[j].forme.lower()
            # Aussi verifier la correction du determinant (peut avoir deja ete corrige)
            det_corr = mots[j].correction.lower() if mots[j].correction else det

            for d in (det, det_corr):
                if nombre_attendu == "p" and d in _DET_PLUR:
                    return True
                if nombre_attendu == "p" and d in _NUMERAUX_PLURIEL:
                    return True
                if nombre_attendu == "s" and d in _DET_SING:
                    return True

        return False

    def _contexte_confirme_homo_structurel(
        self, mots: list[MotV6], mv: MotV6, p2g: str,
    ) -> bool:
        """Verifie le contexte pour les homophones structurels.

        Ces homophones ont des POS differents par nature (et/est, a/a, etc.).
        Des gardes strictes par paire evitent les FP.

        Retourne True si le contexte confirme la correction P2G.
        """
        forme = mv.forme.lower()
        cible = p2g.lower()
        idx = mv.index

        # Pronoms sujets connus
        _PRO_SUJETS = frozenset({
            "je", "j'", "j\u2019", "tu", "il", "elle", "on",
            "nous", "vous", "ils", "elles",
        })

        # --- et -> est : CON -> AUX (verbe etre) ---
        # "il et grand" -> "il est grand"
        # Confirme si un pronom sujet ou NOM precede directement
        if forme == "et" and cible == "est":
            if idx > 0:
                prev_low = mots[idx - 1].forme.lower()
                # Pronom sujet + et -> est : "il et" -> "il est"
                if prev_low in ("il", "elle", "on", "ce", "c'", "c\u2019", "tout"):
                    return True
            # Contexte ADJ apres : "X et grand" -> "X est grand"
            # Seulement si ce n'est PAS une coordination (ADJ et ADJ, NOM et NOM)
            if idx > 0 and idx < len(mots) - 1:
                prev_pos = mots[idx - 1].g2p_pos or ""
                next_pos = mots[idx + 1].g2p_pos or ""
                if next_pos == "ADJ" and not prev_pos.startswith("ADJ"):
                    return True
            return False

        # --- est -> et : AUX -> CON ---
        # "noir est blanc" -> "noir et blanc"
        # Confirme si entoure de deux ADJ/NOM (coordination)
        if forme == "est" and cible == "et":
            if idx > 0 and idx < len(mots) - 1:
                prev_pos = mots[idx - 1].g2p_pos or ""
                next_pos = mots[idx + 1].g2p_pos or ""
                # Deux ADJ ou deux NOM coordonnes
                if (prev_pos.startswith("ADJ") and next_pos.startswith("ADJ")):
                    return True
                if (prev_pos.startswith("NOM") and next_pos.startswith("NOM")):
                    return True
            return False

        # --- a -> à : VER -> PRE ---
        # Confirme si le contexte indique clairement une preposition
        if forme == "a" and cible == "\u00e0":
            if idx == 0 or idx >= len(mots) - 1:
                return False
            prev_low = mots[idx - 1].forme.lower()
            prev_corr = (mots[idx - 1].correction.lower()
                         if mots[idx - 1].correction else prev_low)
            # Bloquer si precede par un pronom sujet → auxiliaire avoir
            if prev_low in _PRO_SUJETS or prev_corr in _PRO_SUJETS:
                return False
            # Bloquer si precede par ne/n' → auxiliaire (n'a pas)
            if prev_low in ("n'", "n\u2019", "ne"):
                return False
            # Bloquer si precede par "qui" → pronom relatif sujet
            if prev_low == "qui":
                return False
            # Confirmer si suivi par un article/determinant → preposition
            next_low = mots[idx + 1].forme.lower()
            _DETERMINANTS = frozenset({
                "la", "le", "l'", "l\u2019", "les",
                "un", "une", "des", "du", "au", "aux",
                "mon", "ton", "son", "ma", "ta", "sa",
                "mes", "tes", "ses", "nos", "vos", "leur", "leurs",
                "ce", "cette", "cet", "ces",
            })
            if next_low in _DETERMINANTS:
                return True
            return False

        # --- a -> a : PRE -> VER ---
        # "elle à mange" -> "elle a mange"
        # Confirme si un pronom sujet precede
        if forme == "\u00e0" and cible == "a":
            if idx > 0:
                prev_low = mots[idx - 1].forme.lower()
                if prev_low in ("il", "elle", "on", "qui"):
                    return True
            return False

        # --- son -> sont : DET -> AUX ---
        # "ils son partis" -> "ils sont partis"
        # Confirme si un pronom sujet pluriel precede
        if forme == "son" and cible == "sont":
            for j in range(max(0, idx - 2), idx):
                prev_low = mots[j].forme.lower()
                if prev_low in ("ils", "elles"):
                    return True
            return False

        # --- sont -> son : AUX -> DET ---
        # "avec sont velo" -> "avec son velo"
        # Confirme si un NOM suit directement
        if forme == "sont" and cible == "son":
            if idx < len(mots) - 1:
                next_pos = mots[idx + 1].g2p_pos or ""
                if next_pos.startswith("NOM"):
                    return True
            return False

        # --- on -> ont : PRO -> AUX ---
        # "les cuisiniers on prepare" -> "ont prepare"
        # Confirme si un NOM pluriel ou pronom pluriel precede + VER/PP suit
        if forme == "on" and cible == "ont":
            has_plural_subject = False
            subject_idx: int | None = None
            for j in range(idx - 1, max(-1, idx - 4), -1):
                # Recherche du plus proche au plus loin
                if j < 0:
                    break
                mj = mots[j]
                mj_low = mj.forme.lower()
                if mj_low in ("ils", "elles"):
                    has_plural_subject = True
                    subject_idx = j
                    break
                if mj.g2p_pos and mj.g2p_pos.startswith("NOM"):
                    nom_nombre = mj.g2p_nombre or mj.p2g_nombre
                    if nom_nombre == "p":
                        has_plural_subject = True
                        subject_idx = j
                        break
            # Barriere : si une preposition, pronom relatif ou verbe
            # conjugue se trouve entre le sujet candidat et "on",
            # c'est une autre clause (ex: "ils sont delicats; on")
            if has_plural_subject and subject_idx is not None:
                for j in range(subject_idx + 1, idx):
                    pos_j = mots[j].g2p_pos or ""
                    if pos_j.startswith("PRE") or pos_j.startswith("PRO:rel"):
                        has_plural_subject = False
                        break
                    # Verbe conjugue = frontiere de clause
                    if pos_j == "AUX" or (
                        pos_j.startswith("VER")
                        and pos_j not in ("VER:infi", "VER:ppre", "VER:pper")
                    ):
                        has_plural_subject = False
                        break
            if has_plural_subject:
                return True
            return False

        # --- ont -> on : AUX -> PRO ---
        # "ont a gagne" -> "on a gagne"
        # Confirme si suivi d'un verbe singulier 3p
        if forme == "ont" and cible == "on":
            if idx < len(mots) - 1:
                next_mv = mots[idx + 1]
                next_low = next_mv.forme.lower()
                # "ont a" -> "on a" : suivi d'auxiliaire singulier
                if next_low in ("a", "est", "va", "peut", "doit", "veut"):
                    return True
            return False

        # --- ou -> où : CON -> ADV (interrogatif/relatif) ---
        # "ou vas-tu" -> "où vas-tu"
        # "la ville ou il habite" -> "où il habite"
        # Confirme si debut de phrase ou apres virgule, suivi d'un verbe
        if forme == "ou" and cible == "o\u00f9":
            # Debut de phrase + suivi d'un verbe/pronom
            if idx == 0 and len(mots) > 1:
                next_low = mots[idx + 1].forme.lower()
                next_pos = mots[idx + 1].g2p_pos or ""
                if (next_pos.startswith("VER") or next_pos == "AUX"
                        or next_low in ("vas", "va", "est", "sont",
                                        "allez", "allons", "es",
                                        "habites", "habite", "habitez")):
                    return True
            return False

        # --- où -> ou : ADV -> CON ---
        if forme == "o\u00f9" and cible == "ou":
            return False

        # --- se -> ce : PRO -> DET ---
        # "se garcon" -> "ce garcon"
        # Confirme si suivi d'un NOM (pas d'un verbe)
        if forme == "se" and cible == "ce":
            if idx < len(mots) - 1:
                next_pos = mots[idx + 1].g2p_pos or ""
                if next_pos.startswith("NOM"):
                    return True
            return False

        # --- ce -> se : DET -> PRO ---
        # Tres risque, desactive
        if forme == "ce" and cible == "se":
            return False

        return False

    # ------------------------------------------------------------------
    # 3b. Accords morphologiques
    # ------------------------------------------------------------------

    def _corriger_accords_v6(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige les accords morphologiques (nombre/genre) ADJ/PART apres NOM.

        Declencheur : div_nombre=True ou div_genre=True pour un mot
        dont le G2P POS est ADJ ou participe, apres un NOM gouverneur.

        Gardes :
        1. Le NOM gouverneur doit etre adjacent (fenetre accord_fenetre)
        2. G2P et P2G s'accordent sur nombre/genre du NOM gouverneur
        3. La forme corrigee doit exister dans le lexique
        4. Pas de correction si ambiguite (ex: "grand" peut etre ADV)
        """
        corrections: list[Correction] = []
        lex = self._lexique
        cfg = self._config
        n = len(mots)

        for i, mv in enumerate(mots):
            # Pas de correction sur mots proteges
            if mv.forme.lower() in _MOTS_PROTEGES:
                continue

            # Garde nom propre : ADJ capitalise (pas en debut de phrase)
            # = probablement partie d'un nom compose (Grande ecole, etc.)
            # NB: mv.forme est lowercase, mv.correction preserve la casse originale.
            if mv.correction and mv.correction[0].isupper() and i > 0:
                prev_corr = mots[i - 1].correction if i > 0 else ""
                if prev_corr and prev_corr[-1] not in ".!?…":
                    continue

            # Le mot doit etre ADJ (strict) — pas les determinants/pronoms
            pos = mv.g2p_pos
            if not pos:
                continue
            is_adj = pos == "ADJ"
            if not is_adj:
                continue

            # Le mot doit aussi etre ADJ dans le lexique (pas seulement G2P)
            # Evite les NOM/VER mal tagges par G2P comme ADJ
            if lex.existe(mv.forme):
                infos_lex = lex.info(mv.forme)
                has_adj_lex = any(
                    e.get("cgram", "").startswith("ADJ") for e in infos_lex
                )
                if not has_adj_lex:
                    continue

            # Chercher le NOM gouverneur (fenetre=1 pour precision)
            nom_mv = self._trouver_nom_gouverneur(mots, i, 1)
            if nom_mv is None:
                continue

            # Garde coordination : si "et" est dans les 5 mots avant l'ADJ,
            # et qu'on veut passer du pluriel au singulier, l'ADJ peut
            # accorder avec un sujet compose (pluriel)
            # "des bottes et un long manteau noirs" -> noirs est correct
            nom_nombre = nom_mv.g2p_nombre or nom_mv.p2g_nombre
            if nom_nombre == "s":
                has_et = any(
                    mots[k].forme.lower() == "et"
                    for k in range(max(0, i - 5), i)
                )
                if has_et:
                    continue

            # Garde 2 : G2P et P2G s'accordent sur nombre/genre du NOM
            if nom_mv.g2p_nombre and nom_mv.p2g_nombre:
                if nom_mv.g2p_nombre != nom_mv.p2g_nombre:
                    continue  # Desaccord sur le NOM -> pas fiable
            if nom_mv.g2p_genre and nom_mv.p2g_genre:
                if nom_mv.g2p_genre != nom_mv.p2g_genre:
                    continue

            # Detecter les divergences (intra-mot ou cross-mot ADJ vs NOM)
            nom_genre = nom_mv.g2p_genre or nom_mv.p2g_genre
            adj_genre = mv.g2p_genre or mv.p2g_genre
            cross_genre_mismatch = (
                nom_genre and adj_genre
                and nom_genre != adj_genre
                and nom_mv.g2p_genre == nom_mv.p2g_genre  # NOM fiable
            )

            # Deja corrige par une etape precedente (ex: P2G) ?
            # Laisser passer si cross-genre mismatch detecte : l'accord
            # peut amender la correction P2G avec le bon genre.
            if mv.regle and not cross_genre_mismatch:
                continue

            # Declencheur : divergence nombre, genre intra-mot, ou cross-mot
            if not mv.div_nombre and not mv.div_genre and not cross_genre_mismatch:
                continue

            # Garde cross-genre : si le seul declencheur est cross_genre_mismatch
            # (pas de divergence intra-mot), ne pas corriger.
            # Les erreurs de genre sont tres rares en francais — c'est presque
            # toujours un mauvais NOM gouverneur (latérales =/= latéraux).
            if cross_genre_mismatch and not mv.div_nombre and not mv.div_genre:
                continue

            # Garde 2b : le nombre du NOM doit etre confirme par un determinant
            nom_nombre = nom_mv.g2p_nombre or nom_mv.p2g_nombre
            if nom_nombre and not self._contexte_confirme_nombre(
                mots, nom_mv.index, nom_nombre,
            ):
                continue

            # Determiner le nombre cible (du NOM gouverneur)
            nombre_cible = nom_mv.g2p_nombre or nom_mv.p2g_nombre

            # Determiner le genre cible
            genre_cible = ""
            if mv.div_genre or cross_genre_mismatch:
                # Utiliser le genre du NOM (deja valide fiable ci-dessus)
                if nom_genre:
                    genre_cible = nom_genre
            if not genre_cible:
                genre_cible = adj_genre or ""

            if not nombre_cible and not genre_cible:
                continue

            # Si pas de nombre_cible, garder le nombre actuel de l'ADJ
            if not nombre_cible:
                nombre_cible = mv.g2p_nombre or mv.p2g_nombre or ""

            # Chercher la forme corrigee dans le lexique
            forme_corrigee = self._trouver_forme_accord(
                mv.forme, pos, nombre_cible, genre_cible,
            )
            if forme_corrigee is None or forme_corrigee == mv.forme:
                continue

            # Garde 3 : la forme corrigee doit exister dans le lexique
            if not lex.existe(forme_corrigee):
                continue

            # Garde 4 : pas de correction si le mot est ambigu
            if self._est_ambigu_accord(mv.forme, pos):
                continue

            # Garde 5 : verifier que l'ADJ dans le lexique n'est PAS deja
            # accorde avec le NOM cible (evite FP sur texte propre)
            if lex.existe(mv.forme):
                adj_infos = lex.info(mv.forme)
                adj_deja_accorde = False
                for entry in adj_infos:
                    if not entry.get("cgram", "").startswith("ADJ"):
                        continue
                    from lectura_correcteur._utils import normaliser_morpho
                    e_n = normaliser_morpho(entry.get("nombre", ""))
                    e_g = normaliser_morpho(entry.get("genre", ""))
                    # Fallback Multext si champs vides ou invalides
                    if e_g not in ("m", "f") or e_n not in ("s", "p"):
                        mt_g, mt_n = self._genre_nombre_from_multext(
                            entry.get("multext", ""),
                        )
                        if e_g not in ("m", "f"):
                            e_g = mt_g
                        if e_n not in ("s", "p"):
                            e_n = mt_n
                    n_ok = not nombre_cible or e_n == nombre_cible
                    g_ok = not genre_cible or e_g == genre_cible
                    if n_ok and g_ok:
                        adj_deja_accorde = True
                        break
                if adj_deja_accorde:
                    continue  # ADJ deja correctement accorde

            # Garde 5b : si G2P predit un nombre pour l'ADJ qui differe
            # du NOM gouverneur, le NOM gouverneur est probablement le
            # mauvais (hors fenetre). Ne pas corriger.
            # Exception : si div_nombre est actif, le desaccord est attendu.
            adj_nombre_g2p = mv.g2p_nombre
            if (adj_nombre_g2p and adj_nombre_g2p != nombre_cible
                    and not mv.div_nombre):
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, forme_corrigee)
            mv.regle = "accord.nombre_genre"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"Accord: '{mv.forme}' -> '{forme_corrigee}' "
                    f"(nombre={nombre_cible}, genre={genre_cible})"
                ),
            ))

        return corrections

    def _trouver_nom_gouverneur(
        self,
        mots: list[MotV6],
        idx: int,
        fenetre: int,
    ) -> MotV6 | None:
        """Cherche le NOM gouverneur le plus proche dans la fenetre a gauche.

        Gardes supplementaires :
        - Ignore les NOM qui sont dans un syntagme prepositionnel
        - Ignore les NOM precedes d'un verbe (sujet d'une autre proposition)
        """
        _PREPOSITIONS = frozenset({
            "de", "du", "des", "d'", "d\u2019",
            "en", "\u00e0", "au", "aux", "par", "pour",
            "avec", "sans", "chez", "vers", "contre",
        })
        # Chercher du plus proche au plus loin
        for j in range(idx - 1, max(-1, idx - fenetre - 1), -1):
            if j < 0:
                break
            mv = mots[j]
            if not mv.g2p_pos or not mv.g2p_pos.startswith("NOM"):
                continue
            # Ignorer les NOM dans un syntagme prepositionnel
            if j > 0 and mots[j - 1].forme.lower() in _PREPOSITIONS:
                continue
            # Aussi detecter prep + article + NOM ("a la fois", "du NOM")
            if j > 1 and mots[j - 1].forme.lower() in (
                "la", "le", "les", "l'", "l\u2019",
            ):
                if mots[j - 2].forme.lower() in _PREPOSITIONS:
                    continue
            return mv
        return None

    @staticmethod
    def _genre_nombre_from_multext(multext: str) -> tuple[str, str]:
        """Extrait genre et nombre depuis un tag Multext pour ADJ/NOM.

        Multext ADJ : A + type + [degree] + genre(m/f) + nombre(s/p)
        Multext NOM : N + type + genre(m/f) + nombre(s/p)

        Genre et nombre sont toujours les 2 derniers caracteres du tag.
        Retourne (genre, nombre) normalises ('m'/'f', 's'/'p') ou ''.
        """
        if not multext or len(multext) < 3:
            return ("", "")
        # Les 2 derniers chars sont genre + nombre
        genre_ch = multext[-2]
        nombre_ch = multext[-1]
        genre = genre_ch if genre_ch in ("m", "f") else ""
        nombre = nombre_ch if nombre_ch in ("s", "p") else ""
        return (genre, nombre)

    def _trouver_forme_accord(
        self,
        forme: str,
        pos: str,
        nombre_cible: str,
        genre_cible: str,
    ) -> str | None:
        """Cherche la forme accordee dans le lexique.

        Utilise formes_de(lemme) si disponible (retourne toutes les flexions),
        sinon fallback sur info(lemme).
        """
        lex = self._lexique
        from lectura_correcteur._utils import normaliser_morpho

        # Ne pas modifier les noms invariables en "s"
        if forme.lower() in _INVARIABLES_S:
            return None

        # Obtenir le lemme du mot
        infos = lex.info(forme)
        if not infos:
            return None

        lemmes = {e.get("lemme", "") for e in infos if e.get("lemme")}
        if not lemmes:
            return None

        for lemme in lemmes:
            if not lemme:
                continue

            # Methode 1 : formes_de(lemme) — retourne toutes les flexions
            if hasattr(lex, "formes_de"):
                try:
                    flexions = lex.formes_de(lemme)
                    for entry in flexions:
                        entry_pos = entry.get("cgram", "")
                        if not entry_pos.startswith(pos[:3]):
                            continue
                        entry_nombre = normaliser_morpho(
                            entry.get("nombre", ""),
                        )
                        entry_genre = normaliser_morpho(
                            entry.get("genre", ""),
                        )
                        # Fallback Multext si genre/nombre vides ou invalides
                        genre_valide = entry_genre in ("m", "f")
                        nombre_valide = entry_nombre in ("s", "p")
                        if not genre_valide or not nombre_valide:
                            mt_g, mt_n = self._genre_nombre_from_multext(
                                entry.get("multext", ""),
                            )
                            if not genre_valide:
                                entry_genre = mt_g
                            if not nombre_valide:
                                entry_nombre = mt_n
                        nombre_ok = not nombre_cible or entry_nombre == nombre_cible
                        genre_ok = not genre_cible or entry_genre == genre_cible
                        if nombre_ok and genre_ok:
                            entry_forme = entry.get("ortho", "")
                            if entry_forme and entry_forme != forme:
                                return entry_forme
                except Exception:
                    pass

            # Methode 2 : fallback info(lemme)
            infos_lemme = lex.info(lemme)
            for entry in infos_lemme:
                entry_pos = entry.get("cgram", "")
                if not entry_pos.startswith(pos[:3]):
                    continue
                entry_nombre = entry.get("nombre", "")
                entry_genre = entry.get("genre", "")
                nombre_ok = not nombre_cible or entry_nombre == nombre_cible
                genre_ok = not genre_cible or entry_genre == genre_cible
                if nombre_ok and genre_ok:
                    entry_forme = entry.get("forme", entry.get("ortho", ""))
                    if entry_forme and entry_forme != forme:
                        return entry_forme

        # Strategie simple : essayer les suffixes courants
        if nombre_cible == "p" and not forme.endswith("s"):
            candidat = forme + "s"
            if lex.existe(candidat):
                return candidat
        if nombre_cible == "s" and forme.endswith("s") and len(forme) > 2:
            candidat = forme[:-1]
            if lex.existe(candidat):
                # Verifier que c'est le meme lemme (eviter grands→grand faux positif)
                infos_cand = lex.info(candidat)
                lemmes_cand = {e.get("lemme") for e in infos_cand if e.get("lemme")}
                if lemmes & lemmes_cand:
                    return candidat

        return None

    def _est_ambigu_accord(self, forme: str, pos_attendu: str) -> bool:
        """Verifie si un mot est ambigu pour l'accord.

        Retourne True si le mot peut etre un autre POS que ADJ/PART.
        """
        lex = self._lexique
        infos = lex.info(forme)
        if not infos:
            return False

        cgrams = {e.get("cgram", "") for e in infos if e.get("cgram")}
        # Si le mot n'a qu'un seul POS, pas ambigu
        if len(cgrams) <= 1:
            return False

        # Si le mot peut etre ADV, PRE, CON, DET, PRO, il est ambigu
        # (NOM et VER ne comptent pas car beaucoup d'ADJ sont aussi NOM)
        pos_ambigus = {
            "ADV", "PRE", "CON", "PRO:per", "PRO:rel",
            "DET", "DET:art", "DET:dem", "DET:pos", "DET:ind",
        }
        if cgrams & pos_ambigus:
            return True

        return False

    # ------------------------------------------------------------------
    # 3c. Participe passe
    # ------------------------------------------------------------------

    def _corriger_participe_v6(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige infinitif -> participe passe apres auxiliaire.

        Declencheur : P2G predit PP la ou G2P predit INF, apres auxiliaire.

        Gardes :
        1. Le mot precedent doit etre auxiliaire
        2. La forme PP doit exister dans le lexique
        3. p2g_confiance >= participe_confiance_min
        """
        corrections: list[Correction] = []
        lex = self._lexique
        cfg = self._config

        for i, mv in enumerate(mots):
            # Deja corrige ?
            if mv.regle:
                continue

            # Declencheur : divergence ortho + le P2G predit une forme differente
            if not mv.div_ortho:
                continue

            # Le G2P doit penser que c'est un verbe (infinitif)
            if not mv.g2p_pos or not mv.g2p_pos.startswith(("VER", "AUX")):
                continue

            # La forme doit finir par -er (infinitif 1er groupe)
            if not mv.forme.endswith("er"):
                continue

            # Le P2G propose une forme differente (potentiellement un PP)
            p2g = mv.p2g_ortho
            if not p2g or p2g == mv.forme:
                continue

            # La forme P2G doit etre un participe passe (finir par -e, -es, -i, -is, -u, -us, etc.)
            if not (p2g.endswith(("\u00e9", "\u00e9e", "\u00e9s", "\u00e9es",
                                 "i", "ie", "is", "ies",
                                 "u", "ue", "us", "ues",
                                 "t", "te", "ts", "tes"))):
                continue

            # Garde 1 : mot precedent doit etre auxiliaire
            if i == 0:
                continue
            prev = mots[i - 1]
            if prev.forme.lower() not in _AUXILIAIRES:
                continue

            # Garde 1b : detecter construction attributive "c'etait/c'est + INF"
            # Dans ce cas l'infinitif est correct (pas de PP)
            if i >= 2:
                prev_prev = mots[i - 2]
                if prev_prev.forme.lower() in ("c'", "c\u2019", "ce"):
                    continue

            # Garde 1c : si la forme -er est aussi ADJ dans le lexique,
            # c'est probablement un adjectif (fier, cher, leger, entier, etc.)
            # "est fier" ≠ "est fié"
            infos_er = lex.info(mv.forme)
            if infos_er and any(
                e.get("cgram", "").startswith("ADJ")
                for e in infos_er
            ):
                continue

            # Garde 2 : la forme PP doit exister dans le lexique
            if not lex.existe(p2g):
                continue

            # Garde 3 : confiance P2G elevee
            if mv.p2g_confiance < cfg.participe_confiance_min:
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = "participe.inf_pp"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"Participe: '{mv.forme}' -> '{p2g}' "
                    f"apres auxiliaire '{prev.forme}' "
                    f"(confiance={mv.p2g_confiance:.2f})"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3c-bis. PP accent (mange -> mange apres auxiliaire)
    # ------------------------------------------------------------------

    def _corriger_pp_accent(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige les PP in-lexique sans accent apres auxiliaire (mange->mange)."""
        from lectura_correcteur.orthographe._suggestions import _est_variante_accent
        corrections: list[Correction] = []
        lex = self._lexique

        for i, mv in enumerate(mots):
            if mv.regle:
                continue
            if not mv.div_ortho:
                continue
            p2g = mv.p2g_ortho
            if not p2g or p2g == mv.forme:
                continue
            if mv.forme.lower() in _MOTS_PROTEGES:
                continue
            # La correction doit etre une variante accent
            if not _est_variante_accent(mv.forme, p2g):
                continue
            # Ne corriger que vers la forme PLUS accentuee (pas l'inverse)
            # Supprimer un accent = FP (entrebâillée→entrebaillée, dû→du)
            _ACCENTED_PP = frozenset("àâäéèêëïîôùûüÿçœæ")
            nb_acc_orig = sum(1 for c in mv.forme if c in _ACCENTED_PP)
            nb_acc_p2g = sum(1 for c in p2g if c in _ACCENTED_PP)
            if nb_acc_p2g <= nb_acc_orig:
                continue
            # Le P2G doit exister dans le lexique
            if not lex.existe(p2g):
                continue
            # Confiance P2G >= 0.85
            if mv.p2g_confiance < 0.85:
                continue
            # Un auxiliaire doit preceder (fenetre 3 mots, ADV/clitiques ok)
            if i == 0:
                continue
            idx_aux = None
            for j in range(max(0, i - 3), i):
                mj_low = mots[j].forme.lower()
                if mj_low in _AUXILIAIRES:
                    idx_aux = j
                    break
            if idx_aux is None:
                continue
            # Verifier que seuls ADV/clitiques/negation sont entre aux et PP
            _pp_ok = True
            for k in range(idx_aux + 1, i):
                mk_low = mots[k].forme.lower()
                if mk_low in _CLITIQUES_OBJETS:
                    continue
                if mk_low in ("ne", "n'", "n\u2019", "pas", "jamais",
                               "plus", "rien"):
                    continue
                if mots[k].g2p_pos and mots[k].g2p_pos.startswith("ADV"):
                    continue
                # "été" autorise pour passif : "a été mange" → "a été mangé"
                if mk_low in ("\u00e9t\u00e9", "ete"):
                    continue
                _pp_ok = False
                break
            if not _pp_ok:
                continue
            # Bloquer si c'est/c'etait + infinitif (construction attributive)
            if idx_aux >= 1 and mots[idx_aux - 1].forme.lower() in (
                "c'", "c\u2019", "ce",
            ):
                continue

            aux_low = mots[idx_aux].forme.lower()
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = "pp.accent"
            corrections.append(Correction(
                index=mv.index, original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=f"PP accent: '{mv.forme}' -> '{p2g}' apres '{aux_low}'",
            ))

        # Pass 2 : PP sans accent apres auxiliaire avoir (sans P2G)
        # Ex: "a mange" -> "a mangé", "a sonne" -> "a sonné"
        # Le mot est un VER présent dont le PP accentué existe.
        for i, mv in enumerate(mots):
            if mv.regle:
                continue
            forme_low = mv.forme.lower()
            if forme_low in _MOTS_PROTEGES:
                continue
            if not lex.existe(forme_low):
                continue

            # Chercher si le mot est un VER present et a un PP homonyme accentue
            infos = lex.info(forme_low)
            is_ver_pre = any(
                e.get("cgram", "").startswith("VER")
                and e.get("mode") in ("ind", "indicatif")
                and e.get("temps") in ("pre", "present")
                for e in infos
            )
            if not is_ver_pre:
                continue

            # Guard: mots dont le NOM est plus frequent que le VER present
            # (plage, porte, place, etc.) — la lecture NOM domine.
            _nom_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos if e.get("cgram", "").startswith("NOM")),
                default=0.0,
            )
            _ver_pre_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos
                 if (e.get("cgram", "").startswith("VER")
                     and e.get("mode") in ("ind", "indicatif")
                     and e.get("temps") in ("pre", "present"))),
                default=0.0,
            )
            _phase2_freq = False
            if _nom_max_freq > _ver_pre_max_freq:
                if self._config.activer_phase2 and _nom_max_freq <= 2.0 * _ver_pre_max_freq:
                    _phase2_freq = True
                else:
                    continue

            # Chercher la variante PP accentuée (mange->mangé, sonne->sonné)
            # Pattern: forme -e -> -é, forme -es -> -és
            pp_candidat = None
            if forme_low.endswith("e") and not forme_low.endswith("\u00e9"):
                pp_test = forme_low[:-1] + "\u00e9"
                if lex.existe(pp_test):
                    pp_infos = lex.info(pp_test)
                    if any(
                        e.get("cgram", "").startswith("VER")
                        and e.get("mode") in ("par", "participe")
                        for e in pp_infos
                    ):
                        pp_candidat = pp_test
            elif forme_low.endswith("es") and not forme_low.endswith("\u00e9s"):
                pp_test = forme_low[:-2] + "\u00e9s"
                if lex.existe(pp_test):
                    pp_infos = lex.info(pp_test)
                    if any(
                        e.get("cgram", "").startswith("VER")
                        and e.get("mode") in ("par", "participe")
                        for e in pp_infos
                    ):
                        pp_candidat = pp_test

            if not pp_candidat:
                continue

            # Un auxiliaire "avoir" doit preceder (fenetre 3)
            idx_aux = None
            for j in range(max(0, i - 3), i):
                mj_low = mots[j].forme.lower()
                if mj_low in _AVOIR_CONJUGUE:
                    idx_aux = j
                    break
            if idx_aux is None:
                continue

            # Verifier que seuls clitiques/negation entre aux et mot
            _ok = True
            for k in range(idx_aux + 1, i):
                mk_low = mots[k].forme.lower()
                if mk_low in _CLITIQUES_OBJETS:
                    continue
                if mk_low in ("ne", "n'", "n\u2019", "pas", "jamais",
                               "plus", "rien"):
                    continue
                if mots[k].g2p_pos and mots[k].g2p_pos.startswith("ADV"):
                    continue
                if mk_low in ("\u00e9t\u00e9", "ete"):
                    continue
                _ok = False
                break
            if not _ok:
                continue

            mv.correction = transferer_casse(mv.correction, pp_candidat)
            mv.regle = "pp.accent.struct"
            if _phase2_freq:
                mv.regle = "p2." + mv.regle
            corrections.append(Correction(
                index=mv.index, original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"PP accent structurel: '{mv.forme}' -> "
                    f"'{pp_candidat}' apres avoir"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3c-ter. INF -> PP apres auxiliaire avoir (manger -> mange)
    # ------------------------------------------------------------------

    def _corriger_inf_pp_avoir(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige les infinitifs utilises a la place du PP apres avoir.

        Cas typique : "il a manger" -> "il a mange"

        Gardes :
        1. Le mot doit avoir div_ortho (P2G diverge de l'entree)
        2. Le P2G doit proposer une forme differente
        3. La forme actuelle doit etre un infinitif (-er dans le lexique)
        4. Un auxiliaire "avoir" conjugue doit preceder
           (fenetre 3 mots, ADV/clitiques entre)
        5. La forme P2G doit exister dans le lexique
        6. P2G confiance >= 0.85 (conservateur)
        7. Pas de verbe entre l'auxiliaire et le mot (evite "a fait manger")
        """
        corrections: list[Correction] = []
        lex = self._lexique

        for i, mv in enumerate(mots):
            if mv.regle:
                continue
            if not mv.div_ortho:
                continue
            p2g = mv.p2g_ortho
            if not p2g or p2g == mv.forme:
                continue
            if mv.forme.lower() in _MOTS_PROTEGES:
                continue

            forme_low = mv.forme.lower()

            # Garde 3 : forme actuelle doit etre un infinitif en -er
            # (on se limite au 1er groupe pour la securite : homophones INF/PP)
            if not forme_low.endswith("er"):
                continue
            # Verifier dans le lexique que c'est bien un VER infinitif
            # Multext: Vmn = verbe main infinitif (3e char = 'n')
            if lex.existe(forme_low):
                infos = lex.info(forme_low)
                est_inf = any(
                    e.get("cgram", "").startswith("VER")
                    and len(e.get("multext") or "") >= 3
                    and (e.get("multext") or "")[2] == "n"
                    for e in infos
                )
                if not est_inf:
                    # Peut etre un NOM en -er (papier, cahier) → skip
                    continue

            # Garde : la forme P2G doit etre un PP (-e, -ee, -es, -ees)
            p2g_low = p2g.lower()
            if not (p2g_low.endswith(("\u00e9", "\u00e9e", "\u00e9s", "\u00e9es"))):
                continue

            # Garde 4 : auxiliaire dans une fenetre de 3 mots
            # (avoir conjugue OU etre pour passif : "est construire" → "construite")
            idx_avoir = None
            for j in range(max(0, i - 3), i):
                mj_low = mots[j].forme.lower()
                if mj_low in _AVOIR_CONJUGUE or mj_low in _AUXILIAIRES:
                    idx_avoir = j
                    break

            if idx_avoir is None:
                continue

            # Garde 7 : entre l'auxiliaire et le mot, seuls ADV, clitiques
            # et "ne/n'" sont autorises (pas d'autre verbe)
            intervening_ok = True
            for k in range(idx_avoir + 1, i):
                mk = mots[k]
                mk_low = mk.forme.lower()
                if mk_low in _CLITIQUES_OBJETS:
                    continue
                if mk_low in ("ne", "n'", "n\u2019", "pas", "jamais",
                               "plus", "rien"):
                    continue
                if mk.g2p_pos and mk.g2p_pos.startswith("ADV"):
                    continue
                # "été" autorise pour passif : "a été manger" → "a été mangé"
                if mk_low in ("\u00e9t\u00e9", "ete"):
                    continue
                # Autre chose (verbe, nom...) → pas un contexte auxiliaire+PP
                intervening_ok = False
                break
            if not intervening_ok:
                continue

            # Garde 5 : la forme P2G doit exister dans le lexique
            if not lex.existe(p2g):
                continue

            # Garde 6 : confiance P2G >= 0.75
            if mv.p2g_confiance < 0.75:
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = "participe.inf_pp"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"INF->PP: '{mv.forme}' -> '{p2g}' "
                    f"apres avoir (confiance={mv.p2g_confiance:.2f})"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3c-quater. INF -> PP structurel (couche 2, sans P2G)
    # ------------------------------------------------------------------

    def _corriger_inf_pp_structurel(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige un infinitif 1er groupe apres auxiliaire sans P2G.

        Cas typique : "il a manger" -> "il a mange"

        Contrairement a _corriger_inf_pp_avoir, cette methode ne depend
        PAS de div_ortho ni du P2G. Elle detecte purement structurellement
        un -er apres un auxiliaire avoir/etre et le remplace par -e.

        Gardes :
        1. mv.regle vide (pas deja corrige)
        2. Forme finit par -er (1er groupe)
        3. Forme est VER infinitif dans le lexique
        4. Forme PP (-e) existe dans le lexique comme VER participe
        5. Auxiliaire avoir/etre dans fenetre 3 mots avant
        6. Entre auxiliaire et mot : uniquement clitiques, ne/pas/jamais, ADV, ete
        7. Guard NOM : si forme -er est aussi NOM frequent, skip
        8. Guard modal : si modal entre auxiliaire et cible, skip
        """
        corrections: list[Correction] = []
        lex = self._lexique

        for i, mv in enumerate(mots):
            if mv.regle:
                continue

            forme_low = mv.forme.lower()

            # Garde 2 : forme finit par -er (1er groupe uniquement)
            if not forme_low.endswith("er") or len(forme_low) < 3:
                continue

            # Garde : mots proteges
            if forme_low in _MOTS_PROTEGES:
                continue

            # Garde 3 : forme est VER infinitif dans le lexique
            if not lex.existe(forme_low):
                continue
            infos = lex.info(forme_low)
            est_inf = any(
                e.get("cgram", "").startswith("VER")
                and len(e.get("multext") or "") >= 3
                and (e.get("multext") or "")[2] == "n"
                for e in infos
            )
            if not est_inf:
                continue

            # Garde 7 : si forme -er est aussi NOM plus frequent que VER, skip
            # (papier, cahier, dossier, etc.)
            nom_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos if e.get("cgram", "").startswith("NOM")),
                default=0.0,
            )
            ver_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos if e.get("cgram", "").startswith("VER")),
                default=0.0,
            )
            if nom_max_freq > ver_max_freq:
                continue

            # Garde 1c : si forme -er est aussi ADJ, skip
            # (fier, cher, leger, entier, etc. — "est fier" ≠ "est fie")
            if any(e.get("cgram", "").startswith("ADJ") for e in infos):
                continue

            # Garde 4 : construire la forme PP en -e et verifier qu'elle
            # existe dans le lexique comme VER participe
            pp_forme = forme_low[:-2] + "\u00e9"
            if not lex.existe(pp_forme):
                continue
            pp_infos = lex.info(pp_forme)
            est_pp = any(
                e.get("cgram", "").startswith("VER")
                and e.get("mode") in ("par", "participe")
                for e in pp_infos
            )
            if not est_pp:
                continue

            # Garde 5 : auxiliaire avoir/etre dans fenetre 3 mots avant
            idx_aux = None
            for j in range(max(0, i - 3), i):
                mj_low = mots[j].forme.lower()
                if mj_low in _AVOIR_CONJUGUE or mj_low in _AUXILIAIRES_ETRE:
                    idx_aux = j
                    break
            if idx_aux is None:
                continue

            # Garde 6+8 : entre auxiliaire et mot, seuls clitiques, negation,
            # ADV, "ete" sont autorises. Si modal → skip (garde 8).
            intervening_ok = True
            for k in range(idx_aux + 1, i):
                mk_low = mots[k].forme.lower()
                if mk_low in _MODAUX_ELARGI:
                    # Modal entre auxiliaire et cible : "a pu manger" → correct
                    intervening_ok = False
                    break
                if mk_low in _CLITIQUES_OBJETS:
                    continue
                if mk_low in ("ne", "n'", "n\u2019", "pas", "jamais",
                               "plus", "rien"):
                    continue
                if mots[k].g2p_pos and mots[k].g2p_pos.startswith("ADV"):
                    continue
                if mk_low in ("\u00e9t\u00e9", "ete"):
                    continue
                intervening_ok = False
                break
            if not intervening_ok:
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, pp_forme)
            mv.regle = "participe.inf_pp.struct"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"INF->PP struct: '{mv.forme}' -> '{pp_forme}' "
                    f"apres auxiliaire (sans P2G)"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3d. Correction verbe via P2G (div_pos acceptable quand homophones)
    # ------------------------------------------------------------------

    def _corriger_verbe_p2g(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige les formes verbales homophones via divergence P2G.

        Cas typiques :
        - "les enfants mange" -> "mangent" (div_ortho + div_pos NOM->VER)
        - "il a mange a la cantine" -> "il a mange a la cantine"
          mais "a la" -> "a la" (a=VER vs a=PRE)

        Declencheur : div_ortho=True ET div_pos=True ET les formes sont homophones.

        Gardes :
        1. Les deux formes doivent etre homophones (meme phoneme G2P)
        2. La forme P2G doit exister dans le lexique
        3. p2g_confiance >= homophone_confiance_min
        4. Le contexte doit confirmer le POS predit par le P2G
           (ex: sujet pluriel avant verbe pluriel)
        5. Le mot original doit exister dans le lexique avec un POS
           different de celui predit par P2G
        """
        corrections: list[Correction] = []
        lex = self._lexique
        cfg = self._config

        for mv in mots:
            # Deja corrige ?
            if mv.regle:
                continue

            # Pas de correction sur mots proteges ou tokens elides
            if mv.forme.lower() in _MOTS_PROTEGES:
                continue
            if mv.forme.endswith(("'", "\u2019")):
                continue

            # Pas de correction sur noms propres (majuscule initiale)
            if mv.correction and mv.correction[0].isupper():
                continue

            # Declencheur : divergence ortho + (pos ou nombre)
            if not mv.div_ortho:
                continue
            if not mv.div_pos and not mv.div_nombre:
                continue

            forme = mv.forme
            p2g = mv.p2g_ortho

            if not p2g or forme == p2g:
                continue

            # Pas de correction si la cible est un mot protege
            if p2g.lower() in _MOTS_PROTEGES:
                continue

            # Pas de correction si forme ou cible est un auxiliaire
            # (changer la conjugaison d'un auxiliaire est trop risque)
            if forme.lower() in _AUXILIAIRES or p2g.lower() in _AUXILIAIRES:
                continue

            # Le P2G doit predire un verbe (on ne corrige que vers des verbes)
            if not mv.p2g_pos or not (
                mv.p2g_pos.startswith("VER") or mv.p2g_pos == "AUX"
                or mv.p2g_pos == "PRE"  # a -> a (prep)
            ):
                continue

            # Garde 1 : les formes doivent etre homophones
            if mv.g2p_phone:
                phone_p2g = self._g2p_phone(p2g)
                if phone_p2g and phone_p2g != mv.g2p_phone:
                    continue

            # Garde 2 : la forme P2G doit exister dans le lexique
            if not lex.existe(p2g):
                continue

            # Garde 3 : confiance P2G suffisante
            if mv.p2g_confiance < cfg.homophone_confiance_min:
                continue

            # Garde 4 : le contexte doit confirmer le POS P2G
            if not self._contexte_confirme_pos_p2g(mots, mv):
                continue

            # Garde 5 : si le mot original existe dans le lexique avec le
            # meme POS que le P2G, verifier que le nombre+personne est
            # different (sinon c'est deja la bonne forme conjuguee).
            # Ex: "mange" est VER 1s/3s -> correction en "mangent" (VER 3p)
            # acceptee car le nombre differe.
            if lex.existe(forme):
                infos = lex.info(forme)
                p2g_pos = mv.p2g_pos
                p2g_nombre = mv.p2g_nombre
                p2g_personne = mv.p2g_personne

                # Verifier si une entree du lexique a le meme POS + nombre + personne
                has_same_flexion = False
                for entry in infos:
                    e_pos = entry.get("cgram", "")
                    e_nombre = entry.get("nombre", "")
                    e_personne = entry.get("personne", "")
                    # Meme famille POS (VER/AUX)
                    same_pos = (e_pos == p2g_pos) or (
                        e_pos.startswith("VER") and p2g_pos.startswith("VER")
                    ) or (
                        e_pos in ("AUX", "VER") and p2g_pos in ("AUX", "VER")
                    )
                    if not same_pos:
                        continue
                    # Comparer nombre + personne
                    nombre_match = (
                        not p2g_nombre or not e_nombre
                        or e_nombre == p2g_nombre
                    )
                    personne_match = (
                        not p2g_personne or not e_personne
                        or e_personne == p2g_personne
                    )
                    if nombre_match and personne_match:
                        has_same_flexion = True
                        break

                if has_same_flexion:
                    continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = f"verbe.p2g.{forme}_{p2g}"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"Verbe P2G: '{forme}' -> '{p2g}' "
                    f"({mv.g2p_pos} -> {mv.p2g_pos}, "
                    f"confiance={mv.p2g_confiance:.2f})"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3d-bis. Detection sujet nominal (pour accord sujet-verbe)
    # ------------------------------------------------------------------

    def _detecter_sujet_nominal(
        self, mots: list[MotV6], idx_verbe: int,
    ) -> str | None:
        """Detecte le nombre du sujet nominal avant le verbe.

        Retourne "s" (singulier), "p" (pluriel), ou None.

        Version conservatrice : exige un DET identifie pour conclure.
        Scanne en arriere en sautant les mots transparents (clitiques),
        puis les groupes prepositionnels (PRE + DET + NOM/ADJ) pour
        trouver le vrai sujet et non un complement de nom.

        Ex: "le chat de mes voisins dort" -> sujet = "le chat" (s)
        """
        # Mots transparents entre sujet et verbe
        _TRANSPARENTS = frozenset({
            "se", "s'", "s\u2019", "ne", "n'", "n\u2019",
            "pas", "plus", "jamais", "rien", "point",
            "y", "en", "me", "m'", "m\u2019",
            "te", "t'", "t\u2019", "le", "la", "l'", "l\u2019",
            "les", "lui", "nous", "vous", "leur",
        })

        # Prepositions (incluant contractions)
        _PREPS = frozenset({
            "de", "du", "d'", "d\u2019",
            "en", "\u00e0", "au", "aux", "par", "pour",
            "avec", "sans", "chez", "vers", "contre",
            "entre", "sur", "sous", "lors",
        })

        # Determinants singuliers et pluriels
        _DET_SING = frozenset({
            "le", "la", "l'", "l\u2019", "un", "une",
            "ce", "cet", "cette", "mon", "ma", "ton", "ta",
            "son", "sa", "notre", "votre", "leur",
            "chaque",
        })
        _DET_PLUR = frozenset({
            "les", "des", "ces", "ses", "mes", "tes",
            "nos", "vos", "leurs", "plusieurs", "quelques",
            "certains", "certaines", "tous", "toutes",
            "deux", "trois", "quatre", "cinq", "six", "sept",
            "huit", "neuf", "dix",
        })

        # Quantifieurs prenant l'accord pluriel
        _QUANTIFIERS_PLUR = frozenset({
            "plupart", "majorit\u00e9", "totalit\u00e9", "moiti\u00e9",
            "trentaine", "vingtaine", "quarantaine",
            "cinquantaine", "soixantaine", "centaine",
            "dizaine", "douzaine", "millier",
        })

        # Fenetre max de scan (tokens depuis le verbe, transparents exclus)
        _MAX_SCAN = 8

        j = idx_verbe - 1

        # Sauter les mots transparents
        while j >= 0 and mots[j].forme.lower() in _TRANSPARENTS:
            j -= 1
        if j < 0:
            return None

        # Borne inferieure de scan (fenetre)
        j_min = max(0, j - _MAX_SCAN)

        # Premier NOM rencontre
        first_nom_j = -1
        crossed_pp = False

        while j >= j_min:
            pos_j = mots[j].g2p_pos or ""
            mot_j = mots[j].forme.lower()

            # NOM, NOM PROPRE
            if pos_j in ("NOM", "NOM PROPRE") or pos_j.startswith("NOM"):
                if first_nom_j < 0:
                    first_nom_j = j
                j -= 1
                continue

            if pos_j == "ADJ" or pos_j.startswith("ADJ"):
                # Demonstratifs tagues ADJ — guard PP
                if mot_j in ("ce", "cet", "cette"):
                    # Si precede d'une preposition, c'est un PP (a cette epoque)
                    if j > 0 and mots[j - 1].forme.lower() in _PREPS:
                        first_nom_j = -1
                        crossed_pp = True
                        j -= 2
                        continue
                    return "s"
                if mot_j == "ces":
                    if j > 0 and mots[j - 1].forme.lower() in _PREPS:
                        first_nom_j = -1
                        crossed_pp = True
                        j -= 2
                        continue
                    return "p"
                if first_nom_j < 0:
                    first_nom_j = j
                j -= 1
                continue

            # ADV transparent
            if pos_j == "ADV" or pos_j.startswith("ADV"):
                j -= 1
                continue

            # Contractions prepositionnelles : "du", "au", "aux"
            if mot_j in ("du", "au", "aux"):
                first_nom_j = -1
                crossed_pp = True
                j -= 1
                continue

            # Determinant (ART, DET, ADJ:pos, ADJ:dem)
            if (
                pos_j.startswith("ART")
                or pos_j.startswith("DET")
                or pos_j in ("ADJ:pos", "ADJ:dem")
            ):
                # Verifier si DET dans un PP
                if j > 0:
                    prev_pos = mots[j - 1].g2p_pos or ""
                    prev_mot = mots[j - 1].forme.lower()
                    if prev_pos == "PRE" or prev_mot in _PREPS or prev_mot == "des":
                        first_nom_j = -1
                        crossed_pp = True
                        j -= 2
                        continue
                    # ADJ/quantifier entre PRE et DET
                    if prev_pos in ("ADJ", "ADJ:pos") and j > 1:
                        pp2_mot = mots[j - 2].forme.lower()
                        pp2_pos = mots[j - 2].g2p_pos or ""
                        if pp2_pos == "PRE" or pp2_mot in _PREPS or pp2_mot == "des":
                            first_nom_j = -1
                            crossed_pp = True
                            j -= 3
                            continue
                    # "des" apres NOM/ADJ/PRO = contraction "de+les"
                    if mot_j == "des" and prev_pos in (
                        "NOM", "ADJ", "NOM PROPRE",
                        "PRO:dem", "PRO:rel", "PRO:ind",
                    ):
                        first_nom_j = -1
                        crossed_pp = True
                        j -= 1
                        continue

                # Pas de preposition devant -> DET du sujet
                if mot_j in _DET_PLUR:
                    # Guard: "un/une des NOM" = singulier
                    if mot_j == "des" and j > 0 and mots[j - 1].forme.lower() in ("un", "une", "l'un", "l'une"):
                        return "s"
                    # Guard: superlatif "les plus/moins ADJ"
                    if mot_j == "les" and j + 1 < len(mots):
                        next_w = mots[j + 1].forme.lower()
                        if next_w in ("plus", "moins"):
                            first_nom_j = -1
                            j -= 1
                            continue
                    # Guard coordination : "DET_sing NOM et DET_plur NOM VERB"
                    # -> le pluriel est fiable, pas besoin de guard
                    return "p"
                if mot_j in _DET_SING:
                    # Guard: quantifieurs pluriel
                    if first_nom_j >= 0:
                        nom_q = mots[first_nom_j].forme.lower()
                        if nom_q in _QUANTIFIERS_PLUR:
                            return "p"
                    # Guard coordination : "X et DET NOM VERB" -> pluriel
                    # Mais pas "entre X et DET NOM" (PP, pas coordination)
                    if j > 0 and mots[j - 1].forme.lower() == "et":
                        # Verifier qu'il n'y a pas "entre" avant "et"
                        _is_entre = False
                        for _ke in range(j - 2, max(-1, j - 6), -1):
                            if mots[_ke].forme.lower() == "entre":
                                _is_entre = True
                                break
                        if not _is_entre:
                            return "p"
                    return "s"
                return None

            # Preposition explicite
            if pos_j == "PRE" or mot_j in _PREPS:
                first_nom_j = -1
                crossed_pp = True
                j -= 1
                continue

            # Autre POS -> on arrete le scan
            break

        # Pas de DET trouve -> pas assez fiable pour corriger
        return None

    # ------------------------------------------------------------------
    # 3d-ter. Accord sujet-verbe (conjugaison par lexique)
    # ------------------------------------------------------------------

    def _corriger_accord_sujet_verbe(
        self, mots: list[MotV6],
    ) -> list[Correction]:
        """Corrige les desaccords sujet-verbe en personne/nombre.

        Detecte les pronoms sujets (il, elle, ils, elles, je, tu, on)
        suivis d'un verbe dont la personne ou le nombre ne correspondent
        pas, et corrige le verbe via le lexique (conjugaison).

        Cette regle bypasse _MOTS_PROTEGES car elle a ses propres gardes
        (detection de sujet fiable + verification dans le lexique).

        Exemples :
        - "il deviens" -> "il devient" (2s -> 3s)
        - "ils est" -> "ils sont" (3s -> 3p)
        - "elle pouvons" -> "elle peut" (1p -> 3s)
        """
        corrections: list[Correction] = []
        lex = self._lexique

        # Pronoms sujets non-ambigus (nous/vous omis : ambigus sujet/objet)
        _PRO = {
            "je": ("1", "s"), "j'": ("1", "s"), "j\u2019": ("1", "s"),
            "tu": ("2", "s"),
            "il": ("3", "s"), "elle": ("3", "s"), "on": ("3", "s"),
            "ils": ("3", "p"), "elles": ("3", "p"),
        }

        # Mots autorises entre sujet et verbe (clitiques + negation)
        _INTER = frozenset({
            "me", "m'", "m\u2019", "te", "t'", "t\u2019",
            "se", "s'", "s\u2019", "le", "la", "les", "l'", "l\u2019",
            "lui", "leur", "nous", "vous", "y", "en",
            "ne", "n'", "n\u2019",
        })

        # Modes verbaux conjugues (codes courts normalises)
        _MODES_CONJ = frozenset({"ind", "sub", "con"})

        for i, mv in enumerate(mots):
            if mv.regle:
                continue
            if i == 0:
                continue

            forme_low = mv.forme.lower()

            # Skip elisions
            if forme_low.endswith(("'", "\u2019")):
                continue

            # Le mot doit etre un verbe conjugue dans le lexique
            if not lex.existe(forme_low):
                continue

            infos = lex.info(forme_low)
            verb_entries = [
                e for e in infos
                if (e.get("cgram", "").startswith("VER")
                    or e.get("cgram") == "AUX")
                and e.get("mode", "") in _MODES_CONJ
            ]
            if not verb_entries:
                continue

            # G2P POS doit confirmer verbe (si disponible)
            g2p_pos = mv.g2p_pos or ""
            if g2p_pos and not (
                g2p_pos.startswith("VER") or g2p_pos.startswith("AUX")
            ):
                continue

            # Guard : pas de participe passe (emis, mis, etc.)
            # G2P POS "VER:par" = participe, pas un verbe conjugue
            if g2p_pos and ":par" in g2p_pos:
                continue

            # Trouver le sujet en amont (pronom puis NOM)
            sujet_pers = ""
            sujet_nombre = ""

            # Phase 1 : pronom sujet non-ambigu (fenetre 3 mots)
            for j in range(i - 1, max(-1, i - 4), -1):
                mj = mots[j]
                mj_low = mj.forme.lower()

                if mj_low in _PRO:
                    sujet_pers, sujet_nombre = _PRO[mj_low]
                    break

                if mj_low in _INTER:
                    continue
                mj_pos = mj.g2p_pos or ""
                if mj_pos.startswith("ADV"):
                    continue
                break

            # Phase 2 : sujet nominal DET+NOM avec saut de PP
            _sujet_from_nom = False
            if not sujet_pers:
                sujet_nombre = self._detecter_sujet_nominal(mots, i)
                if sujet_nombre:
                    sujet_pers = "3"
                    _sujet_from_nom = True

            if not sujet_pers or not sujet_nombre:
                continue

            # Guard : "ce sont" / "c'est" — sujet demonstratif
            # "ce" devant le verbe (skip transparents) = pronom demonstratif,
            # pas le sujet nominal detecte par le scanner.
            if _sujet_from_nom:
                _j_ce = i - 1
                while _j_ce >= 0 and mots[_j_ce].forme.lower() in _INTER:
                    _j_ce -= 1
                if _j_ce >= 0 and mots[_j_ce].forme.lower() in (
                    "ce", "c'", "c\u2019",
                ):
                    continue

            # Guard : inversion (pronom sujet APRES le verbe)
            if i + 1 < len(mots):
                next_low = mots[i + 1].forme.lower()
                if next_low in _PRO or next_low in (
                    "ce", "c'", "c\u2019",
                ):
                    continue

            # Meilleure entree verbe (plus haute frequence)
            verb_entries.sort(
                key=lambda e: float(e.get("freq", 0)), reverse=True,
            )
            best = verb_entries[0]

            v_pers = best.get("personne", "")
            v_nombre_raw = best.get("nombre", "")
            v_lemme = best.get("lemme", "")
            v_mode = best.get("mode", "")
            v_temps = best.get("temps", "")

            if not v_lemme or not v_mode:
                continue

            # Normaliser nombre (info() normalise via LexiqueNormalise)
            v_nombre = normaliser_morpho(v_nombre_raw) if v_nombre_raw else ""

            # Guard NOM sujet : ne corriger que le nombre (pas la personne)
            # Avec un sujet nominal, la personne est toujours 3.
            # Si le verbe est deja P3 mais le nombre differe -> corriger.
            # Si le verbe est P1/P2 -> c'est probablement un imperatif,
            # un participe passe, ou une forme ambigue (P1s=P3s). Trop
            # risque de FP -> ne pas corriger.
            # Exception : indicatif present/imparfait — P1/P2 apres un
            # sujet nominal est toujours faux, car :
            # - pas d'imperatif avec sujet nominal devant
            # - le G2P a deja confirme VER en amont (guard ligne 3396)
            if _sujet_from_nom and v_pers and v_pers != "3":
                if (v_mode == "ind"
                        and v_temps in ("pre", "imp")):
                    # Guard supplementaire : la forme ne doit pas etre
                    # aussi un NOM courant (porte, livre, etc.)
                    _also_nom = any(
                        e.get("cgram", "").startswith("NOM")
                        and float(e.get("freq", 0)) > 10.0
                        for e in infos
                    )
                    if _also_nom:
                        continue
                    # OK — permettre la correction
                else:
                    continue

            # Verifier le desaccord
            if ((not v_pers or v_pers == sujet_pers)
                    and (not v_nombre or v_nombre == sujet_nombre)):
                continue

            # Chercher la forme correcte dans le lexique
            # NB: formes_de() passe par __getattr__ (pas normalise),
            # donc on normalise mode/temps/nombre ici.
            formes = lex.formes_de(v_lemme)
            candidat = None
            candidat_freq = -1.0

            for f in formes:
                f_cgram = f.get("cgram", "")
                if not (f_cgram.startswith("VER") or f_cgram == "AUX"):
                    continue
                f_mode = normaliser_morpho(f.get("mode", ""))
                f_temps = normaliser_morpho(f.get("temps", ""))
                if f_mode != v_mode or f_temps != v_temps:
                    continue

                f_pers = f.get("personne", "")
                f_nombre = normaliser_morpho(f.get("nombre", ""))

                if f_pers == sujet_pers and f_nombre == sujet_nombre:
                    f_freq = float(f.get("freq", 0))
                    if f_freq > candidat_freq:
                        candidat = f.get("ortho", "")
                        candidat_freq = f_freq

            # Fallback: formes_de() omet P3s quand P1s a la meme ortho
            # (verbes -er, certains -ir). On verifie via info() si une
            # forme du bon nombre existe aussi pour la personne cible.
            if candidat is None and sujet_nombre != "p":
                for f in formes:
                    f_cgram = f.get("cgram", "")
                    if not (f_cgram.startswith("VER") or f_cgram == "AUX"):
                        continue
                    f_mode = normaliser_morpho(f.get("mode", ""))
                    f_temps = normaliser_morpho(f.get("temps", ""))
                    if f_mode != v_mode or f_temps != v_temps:
                        continue
                    f_nombre = normaliser_morpho(f.get("nombre", ""))
                    if f_nombre != sujet_nombre:
                        continue
                    cand_ortho = f.get("ortho", "")
                    if not cand_ortho:
                        continue
                    # Verifier via info() que cette forme admet la personne cible
                    cand_infos = lex.info(cand_ortho)
                    if any(
                        (e.get("cgram", "").startswith("VER")
                         or e.get("cgram") == "AUX")
                        and e.get("mode") == v_mode
                        and e.get("temps") == v_temps
                        and e.get("personne") == sujet_pers
                        and normaliser_morpho(
                            e.get("nombre", "")) == sujet_nombre
                        for e in cand_infos
                    ):
                        candidat = cand_ortho
                        break

            if not candidat or candidat.lower() == forme_low:
                continue

            # Guard : candidat de longueur aberrante (ex: "porte-jarretelle")
            if abs(len(candidat) - len(forme_low)) > 3:
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, candidat)
            mv.regle = "accord.sujet_verbe"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"Accord S-V: '{mv.forme}' -> '{candidat}' "
                    f"({v_pers}{v_nombre} -> {sujet_pers}{sujet_nombre})"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3d-ter. Infinitif -> conjugue apres pronom sujet
    # ------------------------------------------------------------------

    def _corriger_infinitif_sujet(
        self, mots: list[MotV6],
    ) -> list[Correction]:
        """Conjugue un infinitif utilise a la place d'un verbe conjugue.

        Detecte les pronoms sujets suivis d'un verbe a l'infinitif
        et conjugue au present de l'indicatif.

        Erreur typique FLE : "je manger" -> "je mange"
        Aussi present chez les natifs : "il finir" -> "il finit"
        """
        corrections: list[Correction] = []
        lex = self._lexique

        _PRO_INF = {
            "je": ("1", "s"), "j'": ("1", "s"), "j\u2019": ("1", "s"),
            "tu": ("2", "s"),
            "il": ("3", "s"), "elle": ("3", "s"), "on": ("3", "s"),
            "nous": ("1", "p"), "vous": ("2", "p"),
            "ils": ("3", "p"), "elles": ("3", "p"),
        }

        # Mots transparents (clitiques, negation) entre sujet et verbe
        _INTER = frozenset({
            "me", "m'", "m\u2019", "te", "t'", "t\u2019",
            "se", "s'", "s\u2019", "le", "la", "les", "l'", "l\u2019",
            "lui", "leur", "nous", "vous", "y", "en",
            "ne", "n'", "n\u2019",
        })

        # Prepositions apres lesquelles un infinitif est correct
        _PREPS_INF = frozenset({
            "de", "d'", "d\u2019", "\u00e0", "a", "pour", "sans",
            "avant", "apr\u00e8s", "afin",
        })

        for i, mv in enumerate(mots):
            if mv.regle:
                continue

            forme_low = mv.forme.lower()

            # Le mot doit etre un infinitif dans le lexique
            if not lex.existe(forme_low):
                continue

            infos = lex.info(forme_low)
            inf_entries = [
                e for e in infos
                if (e.get("cgram", "").startswith("VER")
                    or e.get("cgram") == "AUX")
                and e.get("mode") == "inf"
            ]
            if not inf_entries:
                continue

            # Le mot ne doit PAS aussi etre un verbe conjugue
            # (sinon c'est ambigu : "porte" = inf? non, conjugue)
            has_conj = any(
                (e.get("cgram", "").startswith("VER")
                 or e.get("cgram") == "AUX")
                and e.get("mode", "") in ("ind", "sub", "con", "imp")
                for e in infos
            )
            if has_conj:
                continue

            # NB: pas de guard NOM/ADJ ici — "manger", "avoir", "pouvoir"
            # sont aussi des NOM dans le lexique, mais apres un pronom sujet
            # c'est toujours un verbe. Le guard pronom en aval suffit.

            # Chercher un pronom sujet en amont (fenetre 3)
            sujet_pers = ""
            sujet_nombre = ""
            found_prep = False
            found_modal = False

            for j in range(i - 1, max(-1, i - 4), -1):
                mj_low = mots[j].forme.lower()

                if mj_low in _PRO_INF:
                    sujet_pers, sujet_nombre = _PRO_INF[mj_low]
                    break

                if mj_low in _INTER:
                    continue

                # Preposition avant = infinitif correct ("pour manger")
                if mj_low in _PREPS_INF:
                    found_prep = True
                    break

                # Modal/auxiliaire avant = infinitif correct ("veux manger")
                if mj_low in _MODAUX or mj_low in _AUXILIAIRES:
                    found_modal = True
                    break

                # Autre mot = stop
                break

            if not sujet_pers or found_prep or found_modal:
                continue

            # Guard : verifier que le G2P ne voit pas un infinitif valide
            g2p_pos = mv.g2p_pos or ""
            if g2p_pos and "inf" in g2p_pos.lower():
                # G2P confirme l'infinitif : verifier le contexte
                # Si un modal precede dans une fenetre plus large, ne pas corriger
                for j2 in range(i - 1, max(-1, i - 6), -1):
                    mj2_low = mots[j2].forme.lower()
                    if mj2_low in _MODAUX or mj2_low in _AUXILIAIRES:
                        found_modal = True
                        break
                if found_modal:
                    continue

            # Trouver la forme conjuguee (present indicatif)
            lemme = inf_entries[0].get("lemme", "")
            if not lemme:
                continue

            formes = lex.formes_de(lemme)
            candidat = None
            candidat_freq = -1.0

            for f in formes:
                f_cgram = f.get("cgram", "")
                if not (f_cgram.startswith("VER") or f_cgram == "AUX"):
                    continue
                f_mode = normaliser_morpho(f.get("mode", ""))
                f_temps = normaliser_morpho(f.get("temps", ""))
                if f_mode != "ind" or f_temps != "pre":
                    continue
                f_pers = f.get("personne", "")
                f_nombre = normaliser_morpho(f.get("nombre", ""))
                if f_pers == sujet_pers and f_nombre == sujet_nombre:
                    f_freq = float(f.get("freq", 0))
                    if f_freq > candidat_freq:
                        candidat = f.get("ortho", "")
                        candidat_freq = f_freq

            # Fallback P1s = P3s pour verbes -er
            if candidat is None and sujet_nombre == "s":
                for f in formes:
                    f_cgram = f.get("cgram", "")
                    if not (f_cgram.startswith("VER") or f_cgram == "AUX"):
                        continue
                    f_mode = normaliser_morpho(f.get("mode", ""))
                    f_temps = normaliser_morpho(f.get("temps", ""))
                    if f_mode != "ind" or f_temps != "pre":
                        continue
                    f_nombre = normaliser_morpho(f.get("nombre", ""))
                    if f_nombre != sujet_nombre:
                        continue
                    cand_ortho = f.get("ortho", "")
                    if not cand_ortho:
                        continue
                    cand_infos = lex.info(cand_ortho)
                    if any(
                        (e.get("cgram", "").startswith("VER")
                         or e.get("cgram") == "AUX")
                        and e.get("mode") == "ind"
                        and e.get("temps") == "pre"
                        and e.get("personne") == sujet_pers
                        for e in cand_infos
                    ):
                        candidat = cand_ortho
                        break

            if not candidat or candidat.lower() == forme_low:
                continue

            mv.correction = transferer_casse(mv.correction, candidat)
            mv.regle = "conj.infinitif_sujet"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"Infinitif->conjugue: '{mv.forme}' -> '{candidat}' "
                    f"(sujet {sujet_pers}{sujet_nombre})"
                ),
            ))

        return corrections

    def _contexte_confirme_pos_p2g(
        self, mots: list[MotV6], mv: MotV6,
    ) -> bool:
        """Verifie que le contexte local confirme le POS predit par le P2G.

        Heuristiques contextuelles conservatrices :
        - VER pluriel : sujet pluriel (pronoms, NOM pluriel) dans les 3 mots avant
        - PRE (a -> a) : pas de sujet pronoun/nom juste avant (evite "il a")
        """
        p2g_pos = mv.p2g_pos
        idx = mv.index

        if not p2g_pos:
            return False

        # Cas VER : verifier qu'un sujet coherent precede
        if p2g_pos.startswith("VER") or p2g_pos == "AUX":
            # Pronoms avec nombre + personne
            _pro_info = {
                "je": ("s", "1"), "j'": ("s", "1"),
                "tu": ("s", "2"),
                "il": ("s", "3"), "elle": ("s", "3"), "on": ("s", "3"),
                "nous": ("p", "1"),
                "vous": ("p", "2"),
                "ils": ("p", "3"), "elles": ("p", "3"),
            }

            p2g_nombre = mv.p2g_nombre
            p2g_personne = mv.p2g_personne
            for j in range(max(0, idx - 3), idx):
                mj = mots[j]
                mj_low = mj.forme.lower()
                # Pronom sujet (fiable)
                if mj_low in _pro_info:
                    pro_nombre, pro_personne = _pro_info[mj_low]
                    if p2g_nombre and pro_nombre != p2g_nombre:
                        continue
                    if p2g_personne and pro_personne != p2g_personne:
                        continue
                    return True
                # NOM sujet avec confirmation par determinant ou P2G
                if mj.g2p_pos and mj.g2p_pos.startswith("NOM"):
                    nom_nombre = mj.p2g_nombre or mj.g2p_nombre
                    if p2g_nombre and nom_nombre == p2g_nombre:
                        # Option 1 : DET confirme le nombre du NOM
                        if self._contexte_confirme_nombre(
                            mots, j, nom_nombre,
                        ):
                            return True
                        # Option 2 : pas de DET visible mais P2G du NOM
                        # confirme le nombre du verbe (confiance P2G)
                        if mj.p2g_nombre and mj.p2g_nombre == p2g_nombre:
                            return True
            return False

        # Cas PRE (a -> a) : verifier que ce n'est PAS apres un sujet verbal
        if p2g_pos == "PRE":
            # "a" comme preposition est attendu apres un verbe/nom, pas apres
            # un pronom sujet direct (il a, on a, elle a)
            if idx > 0:
                prev = mots[idx - 1]
                prev_low = prev.forme.lower()
                # Si le mot precedent est un pronom sujet, "a" est probablement
                # auxiliaire -> ne pas corriger en preposition
                if prev_low in {"il", "elle", "on", "qui"}:
                    return False
            # Apres un NOM ou un verbe, "a" preposition est probable
            if idx > 0:
                prev = mots[idx - 1]
                if prev.g2p_pos and (
                    prev.g2p_pos.startswith("NOM")
                    or prev.g2p_pos.startswith("VER")
                    or prev.g2p_pos.startswith("ADJ")
                    or prev.g2p_pos.startswith("ADV")
                ):
                    return True
            return False

        return False

    # ------------------------------------------------------------------
    # 3e. PP -> INF apres modal (travaillé -> travailler)
    # ------------------------------------------------------------------

    def _corriger_pp_inf_modal(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige PP -> INF apres verbe modal.

        Cas typique : "ils doivent travaillé" -> "ils doivent travailler"

        Declencheur : div_ortho=True ET le P2G predit l'infinitif
        ET le mot precedent est un verbe modal.

        Gardes :
        1. Le mot precedent doit etre un verbe modal
        2. La forme INF doit exister dans le lexique
        3. p2g_confiance >= homophone_confiance_min
        """
        corrections: list[Correction] = []
        lex = self._lexique
        cfg = self._config

        for i, mv in enumerate(mots):
            if mv.regle:
                continue
            if not mv.div_ortho:
                continue

            p2g = mv.p2g_ortho
            if not p2g or p2g == mv.forme:
                continue

            # Le P2G doit proposer un infinitif (-er, -ir, -re, -oir)
            if not p2g.endswith(("er", "ir", "re", "oir")):
                continue

            # La forme actuelle doit etre un PP (accent: -é, -ée, -és, -ées, -i, -is, -u, -us)
            forme = mv.forme
            if not (forme.endswith(("\u00e9", "\u00e9e", "\u00e9s", "\u00e9es",
                                   "i", "is", "it", "u", "us", "ut"))):
                continue

            # Garde 1 : modal dans fenetre 3 (avec negation/clitiques/ADV entre)
            trigger_low = None
            for j in range(i - 1, max(-1, i - 4), -1):
                if j < 0:
                    break
                mk_low = mots[j].forme.lower()
                if mk_low in _MODAUX_ELARGI:
                    trigger_low = mk_low
                    break
                # Mots intercalaires autorises entre modal et PP
                if mk_low in _CLITIQUES_OBJETS:
                    continue
                if mk_low in ("ne", "n'", "n\u2019", "pas", "jamais",
                              "plus", "rien"):
                    continue
                mk_pos = mots[j].g2p_pos or ""
                if mk_pos.startswith("ADV"):
                    continue
                break  # Mot non intercalaire → arret
            if trigger_low is None:
                continue

            # Garde 2 : la forme INF doit exister dans le lexique
            if not lex.existe(p2g):
                continue

            # Garde 3 : confiance P2G suffisante (0.65 — seuil abaisse
            # car le contexte modal est un signal fort)
            if mv.p2g_confiance < 0.65:
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = "participe.pp_inf"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"PP->INF: '{forme}' -> '{p2g}' "
                    f"apres modal '{trigger_low}' "
                    f"(confiance={mv.p2g_confiance:.2f})"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3e-bis. PP -> INF structurel (couche 2, sans P2G)
    # ------------------------------------------------------------------

    def _corriger_pp_inf_structurel(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige PP 1er groupe apres modal ou preposition sans P2G.

        Cas typiques :
        - "il doit travaille" -> "il doit travailler"
        - "pour mange" -> "pour manger"
        - "sans oublie" -> "sans oublier"

        Contrairement a _corriger_pp_inf_modal, cette methode ne depend
        PAS de div_ortho ni du P2G. Elle detecte purement structurellement
        un PP (-e/-ee/-es/-ees) apres modal ou preposition.

        Gardes :
        1. mv.regle vide
        2. Forme finit par -e, -ee, -es, -ees (PP 1er groupe)
        3. Forme est VER participe dans le lexique
        4. INF en -er existe dans le lexique comme VER infinitif
        5. Mot precedent est modal ou preposition (fenetre 2 avec clitique)
        6. Guard NOM : si forme -e est aussi NOM frequent, skip
        7. Guard auxiliaire : si mot precedent est auxiliaire, skip
        """
        corrections: list[Correction] = []
        lex = self._lexique

        _PREPS_INF_STRUCT = frozenset({
            "de", "d'", "d\u2019", "\u00e0", "a", "pour", "sans",
            "avant", "apr\u00e8s", "apres", "afin",
        })

        for i, mv in enumerate(mots):
            if mv.regle:
                continue

            forme_low = mv.forme.lower()

            # Garde : mots proteges
            if forme_low in _MOTS_PROTEGES:
                continue

            # Garde 2 : forme finit par -e/-ee/-es/-ees (PP 1er groupe
            # sans accent — typique des erreurs ou l'accent manque)
            # On accepte aussi les formes accentuees -e/-ee/-es/-ees
            pp_suffix = None
            if forme_low.endswith("\u00e9es"):
                pp_suffix = "\u00e9es"
            elif forme_low.endswith("\u00e9e"):
                pp_suffix = "\u00e9e"
            elif forme_low.endswith("\u00e9s"):
                pp_suffix = "\u00e9s"
            elif forme_low.endswith("\u00e9"):
                pp_suffix = "\u00e9"
            # Formes sans accent (e/es au lieu de e/es)
            # "travaille" → -e mais attention aux verbes presents
            # On ne traite que les formes accentuees pour eviter les FP
            # avec les verbes au present (mange = present indicatif)

            if pp_suffix is None:
                continue
            if len(forme_low) < 3:
                continue

            # Garde 3 : forme est VER participe dans le lexique
            if not lex.existe(forme_low):
                continue
            infos = lex.info(forme_low)
            est_pp = any(
                e.get("cgram", "").startswith("VER")
                and e.get("mode") in ("par", "participe")
                for e in infos
            )
            if not est_pp:
                continue

            # Garde 6 : si forme est aussi NOM plus frequent que VER participe, skip
            nom_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos if e.get("cgram", "").startswith("NOM")),
                default=0.0,
            )
            ver_par_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos
                 if (e.get("cgram", "").startswith("VER")
                     and e.get("mode") in ("par", "participe"))),
                default=0.0,
            )
            if nom_max_freq > ver_par_max_freq:
                continue

            # Garde 4 : construire la forme INF en -er et verifier qu'elle
            # existe dans le lexique comme VER infinitif
            inf_forme = forme_low[:-len(pp_suffix)] + "er"
            if not lex.existe(inf_forme):
                continue
            inf_infos = lex.info(inf_forme)
            est_inf = any(
                e.get("cgram", "").startswith("VER")
                and len(e.get("multext") or "") >= 3
                and (e.get("multext") or "")[2] == "n"
                for e in inf_infos
            )
            if not est_inf:
                continue

            # Garde 5 : mot precedent est modal ou preposition (fenetre 2)
            # Un clitique peut s'intercaler : "pour le mange" -> "pour le manger"
            trigger_found = False
            trigger_mot = ""
            if i >= 1:
                prev_low = mots[i - 1].forme.lower()
                # Garde 7 : auxiliaire avoir/etre → ne pas corriger
                if prev_low in _AVOIR_CONJUGUE or prev_low in _AUXILIAIRES_ETRE:
                    continue
                if prev_low in _MODAUX_ELARGI or prev_low in _PREPS_INF_STRUCT:
                    trigger_found = True
                    trigger_mot = prev_low
                elif (prev_low in _CLITIQUES_OBJETS
                      or prev_low in ("ne", "n'", "n\u2019", "pas", "jamais",
                                      "plus", "rien")) and i >= 2:
                    # Clitique ou negation intercale, verifier le mot d'avant
                    pprev_low = mots[i - 2].forme.lower()
                    if pprev_low in _AVOIR_CONJUGUE or pprev_low in _AUXILIAIRES_ETRE:
                        continue
                    if pprev_low in _MODAUX_ELARGI or pprev_low in _PREPS_INF_STRUCT:
                        trigger_found = True
                        trigger_mot = pprev_low

            if not trigger_found:
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, inf_forme)
            mv.regle = "participe.pp_inf.struct"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"PP->INF struct: '{mv.forme}' -> '{inf_forme}' "
                    f"apres '{trigger_mot}' (sans P2G)"
                ),
            ))

        # Pass 2 : formes -e SANS accent (VER present 1er groupe -> INF -er)
        # Ex: "pour mange" -> "pour manger", "sans oublie" -> "sans oublier"
        # Gardes plus strictes que Pass 1 car la forme est ambigue
        # (VER present vs NOM vs ADJ).
        _DETS_BLOCK_INF = frozenset({
            "le", "la", "l'", "l\u2019", "un", "une", "du", "au",
            "les", "des", "aux", "son", "sa", "ses", "mon", "ma",
            "ton", "ta", "ce", "cet", "cette", "ces",
            "notre", "votre", "nos", "vos", "leur", "leurs",
        })

        for i, mv in enumerate(mots):
            if mv.regle:
                continue

            forme_low = mv.forme.lower()

            if forme_low in _MOTS_PROTEGES:
                continue

            # Guard longueur : eviter "se", "ne", etc.
            if len(forme_low) < 4:
                continue

            # Guard nom propre : mot original commencant par majuscule -> skip
            # mv.correction preserve la casse originale (mv.forme peut etre
            # en minuscule apres tokenisation)
            _orig_forme = mv.correction or mv.forme
            if _orig_forme[0].isupper():
                continue

            # Guard : forme finit par -e (sans accent) et PAS par -ee/-e/-ie/-ue
            # On ne traite que -e simple (pas -es qui est trop ambigu)
            if not forme_low.endswith("e"):
                continue
            if forme_low.endswith(("\u00e9", "\u00e9e", "\u00e9s", "\u00e9es")):
                continue
            if forme_low.endswith("ee"):
                continue

            # Guard : le mot doit etre VER present indicatif dans le lexique
            if not lex.existe(forme_low):
                continue
            infos = lex.info(forme_low)
            is_ver_pre = any(
                e.get("cgram", "").startswith("VER")
                and e.get("mode") in ("ind", "indicatif")
                and e.get("temps") in ("pre", "present")
                for e in infos
            )
            if not is_ver_pre:
                continue

            # Guard NOM strict : si NOM existe, exiger VER_pre > 2*NOM
            # (forme sans accent = tres ambigue, il faut un ratio fort)
            _nom_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos if e.get("cgram", "").startswith("NOM")),
                default=0.0,
            )
            _ver_pre_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos
                 if (e.get("cgram", "").startswith("VER")
                     and e.get("mode") in ("ind", "indicatif")
                     and e.get("temps") in ("pre", "present"))),
                default=0.0,
            )
            _phase2_freq = False
            if _nom_max_freq > 0 and _ver_pre_max_freq < 2.0 * _nom_max_freq:
                if self._config.activer_phase2 and _ver_pre_max_freq >= 1.0 * _nom_max_freq:
                    _phase2_freq = True
                else:
                    continue

            # Guard ADJ : si freq ADJ > freq VER present -> skip
            _adj_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos if e.get("cgram", "").startswith("ADJ")),
                default=0.0,
            )
            if _adj_max_freq > _ver_pre_max_freq:
                continue

            # Guard PRE : si le mot est aussi preposition -> skip
            # (ex: "contre" est PRE bien plus souvent que VER)
            _pre_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos if e.get("cgram", "").startswith("PRE")),
                default=0.0,
            )
            if _pre_max_freq > _ver_pre_max_freq:
                continue

            # Guard freq minimum : VER present trop rare -> risque FP
            if _ver_pre_max_freq < 1.0:
                continue

            # Construire la forme INF en -er et verifier qu'elle existe
            inf_forme = forme_low[:-1] + "er"
            if not lex.existe(inf_forme):
                continue
            inf_infos = lex.info(inf_forme)
            est_inf = any(
                e.get("cgram", "").startswith("VER")
                and len(e.get("multext") or "") >= 3
                and (e.get("multext") or "")[2] == "n"
                for e in inf_infos
            )
            if not est_inf:
                continue

            # Guard trigger : mot precedent est prep ou modal (fenetre 2)
            trigger_found = False
            trigger_mot = ""
            if i >= 1:
                prev_low = mots[i - 1].forme.lower()
                # Guard auxiliaire : avoir/etre -> skip (contexte PP)
                if prev_low in _AVOIR_CONJUGUE or prev_low in _AUXILIAIRES_ETRE:
                    continue
                # Guard determinant intercale : "de l'oeuvre", "pour le reste"
                # -> la presence d'un DET avant le mot indique un NOM
                if prev_low in _DETS_BLOCK_INF:
                    continue
                if prev_low in _MODAUX_ELARGI or prev_low in _PREPS_INF_STRUCT:
                    trigger_found = True
                    trigger_mot = prev_low
                elif (prev_low in _CLITIQUES_OBJETS
                      or prev_low in ("ne", "n'", "n\u2019", "pas", "jamais",
                                      "plus", "rien")) and i >= 2:
                    pprev_low = mots[i - 2].forme.lower()
                    if (pprev_low in _AVOIR_CONJUGUE
                            or pprev_low in _AUXILIAIRES_ETRE):
                        continue
                    if (pprev_low in _MODAUX_ELARGI
                            or pprev_low in _PREPS_INF_STRUCT):
                        trigger_found = True
                        trigger_mot = pprev_low

            if not trigger_found:
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, inf_forme)
            mv.regle = "participe.pp_inf.struct.noaccent"
            if _phase2_freq:
                mv.regle = "p2." + mv.regle

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"PP->INF struct (noaccent): '{mv.forme}' -> '{inf_forme}' "
                    f"apres '{trigger_mot}'"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3f. Accord DET-NOM (les chien -> les chiens)
    # ------------------------------------------------------------------

    # Determinants avec nombre connu (pour accord DET-NOM)
    _DET_NOMBRE: dict[str, str] = {
        "le": "s", "la": "s", "l'": "s", "l\u2019": "s",
        "un": "s", "une": "s", "du": "s", "au": "s",
        "ce": "s", "cet": "s", "cette": "s",
        "mon": "s", "ma": "s", "ton": "s", "ta": "s",
        "son": "s", "sa": "s",
        "notre": "s", "votre": "s", "leur": "s",
        "chaque": "s", "quel": "s", "quelle": "s",
        "les": "p", "des": "p", "aux": "p",
        "ces": "p", "mes": "p", "tes": "p", "ses": "p",
        "nos": "p", "vos": "p", "leurs": "p",
        "quelques": "p", "plusieurs": "p",
        "quels": "p", "quelles": "p",
    }

    def _corriger_accord_det_nom(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige l'accord en nombre DET -> NOM : 'les chien' -> 'les chiens'.

        Gardes :
        1. Le mot doit etre un NOM (G2P POS = NOM)
        2. Un DET doit preceder (fenetre=1)
        3. Le nombre du DET doit diverger du nombre du NOM (G2P)
        4. La forme corrigee doit exister dans le lexique avec le meme lemme
        5. Le NOM dans le lexique ne doit PAS avoir deja le bon nombre
        6. P2G confirme le nombre du DET (double check)
        """
        corrections: list[Correction] = []
        lex = self._lexique

        for i, mv in enumerate(mots):
            # Deja corrige ?
            if mv.regle:
                continue

            # Pas de correction sur mots proteges
            if mv.forme.lower() in _MOTS_PROTEGES:
                continue

            # Garde 1 : le mot doit etre un NOM
            if not mv.g2p_pos or not mv.g2p_pos.startswith("NOM"):
                continue

            # Garde : ponctuation entre le mot precedent et celui-ci
            # (ex: "trois, feu!" — le numeral ne gouverne pas le NOM)
            if mv.preceded_by_punct:
                continue

            # Le mot doit exister dans le lexique
            if not lex.existe(mv.forme):
                continue

            # Pas de correction sur mots courts
            if len(mv.forme) <= 2:
                continue

            # Garde : nombres (invariables en francais)
            if mv.forme.lower() in _NUMERAUX_PLURIEL:
                continue

            # Garde : formes contenant des caracteres non-alphabetiques
            # (parentheses, tirets partiels, etc.)
            if not mv.forme.isalpha():
                continue

            # Garde : noms invariables en "s" (fils, bras, corps, etc.)
            if mv.forme.lower() in _INVARIABLES_S:
                continue

            # Garde : NOM capitalise en milieu de phrase = nom propre/demonyme
            # (Burkinabe, Americains, etc.) — souvent invariable ou cas special
            if (i > 0 and mv.correction
                    and mv.correction[0].isupper()):
                continue

            # Garde 1b : pas de correction si le mot suivant est "!" (interjection)
            if i < len(mots) - 1 and mots[i + 1].forme in ("!", "!…"):
                continue

            # Garde 2 : un DET doit preceder (fenetre=1)
            if i == 0:
                continue
            prev = mots[i - 1]
            det_low = prev.forme.lower()
            # Aussi verifier la correction du DET
            det_corr = prev.correction.lower() if prev.correction else det_low

            # Garde : possessifs ambigus (leur/notre/votre sont singuliers
            # en forme mais peuvent accompagner un pluriel semantique)
            # "leur activités" est ambigu → ne pas corriger le NOM
            _possessif_phase2 = False
            if det_low in ("leur", "notre", "votre"):
                if self._config.activer_phase2 and mv.div_ortho:
                    _possessif_phase2 = True
                else:
                    continue

            det_nombre = self._DET_NOMBRE.get(det_low) or self._DET_NOMBRE.get(det_corr)
            if not det_nombre:
                # Essayer les numeraux > 1 (deux, trois, etc.)
                if det_low in _NUMERAUX_PLURIEL or det_corr in _NUMERAUX_PLURIEL:
                    det_nombre = "p"
                else:
                    continue

            # Garde 3 : le nombre LEXICAL du NOM doit diverger du DET
            # NB: on utilise le nombre du lexique (pas G2P/P2G) car G2P est
            # context-aware et predit nombre=p pour "chien" dans "les chien".
            infos_nom = lex.info(mv.forme)
            nom_nombre_lex = None
            for entry in infos_nom:
                if entry.get("cgram", "").startswith("NOM"):
                    nom_nombre_lex = entry.get("nombre", "")
                    break
            if not nom_nombre_lex or nom_nombre_lex == det_nombre:
                continue  # Deja accorde ou pas de nombre lexical

            # Garde 5 : verifier dans le lexique que le NOM n'a PAS deja
            # le bon nombre (certains NOM sont invariables)
            nom_deja_accorde = False
            for entry in infos_nom:
                if not entry.get("cgram", "").startswith("NOM"):
                    continue
                e_n = entry.get("nombre", "")
                if e_n == det_nombre:
                    nom_deja_accorde = True
                    break
            if nom_deja_accorde:
                continue

            # Garde 4 : chercher la forme corrigee dans le lexique
            forme_corrigee = self._trouver_forme_accord_nom(
                mv.forme, det_nombre,
            )
            if forme_corrigee is None or forme_corrigee == mv.forme:
                continue

            # Verifier que la forme corrigee existe dans le lexique
            if not lex.existe(forme_corrigee):
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, forme_corrigee)
            mv.regle = "accord.det_nom"
            if _possessif_phase2:
                mv.regle = "p2." + mv.regle

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"Accord DET-NOM: '{mv.forme}' -> '{forme_corrigee}' "
                    f"(det={det_low}, nombre={det_nombre})"
                ),
            ))

        return corrections

    def _trouver_forme_accord_nom(
        self,
        forme: str,
        nombre_cible: str,
    ) -> str | None:
        """Cherche la forme NOM accordee en nombre dans le lexique.

        Verifie que la forme retournee a le meme lemme que l'original.
        """
        lex = self._lexique

        # Ne pas modifier les noms invariables en "s"
        if forme.lower() in _INVARIABLES_S:
            return None

        # Obtenir les lemmes du NOM original
        infos = lex.info(forme)
        if not infos:
            return None

        lemmes = {e.get("lemme", "") for e in infos if e.get("lemme") and e.get("cgram", "").startswith("NOM")}
        if not lemmes:
            return None

        for lemme in lemmes:
            if not lemme:
                continue

            # Methode 1 : formes_de(lemme)
            if hasattr(lex, "formes_de"):
                try:
                    flexions = lex.formes_de(lemme)
                    for entry in flexions:
                        if not entry.get("cgram", "").startswith("NOM"):
                            continue
                        e_n = normaliser_morpho(entry.get("nombre", ""))
                        if e_n == nombre_cible:
                            entry_forme = entry.get("ortho", "")
                            if entry_forme and entry_forme != forme:
                                return entry_forme
                except Exception:
                    pass

        # Strategie simple : suffixes courants
        if nombre_cible == "p" and not forme.endswith("s"):
            if forme.endswith("al"):
                candidat = forme[:-2] + "aux"
                if lex.existe(candidat):
                    return candidat
            candidat = forme + "s"
            if lex.existe(candidat):
                return candidat
        if nombre_cible == "s" and forme.endswith("s") and len(forme) > 2:
            candidat = forme[:-1]
            if lex.existe(candidat):
                # Verifier que c'est le meme lemme (eviter fils→fil)
                infos_cand = lex.info(candidat)
                lemmes_cand = {e.get("lemme") for e in infos_cand if e.get("lemme")}
                if lemmes & lemmes_cand:
                    return candidat
            # -aux -> -al
            if forme.endswith("aux"):
                candidat = forme[:-3] + "al"
                if lex.existe(candidat):
                    infos_cand = lex.info(candidat)
                    lemmes_cand = {e.get("lemme") for e in infos_cand if e.get("lemme")}
                    if lemmes & lemmes_cand:
                        return candidat

        return None

    # ------------------------------------------------------------------
    # 3f-bis. Accord NOM+ADJ en nombre
    # ------------------------------------------------------------------

    def _corriger_accord_nom_adj(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige un ADJ postpose a un NOM avec desaccord en nombre.

        Cas typiques :
        - "des capteurs energetique" -> "energetiques"
        - "fonctions necessaire" -> "necessaires"

        Gardes :
        1. mv.regle vide
        2. mv.div_ortho=True (le P2G a detecte une divergence)
        3. Le mot est ADJ dans le lexique (g2p_pos commence par ADJ)
        4. Le mot precedent est NOM (g2p_pos commence par NOM)
        5. Le nombre lexique de l'ADJ differe du nombre du NOM precedent
        6. La forme ADJ accordee existe dans le lexique (meme lemme)
        7. Confiance P2G >= 0.80
        8. Guard invariable : si ADJ est invariable, skip
        """
        corrections: list[Correction] = []
        lex = self._lexique
        from lectura_correcteur._utils import normaliser_morpho

        for i, mv in enumerate(mots):
            # Garde 1 : deja corrige ?
            if mv.regle:
                continue

            # Garde 2 : divergence P2G
            if not mv.div_ortho:
                continue

            # Garde 7 : confiance P2G
            if mv.p2g_confiance < 0.70:
                continue

            forme_low = mv.forme.lower()

            # Garde : mots proteges
            if forme_low in _MOTS_PROTEGES:
                continue

            # Garde : mots courts
            if len(forme_low) <= 2:
                continue

            # Garde : ponctuation entre mot precedent et celui-ci
            if mv.preceded_by_punct:
                continue

            # Garde 3 : le mot est ADJ (POS G2P ou P2G)
            pos_adj = False
            if mv.g2p_pos and mv.g2p_pos.startswith("ADJ"):
                pos_adj = True
            elif mv.p2g_pos and mv.p2g_pos.startswith("ADJ"):
                pos_adj = True
            if not pos_adj:
                continue

            # Garde 4 : le mot precedent est NOM
            if i == 0:
                continue
            prev = mots[i - 1]
            prev_is_nom = False
            if prev.g2p_pos and prev.g2p_pos.startswith("NOM"):
                prev_is_nom = True
            elif prev.p2g_pos and prev.p2g_pos.startswith("NOM"):
                prev_is_nom = True
            if not prev_is_nom:
                continue

            # Garde coordination : si "et"/"ou" dans la fenetre 5 mots
            # avant l'ADJ, l'accord peut etre avec des NOM coordonnes
            # → trop risque de corriger
            coord_found = False
            for j in range(max(0, i - 5), i):
                if mots[j].forme.lower() in ("et", "ou"):
                    coord_found = True
                    break
            if coord_found:
                continue

            # Obtenir le nombre et le genre du NOM precedent depuis le lexique
            prev_low = prev.forme.lower()
            if not lex.existe(prev_low):
                continue
            infos_prev = lex.info(prev_low)
            nom_nombre = None
            nom_genre = ""
            for entry in infos_prev:
                if entry.get("cgram", "").startswith("NOM"):
                    n = normaliser_morpho(entry.get("nombre", ""))
                    g = normaliser_morpho(entry.get("genre", ""))
                    if n not in ("s", "p") or g not in ("m", "f"):
                        mt_g, mt_n = self._genre_nombre_from_multext(
                            entry.get("multext", ""),
                        )
                        if n not in ("s", "p"):
                            n = mt_n
                        if g not in ("m", "f"):
                            g = mt_g
                    if n in ("s", "p"):
                        nom_nombre = n
                        nom_genre = g if g in ("m", "f") else ""
                        break
            if not nom_nombre:
                continue

            # Garde 5 : le nombre lexique de l'ADJ differe du NOM
            if not lex.existe(forme_low):
                continue
            infos_adj = lex.info(forme_low)
            adj_nombre = None
            for entry in infos_adj:
                if entry.get("cgram", "").startswith("ADJ"):
                    n = normaliser_morpho(entry.get("nombre", ""))
                    if n not in ("s", "p"):
                        # Fallback Multext
                        _, mt_n = self._genre_nombre_from_multext(
                            entry.get("multext", ""),
                        )
                        n = mt_n
                    if n in ("s", "p"):
                        adj_nombre = n
                        break
            if not adj_nombre:
                continue
            if adj_nombre == nom_nombre:
                continue  # Deja accorde

            # Garde 8 : ADJ invariable (formes identiques en s et p)
            # Verifier si l'ADJ a les deux nombres dans le lexique
            # sous la meme orthographe
            adj_invariable = False
            nombre_set = set()
            for entry in infos_adj:
                if entry.get("cgram", "").startswith("ADJ"):
                    n = normaliser_morpho(entry.get("nombre", ""))
                    if n not in ("s", "p"):
                        _, mt_n = self._genre_nombre_from_multext(
                            entry.get("multext", ""),
                        )
                        n = mt_n
                    if n in ("s", "p"):
                        nombre_set.add(n)
            if "s" in nombre_set and "p" in nombre_set:
                adj_invariable = True
            if adj_invariable:
                continue

            # Garde 6 : trouver la forme ADJ accordee dans le lexique
            forme_accordee = self._trouver_forme_accord(
                forme_low, "ADJ", nom_nombre, nom_genre,
            )
            if not forme_accordee or forme_accordee == forme_low:
                # Strategie simple par suffixe
                if nom_nombre == "p" and not forme_low.endswith("s"):
                    candidat = forme_low + "s"
                    if lex.existe(candidat):
                        # Verifier meme lemme
                        lemmes_orig = {e.get("lemme") for e in infos_adj
                                       if e.get("lemme") and e.get("cgram", "").startswith("ADJ")}
                        infos_cand = lex.info(candidat)
                        lemmes_cand = {e.get("lemme") for e in infos_cand
                                       if e.get("lemme") and e.get("cgram", "").startswith("ADJ")}
                        if lemmes_orig & lemmes_cand:
                            forme_accordee = candidat
                elif nom_nombre == "s" and forme_low.endswith("s") and len(forme_low) > 2:
                    candidat = forme_low[:-1]
                    if lex.existe(candidat):
                        lemmes_orig = {e.get("lemme") for e in infos_adj
                                       if e.get("lemme") and e.get("cgram", "").startswith("ADJ")}
                        infos_cand = lex.info(candidat)
                        lemmes_cand = {e.get("lemme") for e in infos_cand
                                       if e.get("lemme") and e.get("cgram", "").startswith("ADJ")}
                        if lemmes_orig & lemmes_cand:
                            forme_accordee = candidat

            if not forme_accordee or forme_accordee == forme_low:
                continue

            # Verifier que la forme accordee existe dans le lexique
            if not lex.existe(forme_accordee):
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, forme_accordee)
            mv.regle = "accord.nom_adj"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"Accord NOM+ADJ: '{mv.forme}' -> '{forme_accordee}' "
                    f"(nom={prev_low}, nombre={nom_nombre})"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3f-ter. Phase 2 — NOM generique via P2G
    # ------------------------------------------------------------------

    def _corriger_nom_p2g(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige le nombre d'un NOM via le signal P2G (Phase 2 suggestion).

        Cas typique : "les activite" -> "activites" quand P2G propose
        la forme plurielle avec confiance suffisante et changement
        de nombre morphologique confirme.

        Guards :
        1. mv.regle vide (pas deja corrige)
        2. mv.div_ortho = True (P2G diverge)
        3. P2G propose une forme differente
        4. mv.p2g_confiance >= 0.65
        5. mv.g2p_pos.startswith("NOM")
        6. Pas dans _MOTS_PROTEGES, _INVARIABLES_S
        7. len(forme) > 2, alphabetique
        8. Pas nom propre capitalise en milieu de phrase
        9. P2G existe dans le lexique comme NOM
        10. Changement de nombre morphologique (_est_changement_nombre_seul)
        11. Guard contexte : sans DET/PREP, exiger confiance >= 0.90
        12. Pas de ponctuation intercalaire
        """
        corrections: list[Correction] = []
        lex = self._lexique

        for i, mv in enumerate(mots):
            # 1. Pas deja corrige
            if mv.regle:
                continue

            # 2. Divergence ortho
            if not mv.div_ortho:
                continue

            # 3. P2G propose une forme differente
            p2g = mv.p2g_ortho
            if not p2g or p2g == mv.forme:
                continue

            # 4. Confiance P2G suffisante
            if mv.p2g_confiance < 0.65:
                continue

            # 5. G2P dit NOM
            if not mv.g2p_pos or not mv.g2p_pos.startswith("NOM"):
                continue

            forme_low = mv.forme.lower()
            p2g_low = p2g.lower()

            # 6. Pas dans mots proteges ni invariables
            if forme_low in _MOTS_PROTEGES:
                continue
            if forme_low in _INVARIABLES_S:
                continue

            # 7. Longueur et alphabetique
            if len(forme_low) <= 2:
                continue
            if not forme_low.isalpha():
                continue

            # 8. Pas nom propre capitalise en milieu de phrase
            _orig = mv.correction or mv.forme
            if i > 0 and _orig[0].isupper():
                continue

            # 12. Pas de ponctuation intercalaire
            if mv.preceded_by_punct:
                continue

            # Guard contexte : sans DET/PREP confirmant, exiger confiance haute
            _ctx_ok = self._contexte_confirme_nombre_p2g(mots, i, mv, p2g_low)
            if not _ctx_ok and mv.p2g_confiance < 0.90:
                continue

            # 9. P2G existe dans le lexique comme NOM
            if not lex.existe(p2g_low):
                continue
            p2g_infos = lex.info(p2g_low)
            p2g_est_nom = any(
                (e.get("cgram") or "").startswith("NOM") for e in p2g_infos
            )
            if not p2g_est_nom:
                continue

            # 10. Changement de nombre morphologique
            if not self._est_changement_nombre_seul(forme_low, p2g_low):
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = "p2.accord.nom_p2g"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"NOM P2G: '{mv.forme}' -> '{p2g}' "
                    f"(changement nombre)"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3f-quater. ADJ generique via P2G (Phase 2)
    # ------------------------------------------------------------------

    def _corriger_adj_p2g(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige le nombre d'un ADJ via le signal P2G (Phase 2).

        Cas typique : "des resultats important" -> "importants" quand P2G
        propose la forme plurielle avec confiance suffisante et changement
        de nombre morphologique confirme.

        Guards :
        1. mv.regle vide (pas deja corrige)
        2. mv.div_ortho = True (P2G diverge)
        3. P2G propose une forme differente
        4. mv.p2g_confiance >= 0.65
        5. g2p_pos ou p2g_pos commence par "ADJ"
        6. Pas dans _MOTS_PROTEGES
        7. len(forme) > 2, alphabetique
        8. Pas nom propre capitalise en milieu de phrase
        9. Pas de ponctuation intercalaire
        10. P2G existe dans le lexique comme ADJ
        11. Changement de nombre morphologique (_est_changement_nombre_seul)
        12. Guard contexte : sans DET/PREP, exiger confiance >= 0.90
        """
        corrections: list[Correction] = []
        lex = self._lexique

        for i, mv in enumerate(mots):
            # 1. Pas deja corrige
            if mv.regle:
                continue

            # 2. Divergence ortho
            if not mv.div_ortho:
                continue

            # 3. P2G propose une forme differente
            p2g = mv.p2g_ortho
            if not p2g or p2g == mv.forme:
                continue

            # 4. Confiance P2G suffisante
            if mv.p2g_confiance < 0.65:
                continue

            # 5. G2P ou P2G dit ADJ
            g2p_adj = mv.g2p_pos and mv.g2p_pos.startswith("ADJ")
            p2g_adj = mv.p2g_pos and mv.p2g_pos.startswith("ADJ")
            if not g2p_adj and not p2g_adj:
                continue

            forme_low = mv.forme.lower()
            p2g_low = p2g.lower()

            # 6. Pas dans mots proteges
            if forme_low in _MOTS_PROTEGES:
                continue

            # 7. Longueur et alphabetique
            if len(forme_low) <= 2:
                continue
            if not forme_low.isalpha():
                continue

            # 8. Pas nom propre capitalise en milieu de phrase
            _orig = mv.correction or mv.forme
            if i > 0 and _orig[0].isupper():
                continue

            # 9. Pas de ponctuation intercalaire
            if mv.preceded_by_punct:
                continue

            # Guard contexte : sans DET/PREP confirmant, exiger confiance haute
            _ctx_ok = self._contexte_confirme_nombre_p2g(mots, i, mv, p2g_low)
            if not _ctx_ok and mv.p2g_confiance < 0.90:
                continue

            # 10. P2G existe dans le lexique comme ADJ
            if not lex.existe(p2g_low):
                continue
            p2g_infos = lex.info(p2g_low)
            p2g_est_adj = any(
                (e.get("cgram") or "").startswith("ADJ") for e in p2g_infos
            )
            if not p2g_est_adj:
                continue

            # 11. Changement de nombre morphologique
            if not self._est_changement_nombre_seul(forme_low, p2g_low):
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = "p2.accord.adj_p2g"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"ADJ P2G: '{mv.forme}' -> '{p2g}' "
                    f"(changement nombre)"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3g. Accord attribut a travers verbe d'etat
    # ------------------------------------------------------------------

    def _corriger_accord_attribut(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige l'accord de l'attribut avec le sujet a travers un verbe d'etat.

        Cas typiques :
        - "les fleurs sont tres belle" -> "belles" (sujet pluriel)
        - "cette robe est trop petit" -> "petite" (sujet feminin)

        Declencheur : div_ortho=True pour un ADJ apres un verbe d'etat,
        OU le P2G predit une variante morphologique du meme ADJ.

        Gardes :
        1. Un verbe d'etat doit preceder l'ADJ (fenetre 3 mots, ADV entre ok)
        2. Un sujet NOM/PRO doit preceder le verbe d'etat
        3. La forme corrigee doit exister dans le lexique
        4. p2g_confiance >= homophone_confiance_min
        """
        corrections: list[Correction] = []
        lex = self._lexique
        cfg = self._config

        for i, mv in enumerate(mots):
            if mv.regle:
                continue

            # Pas de correction sur mots proteges
            if mv.forme.lower() in _MOTS_PROTEGES:
                continue

            # Garde passe simple : ne pas modifier les formes verbales
            _fl_attr = mv.forme.lower()
            if _fl_attr in _PASSE_SIMPLE_SUBJ:
                continue
            if (_fl_attr.endswith(_PASSE_SIMPLE_SUFFIXES)
                    and lex.existe(_fl_attr)):
                _infos_ps = lex.info(_fl_attr)
                if any(e.get("cgram", "").startswith("VER") for e in _infos_ps):
                    continue

            # Le mot doit etre un ADJ (pas ADJ:dem, ADJ:pos, ADJ:ind = determinants)
            if not mv.g2p_pos or not mv.g2p_pos.startswith("ADJ"):
                continue
            if mv.g2p_pos.startswith(("ADJ:pos", "ADJ:dem", "ADJ:ind")):
                continue

            # Pas de correction sur mots courts (determinants mal tagges)
            if len(mv.forme) <= 2:
                continue

            # Pas de correction si le mot est ambigu (peut etre ADV, NOM, etc.)
            if self._est_ambigu_accord(mv.forme, "ADJ"):
                continue

            # Chercher un verbe d'etat dans les 3 mots precedents
            # (autorise ADV entre : "est tres belle", "sont trop petit")
            idx_vetat = None
            for j in range(max(0, i - 3), i):
                mj_low = mots[j].forme.lower()
                if mj_low in _VERBES_ETAT:
                    idx_vetat = j
                    break
                # Seuls les ADV sont autorises entre verbe d'etat et ADJ
                if mots[j].g2p_pos and not mots[j].g2p_pos.startswith("ADV"):
                    if j > max(0, i - 3):
                        break  # non-ADV entre le verbe d'etat et l'ADJ -> stop

            if idx_vetat is None:
                continue

            # Verifier que seuls des ADV sont entre le verbe d'etat et l'ADJ
            # (bloque "sont de grands" ou "est en bonne")
            _intervening_ok = True
            for _k in range(idx_vetat + 1, i):
                _mk = mots[_k]
                if not _mk.g2p_pos or not _mk.g2p_pos.startswith("ADV"):
                    _intervening_ok = False
                    break
            if not _intervening_ok:
                continue

            # Chercher le sujet avant le verbe d'etat
            sujet = self._trouver_sujet(mots, idx_vetat)
            if sujet is None:
                continue

            # Le sujet doit avoir un nombre OU genre determine
            # et G2P/P2G doivent s'accorder
            nombre_sujet = sujet.p2g_nombre or sujet.g2p_nombre
            genre_sujet = sujet.p2g_genre or sujet.g2p_genre

            if not nombre_sujet and not genre_sujet:
                continue

            # Si les deux modeles divergent sur le sujet -> pas fiable
            if (sujet.g2p_nombre and sujet.p2g_nombre
                    and sujet.g2p_nombre != sujet.p2g_nombre):
                continue
            if (sujet.g2p_genre and sujet.p2g_genre
                    and sujet.g2p_genre != sujet.p2g_genre):
                continue

            # Si le sujet est un NOM, confirmer son nombre par determinant
            if sujet.g2p_pos and sujet.g2p_pos.startswith("NOM"):
                if nombre_sujet and not self._contexte_confirme_nombre(
                    mots, sujet.index, nombre_sujet,
                ):
                    continue

            # Verifier que l'ADJ ne correspond pas deja au sujet
            adj_ok = True
            if nombre_sujet:
                adj_nombre = mv.g2p_nombre or mv.p2g_nombre
                if adj_nombre and adj_nombre != nombre_sujet:
                    adj_ok = False
            if genre_sujet:
                adj_genre = mv.g2p_genre or mv.p2g_genre
                if adj_genre and adj_genre != genre_sujet:
                    adj_ok = False
            if adj_ok:
                continue  # L'ADJ est deja accorde

            # Garde supplementaire : verifier dans le lexique que l'ADJ
            # n'est PAS deja accorde avec le sujet
            if lex.existe(mv.forme):
                adj_lex_infos = lex.info(mv.forme)
                adj_lex_ok = False
                for entry in adj_lex_infos:
                    if not entry.get("cgram", "").startswith("ADJ"):
                        continue
                    e_n = entry.get("nombre", "")
                    e_g = entry.get("genre", "")
                    # Fallback Multext
                    if e_g not in ("m", "f") or e_n not in ("s", "p"):
                        mt_g, mt_n = self._genre_nombre_from_multext(entry.get("multext", ""))
                        if e_g not in ("m", "f"):
                            e_g = mt_g
                        if e_n not in ("s", "p"):
                            e_n = mt_n
                    n_ok = not nombre_sujet or e_n == nombre_sujet
                    g_ok = not genre_sujet or e_g == genre_sujet
                    if n_ok and g_ok:
                        adj_lex_ok = True
                        break
                if adj_lex_ok:
                    continue  # Lexique confirme ADJ deja accorde

            # Cas 1 : le P2G propose directement la bonne forme
            if mv.div_ortho and mv.p2g_ortho and mv.p2g_ortho != mv.forme:
                p2g = mv.p2g_ortho
                if lex.existe(p2g) and mv.p2g_confiance >= cfg.homophone_confiance_min:
                    # Verifier que la forme P2G a le bon nombre/genre
                    infos_p2g = lex.info(p2g)
                    for entry in infos_p2g:
                        if not entry.get("cgram", "").startswith("ADJ"):
                            continue
                        e_n = entry.get("nombre", "")
                        e_g = entry.get("genre", "")
                        # Fallback Multext
                        if e_g not in ("m", "f") or e_n not in ("s", "p"):
                            mt_g, mt_n = self._genre_nombre_from_multext(entry.get("multext", ""))
                            if e_g not in ("m", "f"):
                                e_g = mt_g
                            if e_n not in ("s", "p"):
                                e_n = mt_n
                        n_ok = not nombre_sujet or e_n == nombre_sujet
                        g_ok = not genre_sujet or e_g == genre_sujet
                        if n_ok and g_ok:
                            mv.correction = transferer_casse(mv.correction, p2g)
                            mv.regle = "accord.attribut"
                            corrections.append(Correction(
                                index=mv.index,
                                original=mv.forme,
                                corrige=mv.correction,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle=mv.regle,
                                explication=(
                                    f"Accord attribut: '{mv.forme}' -> '{p2g}' "
                                    f"(sujet nombre={nombre_sujet}, genre={genre_sujet})"
                                ),
                            ))
                            break
                if mv.regle:
                    continue

            # Cas 2 : le P2G ne propose pas de correction, mais le lexique
            # a une variante avec le bon nombre/genre
            forme_corrigee = self._trouver_forme_accord(
                mv.forme, "ADJ", nombre_sujet, genre_sujet,
            )
            if forme_corrigee and forme_corrigee != mv.forme and lex.existe(forme_corrigee):
                # Verifier que la forme originale a bien le mauvais genre/nombre
                # (confirmation lexique que c'est une erreur, pas d'ambiguite)
                infos_orig = lex.info(mv.forme)
                orig_a_meme_genre_nombre = False
                for entry in infos_orig:
                    if not entry.get("cgram", "").startswith("ADJ"):
                        continue
                    e_n = entry.get("nombre", "")
                    e_g = entry.get("genre", "")
                    # Fallback Multext
                    if e_g not in ("m", "f") or e_n not in ("s", "p"):
                        mt_g, mt_n = self._genre_nombre_from_multext(entry.get("multext", ""))
                        if e_g not in ("m", "f"):
                            e_g = mt_g
                        if e_n not in ("s", "p"):
                            e_n = mt_n
                    n_match = not nombre_sujet or e_n == nombre_sujet
                    g_match = not genre_sujet or e_g == genre_sujet
                    if n_match and g_match:
                        orig_a_meme_genre_nombre = True
                        break

                if not orig_a_meme_genre_nombre:
                    mv.correction = transferer_casse(mv.correction, forme_corrigee)
                    mv.regle = "accord.attribut"
                    corrections.append(Correction(
                        index=mv.index,
                        original=mv.forme,
                        corrige=mv.correction,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle=mv.regle,
                        explication=(
                            f"Accord attribut: '{mv.forme}' -> '{forme_corrigee}' "
                            f"(sujet nombre={nombre_sujet}, genre={genre_sujet})"
                        ),
                    ))

        return corrections

    def _trouver_sujet(
        self,
        mots: list[MotV6],
        idx_verbe: int,
    ) -> MotV6 | None:
        """Cherche le sujet (NOM ou PRO) avant un verbe."""
        # Pronoms sujets
        _PRONOMS_SUJET = {
            "je", "j'", "tu", "il", "elle", "on",
            "nous", "vous", "ils", "elles",
        }
        _PRONOM_NOMBRE = {
            "je": "s", "j'": "s", "tu": "s", "il": "s", "elle": "s", "on": "s",
            "nous": "p", "vous": "p", "ils": "p", "elles": "p",
        }
        _PRONOM_GENRE = {
            "il": "m", "elle": "f", "ils": "m", "elles": "f",
        }

        for j in range(max(0, idx_verbe - 3), idx_verbe):
            mj = mots[j]
            mj_low = mj.forme.lower()

            if mj_low in _PRONOMS_SUJET:
                # Creer un sujet synthetique avec nombre/genre du pronom
                mj.p2g_nombre = mj.p2g_nombre or _PRONOM_NOMBRE.get(mj_low, "")
                mj.g2p_nombre = mj.g2p_nombre or _PRONOM_NOMBRE.get(mj_low, "")
                mj.p2g_genre = mj.p2g_genre or _PRONOM_GENRE.get(mj_low, "")
                mj.g2p_genre = mj.g2p_genre or _PRONOM_GENRE.get(mj_low, "")
                return mj

            if mj.g2p_pos and mj.g2p_pos.startswith("NOM"):
                return mj

        return None

    # ------------------------------------------------------------------
    # 3h. Accord PP + etre (elles sont arrive -> arrivees)
    # ------------------------------------------------------------------

    def _corriger_pp_etre(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige l'accord du participe passe avec le sujet apres auxiliaire etre.

        Cas typiques :
        - "elles sont arrive" -> "arrivees"
        - "nous sommes alle" -> "alles"
        - "elle est tombe" -> "tombee"

        Gardes :
        1. Le mot doit avoir div_ortho=True et P2G propose une forme differente
        2. Auxiliaire etre dans fenetre 3 mots (ADV/negation/clitiques intercales)
        3. Un sujet (pronom ou NOM+DET) doit preceder l'auxiliaire
        4. Le nombre/genre du sujet ne correspond pas a la forme actuelle du PP
        5. La forme PP accordee doit exister dans le lexique (meme lemme)
        6. Confiance P2G >= 0.75
        """
        corrections: list[Correction] = []
        lex = self._lexique

        for i, mv in enumerate(mots):
            # Deja corrige ?
            if mv.regle:
                continue

            # Pas de correction sur mots proteges
            if mv.forme.lower() in _MOTS_PROTEGES:
                continue

            # Garde passe simple : ne pas modifier les formes verbales
            # du passe simple (apparut, sortit, etc.)
            _fl_pp = mv.forme.lower()
            if _fl_pp in _PASSE_SIMPLE_SUBJ:
                continue
            if (_fl_pp.endswith(_PASSE_SIMPLE_SUFFIXES)
                    and lex.existe(_fl_pp)):
                _infos_ps = lex.info(_fl_pp)
                if any(e.get("cgram", "").startswith("VER") for e in _infos_ps):
                    continue

            # Garde 1 : divergence ortho et P2G propose une forme differente
            if not mv.div_ortho:
                continue
            p2g = mv.p2g_ortho
            if not p2g or p2g == mv.forme:
                continue

            # Garde 6 : confiance P2G suffisante
            # Seuil abaisse a 0.75 : le contexte etre+sujet est un signal
            # structurel fort qui compense une confiance P2G moindre
            if mv.p2g_confiance < 0.75:
                continue

            # Garde 2 : auxiliaire etre dans fenetre 3 mots (5 en Phase 2)
            # (accepte ADV, negation, clitiques intercales)
            # Detecte aussi le passif compose : "a ete + PP" / "ont ete + PP"
            _fenetre_etre = 5 if self._config.activer_phase2 else 3
            idx_aux = None
            for j in range(max(0, i - _fenetre_etre), i):
                mj_low = mots[j].forme.lower()
                if mj_low in _AUXILIAIRES_ETRE:
                    # Verifier que les mots entre etre et le PP sont des
                    # intercalaires autorises (ADV, negation, clitiques)
                    intercalaire_ok = True
                    for k in range(j + 1, i):
                        mk_low = mots[k].forme.lower()
                        if mk_low in _CLITIQUES_OBJETS:
                            continue
                        if mk_low in ("ne", "n'", "n\u2019", "pas", "jamais",
                                      "plus", "rien", "point", "guere"):
                            continue
                        mk_pos = mots[k].g2p_pos or ""
                        if mk_pos.startswith("ADV"):
                            continue
                        intercalaire_ok = False
                        break
                    if intercalaire_ok:
                        idx_aux = j
                        break
                # Passif compose : "a ete" / "ont ete" / "avait ete" etc.
                # Le PP s'accorde avec le sujet (comme etre, pas comme avoir)
                if mj_low in ("\u00e9t\u00e9", "ete") and j >= 1:
                    prev_j_low = mots[j - 1].forme.lower()
                    if prev_j_low in _AVOIR_CONJUGUE:
                        # Verifier intercalaires entre ete+1 et i
                        intercalaire_ok = True
                        for k in range(j + 1, i):
                            mk_low = mots[k].forme.lower()
                            if mk_low in _CLITIQUES_OBJETS:
                                continue
                            if mk_low in ("ne", "n'", "n\u2019", "pas",
                                          "jamais", "plus", "rien",
                                          "point", "guere"):
                                continue
                            mk_pos = mots[k].g2p_pos or ""
                            if mk_pos.startswith("ADV"):
                                continue
                            intercalaire_ok = False
                            break
                        if intercalaire_ok:
                            # Pointer vers avoir pour trouver le sujet
                            idx_aux = j - 1
                            break
            if idx_aux is None:
                continue
            _etre_distant = (i - idx_aux) > 3

            # Verifier que le mot actuel est un PP ou un verbe
            # (G2P le tague comme VER ou ADJ — les PP sont souvent ADJ)
            pos_ok = (
                (mv.g2p_pos and mv.g2p_pos.startswith(("VER", "ADJ")))
                or (mv.p2g_pos and mv.p2g_pos.startswith(("VER", "ADJ")))
            )
            if not pos_ok:
                continue

            # Garde 3 : trouver le sujet avant l'auxiliaire
            sujet = self._trouver_sujet(mots, idx_aux)
            if sujet is None:
                continue

            nombre_sujet = sujet.p2g_nombre or sujet.g2p_nombre
            genre_sujet = sujet.p2g_genre or sujet.g2p_genre

            if not nombre_sujet and not genre_sujet:
                continue

            # Garde 4 : verifier que l'ADJ/PP actuel ne correspond PAS au sujet
            # (sinon il est deja accorde)
            if lex.existe(mv.forme):
                infos_pp = lex.info(mv.forme)
                pp_deja_accorde = False
                for entry in infos_pp:
                    cgram = entry.get("cgram", "")
                    if not (cgram.startswith("VER") or cgram.startswith("ADJ")):
                        continue
                    e_n = entry.get("nombre", "")
                    e_g = entry.get("genre", "")
                    # Fallback Multext
                    if e_g not in ("m", "f") or e_n not in ("s", "p"):
                        mt_g, mt_n = self._genre_nombre_from_multext(entry.get("multext", ""))
                        if e_g not in ("m", "f"):
                            e_g = mt_g
                        if e_n not in ("s", "p"):
                            e_n = mt_n
                    n_ok = not nombre_sujet or e_n == nombre_sujet
                    g_ok = not genre_sujet or e_g == genre_sujet
                    if n_ok and g_ok:
                        pp_deja_accorde = True
                        break
                if pp_deja_accorde:
                    continue

            # Garde 5 : verifier que la forme P2G existe et a le bon accord
            if not lex.existe(p2g):
                continue

            # Verifier que la forme P2G a le bon nombre/genre
            infos_p2g = lex.info(p2g)
            p2g_accord_ok = False
            for entry in infos_p2g:
                cgram = entry.get("cgram", "")
                if not (cgram.startswith("VER") or cgram.startswith("ADJ")):
                    continue
                e_n = entry.get("nombre", "")
                e_g = entry.get("genre", "")
                # Fallback Multext
                if e_g not in ("m", "f") or e_n not in ("s", "p"):
                    mt_g, mt_n = self._genre_nombre_from_multext(entry.get("multext", ""))
                    if e_g not in ("m", "f"):
                        e_g = mt_g
                    if e_n not in ("s", "p"):
                        e_n = mt_n
                n_ok = not nombre_sujet or e_n == nombre_sujet
                g_ok = not genre_sujet or e_g == genre_sujet
                if n_ok and g_ok:
                    p2g_accord_ok = True
                    break
            if not p2g_accord_ok:
                # Fallback : chercher la forme accordee dans le lexique
                # via le lemme du PP
                forme_accord = self._trouver_forme_accord(
                    mv.forme, "VER", nombre_sujet, genre_sujet,
                )
                if forme_accord and forme_accord != mv.forme and lex.existe(forme_accord):
                    p2g = forme_accord
                else:
                    continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = "accord.pp_etre"
            if _etre_distant:
                mv.regle = "p2." + mv.regle

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"PP+etre: '{mv.forme}' -> '{p2g}' "
                    f"(sujet nombre={nombre_sujet}, genre={genre_sujet})"
                ),
            ))

        # Pass 2 : PP+etre structurel SANS div_ortho
        # Detecte les VER present (-e/-es 1er groupe) apres etre et les
        # corrige en PP accentue accorde avec le sujet.
        # Ex: "elle est tombe" -> "tombee", "ils sont monte" -> "montes"
        _PREPS_BLOCK_PP_ETRE = frozenset({
            "de", "d'", "d\u2019", "pour", "sans", "avant",
            "apr\u00e8s", "apres", "afin", "\u00e0", "a",
        })

        _DETS_BLOCK_PP_ETRE = frozenset({
            "le", "la", "l'", "l\u2019", "un", "une", "du", "au",
            "les", "des", "aux", "son", "sa", "ses", "mon", "ma",
            "ton", "ta", "ce", "cet", "cette", "ces",
            "notre", "votre", "nos", "vos", "leur", "leurs",
        })

        for i, mv in enumerate(mots):
            if mv.regle:
                continue

            forme_low = mv.forme.lower()

            if forme_low in _MOTS_PROTEGES:
                continue

            # Guard longueur
            if len(forme_low) < 3:
                continue

            # Guard nom propre : mot original commencant par majuscule -> skip
            _orig_forme2 = mv.correction or mv.forme
            if _orig_forme2[0].isupper():
                continue

            # Guard passe simple
            if forme_low in _PASSE_SIMPLE_SUBJ:
                continue
            if (forme_low.endswith(_PASSE_SIMPLE_SUFFIXES)
                    and lex.existe(forme_low)):
                _infos_ps = lex.info(forme_low)
                if any(e.get("cgram", "").startswith("VER")
                       for e in _infos_ps):
                    continue

            # Guard : la forme doit finir par -e ou -es (sans accent)
            # et NE PAS finir par -e/-ee/-es/-ees (deja traite par Pass 1)
            if forme_low.endswith(("\u00e9", "\u00e9e", "\u00e9s", "\u00e9es")):
                continue
            if not (forme_low.endswith("e") or forme_low.endswith("es")):
                continue
            # Exclure -ie (vie, serie, etc.) et -ue (vue, rue, etc.)
            # sauf si VER present freq > NOM freq (vérifiee apres)
            _ends_ie = forme_low.endswith("ie") or forme_low.endswith("ies")
            _ends_ue = forme_low.endswith("ue") or forme_low.endswith("ues")

            # Guard : forme doit etre VER present indicatif dans le lexique
            if not lex.existe(forme_low):
                continue
            infos = lex.info(forme_low)
            is_ver_pre = any(
                e.get("cgram", "").startswith("VER")
                and e.get("mode") in ("ind", "indicatif")
                and e.get("temps") in ("pre", "present")
                for e in infos
            )
            if not is_ver_pre:
                continue

            # Guard NOM strict : si NOM existe, exiger VER_pre > 1.5*NOM
            _nom_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos if e.get("cgram", "").startswith("NOM")),
                default=0.0,
            )
            _ver_pre_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos
                 if (e.get("cgram", "").startswith("VER")
                     and e.get("mode") in ("ind", "indicatif")
                     and e.get("temps") in ("pre", "present"))),
                default=0.0,
            )
            _phase2_freq = False
            if _nom_max_freq > 0 and _ver_pre_max_freq < 1.5 * _nom_max_freq:
                if self._config.activer_phase2 and _ver_pre_max_freq >= 0.5 * _nom_max_freq:
                    _phase2_freq = True
                else:
                    continue

            # Guard ADJ : si freq ADJ > freq VER present -> skip
            _adj_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos if e.get("cgram", "").startswith("ADJ")),
                default=0.0,
            )
            if _adj_max_freq > _ver_pre_max_freq:
                continue

            # Guard -ie/-ue : pour ces suffixes, exiger VER freq nettement > NOM
            if (_ends_ie or _ends_ue) and _ver_pre_max_freq <= _nom_max_freq:
                continue

            # Guard prep : si mot precedent est preposition -> skip (cas INF)
            if i >= 1:
                prev_low = mots[i - 1].forme.lower()
                if prev_low in _PREPS_BLOCK_PP_ETRE:
                    continue
                # Guard determinant : DET avant le mot -> c'est un NOM
                if prev_low in _DETS_BLOCK_PP_ETRE:
                    continue

            # Construire le PP accentue de base (forme -e -> -e)
            if forme_low.endswith("es"):
                pp_base = forme_low[:-2] + "\u00e9"
            elif forme_low.endswith("e"):
                pp_base = forme_low[:-1] + "\u00e9"
            else:
                continue

            # Verifier que pp_base existe comme VER participe
            if not lex.existe(pp_base):
                continue
            pp_infos = lex.info(pp_base)
            est_pp = any(
                e.get("cgram", "").startswith("VER")
                and e.get("mode") in ("par", "participe")
                for e in pp_infos
            )
            if not est_pp:
                continue

            # Chercher auxiliaire etre dans fenetre 3 (5 en Phase 2)
            # (inclut aussi passif compose "a ete")
            _fenetre_etre = 5 if self._config.activer_phase2 else 3
            idx_aux = None
            for j in range(max(0, i - _fenetre_etre), i):
                _mj_low = mots[j].forme.lower()
                if _mj_low in _AUXILIAIRES_ETRE:
                    intercalaire_ok = True
                    for k in range(j + 1, i):
                        mk_low = mots[k].forme.lower()
                        if mk_low in _CLITIQUES_OBJETS:
                            continue
                        if mk_low in ("ne", "n'", "n\u2019", "pas",
                                      "jamais", "plus", "rien",
                                      "point", "guere"):
                            continue
                        mk_pos = mots[k].g2p_pos or ""
                        if mk_pos.startswith("ADV"):
                            continue
                        intercalaire_ok = False
                        break
                    if intercalaire_ok:
                        idx_aux = j
                        break
                # Passif compose : "a ete + PP"
                if _mj_low in ("\u00e9t\u00e9", "ete") and j >= 1:
                    _prev_j_low = mots[j - 1].forme.lower()
                    if _prev_j_low in _AVOIR_CONJUGUE:
                        intercalaire_ok = True
                        for k in range(j + 1, i):
                            mk_low = mots[k].forme.lower()
                            if mk_low in _CLITIQUES_OBJETS:
                                continue
                            if mk_low in ("ne", "n'", "n\u2019", "pas",
                                          "jamais", "plus", "rien",
                                          "point", "guere"):
                                continue
                            mk_pos = mots[k].g2p_pos or ""
                            if mk_pos.startswith("ADV"):
                                continue
                            intercalaire_ok = False
                            break
                        if intercalaire_ok:
                            idx_aux = j - 1
                            break
            if idx_aux is None:
                continue
            _etre_distant = (i - idx_aux) > 3

            # Trouver le sujet
            sujet = self._trouver_sujet(mots, idx_aux)
            if sujet is None:
                continue

            nombre_sujet = sujet.p2g_nombre or sujet.g2p_nombre
            genre_sujet = sujet.p2g_genre or sujet.g2p_genre

            if not nombre_sujet and not genre_sujet:
                continue

            # Trouver la forme PP accordee
            forme_accord = self._trouver_forme_accord(
                pp_base, "VER", nombre_sujet, genre_sujet,
            )
            if not forme_accord:
                # Fallback : pp_base lui-meme si ms correspond au sujet
                if nombre_sujet in ("s", "") and genre_sujet in ("m", ""):
                    forme_accord = pp_base
                else:
                    continue

            # Guard : la forme accordee doit differer de la forme actuelle
            if forme_accord == forme_low:
                continue

            # Guard : la forme accordee doit exister dans le lexique
            if not lex.existe(forme_accord):
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, forme_accord)
            mv.regle = "accord.pp_etre.struct"
            if _phase2_freq or _etre_distant:
                mv.regle = "p2." + mv.regle

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"PP+etre struct: '{mv.forme}' -> '{forme_accord}' "
                    f"(sujet nombre={nombre_sujet}, genre={genre_sujet})"
                ),
            ))

        # Pass 3 : PP+etre structurel TOUS GROUPES sans div_ortho
        # Detecte les mots qui sont PP (multext Vmps*) apres etre et corrige
        # l'accord avec le sujet. Couvre les 2e/3e groupes :
        # "elle est connu" -> "connue", "ils sont arrive" -> "arrives"
        for i, mv in enumerate(mots):
            if mv.regle:
                continue

            forme_low = mv.forme.lower()

            if forme_low in _MOTS_PROTEGES:
                continue

            # Guard longueur
            if len(forme_low) < 3:
                continue

            # Guard nom propre
            _orig_forme3 = mv.correction or mv.forme
            if _orig_forme3[0].isupper():
                continue

            # Guard passe simple
            if forme_low in _PASSE_SIMPLE_SUBJ:
                continue
            if (forme_low.endswith(_PASSE_SIMPLE_SUFFIXES)
                    and lex.existe(forme_low)):
                _infos_ps = lex.info(forme_low)
                if any((e.get("cgram") or "").startswith("VER")
                       and (e.get("mode") or "") in ("ind", "indicatif",
                                                     "sub", "subjonctif")
                       for e in _infos_ps):
                    continue

            # Le mot doit etre un VER PP dans le lexique (multext Vmps*)
            if not lex.existe(forme_low):
                continue
            infos = lex.info(forme_low)
            est_pp = False
            for entry in infos:
                mt = entry.get("multext") or ""
                if mt.startswith("Vmps"):
                    est_pp = True
                    break
                cgram = entry.get("cgram") or ""
                mode = entry.get("mode") or ""
                if (cgram.startswith("VER")
                        and mode in ("par", "participe")):
                    est_pp = True
                    break
            if not est_pp:
                continue

            # Guard G2P POS : le G2P doit confirmer VER ou ADJ
            # (les PP sont souvent tagues ADJ par le G2P)
            _g2p_pos3 = mv.g2p_pos or ""
            if _g2p_pos3 and not _g2p_pos3.startswith(("VER", "ADJ")):
                continue

            # Guard DET : si un DET precede immediatement -> NOM/ADJ, pas PP
            if i >= 1:
                prev_low = mots[i - 1].forme.lower()
                if prev_low in _DETS_BLOCK_PP_ETRE:
                    continue
                # Guard prep
                if prev_low in _PREPS_BLOCK_PP_ETRE:
                    continue

            # Guard NOM strict : si NOM existe avec freq > VER PP freq -> skip
            _pp_max_freq = max(
                (float(e.get("freq", 0))
                 for e in infos
                 if ((e.get("cgram") or "").startswith("VER")
                     and (e.get("mode") or "") in ("par", "participe"))),
                default=0.0,
            )
            _nom_max_freq3 = max(
                (float(e.get("freq", 0))
                 for e in infos
                 if (e.get("cgram") or "").startswith("NOM")),
                default=0.0,
            )
            _phase2_freq = False
            if _nom_max_freq3 > 0 and _pp_max_freq < 2.0 * _nom_max_freq3:
                if self._config.activer_phase2 and _pp_max_freq >= 0.5 * _nom_max_freq3:
                    _phase2_freq = True
                else:
                    continue

            # Guard ADJ predominant : freq ADJ > 2 * freq VER PP
            _adj_max_freq3 = max(
                (float(e.get("freq", 0))
                 for e in infos
                 if (e.get("cgram") or "").startswith("ADJ")),
                default=0.0,
            )
            if _adj_max_freq3 > 2.0 * _pp_max_freq and _pp_max_freq > 0:
                continue

            # Guard VER conjugue : si aussi VER indicatif avec freq > PP -> skip
            _ver_conj_freq = max(
                (float(e.get("freq", 0))
                 for e in infos
                 if ((e.get("cgram") or "").startswith("VER")
                     and (e.get("mode") or "") in ("ind", "indicatif"))),
                default=0.0,
            )
            if _ver_conj_freq > _pp_max_freq:
                continue

            # Chercher auxiliaire etre dans fenetre 3 (5 en Phase 2)
            _fenetre_etre = 5 if self._config.activer_phase2 else 3
            idx_aux = None
            for j in range(max(0, i - _fenetre_etre), i):
                _mj_low = mots[j].forme.lower()
                if _mj_low in _AUXILIAIRES_ETRE:
                    intercalaire_ok = True
                    for k in range(j + 1, i):
                        mk_low = mots[k].forme.lower()
                        if mk_low in _CLITIQUES_OBJETS:
                            continue
                        if mk_low in ("ne", "n'", "n\u2019", "pas",
                                      "jamais", "plus", "rien",
                                      "point", "guere"):
                            continue
                        mk_pos = mots[k].g2p_pos or ""
                        if mk_pos.startswith("ADV"):
                            continue
                        intercalaire_ok = False
                        break
                    if intercalaire_ok:
                        idx_aux = j
                        break
                # Passif compose : "a ete + PP"
                if _mj_low in ("\u00e9t\u00e9", "ete") and j >= 1:
                    _prev_j_low = mots[j - 1].forme.lower()
                    if _prev_j_low in _AVOIR_CONJUGUE:
                        intercalaire_ok = True
                        for k in range(j + 1, i):
                            mk_low = mots[k].forme.lower()
                            if mk_low in _CLITIQUES_OBJETS:
                                continue
                            if mk_low in ("ne", "n'", "n\u2019", "pas",
                                          "jamais", "plus", "rien",
                                          "point", "guere"):
                                continue
                            mk_pos = mots[k].g2p_pos or ""
                            if mk_pos.startswith("ADV"):
                                continue
                            intercalaire_ok = False
                            break
                        if intercalaire_ok:
                            idx_aux = j - 1
                            break
            if idx_aux is None:
                continue
            _etre_distant = (i - idx_aux) > 3

            # Trouver le sujet — pronom OU nominal avec DET confirmant
            sujet = self._trouver_sujet(mots, idx_aux)
            if sujet is None:
                continue

            _PRO_SUJET_PP3 = frozenset({
                "je", "j'", "j\u2019", "tu",
                "il", "elle", "on",
                "nous", "vous", "ils", "elles",
            })
            sujet_low = sujet.forme.lower()
            if sujet_low not in _PRO_SUJET_PP3:
                # Sujet nominal : exiger DET confirmant devant le NOM
                sujet_idx = None
                for k in range(max(0, idx_aux - 3), idx_aux):
                    if mots[k] is sujet:
                        sujet_idx = k
                        break
                if sujet_idx is None:
                    continue
                # Chercher DET dans les 2 mots avant le NOM sujet
                det_confirme = False
                for k in range(max(0, sujet_idx - 2), sujet_idx):
                    mk_low = mots[k].forme.lower()
                    if mk_low in self._DET_NOMBRE:
                        det_confirme = True
                        break
                if not det_confirme:
                    continue

            nombre_sujet = sujet.p2g_nombre or sujet.g2p_nombre
            genre_sujet = sujet.p2g_genre or sujet.g2p_genre

            if not nombre_sujet and not genre_sujet:
                continue

            # Verifier que le PP actuel ne correspond PAS au sujet
            pp_deja_accorde = False
            for entry in infos:
                cgram = entry.get("cgram") or ""
                if not cgram.startswith("VER"):
                    continue
                e_n = entry.get("nombre") or ""
                e_g = entry.get("genre") or ""
                if e_g not in ("m", "f") or e_n not in ("s", "p"):
                    mt_g, mt_n = self._genre_nombre_from_multext(
                        entry.get("multext") or "")
                    if e_g not in ("m", "f"):
                        e_g = mt_g
                    if e_n not in ("s", "p"):
                        e_n = mt_n
                n_ok = not nombre_sujet or e_n == nombre_sujet
                g_ok = not genre_sujet or e_g == genre_sujet
                if n_ok and g_ok:
                    pp_deja_accorde = True
                    break
            if pp_deja_accorde:
                continue

            # Trouver la forme PP accordee
            forme_accord = self._trouver_forme_accord(
                forme_low, "VER", nombre_sujet, genre_sujet,
            )
            if not forme_accord or forme_accord == forme_low:
                continue

            # Guard : la forme accordee doit exister dans le lexique
            if not lex.existe(forme_accord):
                continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, forme_accord)
            mv.regle = "accord.pp_etre.struct.all"
            if _phase2_freq or _etre_distant:
                mv.regle = "p2." + mv.regle

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"PP+etre struct all: '{mv.forme}' -> '{forme_accord}' "
                    f"(sujet nombre={nombre_sujet}, genre={genre_sujet})"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3h-bis. PP+avoir genre invariable (la regle a emergee -> emerge)
    # ------------------------------------------------------------------

    def _corriger_pp_avoir_genre(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige un PP feminin/pluriel apres avoir en masculin singulier.

        Regle : apres auxiliaire avoir, le PP ne s'accorde pas avec le sujet
        (il reste masculin singulier), SAUF si un COD est antepose.

        Cas typiques :
        - "elle a emergee" -> "emerge"
        - "ils ont provoquees" -> "provoque"

        Gardes :
        1. mv.regle vide
        2. mv.div_ortho=True et P2G propose une forme differente
        3. Forme est VER participe dans le lexique avec genre=f ou nombre=p
        4. Auxiliaire avoir dans fenetre 3 mots
        5. Entre auxiliaire et PP : clitiques/negation/ADV seulement
        6. Guard COD antepose : si "que/qu'" ou pronom COD (la, les, l')
           precede l'auxiliaire, skip
        7. PP masculin singulier existe dans le lexique (meme lemme)
        8. Confiance P2G >= 0.80
        9. Guard etre : si auxiliaire etre, skip
        """
        corrections: list[Correction] = []
        lex = self._lexique

        _COD_PRONOUNS = frozenset({
            "la", "les", "l'", "l\u2019",
        })
        _COD_REL = frozenset({
            "que", "qu'", "qu\u2019",
        })
        _INTERCALAIRES_AVOIR = frozenset(
            _CLITIQUES_OBJETS
            | {"ne", "n'", "n\u2019", "pas", "jamais", "plus", "rien",
               "bien", "mal", "aussi", "encore", "toujours", "d\u00e9j\u00e0", "deja",
               "vraiment", "souvent", "longtemps"}
        )

        for i, mv in enumerate(mots):
            # Garde 1 : deja corrige ?
            if mv.regle:
                continue

            # Garde 2 : divergence P2G
            if not mv.div_ortho:
                continue
            p2g = mv.p2g_ortho
            if not p2g or p2g == mv.forme:
                continue

            # Garde 8 : confiance P2G
            if mv.p2g_confiance < 0.80:
                continue

            forme_low = mv.forme.lower()

            # Garde : mots proteges
            if forme_low in _MOTS_PROTEGES:
                continue

            # Garde 3 : forme est VER participe avec genre=f ou nombre=p
            if not lex.existe(forme_low):
                continue
            infos = lex.info(forme_low)
            est_pp_accorde = False
            for entry in infos:
                cgram = entry.get("cgram", "")
                if not cgram.startswith("VER"):
                    continue
                mode = entry.get("mode", "")
                if mode not in ("par", "participe"):
                    continue
                e_g = entry.get("genre", "")
                e_n = entry.get("nombre", "")
                # Fallback Multext
                if e_g not in ("m", "f") or e_n not in ("s", "p"):
                    mt_g, mt_n = self._genre_nombre_from_multext(
                        entry.get("multext", ""),
                    )
                    if e_g not in ("m", "f"):
                        e_g = mt_g
                    if e_n not in ("s", "p"):
                        e_n = mt_n
                if e_g == "f" or e_n == "p":
                    est_pp_accorde = True
                    break
            if not est_pp_accorde:
                continue

            # Garde 4 : auxiliaire avoir dans fenetre 3 mots avant
            idx_avoir = -1
            for j in range(i - 1, max(i - 4, -1), -1):
                if j < 0:
                    break
                mj_low = mots[j].forme.lower()
                if mj_low in _AVOIR_CONJUGUE:
                    idx_avoir = j
                    break
                # Garde 9 : si on tombe sur etre, skip (couvert par pp_etre)
                if mj_low in _AUXILIAIRES_ETRE:
                    break
                # Garde 5 : entre auxiliaire et PP, seulement intercalaires
                if mj_low not in _INTERCALAIRES_AVOIR:
                    break
            if idx_avoir < 0:
                continue

            # Garde 6 : COD antepose (que/qu' ou pronom COD avant auxiliaire)
            cod_antepose = False
            for j in range(idx_avoir - 1, max(idx_avoir - 6, -1), -1):
                if j < 0:
                    break
                mj_low = mots[j].forme.lower()
                if mj_low in _COD_REL or mj_low in _COD_PRONOUNS:
                    cod_antepose = True
                    break
                # Arreter au premier mot non-intercalaire
                if (mj_low not in _CLITIQUES_OBJETS
                        and mj_low not in ("ne", "n'", "n\u2019")):
                    break
            if cod_antepose:
                continue

            # Garde 7 : PP masculin singulier existe dans le lexique (meme lemme)
            forme_ms = self._trouver_forme_accord(
                forme_low, "VER", "s", "m",
            )
            if not forme_ms or forme_ms == forme_low:
                # Essayer aussi avec la forme P2G si elle est masc sing
                if p2g and lex.existe(p2g):
                    infos_p2g = lex.info(p2g)
                    p2g_is_ms = False
                    for entry in infos_p2g:
                        cgram = entry.get("cgram", "")
                        if not cgram.startswith("VER"):
                            continue
                        mode = entry.get("mode", "")
                        if mode not in ("par", "participe"):
                            continue
                        e_g = entry.get("genre", "")
                        e_n = entry.get("nombre", "")
                        if e_g not in ("m", "f") or e_n not in ("s", "p"):
                            mt_g, mt_n = self._genre_nombre_from_multext(
                                entry.get("multext", ""),
                            )
                            if e_g not in ("m", "f"):
                                e_g = mt_g
                            if e_n not in ("s", "p"):
                                e_n = mt_n
                        if e_g == "m" and e_n == "s":
                            p2g_is_ms = True
                            break
                    if p2g_is_ms:
                        forme_ms = p2g
                    else:
                        continue
                else:
                    continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, forme_ms)
            mv.regle = "participe.pp_avoir_genre"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"PP+avoir genre: '{mv.forme}' -> '{forme_ms}' "
                    f"(avoir a pos {idx_avoir}, PP invariable apres avoir)"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3i. Accent via P2G (foret -> foret, fatigue -> fatigue)
    # ------------------------------------------------------------------

    def _corriger_accent_p2g(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige les mots in-lexique avec mauvais accent via P2G.

        Cas typiques :
        - "foret" (freq=0.55, NOM) -> "foret" (P2G conf=0.96)
        - "fatigue" (ADJ/NOM) -> "fatigue" ou "fatigue" selon contexte
        - "tres" -> "tres"

        Gardes :
        1. div_ortho=True et div_pos=False (meme POS)
        2. La forme P2G est une variante accent de l'original
        3. La forme P2G existe dans le lexique
        4. La frequence de la forme P2G > frequence original * 3
           (ou original freq < seuil)
        5. Confidence P2G >= 0.85
        """
        from lectura_correcteur.orthographe._suggestions import _est_variante_accent

        corrections: list[Correction] = []
        lex = self._lexique

        def _freq(mot: str) -> float:
            return lex.frequence(mot) if hasattr(lex, "frequence") else 0.0

        for mv in mots:
            # Deja corrige ?
            if mv.regle:
                continue

            # Pas de correction sur mots proteges
            if mv.forme.lower() in _MOTS_PROTEGES:
                continue

            # Declencheur : divergence orthographique
            if not mv.div_ortho:
                continue

            forme = mv.forme
            p2g = mv.p2g_ortho

            if not p2g or forme == p2g:
                continue

            # Garde 5 : confiance P2G suffisante
            if mv.p2g_confiance < 0.85:
                continue

            # Garde 1 : meme POS (div_pos=False)
            # Les variantes accent sont le meme mot, donc le POS doit etre stable
            # Exception : autoriser si ratio de frequence ecrasant (>100x)
            # Ex: tres (NOM freq 0.1) vs très (ADV freq 4000) -> div_pos attendu
            if mv.div_pos:
                freq_orig_g = _freq(forme)
                freq_p2g_g = _freq(p2g) if p2g else 0.0
                if freq_orig_g >= 0.1 and freq_p2g_g < freq_orig_g * 100:
                    continue
                if freq_orig_g < 0.1 and freq_p2g_g < 100.0:
                    continue

            # Garde 2 : la forme P2G doit etre une variante accent de l'original
            if not _est_variante_accent(forme, p2g):
                continue

            # Garde 2b : ne corriger que vers la forme PLUS accentuee
            # Supprimer un accent = FP (entrebâillée→entrebaillée, dû→du)
            _ACCENTED_P2G = frozenset("àâäéèêëïîôùûüÿçœæ")
            nb_acc_orig = sum(1 for c in forme if c in _ACCENTED_P2G)
            nb_acc_p2g = sum(1 for c in p2g if c in _ACCENTED_P2G)
            if nb_acc_p2g <= nb_acc_orig:
                continue

            # Garde 3 : la forme P2G doit exister dans le lexique
            if not lex.existe(p2g):
                continue

            # L'original doit exister dans le lexique (sinon c'est un OOV
            # qui devrait etre corrige par l'etape 1)
            if not lex.existe(forme):
                continue

            # Garde 4 : la frequence de la forme P2G doit etre significativement
            # plus elevee que celle de l'original
            freq_orig = _freq(forme)
            freq_p2g = _freq(p2g)

            # Seuil conservateur : ratio adaptatif ou forme originale quasi-inexistante
            ratio_min = 5 if mv.p2g_confiance >= 0.90 else 20
            if freq_orig >= 0.1:
                # Mot existant avec frequence mesurable -> exiger ratio adaptatif
                if freq_p2g < freq_orig * ratio_min:
                    continue
            else:
                # Mot a frequence quasi-nulle -> seuil absolu
                if freq_p2g < 10.0:
                    continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, p2g)
            mv.regle = f"accent.p2g.{forme}_{p2g}"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.HORS_LEXIQUE,
                regle=mv.regle,
                explication=(
                    f"Accent P2G: '{forme}' -> '{p2g}' "
                    f"(freq {freq_orig:.1f} -> {freq_p2g:.1f}, "
                    f"confiance={mv.p2g_confiance:.2f})"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3i-bis. Accent lexique fallback
    # ------------------------------------------------------------------

    def _corriger_accent_lexique(self, mots: list[MotV6]) -> list[Correction]:
        """Corrige les accents/cedilles manquants via le lexique.

        Cible : mots in-lexique dont la variante accentuee est beaucoup
        plus frequente. Ex: ca (597) -> ca (8934), meme (17) -> meme (908),
        riviere (0.4) -> riviere (28), bibliotheque (0.4) -> bibliotheque (18).
        """
        from lectura_correcteur.orthographe._suggestions import _est_variante_accent
        corrections: list[Correction] = []
        lex = self._lexique

        def _freq(mot: str) -> float:
            return lex.frequence(mot) if hasattr(lex, "frequence") else 0.0

        _ACCENTED = frozenset("àâäéèêëïîôùûüÿçœæ")

        for i_mv, mv in enumerate(mots):
            if mv.regle:
                continue
            if mv.forme.lower() in _MOTS_PROTEGES:
                continue

            # Garde nom propre : mot capitalise pas en debut de phrase
            # = probablement un nom propre (Angelo, Boutonnat, etc.)
            # NB: mv.forme est lowercase, mv.correction preserve la casse originale.
            if mv.correction and mv.correction[0].isupper() and i_mv > 0:
                continue
            # En debut de phrase : verifier que ce n'est pas un nom propre
            if mv.correction and mv.correction[0].isupper() and i_mv == 0:
                infos_forme = lex.info(mv.forme)
                if infos_forme:
                    has_propre = any(
                        "PROPRE" in e.get("cgram", "").upper()
                        or e.get("cgram", "").startswith("NAM")
                        for e in infos_forme
                    )
                    has_common = any(
                        e.get("cgram", "") in ("NOM", "VER", "ADJ", "ADV", "PRO",
                                                "PRE", "CON", "DET", "ONO")
                        for e in infos_forme
                    )
                    if has_propre and not has_common:
                        continue  # Nom propre uniquement, pas de correction

            forme = mv.forme.lower()
            if not lex.existe(forme):
                continue

            # Garde homographes : mots dont la forme sans accent est un mot
            # valide avec un sens different (foret/foret, soul/soul, etc.)
            if forme in _HOMOGRAPHES_ACCENT:
                continue

            # Tenter variante accent via la methode OOV
            candidat = self._trouver_variante_accent_oov(forme)
            if candidat is None or candidat == forme:
                continue
            if not lex.existe(candidat):
                continue

            # Verifier que c'est bien une variante accent
            if not _est_variante_accent(forme, candidat):
                continue

            # Ne corriger que vers la forme PLUS accentuee (pas l'inverse)
            nb_acc_orig = sum(1 for c in forme if c in _ACCENTED)
            nb_acc_cand = sum(1 for c in candidat if c in _ACCENTED)
            if nb_acc_cand <= nb_acc_orig:
                continue

            # Garde lemme : ne pas corriger si original et candidat partagent
            # un lemme (= ambiguite morphologique du meme mot, pas accent manquant)
            # Ex: confectionne/confectionné (meme lemme "confectionner")
            infos_orig = lex.info(forme)
            infos_cand = lex.info(candidat)
            if infos_orig and infos_cand:
                lemmes_orig = {e.get("lemme") for e in infos_orig if e.get("lemme")}
                lemmes_cand = {e.get("lemme") for e in infos_cand if e.get("lemme")}
                if lemmes_orig & lemmes_cand:
                    continue  # Meme lemme = variante morpho, pas accent manquant

            freq_orig = _freq(forme)
            freq_candidat = _freq(candidat)

            # Garde : candidat a frequence nulle → pas de correction fiable
            if freq_candidat <= 0.0:
                continue

            # Ratio adaptatif selon la frequence de l'original :
            # - freq_orig < 5 (quasi-inconnu) : ratio 10x ou freq_cand >= 10
            # - freq_orig 5-100 (peu courant) : ratio 10x
            # - freq_orig > 100 (courant) : ratio 5x
            # Mots courts (<=3) : toujours exiger un ratio plus eleve
            if len(forme) <= 2:
                # Mots de 1-2 chars : tres risque (a, y, etc.)
                if freq_candidat < freq_orig * 10:
                    continue
            elif len(forme) <= 3:
                # 3 chars : ca->ca, git->git, etc.
                if freq_candidat < freq_orig * 5:
                    continue
            elif freq_orig < 5.0:
                if freq_candidat < freq_orig * 10 and freq_candidat < 10.0:
                    continue
            elif freq_orig < 100.0:
                if freq_candidat < freq_orig * 10:
                    continue
            else:
                # Mot frequent (>100) : exiger ratio 5x
                if freq_candidat < freq_orig * 5:
                    continue

            # Appliquer
            mv.correction = transferer_casse(mv.correction, candidat)
            mv.regle = f"accent.lex.{forme}_{candidat}"
            corrections.append(Correction(
                index=mv.index, original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.HORS_LEXIQUE,
                regle=mv.regle,
                explication=(
                    f"Accent lexique: '{forme}' -> '{candidat}' "
                    f"(freq {freq_orig:.1f} -> {freq_candidat:.1f})"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3j. Negation (il veut pas -> il ne veut pas)
    # ------------------------------------------------------------------

    def _corriger_negation(self, mots: list[MotV6]) -> list[Correction]:
        """Insere ne/n' devant un verbe quand il manque dans la negation.

        Cas typiques :
        - "il veut pas" -> "il ne veut pas"
        - "elle mange jamais" -> "elle ne mange jamais"
        - "il aime plus" -> "il n'aime plus"

        Gardes :
        1. Detecter une particule de negation (pas, plus, jamais, etc.)
        2. Un verbe doit preceder dans les 2 mots avant
        3. Il n'y a PAS deja un ne/n' avant le verbe
        """
        corrections: list[Correction] = []
        n = len(mots)

        for i, mv in enumerate(mots):
            # Detecter une particule de negation
            if mv.forme.lower() not in _NEGATION_PARTICULES:
                continue

            # Particule precedee de ponctuation (virgule, point-virgule) :
            # c'est une juxtaposition, pas une negation verbale
            # Ex: "elle avait vu, pas pensé" → "pas" ≠ negation de "avait"
            if mv.preceded_by_punct:
                continue

            # "non pas/point" = construction de substitution, pas negation verbale
            # Ex: "considère non pas ce graphe" → ne PAS ajouter "ne"
            if i > 0 and mots[i - 1].forme.lower() == "non":
                continue

            # "pas" comme nom (= un pas, quelques pas, du pas lourd)
            # Detecter via le mot precedent : DET, NUM, quantifieur, adj
            if mv.forme.lower() == "pas":
                if i > 0:
                    prev_low = mots[i - 1].forme.lower()
                    prev_pos = mots[i - 1].g2p_pos or ""
                    if (prev_low in ("un", "le", "du", "ce", "son", "mon",
                                     "ton", "chaque", "quelques", "premier",
                                     "premiers", "petit", "petits", "grand",
                                     "grands", "deux", "trois", "faux",
                                     "premier", "dernier",
                                     "nos", "mes", "tes", "ses", "vos",
                                     "leurs", "ces", "des", "aux",
                                     "cent", "mille", "nombreux",
                                     "quelque", "certains", "plusieurs")
                            or prev_pos.startswith("DET")
                            or prev_pos.startswith("NUM")):
                        continue

            # "jamais" en sens positif : "si jamais", "a jamais", "pour jamais"
            # + subjonctif sans negation : "le plus beau qu'on ait jamais vu"
            if mv.forme.lower() == "jamais":
                if i > 0:
                    prev_low = mots[i - 1].forme.lower()
                    if prev_low in ("si", "à", "a", "pour", "sans"):
                        continue
                # Subjonctif apres superlatif/restrictif : "que X ait jamais"
                # "jamais" en sens positif (= un jour/ever)
                if i > 0:
                    prev_pos = mots[i - 1].g2p_pos or ""
                    if prev_pos.startswith("VER") or prev_pos == "AUX":
                        # Verbe au subjonctif avant "jamais" → sens positif
                        prev_low2 = mots[i - 1].forme.lower()
                        if prev_low2 in ("ait", "aie", "aies",
                                         "eût", "eusse", "eusses",
                                         "fût", "fusse", "fusses",
                                         "soit", "sois",
                                         "aient", "eussent", "fussent",
                                         "soient", "puisse", "puissent"):
                            continue

            # "point" : tres rare comme negation, beaucoup de FP
            # "au point", "mis au point", "point de vue" etc.
            if mv.forme.lower() == "point":
                if i > 0:
                    prev_low = mots[i - 1].forme.lower()
                    if prev_low in ("au", "du", "un", "le", "ce", "son",
                                    "quel", "tout"):
                        continue

            # Exclure "plus" non-negatif :
            # - quantitatif : "je veux plus de pain"
            # - superlatif  : "la plus grande", "au plus profond"
            # - comparatif  : "plus prudent", "plus grand"
            if mv.forme.lower() == "plus":
                # Quantitatif : suivi de de/du/des
                if i < n - 1:
                    next_low = mots[i + 1].forme.lower()
                    if next_low in ("de", "du", "des", "d'", "d\u2019"):
                        continue
                # Superlatif/intensif : precede de article/prep/adverbe
                if i > 0:
                    prev_low = mots[i - 1].forme.lower()
                    if prev_low in ("la", "le", "les", "au", "du", "de",
                                    "encore", "bien", "beaucoup",
                                    "toujours", "parfois"):
                        continue
                # Comparatif : suivi d'un adjectif, adverbe ou participe passe
                # (VER:pper fonctionne souvent comme adjectif : sacre, fatigue…)
                if i < n - 1:
                    next_pos = mots[i + 1].g2p_pos or ""
                    if (next_pos.startswith("ADJ") or next_pos.startswith("ADV")
                            or next_pos == "VER:pper"):
                        continue

            # Chercher le verbe avant la particule.
            # Pour "plus", le verbe doit etre juste avant (distance 1)
            # pour eviter les correlatifs "plus X est Y, plus Z".
            # Pour les autres particules, on cherche dans les 2 mots avant.
            verb_window = 1 if mv.forme.lower() == "plus" else 2
            idx_verbe = None
            for j in range(max(0, i - verb_window), i):
                mj = mots[j]
                if mj.g2p_pos and (
                    mj.g2p_pos.startswith("VER")
                    or mj.g2p_pos == "AUX"
                    or mj.forme.lower() in _AUXILIAIRES
                ):
                    idx_verbe = j
                    break

            if idx_verbe is None:
                continue

            # Ne pas modifier un verbe deja corrige par un autre module
            if mots[idx_verbe].regle:
                continue

            # Garde stricte : exiger un pronom personnel ou un NOM sujet
            # immediatement avant le verbe pour confirmer la negation.
            # "il veut pas" → OK (pronom "il" devant "veut")
            # "la mer était déserte" + "Pas" → bloqué (pas de pronom)
            _PRONOMS_SUJET = frozenset({
                "je", "j'", "tu", "il", "elle", "on",
                "nous", "vous", "ils", "elles",
                "ce", "c'", "ça", "cela",
                "qui", "qu'",
            })
            has_sujet = False
            if idx_verbe > 0:
                prev_v = mots[idx_verbe - 1].forme.lower()
                prev_pos = mots[idx_verbe - 1].g2p_pos or ""
                if prev_v in _PRONOMS_SUJET or prev_pos.startswith("PRO:per"):
                    has_sujet = True
            if not has_sujet:
                continue

            # Ne pas inserer ne/n' quand le sujet est un pronom elide
            # (c'est pas, j'sais pas, s'agit pas, t'as pas, etc.)
            # L'absence de "ne" est volontaire dans ce registre oral,
            # et l'insertion produirait des formes absurdes (c'n'est).
            if idx_verbe > 0:
                prev_forme = mots[idx_verbe - 1].forme
                if prev_forme.endswith(("'", "\u2019")):
                    continue

            # Ne pas inserer ne/n' devant un participe (present ou passe)
            # ou apres une preposition (structure infinitive, pas negation)
            verbe_pos = mots[idx_verbe].g2p_pos or ""
            if verbe_pos in ("VER:ppre", "VER:G", "VER:pper"):
                continue
            # Ne pas inserer devant un verbe OOV (artefact typographique
            # comme "au-rait" = cesure de ligne du XIXe)
            if not self._lexique.existe(mots[idx_verbe].forme.lower()):
                continue
            if idx_verbe > 0:
                prev_forme = mots[idx_verbe - 1].forme.lower()
                if prev_forme in ("de", "à", "sans", "pour", "en",
                                  "d'", "d\u2019",
                                  "l'", "l\u2019"):
                    continue
            # Aussi verifier 2 positions avant (prep + pronom : "de l'avoir")
            if idx_verbe > 1:
                prev2_forme = mots[idx_verbe - 2].forme.lower()
                if prev2_forme in ("de", "à", "sans", "pour", "en",
                                   "d'", "d\u2019"):
                    continue

            # Verifier qu'il n'y a PAS deja un ne/n' dans le groupe verbal
            # (fenetre elargie : 5 mots avant le verbe ET entre verbe et particule
            # pour couvrir "pourrait ne pas" et constructions similaires)
            has_ne = False
            for j in range(max(0, idx_verbe - 5), i):
                mj_low = mots[j].forme.lower()
                if mj_low in ("ne", "n'", "n\u2019"):
                    has_ne = True
                    break
            if has_ne:
                continue

            # Barriere de phrase : les MotV6 ne contiennent pas de ponctuation,
            # donc on utilise la casse (champ `correction` qui preserve
            # la casse originale) comme signal de frontiere de phrase.
            #
            # 1. Particule en majuscule = debut de nouvelle phrase
            # Ex: "la mer était déserte. Pas un bateau" → "Pas" = nouvelle phrase
            corr_part = mv.correction or mv.forme
            if corr_part and corr_part[0].isupper() and i > 0:
                continue
            # 2. Mot en majuscule entre le verbe et la particule = frontiere
            # Ex: "qui vous le dénie? Ai-je jamais" → "Ai-je" entre verbe et particule
            has_upper_barrier = False
            for j in range(idx_verbe + 1, i):
                corr_j = mots[j].correction or mots[j].forme
                if corr_j and corr_j[0].isupper():
                    has_upper_barrier = True
                    break
            if has_upper_barrier:
                continue

            # Garde forward : si ne/n' existe dans les 3 mots APRES la
            # particule, elle appartient a l'autre proposition
            # Ex: "rien n'empechera", "que rien n'est"
            has_ne_forward = False
            for j in range(i + 1, min(i + 4, n)):
                mj_low = mots[j].forme.lower()
                if mj_low in ("ne", "n'", "n\u2019"):
                    has_ne_forward = True
                    break
                # n' colle au verbe : "n'empêchera"
                if mj_low.startswith("n'") or mj_low.startswith("n\u2019"):
                    has_ne_forward = True
                    break
            if has_ne_forward:
                continue

            # Garde "que" barrier : "il espère que rien", "je veux dire que"
            # Le "que" introduit une proposition subordonnee — la negation
            # ne peut pas remonter au verbe de la principale
            has_que_barrier = False
            for j in range(idx_verbe + 1, i):
                if mots[j].forme.lower() in ("que", "qu'", "qu\u2019"):
                    has_que_barrier = True
                    break
            if has_que_barrier:
                continue

            # Determiner la forme a inserer : ne ou n'
            verbe_forme = mots[idx_verbe].forme
            if verbe_forme and verbe_forme[0].lower() in _VOYELLES:
                ne_forme = "n'"
                # Coller le n' au verbe : "n'aime"
                mots[idx_verbe].correction = ne_forme + mots[idx_verbe].correction
            else:
                ne_forme = "ne"
                # Inserer "ne " avant le verbe
                mots[idx_verbe].correction = ne_forme + " " + mots[idx_verbe].correction

            mots[idx_verbe].regle = "negation.ne"

            corrections.append(Correction(
                index=idx_verbe,
                original=mots[idx_verbe].forme,
                corrige=mots[idx_verbe].correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="negation.ne",
                explication=(
                    f"Negation: insertion de '{ne_forme}' devant "
                    f"'{verbe_forme}'"
                ),
            ))

        return corrections

    # ------------------------------------------------------------------
    # 3k. Homophones structurels (regles sans P2G)
    # ------------------------------------------------------------------

    def _corriger_homophones_structurels(
        self, mots: list[MotV6],
    ) -> list[Correction]:
        """Corrige les homophones par regles structurelles (sans P2G).

        Couvre les cas ou le P2G ne predit pas la bonne forme :
        - se + NOM -> ce + NOM
        - ou en debut de phrase + verbe -> où + verbe
        """
        corrections: list[Correction] = []
        n = len(mots)

        for i, mv in enumerate(mots):
            if mv.regle:
                continue

            forme = mv.forme.lower()

            # --- se -> ce : si suivi d'un NOM (pas VER/ART/PRO) ---
            if forme == "se" and i < n - 1:
                next_mv = mots[i + 1]
                next_pos = next_mv.g2p_pos or ""
                next_forme = next_mv.correction.lower() if next_mv.correction else next_mv.forme.lower()
                # Garde reflexif : "se" + auxiliaire = pronominal (se sont, s'est)
                if next_forme in _AUXILIAIRES:
                    continue
                # Garde : "se" apres preposition = reflexif (pour se, de se, a se)
                if i > 0:
                    prev_forme = mots[i - 1].forme.lower()
                    if prev_forme in ("pour", "de", "d'", "d\u2019",
                                      "\u00e0", "sans", "avant"):
                        continue
                # "se" devant un NOM pur est "ce"
                # Mais pas si le mot est aussi VER/ART/PRO
                next_is_nom_only = False
                if self._lexique.existe(next_forme):
                    infos_next = self._lexique.info(next_forme)
                    has_nom = any(
                        e.get("cgram", "").startswith("NOM") for e in infos_next
                    )
                    has_ver = any(
                        e.get("cgram", "").startswith("VER") for e in infos_next
                    )
                    has_art = any(
                        e.get("cgram", "").startswith("ART") for e in infos_next
                    )
                    has_pro = any(
                        e.get("cgram", "").startswith("PRO") for e in infos_next
                    )
                    next_is_nom_only = has_nom and not has_ver and not has_art and not has_pro
                if next_is_nom_only:
                    mv.correction = transferer_casse(mv.correction, "ce")
                    mv.regle = "homophone.struct.se_ce"
                    corrections.append(Correction(
                        index=mv.index, original=mv.forme,
                        corrige=mv.correction,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle=mv.regle,
                        explication=f"Homophone structurel: 'se' -> 'ce' (suivi de NOM)",
                    ))
                    continue
                # "se" + qui/que -> "ce" + qui/que (zero risque FP)
                if next_forme in ("qui", "que", "qu'", "qu\u2019",
                                  "dont", "sera", "serait"):
                    mv.correction = transferer_casse(mv.correction, "ce")
                    mv.regle = "homophone.struct.se_ce"
                    corrections.append(Correction(
                        index=mv.index, original=mv.forme,
                        corrige=mv.correction,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle=mv.regle,
                        explication=(
                            f"Homophone structurel: 'se' -> 'ce' "
                            f"(suivi de '{next_forme}')"
                        ),
                    ))
                    continue
                # "se" + ADJ pur (pas VER infinitif) -> "ce"
                # Ex: "se dernier", "se meme"
                if self._lexique.existe(next_forme):
                    infos_next = self._lexique.info(next_forme)
                    has_adj_only = any(
                        e.get("cgram", "").startswith("ADJ")
                        for e in infos_next
                    )
                    has_ver2 = any(
                        e.get("cgram", "").startswith("VER")
                        for e in infos_next
                    )
                    has_pro2 = any(
                        e.get("cgram", "").startswith("PRO")
                        for e in infos_next
                    )
                    if has_adj_only and not has_ver2 and not has_pro2:
                        mv.correction = transferer_casse(
                            mv.correction, "ce")
                        mv.regle = "homophone.struct.se_ce"
                        corrections.append(Correction(
                            index=mv.index, original=mv.forme,
                            corrige=mv.correction,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle=mv.regle,
                            explication=(
                                "Homophone structurel: 'se' -> 'ce' "
                                "(suivi de ADJ)"
                            ),
                        ))
                        continue

            # --- là -> la : article (P2G confirme) ---
            if forme == "l\u00e0" and mv.div_ortho and mv.p2g_ortho:
                p2g_low = mv.p2g_ortho.lower()
                if p2g_low == "la" and mv.p2g_confiance >= 0.90:
                    # Confirmation structurelle : suivi d'un NOM/ADJ
                    struct_ok = False
                    if i < n - 1:
                        next_pos = mots[i + 1].g2p_pos or ""
                        if next_pos.startswith(("NOM", "ADJ")):
                            struct_ok = True
                    # Ou P2G predit article (ART:def)
                    if mv.p2g_pos and mv.p2g_pos.startswith("ART"):
                        struct_ok = True
                    if struct_ok:
                        mv.correction = transferer_casse(mv.correction, "la")
                        mv.regle = "homophone.struct.l\u00e0_la"
                        corrections.append(Correction(
                            index=mv.index, original=mv.forme,
                            corrige=mv.correction,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle=mv.regle,
                            explication="Homophone structurel: 'l\u00e0' -> 'la' (article)",
                        ))
                        continue

            # --- ça -> sa : possessif devant NOM (P2G confirme) ---
            if forme == "\u00e7a" and mv.div_ortho and mv.p2g_ortho:
                p2g_low = mv.p2g_ortho.lower()
                if p2g_low == "sa" and mv.p2g_confiance >= 0.90:
                    # Le P2G predit possessif + suivi d'un NOM/ADJ
                    struct_ok = False
                    if i < n - 1:
                        next_pos = mots[i + 1].g2p_pos or ""
                        if next_pos.startswith(("NOM", "ADJ")):
                            struct_ok = True
                    if mv.p2g_pos and mv.p2g_pos.startswith("ADJ:pos"):
                        struct_ok = True
                    if struct_ok:
                        mv.correction = transferer_casse(mv.correction, "sa")
                        mv.regle = "homophone.struct.\u00e7a_sa"
                        corrections.append(Correction(
                            index=mv.index, original=mv.forme,
                            corrige=mv.correction,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle=mv.regle,
                            explication="Homophone structurel: '\u00e7a' -> 'sa' (possessif)",
                        ))
                        continue

            # --- ça -> sa : structurel sans P2G (devant NOM feminin) ---
            if forme == "ça" and i < n - 1 and not mv.regle:
                next_mv = mots[i + 1]
                next_forme = (next_mv.correction.lower()
                              if next_mv.correction
                              else next_mv.forme.lower())
                next_pos = next_mv.g2p_pos or ""
                # Guard: ça + verbe = correct (ça va, ça suffit)
                if not next_pos.startswith(("VER", "AUX")):
                    if self._lexique.existe(next_forme):
                        infos_n = self._lexique.info(next_forme)
                        has_nom_f = any(
                            e.get("cgram", "").startswith("NOM")
                            and e.get("genre") == "f"
                            for e in infos_n
                        )
                        has_ver = any(
                            e.get("cgram", "").startswith("VER")
                            for e in infos_n
                        )
                        if has_nom_f and not has_ver:
                            mv.correction = transferer_casse(
                                mv.correction, "sa")
                            mv.regle = "homophone.struct.ça_sa"
                            corrections.append(Correction(
                                index=mv.index, original=mv.forme,
                                corrige=mv.correction,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle=mv.regle,
                                explication=(
                                    "Homophone structurel: 'ça' -> 'sa' "
                                    "(devant NOM feminin)"
                                ),
                            ))
                            continue

            # --- ou -> où : debut de phrase suivi d'un verbe ---
            if forme == "ou" and i == 0 and n > 1:
                next_mv = mots[i + 1]
                next_pos = next_mv.g2p_pos or ""
                next_low = next_mv.forme.lower()
                # Detecter verbe : G2P POS, auxiliaire, ou forme inversee (vas-tu, etc.)
                next_is_verb = (
                    next_pos.startswith("VER") or next_pos == "AUX"
                    or next_low in _AUXILIAIRES
                    or ("-" in next_low and next_low.split("-")[0] in (
                        "vas", "va", "est", "sont", "suis", "es",
                        "allons", "allez", "vont",
                        "habites", "habite", "habitez",
                        "viens", "vient", "venez",
                    ))
                )
                if next_is_verb:
                    mv.correction = transferer_casse(mv.correction, "o\u00f9")
                    mv.regle = "homophone.struct.ou_où"
                    corrections.append(Correction(
                        index=mv.index, original=mv.forme,
                        corrige=mv.correction,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle=mv.regle,
                        explication=f"Homophone structurel: 'ou' -> 'où' (debut + verbe)",
                    ))
                    continue

            # --- ou -> où : expressions figees ---
            if forme == "ou" and i > 0 and not mv.regle:
                prev_low = mots[i - 1].forme.lower()
                corriger_ou = False
                expl_ctx = ""

                # "par ou" -> "par où"
                if prev_low == "par":
                    corriger_ou = True
                    expl_ctx = "par ou"
                # "là ou" -> "là où" (seulement avec accent, pas "la" article)
                elif prev_low == "là":
                    corriger_ou = True
                    expl_ctx = "là ou"
                # "d'ou" -> "d'où"
                elif prev_low in ("d'", "d\u2019"):
                    corriger_ou = True
                    expl_ctx = "d'ou"
                # "n'importe ou" -> "n'importe où"
                elif prev_low == "importe" and i >= 2:
                    prev2_low = mots[i - 2].forme.lower()
                    if prev2_low in ("n'", "n\u2019"):
                        corriger_ou = True
                        expl_ctx = "n'importe ou"
                # "au cas ou" -> "au cas où"
                elif prev_low == "cas" and i >= 2:
                    prev2_low = mots[i - 2].forme.lower()
                    if prev2_low == "au":
                        corriger_ou = True
                        expl_ctx = "au cas ou"

                if corriger_ou:
                    mv.correction = transferer_casse(mv.correction, "où")
                    mv.regle = "homophone.struct.ou_où"
                    corrections.append(Correction(
                        index=mv.index, original=mv.forme,
                        corrige=mv.correction,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle=mv.regle,
                        explication=(
                            f"Homophone structurel: 'ou' -> 'où' "
                            f"(expression '{expl_ctx}')"
                        ),
                    ))
                    continue

            # --- et -> est : regles structurelles ---
            if forme == "et" and i > 0 and i < n - 1:
                next_mv = mots[i + 1]
                next_forme = (next_mv.correction.lower()
                              if next_mv.correction
                              else next_mv.forme.lower())
                next_pos = next_mv.g2p_pos or ""
                prev_forme = mots[i - 1].forme.lower()

                # Regle 3 — n' + et + pas (zero risque FP)
                if prev_forme in ("n'", "n\u2019") and next_forme == "pas":
                    mv.correction = transferer_casse(mv.correction, "est")
                    mv.regle = "homophone.struct.et_est"
                    corrections.append(Correction(
                        index=mv.index, original=mv.forme,
                        corrige=mv.correction,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle=mv.regle,
                        explication="Homophone structurel: 'et' -> 'est' (n'et pas)",
                    ))
                    continue

                # Regle 1 — Pronom sujet + et + non-coordination
                # "il et content" -> "il est content"
                if prev_forme in _PRO_SUJET_EST:
                    # Le mot suivant ne doit pas etre signe de coordination
                    # (NOM pur, DET, ART, PRO, nombre)
                    next_is_coord = False
                    if next_pos.startswith(("ART", "DET", "PRO", "NUM")):
                        next_is_coord = True
                    elif next_forme in _NUMERAUX_PLURIEL:
                        next_is_coord = True
                    elif self._lexique.existe(next_forme):
                        infos_n = self._lexique.info(next_forme)
                        has_nom = any(
                            e.get("cgram", "").startswith("NOM")
                            for e in infos_n
                        )
                        has_adj = any(
                            e.get("cgram", "").startswith("ADJ")
                            for e in infos_n
                        )
                        has_ver = any(
                            e.get("cgram", "").startswith("VER")
                            or e.get("cgram") == "AUX"
                            for e in infos_n
                        )
                        has_pp = any(
                            e.get("cgram", "").startswith("VER")
                            and e.get("mode") == "par"
                            for e in infos_n
                        )
                        # NOM pur sans ADJ/VER/PP = coordination
                        if has_nom and not has_adj and not has_ver:
                            next_is_coord = True
                    if not next_is_coord:
                        # Guard : pas de NOM propre apres (risque coord)
                        if not next_mv.forme[0].isupper() or i + 1 == 0:
                            mv.correction = transferer_casse(
                                mv.correction, "est")
                            mv.regle = "homophone.struct.et_est"
                            corrections.append(Correction(
                                index=mv.index, original=mv.forme,
                                corrige=mv.correction,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle=mv.regle,
                                explication=(
                                    f"Homophone structurel: 'et' -> 'est' "
                                    f"(pronom sujet '{prev_forme}' + non-coord)"
                                ),
                            ))
                            continue

                # Regle 2 — DET_SING + NOM + et + ADJ/PP
                # "sa voiture et rouge" -> "sa voiture est rouge"
                if i >= 2:
                    prev2_forme = mots[i - 2].forme.lower()
                    prev_pos = mots[i - 1].g2p_pos or ""
                    if (prev2_forme in _DET_SING_EST
                            and prev_pos.startswith("NOM")):
                        # Le mot suivant doit etre ADJ ou PP (pas VER inf)
                        next_is_adj_pp = False
                        if self._lexique.existe(next_forme):
                            infos_n = self._lexique.info(next_forme)
                            has_adj = any(
                                e.get("cgram", "").startswith("ADJ")
                                for e in infos_n
                            )
                            has_pp = any(
                                e.get("cgram", "").startswith("VER")
                                and e.get("mode") == "par"
                                for e in infos_n
                            )
                            has_ver_inf = any(
                                e.get("cgram", "").startswith("VER")
                                and e.get("mode") == "inf"
                                for e in infos_n
                            )
                            next_is_adj_pp = (
                                (has_adj or has_pp) and not has_ver_inf
                            )
                        if next_is_adj_pp:
                            mv.correction = transferer_casse(
                                mv.correction, "est")
                            mv.regle = "homophone.struct.et_est"
                            corrections.append(Correction(
                                index=mv.index, original=mv.forme,
                                corrige=mv.correction,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle=mv.regle,
                                explication=(
                                    f"Homophone structurel: 'et' -> 'est' "
                                    f"(DET+NOM+et+ADJ)"
                                ),
                            ))
                            continue

            # --- a -> à : preposition devant DET/locution ---
            if forme == "a" and i > 0 and i < n - 1:
                next_mv = mots[i + 1]
                next_forme = (next_mv.correction.lower()
                              if next_mv.correction
                              else next_mv.forme.lower())
                prev_forme = mots[i - 1].forme.lower()
                prev_pos = mots[i - 1].g2p_pos or ""

                # Guard : pronom sujet avant = auxiliaire "a" (il a, elle a)
                # On ne corrige que si le mot avant N'EST PAS un pronom sujet
                prev_is_sujet = prev_forme in (
                    "il", "elle", "on", "qui", "j'", "j\u2019",
                    "je", "tu",
                )
                if not prev_is_sujet:
                    # Pattern: a + DET_ARTICLE (la, le, l', les, un, une)
                    # Ex: "il va a la maison" -> "il va à la maison"
                    if next_forme in ("la", "le", "l'", "l\u2019",
                                      "les", "un", "une"):
                        mv.correction = transferer_casse(
                            mv.correction, "\u00e0")
                        mv.regle = "homophone.struct.a_\u00e0"
                        corrections.append(Correction(
                            index=mv.index, original=mv.forme,
                            corrige=mv.correction,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle=mv.regle,
                            explication=(
                                "Homophone structurel: 'a' -> '\u00e0' "
                                "(suivi de DET)"
                            ),
                        ))
                        continue
                    # Pattern: a + locution prepositive
                    # Ex: "a côté", "a cause", "a partir", "a travers"
                    if next_forme in (
                        "c\u00f4t\u00e9", "cote", "cause", "partir",
                        "travers", "droite", "gauche",
                        "nouveau", "c\u00f4te",
                        # Locutions supplementaires
                        "condition", "peine", "moins", "force",
                        "priori", "posteriori", "fortiori",
                        "pr\u00e9sent", "present", "bord",
                        "d\u00e9faut", "defaut", "propos",
                    ):
                        mv.correction = transferer_casse(
                            mv.correction, "\u00e0")
                        mv.regle = "homophone.struct.a_\u00e0"
                        corrections.append(Correction(
                            index=mv.index, original=mv.forme,
                            corrige=mv.correction,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle=mv.regle,
                            explication=(
                                "Homophone structurel: 'a' -> '\u00e0' "
                                f"(locution 'a {next_forme}')"
                            ),
                        ))
                        continue

            # --- a -> \u00e0 : apres locution (gr\u00e2ce a, quant a) ---
            if forme == "a" and i > 0 and not mv.regle:
                prev_low = mots[i - 1].forme.lower()
                if prev_low in ("gr\u00e2ce", "grace", "quant"):
                    mv.correction = transferer_casse(
                        mv.correction, "\u00e0")
                    mv.regle = "homophone.struct.a_\u00e0"
                    corrections.append(Correction(
                        index=mv.index, original=mv.forme,
                        corrige=mv.correction,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle=mv.regle,
                        explication=(
                            f"Homophone structurel: 'a' -> '\u00e0' "
                            f"(apres '{prev_low}')"
                        ),
                    ))
                    continue

            # --- on -> ont : pronom devant PP = auxiliaire ---
            if forme == "on" and i > 0 and i < n - 1:
                next_mv = mots[i + 1]
                next_forme = (next_mv.correction.lower()
                              if next_mv.correction
                              else next_mv.forme.lower())
                # "on" apres sujet nominal/pronom pluriel = "ont"
                # Ex: "les parents on prepare" -> "ont"
                # Guard: verifier que le mot suivant est un PP (VER:par)
                prev_pos = mots[i - 1].g2p_pos or ""
                next_pos = next_mv.g2p_pos or ""
                # Pattern: sujet pluriel + on + PP
                if (prev_pos.startswith("NOM")
                        and (next_pos.startswith("VER")
                             and ":par" in next_pos)):
                    mv.correction = transferer_casse(mv.correction, "ont")
                    mv.regle = "homophone.struct.on_ont"
                    corrections.append(Correction(
                        index=mv.index, original=mv.forme,
                        corrige=mv.correction,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle=mv.regle,
                        explication=(
                            "Homophone structurel: 'on' -> 'ont' "
                            "(NOM + on + PP)"
                        ),
                    ))
                    continue
                # "on" suivi de PP (mode=par dans lexique) sans sujet
                # apres pronom pluriel (ils, elles) ou NOM pluriel
                if self._lexique.existe(next_forme):
                    infos_n = self._lexique.info(next_forme)
                    has_pp = any(
                        e.get("cgram", "").startswith("VER")
                        and e.get("mode") == "par"
                        and e.get("temps") == "pass"
                        for e in infos_n
                    )
                    has_inf = any(
                        e.get("cgram", "").startswith("VER")
                        and e.get("mode") == "inf"
                        for e in infos_n
                    )
                    # PP pur (pas aussi infinitif) après un NOM/PRO pluriel
                    if has_pp and not has_inf:
                        # Verifier contexte amont : sujet pluriel
                        prev_low = mots[i - 1].forme.lower()
                        is_plur_subj = (
                            prev_low in ("ils", "elles")
                            or prev_pos.startswith("NOM")
                        )
                        if is_plur_subj:
                            mv.correction = transferer_casse(
                                mv.correction, "ont")
                            mv.regle = "homophone.struct.on_ont"
                            corrections.append(Correction(
                                index=mv.index, original=mv.forme,
                                corrige=mv.correction,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle=mv.regle,
                                explication=(
                                    "Homophone structurel: 'on' -> 'ont' "
                                    "(sujet pluriel + PP)"
                                ),
                            ))
                            continue

            # --- ont -> on : suivi de clitique/pronom reflexif ---
            if forme == "ont" and i < n - 1 and not mv.regle:
                next_low = mots[i + 1].forme.lower()
                # "ont se/ne/me/te" -> "on" (reflexif/negatif apres pronom)
                if next_low in ("se", "s'", "s\u2019", "ne", "n'", "n\u2019",
                                "me", "m'", "m\u2019", "te", "t'", "t\u2019"):
                    # Guard: pas de sujet pluriel avant
                    prev_is_plur = False
                    if i > 0:
                        prev_low = mots[i - 1].forme.lower()
                        if prev_low in ("ils", "elles"):
                            prev_is_plur = True
                    if not prev_is_plur:
                        mv.correction = transferer_casse(
                            mv.correction, "on")
                        mv.regle = "homophone.struct.ont_on"
                        corrections.append(Correction(
                            index=mv.index, original=mv.forme,
                            corrige=mv.correction,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle=mv.regle,
                            explication=(
                                "Homophone structurel: 'ont' -> 'on' "
                                f"(suivi de '{next_low}')"
                            ),
                        ))
                        continue

            # --- son -> sont : possessif au lieu d'auxiliaire ---
            if forme == "son" and i > 0 and not mv.regle:
                prev_low = mots[i - 1].forme.lower()
                # "ils/elles son" -> "sont"
                if prev_low in ("ils", "elles"):
                    mv.correction = transferer_casse(
                        mv.correction, "sont")
                    mv.regle = "homophone.struct.son_sont"
                    corrections.append(Correction(
                        index=mv.index, original=mv.forme,
                        corrige=mv.correction,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle=mv.regle,
                        explication=(
                            f"Homophone structurel: 'son' -> 'sont' "
                            f"(apres '{prev_low}')"
                        ),
                    ))
                    continue
                # "ne/n' son pas/plus/jamais" -> "sont"
                if prev_low in ("ne", "n'", "n\u2019") and i < n - 1:
                    next_low = mots[i + 1].forme.lower()
                    if next_low in ("pas", "plus", "jamais", "rien",
                                    "gu\u00e8re", "guere", "point"):
                        mv.correction = transferer_casse(
                            mv.correction, "sont")
                        mv.regle = "homophone.struct.son_sont"
                        corrections.append(Correction(
                            index=mv.index, original=mv.forme,
                            corrige=mv.correction,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle=mv.regle,
                            explication=(
                                "Homophone structurel: 'son' -> 'sont' "
                                f"('{prev_low} son {next_low}')"
                            ),
                        ))
                        continue

            # --- leurs -> leur : possessif pluriel devant verbe ---
            if forme == "leurs" and i < n - 1 and not mv.regle:
                next_mv = mots[i + 1]
                next_forme = (next_mv.correction.lower()
                              if next_mv.correction
                              else next_mv.forme.lower())
                next_pos = next_mv.g2p_pos or ""
                # G2P detecte verbe
                next_is_ver = next_pos.startswith(("VER", "AUX"))
                # Confirmation lexique : VER pur (pas NOM/ADJ)
                if next_is_ver and self._lexique.existe(next_forme):
                    infos_n = self._lexique.info(next_forme)
                    has_ver = any(
                        e.get("cgram", "").startswith("VER")
                        or e.get("cgram") == "AUX"
                        for e in infos_n
                    )
                    has_nom = any(
                        e.get("cgram", "").startswith("NOM")
                        for e in infos_n
                    )
                    has_adj = any(
                        e.get("cgram", "").startswith("ADJ")
                        for e in infos_n
                    )
                    if has_ver and not has_nom and not has_adj:
                        mv.correction = transferer_casse(
                            mv.correction, "leur")
                        mv.regle = "homophone.struct.leurs_leur"
                        corrections.append(Correction(
                            index=mv.index, original=mv.forme,
                            corrige=mv.correction,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle=mv.regle,
                            explication=(
                                "Homophone structurel: 'leurs' -> 'leur' "
                                f"(devant verbe '{next_forme}')"
                            ),
                        ))
                        continue

        return corrections

    # ------------------------------------------------------------------
    # 3l. Locutions figees plurielles
    # ------------------------------------------------------------------

    _LOCUTIONS_PLURIEL: dict[tuple[str, str], str] = {
        ("en", "terme"): "termes",       # "en termes de"
        ("en", "matiere"): "mati\u00e8res",  # "en matieres de"
        ("en", "mati\u00e8re"): "mati\u00e8res",
        ("d'", "offre"): "offres",       # "appel d'offres"
        ("d\u2019", "offre"): "offres",
    }

    def _corriger_locutions_pluriel(
        self, mots: list[MotV6],
    ) -> list[Correction]:
        """Corrige les locutions figees qui exigent un pluriel."""
        corrections: list[Correction] = []

        for i, mv in enumerate(mots):
            if mv.regle:
                continue
            if i == 0:
                continue

            forme_low = mv.forme.lower()
            prev_low = mots[i - 1].forme.lower()

            cible = self._LOCUTIONS_PLURIEL.get((prev_low, forme_low))
            if not cible:
                continue

            # Ne corriger que si la forme actuelle differe de la cible
            if forme_low == cible:
                continue

            mv.correction = transferer_casse(mv.correction, cible)
            mv.regle = "accord.locution_pluriel"

            corrections.append(Correction(
                index=mv.index,
                original=mv.forme,
                corrige=mv.correction,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle=mv.regle,
                explication=(
                    f"Locution figee: '{prev_low} {mv.forme}' "
                    f"-> '{prev_low} {cible}'"
                ),
            ))

        return corrections
