"""Lectura Morpho Tagger — Analyse morphologique CRF pour le français.

Fichier unique, autonome, zéro dépendance externe.
Utilise un modèle CRF avec décodage Viterbi pur Python.
Prédit en une seule passe : POS + Genre + Nombre + Temps + Mode + Personne,
puis lemmatise par règles.

Usage rapide :
    from lectura_morpho import MorphoTagger

    tagger = MorphoTagger("modele/morpho_model_crf.json")
    result = tagger.tag("Les chats mangent les souris")
    # [{"mot": "Les", "pos": "ART:def", "tag_complet": "ART:def|Plur",
    #   "genre": None, "nombre": "Plur", ...}, ...]

Copyright (c) 2025 Lectura — Licence CC BY-SA 4.0.
Voir LICENCE.txt et ATTRIBUTION.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__version__ = "1.0.0"

# ── Tokenisation simple ─────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"""
    (?P<elision>
        (?:[CcDdJjLlMmNnSsTt]|[Qq]u|[Jj]usqu|[Ll]orsqu|[Pp]uisqu|[Qq]uelqu)
        ['']
    )
    | (?P<word>   [A-ZÀ-Üa-zà-ÿœŒæÆ](?:[A-ZÀ-Üa-zà-ÿœŒæÆ''-]*[A-ZÀ-Üa-zà-ÿœŒæÆ])? )
    | (?P<other> \S )
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[tuple[str, bool]]:
    """Découpe un texte en tokens (mot, is_word).

    Gère les élisions françaises : « l'école » → [("l'", True), ("école", True)]
    tout en gardant « aujourd'hui » comme un seul token.
    """
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    tokens: list[tuple[str, bool]] = []
    for m in _TOKEN_RE.finditer(text):
        if m.group("elision"):
            tokens.append((m.group("elision"), True))
        elif m.group("word"):
            tokens.append((m.group("word"), True))
        else:
            tokens.append((m.group("other"), False))
    return tokens


# ── Extraction de features ──────────────────────────────────────────────────

def _extract_word_features(
    words: list[str],
    idx: int,
) -> dict[str, str | float]:
    """Extrait les features pour le mot à la position idx.

    Features identiques à POS_CRF + suf4/suf5 pour capter les terminaisons
    verbales longues (-aient, -ions, -ement).
    """
    word = words[idx]
    low = word.lower()
    feats: dict[str, str | float] = {
        "bias": 1.0,
        "word": low,
        "len": float(len(low)),
        "suf2": low[-2:] if len(low) >= 2 else low,
        "suf3": low[-3:] if len(low) >= 3 else low,
        "suf4": low[-4:] if len(low) >= 4 else low,
        "suf5": low[-5:] if len(low) >= 5 else low,
        "pre2": low[:2] if len(low) >= 2 else low,
        "pre3": low[:3] if len(low) >= 3 else low,
        "is_upper": 1.0 if word.isupper() else 0.0,
        "is_title": 1.0 if word.istitle() else 0.0,
        "is_digit": 1.0 if word.isdigit() else 0.0,
    }

    # Position
    feats["BOS"] = 1.0 if idx == 0 else 0.0
    feats["EOS"] = 1.0 if idx == len(words) - 1 else 0.0

    # Contexte (bigramme)
    feats["w-1"] = words[idx - 1].lower() if idx > 0 else "__BOS__"
    feats["w+1"] = words[idx + 1].lower() if idx < len(words) - 1 else "__EOS__"

    return feats


def _extract_sequence_features(
    words: list[str],
) -> list[dict[str, str | float]]:
    """Extrait les features pour toute la séquence."""
    return [_extract_word_features(words, i) for i in range(len(words))]


# ── Modèle CRF + Viterbi ────────────────────────────────────────────────────

class CrfModel:
    """Modèle CRF avec décodage Viterbi pur Python.

    Le modèle est chargé depuis un fichier JSON contenant :
      - state_features : dict[feature, dict[tag, poids]]
      - transitions    : dict[tag_précédent, dict[tag_courant, poids]]
      - tags           : list[str]
    """

    __slots__ = ("state_features", "transitions", "tags", "_tag_idx")

    def __init__(
        self,
        state_features: dict[str, dict[str, float]],
        transitions: dict[str, dict[str, float]],
        tags: list[str],
    ) -> None:
        self.state_features = state_features
        self.transitions = transitions
        self.tags = tags
        self._tag_idx = {t: i for i, t in enumerate(tags)}

    @classmethod
    def load(cls, path: Path | str) -> CrfModel:
        """Charge un modèle depuis un fichier JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            state_features=data["state_features"],
            transitions=data["transitions"],
            tags=data["tags"],
        )

    def _score_state(self, feats: dict[str, str | float], tag: str) -> float:
        """Calcule le score d'état pour un tag donné les features."""
        score = 0.0
        for feat_key, feat_val in feats.items():
            lookup = f"{feat_key}:{feat_val}"
            weights = self.state_features.get(lookup)
            if weights and tag in weights:
                score += weights[tag]
        return score

    def _viterbi(
        self,
        observations: list[dict[str, str | float]],
    ) -> list[str]:
        """Décodage Viterbi sur la séquence d'observations."""
        n = len(observations)
        if n == 0:
            return []

        tags = self.tags
        n_tags = len(tags)
        NEG_INF = -1e30

        # Initialisation (t=0)
        scores_0: list[float] = []
        for tag in tags:
            s = self._score_state(observations[0], tag)
            s += self.transitions.get("__BOS__", {}).get(tag, 0.0)
            scores_0.append(s)

        viterbi: list[list[float]] = [scores_0]
        backptr: list[list[int]] = [[0] * n_tags]

        # Récurrence
        for t in range(1, n):
            scores_t: list[float] = []
            bp_t: list[int] = []
            for j, tag_j in enumerate(tags):
                state_score = self._score_state(observations[t], tag_j)
                best_score = NEG_INF
                best_prev = 0

                for i, tag_i in enumerate(tags):
                    prev = viterbi[t - 1][i]
                    if prev <= NEG_INF:
                        continue
                    trans = self.transitions.get(tag_i, {}).get(tag_j, 0.0)
                    score = prev + trans + state_score
                    if score > best_score:
                        best_score = score
                        best_prev = i

                scores_t.append(best_score)
                bp_t.append(best_prev)

            viterbi.append(scores_t)
            backptr.append(bp_t)

        # Backtrack
        best_last = max(range(n_tags), key=lambda j: viterbi[n - 1][j])
        result = [0] * n
        result[n - 1] = best_last
        for t in range(n - 2, -1, -1):
            result[t] = backptr[t + 1][result[t + 1]]

        return [tags[idx] for idx in result]

    def predict(self, words: list[str]) -> list[str]:
        """Prédit la séquence de tags composites pour une liste de mots."""
        if not words:
            return []
        features = _extract_sequence_features(words)
        return self._viterbi(features)


# ── Mini-lexique (correction post-CRF) ──────────────────────────────────────

def _load_mini_lexicon(path: Path | str) -> dict[str, str]:
    """Charge le mini-lexique depuis un fichier JSON.

    Le JSON contient des catégories (contractions, prepositions, …),
    chacune étant un dict mot → tag POS de base. Les clés commençant par '_'
    sont ignorées.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    lexicon: dict[str, str] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            for word, tag in value.items():
                if not word.startswith("_"):
                    lexicon[word] = tag
        elif isinstance(value, str):
            lexicon[key] = value
    return lexicon


def _apply_mini_lexicon(
    words: list[str],
    tags: list[str],
    lexicon: dict[str, str],
) -> list[str]:
    """Corrige la partie POS des tags composites via le mini-lexique.

    Le mini-lexique contient des POS de base (ex: "PRE"). Si le tag prédit
    a une POS différente, on le remplace en gardant la POS du lexique.
    """
    corrected = list(tags)
    for i, word in enumerate(words):
        low = word.lower()
        if low in lexicon:
            lex_pos = lexicon[low]
            pred_pos = tags[i].split("|")[0]
            if pred_pos != lex_pos:
                corrected[i] = lex_pos
    return corrected


# ── Décomposition des tags composites ────────────────────────────────────────

def _decompose_tag(tag: str) -> dict:
    """Parse un tag composite en dict structuré.

    Exemples :
        "VER|Ind|Pres|3|Plur" → {"pos": "VER", "mode": "Ind", "temps": "Pres",
                                   "personne": "3", "nombre": "Plur", "genre": None}
        "NOM|Masc|Sing"       → {"pos": "NOM", "genre": "Masc", "nombre": "Sing", ...}
        "PRE"                 → {"pos": "PRE", "genre": None, "nombre": None, ...}
    """
    parts = tag.split("|")
    pos = parts[0]
    traits = parts[1:]

    result = {
        "pos": pos,
        "genre": None,
        "nombre": None,
        "temps": None,
        "mode": None,
        "personne": None,
    }

    _GENDERS = {"Masc", "Fem"}
    _NUMBERS = {"Sing", "Plur"}
    _TENSES = {"Pres", "Imp", "Past", "Fut"}
    _MOODS = {"Ind", "Sub", "Cnd", "Imp", "Part", "Inf", "Ger"}
    _PERSONS = {"1", "2", "3"}

    for t in traits:
        if t in _GENDERS:
            result["genre"] = t
        elif t in _NUMBERS:
            result["nombre"] = t
        elif t in _TENSES:
            result["temps"] = t
        elif t in _MOODS:
            result["mode"] = t
        elif t in _PERSONS:
            result["personne"] = t

    return result


# ── Lemmatisation par règles ─────────────────────────────────────────────────

_IRREGULARS: dict[str, str] = {
    # être
    "suis": "être", "es": "être", "est": "être", "sommes": "être",
    "êtes": "être", "sont": "être", "étais": "être", "était": "être",
    "étions": "être", "étiez": "être", "étaient": "être", "fus": "être",
    "fut": "être", "fûmes": "être", "fûtes": "être", "furent": "être",
    "serai": "être", "seras": "être", "sera": "être", "serons": "être",
    "serez": "être", "seront": "être", "sois": "être", "soit": "être",
    "soyons": "être", "soyez": "être", "soient": "être",
    "serais": "être", "serait": "être", "serions": "être",
    "seriez": "être", "seraient": "être", "été": "être",
    "étant": "être",
    # avoir
    "ai": "avoir", "as": "avoir", "a": "avoir", "avons": "avoir",
    "avez": "avoir", "ont": "avoir", "avais": "avoir", "avait": "avoir",
    "avions": "avoir", "aviez": "avoir", "avaient": "avoir",
    "eus": "avoir", "eut": "avoir", "eûmes": "avoir", "eûtes": "avoir",
    "eurent": "avoir", "aurai": "avoir", "auras": "avoir", "aura": "avoir",
    "aurons": "avoir", "aurez": "avoir", "auront": "avoir",
    "aie": "avoir", "aies": "avoir", "ait": "avoir", "ayons": "avoir",
    "ayez": "avoir", "aient": "avoir", "aurais": "avoir",
    "aurait": "avoir", "aurions": "avoir", "auriez": "avoir",
    "auraient": "avoir", "eu": "avoir", "ayant": "avoir",
    # aller
    "vais": "aller", "vas": "aller", "va": "aller", "allons": "aller",
    "allez": "aller", "vont": "aller", "irai": "aller", "iras": "aller",
    "ira": "aller", "irons": "aller", "irez": "aller", "iront": "aller",
    "aille": "aller", "ailles": "aller", "aillent": "aller",
    "allé": "aller", "allée": "aller", "allés": "aller", "allées": "aller",
    # faire
    "fais": "faire", "fait": "faire", "faisons": "faire", "faites": "faire",
    "font": "faire", "faisais": "faire", "faisait": "faire",
    "faisions": "faire", "faisiez": "faire", "faisaient": "faire",
    "fis": "faire", "fit": "faire", "fîmes": "faire", "fîtes": "faire",
    "firent": "faire", "ferai": "faire", "feras": "faire",
    "fera": "faire", "ferons": "faire", "ferez": "faire", "feront": "faire",
    "fasse": "faire", "fasses": "faire", "fassent": "faire",
    "ferais": "faire", "ferait": "faire", "ferions": "faire",
    "feriez": "faire", "feraient": "faire", "faisant": "faire",
    "faite": "faire", "faits": "faire", "faites": "faire",
    # pouvoir
    "peux": "pouvoir", "peut": "pouvoir", "pouvons": "pouvoir",
    "pouvez": "pouvoir", "peuvent": "pouvoir", "pouvais": "pouvoir",
    "pouvait": "pouvoir", "pouvions": "pouvoir", "pouviez": "pouvoir",
    "pouvaient": "pouvoir", "pus": "pouvoir", "put": "pouvoir",
    "pourrai": "pouvoir", "pourras": "pouvoir", "pourra": "pouvoir",
    "pourrons": "pouvoir", "pourrez": "pouvoir", "pourront": "pouvoir",
    "puisse": "pouvoir", "puisses": "pouvoir", "puissent": "pouvoir",
    "pu": "pouvoir",
    # vouloir
    "veux": "vouloir", "veut": "vouloir", "voulons": "vouloir",
    "voulez": "vouloir", "veulent": "vouloir", "voulu": "vouloir",
    "voudrai": "vouloir", "voudras": "vouloir", "voudra": "vouloir",
    # savoir
    "sais": "savoir", "sait": "savoir", "savons": "savoir",
    "savez": "savoir", "savent": "savoir", "su": "savoir",
    "saurai": "savoir", "sauras": "savoir", "saura": "savoir",
    "sache": "savoir", "saches": "savoir", "sachent": "savoir",
    # devoir
    "dois": "devoir", "doit": "devoir", "devons": "devoir",
    "devez": "devoir", "doivent": "devoir", "dû": "devoir",
    "devrai": "devoir", "devras": "devoir", "devra": "devoir",
    # dire
    "dis": "dire", "dit": "dire", "disons": "dire",
    "dites": "dire", "disent": "dire", "dite": "dire",
    "dits": "dire", "disant": "dire",
    # prendre
    "prends": "prendre", "prend": "prendre", "prenons": "prendre",
    "prenez": "prendre", "prennent": "prendre", "pris": "prendre",
    "prise": "prendre", "prises": "prendre",
    # venir
    "viens": "venir", "vient": "venir", "venons": "venir",
    "venez": "venir", "viennent": "venir", "venu": "venir",
    "venue": "venir", "venus": "venir", "venues": "venir",
    "viendrai": "venir", "viendras": "venir", "viendra": "venir",
    # élisions
    "l'": "le", "d'": "de", "s'": "se", "n'": "ne", "j'": "je",
    "m'": "me", "t'": "te", "c'": "ce", "qu'": "que",
    "jusqu'": "jusque", "lorsqu'": "lorsque", "puisqu'": "puisque",
    "quelqu'": "quelque",
}


def _lemmatize_by_rules(word: str, pos: str, traits: dict) -> str:
    """Lemmatise par règles de suffixation.

    Niveaux :
    1. Table des irréguliers (~120 formes)
    2. Règles de suffixation (pluriel, féminin, conjugaisons)
    3. Fallback: word.lower()
    """
    low = word.lower()

    # 1. Table irréguliers
    if low in _IRREGULARS:
        return _IRREGULARS[low]

    core_pos = pos.split(":")[0]

    # 2. Règles de suffixation
    mode = traits.get("mode")
    nombre = traits.get("nombre")
    genre = traits.get("genre")

    # Verbes
    if core_pos in ("VER", "AUX"):
        if mode == "Inf":
            return low
        if mode == "Ger":
            if low.endswith("ant"):
                stem = low[:-3]
                if stem.endswith("e"):
                    return stem + "er"
                return stem + "er"
            return low
        if mode == "Part":
            if low.endswith("ée"):
                return low[:-2] + "é"
            if low.endswith("ées"):
                return low[:-3] + "é"
            if low.endswith("és"):
                return low[:-2] + "é"
            if low.endswith("ies"):
                return low[:-3] + "i"
            if low.endswith("ie"):
                return low[:-2] + "i"
            if low.endswith("is"):
                return low[:-1] + "re"
            if low.endswith("ise"):
                return low[:-3] + "ire"
            if low.endswith("ite"):
                return low[:-3] + "ire"
            if low.endswith("ites"):
                return low[:-4] + "ire"
            if low.endswith("tes"):
                return low[:-2]
            if low.endswith("ts"):
                return low[:-1]
            if low.endswith("te"):
                return low[:-1]
            return low
        # Conjugated verb: try to recover infinitive
        if low.endswith("ent"):
            stem = low[:-3]
            return stem + "er"
        if low.endswith("es"):
            return low[:-2] + "er"
        if low.endswith("ez"):
            return low[:-2] + "er"
        if low.endswith("ons"):
            return low[:-3] + "er"
        if low.endswith("ais") or low.endswith("ait"):
            return low[:-3] + "er"
        if low.endswith("aient"):
            return low[:-5] + "er"
        if low.endswith("ions"):
            return low[:-4] + "er"
        if low.endswith("iez"):
            return low[:-3] + "er"
        if low.endswith("era") or low.endswith("erai"):
            return low.split("er")[0] + "er" if "er" in low else low
        if low.endswith("e"):
            return low[:-1] + "er"
        return low

    # Noms et adjectifs: pluriel
    if core_pos in ("NOM", "ADJ"):
        if nombre == "Plur":
            if low.endswith("aux"):
                return low[:-3] + "al"
            if low.endswith("eaux"):
                return low[:-4] + "eau"
            if low.endswith("s") and not low.endswith("ss"):
                return low[:-1]
            if low.endswith("x"):
                return low[:-1]

    # Adjectifs: féminin
    if core_pos == "ADJ" and genre == "Fem":
        if low.endswith("euse"):
            return low[:-4] + "eux"
        if low.endswith("euses"):
            return low[:-5] + "eux"
        if low.endswith("trice"):
            return low[:-5] + "teur"
        if low.endswith("trices"):
            return low[:-6] + "teur"
        if low.endswith("ive"):
            return low[:-3] + "if"
        if low.endswith("ives"):
            return low[:-4] + "if"
        if low.endswith("ée"):
            return low[:-2] + "é"
        if low.endswith("ées"):
            return low[:-3] + "é"
        if low.endswith("elle"):
            return low[:-4] + "el"
        if low.endswith("elles"):
            return low[:-5] + "el"
        if low.endswith("enne"):
            return low[:-4] + "en"
        if low.endswith("ennes"):
            return low[:-5] + "en"
        if low.endswith("e") and not low.endswith("ee"):
            return low[:-1]

    return low


# ── Lexique GLAFF (fallback optionnel) ──────────────────────────────────────

def _load_lexicon(path: Path | str) -> dict[tuple[str, str], str]:
    """Charge un lexique JSON forme|POS → lemme.

    Format attendu : {"forme|POS": "lemme", ...} ou dict imbriqué.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    lexicon: dict[tuple[str, str], str] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            for entry_key, lemma in value.items():
                if entry_key.startswith("_"):
                    continue
                if "|" in entry_key:
                    form, pos = entry_key.rsplit("|", 1)
                    lexicon[(form.lower(), pos)] = lemma
                else:
                    lexicon[(entry_key.lower(), "")] = lemma
        elif isinstance(value, str):
            if "|" in key:
                form, pos = key.rsplit("|", 1)
                lexicon[(form.lower(), pos)] = value
            else:
                lexicon[(key.lower(), "")] = value
    return lexicon


def _apply_lexicon(
    word: str,
    pos: str,
    lexicon: dict[tuple[str, str], str],
) -> str | None:
    """Cherche un lemme dans le lexique pour (forme, POS)."""
    low = word.lower()
    result = lexicon.get((low, pos))
    if result:
        return result
    core_pos = pos.split(":")[0]
    result = lexicon.get((low, core_pos))
    if result:
        return result
    result = lexicon.get((low, ""))
    if result:
        return result
    return None


# ── API publique ─────────────────────────────────────────────────────────────

class MorphoTagger:
    """Analyseur morphologique CRF pour le français.

    Prédit en une seule passe : POS + Genre + Nombre + Temps + Mode + Personne,
    puis lemmatise par règles. Zéro dépendance externe.

    Args:
        model_path: Chemin vers le modèle CRF (JSON).
        lexicon_path: Chemin vers le lexique GLAFF (JSON, optionnel).
            Si fourni, sert de fallback pour la lemmatisation.
        mini_lexicon_path: Chemin vers le mini-lexique POS (JSON, optionnel).
            Si fourni, corrige les POS des mots-outils non ambigus.

    Exemple :
        >>> tagger = MorphoTagger("modele/morpho_model_crf.json")
        >>> tagger.tag("Les chats mangent les souris")
        [{"mot": "Les", "pos": "ART:def", "tag_complet": "ART:def|Plur", ...}, ...]
    """

    def __init__(
        self,
        model_path: str | Path,
        lexicon_path: str | Path | None = None,
        mini_lexicon_path: str | Path | None = None,
    ) -> None:
        self.model = CrfModel.load(model_path)
        self.lexicon: dict[tuple[str, str], str] = {}
        if lexicon_path is not None:
            p = Path(lexicon_path)
            if p.exists():
                self.lexicon = _load_lexicon(p)
        self.mini_lexicon: dict[str, str] = {}
        if mini_lexicon_path is not None:
            p = Path(mini_lexicon_path)
            if p.exists():
                self.mini_lexicon = _load_mini_lexicon(p)

    def _predict(self, words: list[str]) -> list[str]:
        """Prédit les tags composites, puis applique le mini-lexique."""
        tags = self.model.predict(words)
        if self.mini_lexicon:
            tags = _apply_mini_lexicon(words, tags, self.mini_lexicon)
        return tags

    def _lemmatize(self, word: str, pos: str, traits: dict) -> str:
        """Lemmatise un mot en utilisant les 3 niveaux."""
        # 1. Lexique GLAFF (si chargé)
        if self.lexicon:
            lex_lemma = _apply_lexicon(word, pos, self.lexicon)
            if lex_lemma:
                return lex_lemma
        # 2 + 3. Irréguliers + règles de suffixation
        return _lemmatize_by_rules(word, pos, traits)

    def tag(self, text: str) -> list[dict]:
        """Tokenise et analyse morphologiquement un texte brut.

        Args:
            text: Texte en français (ex: "Les chats mangent les souris").

        Returns:
            Liste de dicts avec clés : mot, pos, tag_complet, genre, nombre,
            temps, mode, personne, lemme.
        """
        tokens = tokenize(text)
        words = [tok for tok, is_word in tokens if is_word]
        if not words:
            return []
        composite_tags = self._predict(words)
        return self._build_results(words, composite_tags)

    def tag_words(self, words: list[str]) -> list[dict]:
        """Analyse morphologiquement une liste de mots déjà tokenisés.

        Args:
            words: Liste de mots (ex: ["Les", "chats", "mangent"]).

        Returns:
            Liste de dicts avec clés : mot, pos, tag_complet, genre, nombre,
            temps, mode, personne, lemme.
        """
        if not words:
            return []
        composite_tags = self._predict(words)
        return self._build_results(words, composite_tags)

    def _build_results(self, words: list[str], composite_tags: list[str]) -> list[dict]:
        """Construit les dicts résultat depuis les mots et tags composites."""
        results = []
        for word, ctag in zip(words, composite_tags):
            traits = _decompose_tag(ctag)
            lemma = self._lemmatize(word, traits["pos"], traits)
            results.append({
                "mot": word,
                "pos": traits["pos"],
                "tag_complet": ctag,
                "genre": traits["genre"],
                "nombre": traits["nombre"],
                "temps": traits["temps"],
                "mode": traits["mode"],
                "personne": traits["personne"],
                "lemme": lemma,
            })
        return results

    def tag_formatted(self, text: str) -> str:
        """Retourne un résultat formaté lisible.

        Returns:
            Texte formaté avec un mot par ligne, montrant POS, traits et lemme.
        """
        results = self.tag(text)
        if not results:
            return "(aucun mot détecté)"
        max_mot = max(len(r["mot"]) for r in results)
        max_tag = max(len(r["tag_complet"]) for r in results)
        max_lemme = max(len(r["lemme"]) for r in results)
        lines = []
        for r in results:
            traits_parts = []
            if r["genre"]:
                traits_parts.append(f"G={r['genre']}")
            if r["nombre"]:
                traits_parts.append(f"N={r['nombre']}")
            if r["mode"]:
                traits_parts.append(f"M={r['mode']}")
            if r["temps"]:
                traits_parts.append(f"T={r['temps']}")
            if r["personne"]:
                traits_parts.append(f"P={r['personne']}")
            traits_str = " ".join(traits_parts) if traits_parts else "-"
            lines.append(
                f"  {r['mot']:<{max_mot}}  {r['tag_complet']:<{max_tag}}"
                f"  → {r['lemme']:<{max_lemme}}  [{traits_str}]"
            )
        return "\n".join(lines)
