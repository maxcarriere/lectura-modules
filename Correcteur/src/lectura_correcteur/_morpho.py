"""Analyseur morphologique CRF pour le français.

Adapte depuis lectura_morpho.py (MORPHO_CRF).
Fichier autonome, zero dependance externe.
Utilise un modele CRF avec decodage Viterbi pur Python.
Predit en une seule passe : POS + Genre + Nombre + Temps + Mode + Personne,
puis lemmatise par regles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__version__ = "1.0.0"

# -- Tokenisation simple ---

_TOKEN_RE = re.compile(
    r"""
    (?P<elision>
        (?:[CcDdJjLlMmNnSsTt]|[Qq]u|[Jj]usqu|[Ll]orsqu|[Pp]uisqu|[Qq]uelqu)
        ['\u2019]
    )
    | (?P<word>   [A-Z\u00c0-\u00dca-z\u00e0-\u00ff\u0153\u0152\u00e6\u00c6](?:[A-Z\u00c0-\u00dca-z\u00e0-\u00ff\u0153\u0152\u00e6\u00c6''\u2019-]*[A-Z\u00c0-\u00dca-z\u00e0-\u00ff\u0153\u0152\u00e6\u00c6])? )
    | (?P<other> \S )
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[tuple[str, bool]]:
    """Decoupe un texte en tokens (mot, is_word).

    Gere les elisions francaises : "l'ecole" -> [("l'", True), ("ecole", True)]
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


# -- Extraction de features ---

def _extract_word_features(
    words: list[str],
    idx: int,
) -> dict[str, str | float]:
    """Extrait les features pour le mot a la position idx."""
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

    feats["BOS"] = 1.0 if idx == 0 else 0.0
    feats["EOS"] = 1.0 if idx == len(words) - 1 else 0.0

    feats["w-1"] = words[idx - 1].lower() if idx > 0 else "__BOS__"
    feats["w+1"] = words[idx + 1].lower() if idx < len(words) - 1 else "__EOS__"

    return feats


def _extract_sequence_features(
    words: list[str],
) -> list[dict[str, str | float]]:
    """Extrait les features pour toute la sequence."""
    return [_extract_word_features(words, i) for i in range(len(words))]


# -- Modele CRF + Viterbi ---

class CrfModel:
    """Modele CRF avec decodage Viterbi pur Python."""

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
        """Charge un modele depuis un fichier JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            state_features=data["state_features"],
            transitions=data["transitions"],
            tags=data["tags"],
        )

    def _score_state(self, feats: dict[str, str | float], tag: str) -> float:
        """Calcule le score d'etat pour un tag donne les features."""
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
        """Decodage Viterbi sur la sequence d'observations."""
        n = len(observations)
        if n == 0:
            return []

        tags = self.tags
        n_tags = len(tags)
        NEG_INF = -1e30

        scores_0: list[float] = []
        for tag in tags:
            s = self._score_state(observations[0], tag)
            s += self.transitions.get("__BOS__", {}).get(tag, 0.0)
            scores_0.append(s)

        viterbi: list[list[float]] = [scores_0]
        backptr: list[list[int]] = [[0] * n_tags]

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

        best_last = max(range(n_tags), key=lambda j: viterbi[n - 1][j])
        result = [0] * n
        result[n - 1] = best_last
        for t in range(n - 2, -1, -1):
            result[t] = backptr[t + 1][result[t + 1]]

        return [tags[idx] for idx in result]

    def predict(self, words: list[str]) -> list[str]:
        """Predit la sequence de tags composites pour une liste de mots."""
        if not words:
            return []
        features = _extract_sequence_features(words)
        return self._viterbi(features)


# -- Decomposition des tags composites ---

def _decompose_tag(tag: str) -> dict:
    """Parse un tag composite en dict structure.

    Exemples :
        "VER|Ind|Pres|3|Plur" -> {"pos": "VER", "mode": "Ind", ...}
        "NOM|Masc|Sing"       -> {"pos": "NOM", "genre": "Masc", ...}
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


# -- Lemmatisation par regles ---

_IRREGULARS: dict[str, str] = {
    # etre
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
    "faite": "faire", "faits": "faire",
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
    # elisions
    "l'": "le", "d'": "de", "s'": "se", "n'": "ne", "j'": "je",
    "m'": "me", "t'": "te", "c'": "ce", "qu'": "que",
    "jusqu'": "jusque", "lorsqu'": "lorsque", "puisqu'": "puisque",
    "quelqu'": "quelque",
}


def _lemmatize_by_rules(word: str, pos: str, traits: dict) -> str:
    """Lemmatise par regles de suffixation."""
    low = word.lower()

    if low in _IRREGULARS:
        return _IRREGULARS[low]

    core_pos = pos.split(":")[0]
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

    # Adjectifs: feminin
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


# -- API publique ---

_DEFAULT_MODEL: Path = Path(__file__).parent / "data" / "morpho_model_crf.json"


class MorphoTagger:
    """Analyseur morphologique CRF pour le francais.

    Predit en une seule passe : POS + Genre + Nombre + Temps + Mode + Personne,
    puis lemmatise par regles. Zero dependance externe.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
    ) -> None:
        path = Path(model_path) if model_path else _DEFAULT_MODEL
        self.model = CrfModel.load(path)

    def tag(self, text: str) -> list[dict]:
        """Tokenise et analyse morphologiquement un texte brut.

        Returns:
            Liste de dicts avec cles : mot, pos, tag_complet, genre, nombre,
            temps, mode, personne, lemme.
        """
        tokens = tokenize(text)
        words = [tok for tok, is_word in tokens if is_word]
        if not words:
            return []
        composite_tags = self.model.predict(words)
        return self._build_results(words, composite_tags)

    def tag_words(self, words: list[str]) -> list[dict]:
        """Analyse morphologiquement une liste de mots deja tokenises."""
        if not words:
            return []
        composite_tags = self.model.predict(words)
        return self._build_results(words, composite_tags)

    def tokenize(self, text: str) -> list[tuple[str, bool]]:
        """Expose la tokenisation pour usage externe."""
        return tokenize(text)

    def _build_results(self, words: list[str], composite_tags: list[str]) -> list[dict]:
        """Construit les dicts resultat depuis les mots et tags composites."""
        results = []
        for word, ctag in zip(words, composite_tags):
            traits = _decompose_tag(ctag)
            lemma = _lemmatize_by_rules(word, traits["pos"], traits)
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
