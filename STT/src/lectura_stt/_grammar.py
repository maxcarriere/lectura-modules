"""Post-traitement grammatical par regles structurelles + lexique.

Opere sur les ortho_words finaux (sortie du pipeline, apres merge/elision/spellcheck)
en consultant le lexique morphologique pour le POS/nombre des mots de contexte.

Groupes de regles :
  A. Accord pronom-verbe : on + verbe_3p → verbe_3s
  A2. Accord pronom-verbe il/ils : phone identique → singulier,
      phone different → confiance au verbe, corriger le pronom
  B. Confusion est/et : apres clitique, qui, ne/n'
  C. Confusion leur/leurs : leurs + verbe
  D. Confusion a/a : apres pronom sujet 3s, n', y
  E. Confusion ou/ou : apres d'
  F. Accord nombre : determinant pluriel/singulier + nom/adj
  G. Avoir + infinitif -er → participe passe -e
  H. Possessif + fil → fils
  I. tu + verbe present sans -s → ajouter -s (2e personne)
  J. Etre + infinitif -er → participe passe -e
  K. Correction accents pour mots inconnus du lexique
  L. Preposition + PP -e → infinitif -er
  M. Consonne finale muette manquante : temp → temps, corp → corps

Benchmark : +0.12% WER sur 5000 exemples courants (60 gains, 12 pertes).

Copyright (C) 2025-2026 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

import logging
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────

# Pronoms sujets 3e personne du singulier (prennent "a" du verbe avoir)
_SUBJ_3S = frozenset({
    "il", "elle", "on", "qui", "ça", "cela", "ce",
    "rien", "chacun", "chacune", "quiconque", "personne", "nul",
})

# Clitiques elidees : apres ces mots, "et" ne peut pas etre correct → "est"
_CLITICS_BEFORE_EST = frozenset({"c'", "s'", "l'", "qu'", "m'", "t'"})

# Determinants pluriels (forcent le nom/adj suivant au pluriel)
_PLUR_DETS = frozenset({
    "les", "des", "ces", "ses", "mes", "tes", "nos", "vos", "leurs",
    "aux", "plusieurs", "quelques",
    "certains", "certaines",
    "différents", "différentes",
    "divers", "diverses",
})

# Determinants singuliers (forcent le nom/adj suivant au singulier)
_SING_DETS = frozenset({
    "le", "la", "un", "une", "ce", "cet", "cette",
    "son", "sa", "mon", "ma", "ton", "ta",
    "notre", "votre", "leur",
    "au", "chaque", "aucun", "aucune",
})

# Formes conjuguees non-ambigues d'avoir (pour regle G : avoir + inf → PP)
_AVOIR_FORMS_SAFE = frozenset({
    "ai", "as", "avons", "avez",                     # present
    "avais", "avait", "avions", "aviez", "avaient",   # imparfait
    "aurai", "auras", "aura", "aurons", "aurez", "auront",  # futur
    "aurais", "aurait", "aurions", "auriez", "auraient",     # conditionnel
    "aie", "aies", "ait", "ayons", "ayez", "aient",          # subj. present
})
_AVOIR_A_SUBJECTS = frozenset({
    "il", "elle", "on", "qui", "ça", "cela", "l'",
})
_AVOIR_ONT_SUBJECTS = frozenset({"ils", "elles"})

# Determinants possessifs (pour regle H : possessif + fil → fils)
_POSSESSIFS = frozenset({
    "son", "sa", "mon", "ma", "ton", "ta",
    "leur", "notre", "votre",
})

# Clitiques objets (peuvent s'intercaler entre sujet et verbe)
_CLITIC_OBJECTS = frozenset({
    "me", "m'", "te", "t'", "se", "s'", "le", "la", "l'",
    "lui", "nous", "vous", "les", "leur", "en", "y",
    "ne", "n'",
})

# Formes d'etre non-ambigues (pour regle etre + inf → PP)
_ETRE_FORMS_SAFE = frozenset({
    "sommes", "êtes",                                         # present
    "étais", "était", "étions", "étiez", "étaient",           # imparfait
    "serai", "seras", "sera", "serons", "serez", "seront",   # futur
    "serais", "serait", "serions", "seriez", "seraient",      # conditionnel
    "sois", "soit", "soyons", "soyez", "soient",              # subj. present
    "suis", "es",                                              # present 1s/2s
})
_ETRE_EST_SUBJECTS = frozenset({
    "il", "elle", "on", "qui", "ça", "cela",
})
_ETRE_SONT_SUBJECTS = frozenset({"ils", "elles"})


# ── Helpers ───────────────────────────────────────────────────

def _preserve_case(original: str, replacement: str) -> str:
    """Preserve la casse du mot original dans le remplacement."""
    if original and original[0].isupper() and replacement and replacement[0].islower():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _strip_accents(s: str) -> str:
    """Retire les diacritiques (accents, cedille) d'une chaine."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# ── GrammarLookup ────────────────────────────────────────────

class GrammarLookup:
    """Cache en memoire des POS/morpho depuis la DB lexique.

    Charge toutes les entrees de la table ``lexique`` et construit
    un index en memoire pour les requetes rapides utilisees par
    :func:`corriger_grammatical`.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._entries: dict[str, list[tuple[str, str, str, str, float, str]]] = defaultdict(list)

        # Index lemme → formes verbales pour lookup par conjugaison
        self._verb_by_lemme: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        # lemme_lower → [(ortho_lower, multext, phone), ...]
        self._word_to_verb_lemmes: dict[str, set[str]] = defaultdict(set)
        # ortho_lower → {lemme_lower, ...}

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT ortho, cgram, nombre, genre, multext, freq, phone, lemme "
                "FROM lexique WHERE ortho IS NOT NULL AND ortho != ''"
            )
        except sqlite3.OperationalError:
            conn.close()
            return

        for ortho, cgram, nombre, genre, multext, freq, phone, lemme in cur:
            lower = ortho.lower()
            self._entries[lower].append((
                cgram or "",
                nombre or "",
                genre or "",
                multext or "",
                freq or 0.0,
                phone or "",
            ))
            # Index lemme pour VER/AUX
            if cgram in ("VER", "AUX") and lemme and multext:
                lemme_lower = lemme.lower()
                self._word_to_verb_lemmes[lower].add(lemme_lower)
                self._verb_by_lemme[lemme_lower].append((
                    lower, multext, phone or "",
                ))
        conn.close()

        for key in self._entries:
            self._entries[key].sort(key=lambda e: e[4], reverse=True)

        n_entries = len(self._entries)
        logger.info("GrammarLookup: %d formes, %d lemmes verbaux",
                     n_entries, len(self._verb_by_lemme))

        # Carte accent : stripped_form → {accented_form: total_freq}
        self._accent_map: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float),
        )
        for word, ents in self._entries.items():
            stripped = _strip_accents(word)
            if stripped != word:
                total_freq = sum(e[4] for e in ents)
                self._accent_map[stripped][word] += total_freq

    # ── Requetes de base ──

    def entries(self, word: str) -> list[tuple[str, str, str, str, float]]:
        return self._entries.get(word.lower(), [])

    def pos(self, word: str) -> str | None:
        ents = self.entries(word)
        return ents[0][0] if ents else None

    def nombre(self, word: str) -> str | None:
        for _cgram, nombre, _genre, _multext, _freq, _phone in self.entries(word):
            if nombre:
                return nombre
        return None

    # ── Verbes ──

    def is_verb_3p(self, word: str) -> bool:
        for cgram, _nombre, _genre, multext, _freq, _phone in self.entries(word):
            if cgram in ("VER", "AUX") and "3p" in multext:
                return True
        return False

    def is_verb_3s(self, word: str) -> bool:
        for cgram, _nombre, _genre, multext, _freq, _phone in self.entries(word):
            if cgram in ("VER", "AUX") and "3s" in multext:
                return True
        return False

    def is_verb(self, word: str) -> bool:
        for cgram, _nombre, _genre, _multext, _freq, _phone in self.entries(word):
            if cgram in ("VER", "AUX"):
                return True
        return False

    # ── Lookup par lemme ──

    def _find_verb_forms(self, word: str, to_person: str) -> list[tuple[str, str, str]]:
        """Trouve les formes verbales d'un meme lemme pour une personne cible.

        Returns list of (ortho_lower, multext, phone).
        """
        lower = word.lower()
        lemmes = self._word_to_verb_lemmes.get(lower, set())
        results: list[tuple[str, str, str]] = []
        for lemme in lemmes:
            for ortho, multext, phone in self._verb_by_lemme.get(lemme, []):
                if to_person in multext:
                    results.append((ortho, multext, phone))
        return results

    def _mood_tense_prefixes(self, word: str, person: str) -> list[tuple[str, float]]:
        """Retourne les prefixes mode+temps avec frequence, tries par freq desc.

        Ex: [('Vmii', 266.0), ('Vmip', 0.5)] pour 'étaient'.
        """
        prefixes: dict[str, float] = {}
        for cgram, _n, _g, multext, freq, _ph in self.entries(word):
            if cgram in ("VER", "AUX") and person in multext and len(multext) >= 4:
                mt = multext[:4]
                if mt not in prefixes or freq > prefixes[mt]:
                    prefixes[mt] = freq
        return sorted(prefixes.items(), key=lambda x: -x[1])

    def _find_form_by_mt(
        self, forms: list[tuple[str, str, str]],
        mt_prefixes: list[tuple[str, float]],
        exclude_person: str,
    ) -> str | None:
        """Trouve la meilleure forme parmi les candidats, par mode+temps.

        Priorite : meme mode+temps trie par frequence du mot source.
        ``exclude_person`` filtre les formes ambigues.
        """
        # Parcourir les prefixes mode+temps par freq desc
        for mt, _freq in mt_prefixes:
            for ortho, multext, _phone in forms:
                if multext.startswith(mt) and not self._has_person(ortho, exclude_person):
                    return ortho
        # Fallback : n'importe quelle forme exclusive
        for ortho, _multext, _phone in forms:
            if not self._has_person(ortho, exclude_person):
                return ortho
        return None

    def _has_person(self, word: str, person: str) -> bool:
        """Verifie si un mot a une entree VER/AUX pour une personne donnee."""
        for cgram, _n, _g, multext, _f, _ph in self.entries(word):
            if cgram in ("VER", "AUX") and person in multext:
                return True
        return False

    def verb_3p_to_3s(self, word: str) -> str | None:
        """Convertit un verbe 3p exclusif en sa forme 3s via lemme.

        Cherche la forme 3s avec le meme mode+temps (Vmii3p → Vmii3s),
        en priorisant l'indicatif present.
        """
        lower = word.lower()
        if not self.is_verb_3p(word) or self.is_verb_3s(word):
            return None
        if self._max_nom_adj_freq(lower) > self._max_verb_freq(lower) * 2:
            return None

        mt_prefixes = self._mood_tense_prefixes(word, "3p")
        forms_3s = self._find_verb_forms(word, "3s")
        best = self._find_form_by_mt(forms_3s, mt_prefixes, exclude_person="3p")
        if best is not None:
            return _preserve_case(word, best)
        return None

    def verb_3s_to_3p(self, word: str) -> str | None:
        """Convertit un verbe 3s exclusif en sa forme 3p via lemme.

        Cherche la forme 3p avec le meme mode+temps.
        """
        lower = word.lower()
        if not self.is_verb_3s(word) or self.is_verb_3p(word):
            return None
        if self._max_nom_adj_freq(lower) > self._max_verb_freq(lower) * 2:
            return None

        mt_prefixes = self._mood_tense_prefixes(word, "3s")
        forms_3p = self._find_verb_forms(word, "3p")
        best = self._find_form_by_mt(forms_3p, mt_prefixes, exclude_person="3s")
        if best is not None:
            return _preserve_case(word, best)
        return None

    # ── Comparaison phonetique 3s/3p ──

    def verb_3s_3p_same_phone(self, word_3p: str, word_3s: str) -> bool:
        """Verifie si les formes 3s et 3p du meme mode+temps partagent un phone.

        Compare par mode+temps trie par frequence : si faisaient (Vmii3p)
        et faisait (Vmii3s) ont le meme phone → True (ambigu).
        Si disent (Vmip3p) et dit (Vmip3s) ont des phones differents
        → False (CTC peut distinguer).
        Seul le mode+temps le plus frequent est considere.
        """
        mt_prefixes = self._mood_tense_prefixes(word_3p, "3p")
        if not mt_prefixes:
            return True  # pas d'info → prudent

        forms_3s = self._find_verb_forms(word_3p, "3s")

        # Verifier le mode+temps le plus frequent (premier dans la liste)
        mt_top = mt_prefixes[0][0]

        phones_3p: set[str] = set()
        for cgram, _n, _g, multext, _f, phone in self.entries(word_3p):
            if cgram in ("VER", "AUX") and multext.startswith(mt_top) and "3p" in multext and phone:
                phones_3p.add(phone)

        phones_3s: set[str] = set()
        for ortho, multext, phone in forms_3s:
            if multext.startswith(mt_top) and phone:
                phones_3s.add(phone)

        if phones_3p and phones_3s and not (phones_3p & phones_3s):
            return False

        return True

    # ── Noms / adjectifs ──

    def is_adj_or_pp(self, word: str) -> bool:
        for cgram, _nombre, _genre, multext, _freq, _phone in self.entries(word):
            if cgram == "ADJ":
                return True
            if cgram == "VER" and multext.startswith("Vmps"):
                return True
        return False

    def is_noun(self, word: str) -> bool:
        for cgram, _nombre, _genre, _multext, _freq, _phone in self.entries(word):
            if cgram == "NOM":
                return True
        return False

    def _nom_adj_nombres(self, word: str) -> set[str]:
        result: set[str] = set()
        for cgram, nombre, _genre, _multext, _freq, _phone in self.entries(word):
            if cgram in ("NOM", "ADJ") and nombre:
                result.add(nombre)
        return result

    def _max_verb_freq(self, word: str) -> float:
        mx = 0.0
        for cgram, _nombre, _genre, _multext, freq, _phone in self.entries(word):
            if cgram in ("VER", "AUX") and freq > mx:
                mx = freq
        return mx

    def _max_nom_adj_freq(self, word: str) -> float:
        mx = 0.0
        for cgram, _nombre, _genre, _multext, freq, _phone in self.entries(word):
            if cgram in ("NOM", "ADJ") and freq > mx:
                mx = freq
        return mx

    def try_pluralize(self, word: str) -> str | None:
        lower = word.lower()
        nombres = self._nom_adj_nombres(lower)
        if "s" not in nombres or "p" in nombres:
            return None
        if self._max_verb_freq(lower) > self._max_nom_adj_freq(lower) * 2:
            return None
        if lower.endswith(("s", "x", "z")):
            return None

        cand = lower + "s"
        if "p" in self._nom_adj_nombres(cand):
            return word + "s"
        if lower.endswith(("eau", "au", "eu")):
            cand = lower + "x"
            if "p" in self._nom_adj_nombres(cand):
                return word + "x"
        if lower.endswith("al"):
            cand = lower[:-2] + "aux"
            if "p" in self._nom_adj_nombres(cand):
                return word[:-2] + "aux"
        return None

    def try_singularize(self, word: str) -> str | None:
        lower = word.lower()
        nombres = self._nom_adj_nombres(lower)
        if "p" not in nombres or "s" in nombres:
            return None
        if self._max_verb_freq(lower) > self._max_nom_adj_freq(lower) * 2:
            return None

        if lower.endswith("s") and len(lower) > 2:
            cand = lower[:-1]
            if "s" in self._nom_adj_nombres(cand):
                return word[:-1]
        if lower.endswith("aux") and len(lower) > 3:
            cand = lower[:-3] + "al"
            if "s" in self._nom_adj_nombres(cand):
                return word[:-3] + "al"
        if lower.endswith("x") and len(lower) > 2:
            cand = lower[:-1]
            if "s" in self._nom_adj_nombres(cand):
                return word[:-1]
        return None

    # ── Infinitif / Participe passe ──

    def is_infinitive(self, word: str) -> bool:
        for cgram, _nombre, _genre, multext, _freq, _phone in self.entries(word):
            if cgram == "VER" and multext.startswith("Vmn"):
                return True
        return False

    def is_infinitive_er(self, word: str) -> bool:
        lower = word.lower()
        if not lower.endswith("er") or len(lower) < 3:
            return False
        for cgram, _nombre, _genre, multext, _freq, _phone in self.entries(word):
            if cgram == "VER" and multext.startswith("Vmn"):
                return True
        return False

    def infinitive_to_pp(self, word: str) -> str | None:
        lower = word.lower()
        if not lower.endswith("er") or len(lower) < 3:
            return None
        if not self.is_infinitive_er(word):
            return None
        if self._max_nom_adj_freq(lower) > self._max_verb_freq(lower) * 5:
            return None

        cand = lower[:-2] + "é"
        for cgram, _nombre, _genre, multext, _freq, _phone in self._entries.get(cand, []):
            if cgram == "VER" and multext.startswith("Vmps"):
                return word[:-2] + "é"
        return None

    def pp_to_infinitive(self, word: str) -> str | None:
        lower = word.lower()
        if not lower.endswith("é") or len(lower) < 3:
            return None

        is_pp = False
        for cgram, _nombre, _genre, multext, _freq, _phone in self.entries(word):
            if cgram == "VER" and multext.startswith("Vmps"):
                is_pp = True
                break
        if not is_pp:
            return None
        if self._max_nom_adj_freq(lower) > self._max_verb_freq(lower) * 5:
            return None

        cand = lower[:-1] + "er"
        for cgram, _nombre, _genre, multext, _freq, _phone in self._entries.get(cand, []):
            if cgram == "VER" and multext.startswith("Vmn"):
                return word[:-1] + "er"
        return None

    # ── 2e personne singulier ──

    def verb_add_s_2s(self, word: str) -> str | None:
        lower = word.lower()
        if lower.endswith("s"):
            return None

        is_verb_1s3s = False
        for cgram, _nombre, _genre, multext, _freq, _phone in self.entries(word):
            if cgram in ("VER", "AUX") and ("1s" in multext or "3s" in multext):
                if multext.startswith("Vmip") or multext.startswith("Vaip"):
                    is_verb_1s3s = True
        if not is_verb_1s3s:
            return None
        if self._max_nom_adj_freq(lower) > self._max_verb_freq(lower) * 2:
            return None

        cand = lower + "s"
        for cgram, _nombre, _genre, multext, _freq, _phone in self._entries.get(cand, []):
            if cgram in ("VER", "AUX") and "2s" in multext:
                return word + "s"
        return None

    # ── Correction accents ──

    def try_fix_accent(self, word: str) -> str | None:
        lower = word.lower()
        if lower in self._entries:
            return None

        stripped = _strip_accents(lower)
        candidates = self._accent_map.get(stripped)
        if not candidates:
            return None

        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
        best_form, best_freq = sorted_cands[0]

        if best_form == lower:
            return None

        if len(sorted_cands) == 1:
            return _preserve_case(word, best_form)

        second_freq = sorted_cands[1][1]
        if best_freq > max(second_freq, 0.01) * 10:
            return _preserve_case(word, best_form)
        return None

    # ── Correction consonne finale muette ──

    _FINAL_CONSONANTS = ("s", "x", "t")
    _MIN_FREQ_EXTENDED = 50.0
    _MAX_FREQ_TRUNCATED = 3.0
    _MIN_RATIO = 30.0

    def try_fix_final_consonant(self, word: str) -> str | None:
        """Corrige un mot auquel il manque une consonne finale muette.

        Ex: "temp" → "temps", "corp" → "corps", "alor" → "alors"
        """
        lower = word.lower()
        if len(lower) < 3:
            return None

        # Ne pas corriger si le mot est une forme verbale ou adjectivale
        # connue du lexique — c'est un mot reel, pas une troncature
        ents = self._entries.get(lower, [])
        for e in ents:
            if e[0] in ("VER", "AUX", "ADJ"):
                return None

        freq_current = max((e[4] for e in ents), default=0.0)
        if freq_current > self._MAX_FREQ_TRUNCATED:
            return None

        best_form = None
        best_freq = 0.0

        for suffix in self._FINAL_CONSONANTS:
            candidate = lower + suffix
            cand_ents = self._entries.get(candidate, [])
            if not cand_ents:
                continue
            freq_cand = max(e[4] for e in cand_ents)
            if freq_cand > best_freq:
                best_freq = freq_cand
                best_form = candidate

        if best_form is None:
            return None
        if best_freq < self._MIN_FREQ_EXTENDED:
            return None

        ratio = best_freq / max(freq_current, 0.01)
        if ratio < self._MIN_RATIO:
            return None

        return _preserve_case(word, best_form)


# ── Fonction principale ──────────────────────────────────────

def corriger_grammatical(
    ortho_words: list[str],
    lex: GrammarLookup,
    skip_positions: set[int] | None = None,
) -> tuple[list[str], list[dict]]:
    """Post-traitement grammatical par regles structurelles + lexique.

    Args:
        ortho_words: Mots orthographiques (sortie du pipeline, avant rejoin).
        lex: Instance GrammarLookup pour consulter le lexique morphologique.
        skip_positions: Positions a ignorer (ex: formules reconnues).

    Returns:
        (ortho_corrige, corrections_log) ou corrections_log est une liste de
        dicts {regle, position, avant, apres, contexte}.
    """
    if not ortho_words:
        return ortho_words, []

    result = list(ortho_words)
    corrections: list[dict] = []
    skip = skip_positions or set()
    n = len(result)
    modified: set[int] = set()

    def _correct(i: int, new_word: str, regle: str, contexte: str) -> None:
        result[i] = new_word
        modified.add(i)
        corrections.append({
            "regle": regle,
            "position": i,
            "avant": ortho_words[i],
            "apres": new_word,
            "contexte": contexte,
        })

    for i in range(n):
        if i in skip:
            continue

        lower_i = result[i].lower()
        prev_lower = result[i - 1].lower() if i > 0 else ""
        next_word = result[i + 1] if i + 1 < n else ""
        next_lower = next_word.lower() if next_word else ""

        # ── Groupe A : on + verbe_3p → verbe_3s ──

        if prev_lower == "on" and i > 0 and i - 1 not in skip:
            form_3s = lex.verb_3p_to_3s(result[i])
            if form_3s is not None:
                _correct(i, form_3s, "on_verb_3s",
                         f"on {result[i]} → {form_3s}")
                continue

        # ── Groupe A2 : Accord pronom-verbe il/ils, elle/elles ──
        #   Cas 1 : "il" + verbe_3p_seul → meme phone ? singulier : corriger pronom
        #   Cas 2 : "ils" + verbe_3s_seul → meme phone ? singulier : corriger pronom

        if lower_i in ("il", "elle") and next_word and i + 1 not in skip:
            # "il/elle" + verbe exclusivement 3p
            if lex.is_verb_3p(next_word) and not lex.is_verb_3s(next_word):
                form_3s = lex.verb_3p_to_3s(next_word)
                if form_3s is not None:
                    if lex.verb_3s_3p_same_phone(next_word, form_3s):
                        # Meme phone → singulier (corriger le verbe)
                        _correct(i + 1, form_3s, "il_verb_accord",
                                 f"{result[i]} {next_word} → {form_3s}")
                        continue
                    else:
                        # Phone different → confiance au verbe (corriger pronom)
                        new_pron = "ils" if lower_i == "il" else "elles"
                        _correct(i, _preserve_case(result[i], new_pron),
                                 "il_verb_accord",
                                 f"{result[i]} {next_word} → {new_pron}")
                        continue

        if lower_i in ("ils", "elles") and next_word and i + 1 not in skip:
            # "ils/elles" + verbe exclusivement 3s → toujours singulier
            # (phone identique → ambigu → singulier par defaut ;
            #  phone different → confiance au verbe 3s → singulier)
            if lex.is_verb_3s(next_word) and not lex.is_verb_3p(next_word):
                if not lex.is_noun(next_word) or lex._max_verb_freq(next_lower) > lex._max_nom_adj_freq(next_lower) * 2:
                    new_pron = "il" if lower_i == "ils" else "elle"
                    _correct(i, _preserve_case(result[i], new_pron),
                             "il_verb_accord",
                             f"{result[i]} {next_word} → {new_pron}")
                    continue

        # ── Groupe B : Confusion est/et ──

        if lower_i == "et" and i > 0 and i - 1 not in skip:
            if prev_lower == "qui":
                _correct(i, "est", "qui_et", "qui et")
                continue
            if prev_lower in ("ne", "n'"):
                _correct(i, "est", "ne_et", f"{result[i-1]} et")
                continue
            if prev_lower in _CLITICS_BEFORE_EST:
                _correct(i, "est", "clitic_et", f"{result[i-1]} et")
                continue
            if (i >= 2
                    and result[i - 2].lower() in ("ne", "n'")
                    and prev_lower in _CLITIC_OBJECTS
                    and i - 2 not in skip):
                _correct(i, "est", "ne_clitic_et",
                         f"{result[i-2]} {result[i-1]} et")
                continue

            # B5 : PRONOM_SUJET_3S + et + ADJ/PP → est
            if (prev_lower in ("il", "elle", "on", "ce")
                    and next_word and i + 1 not in skip
                    and lex.is_adj_or_pp(next_word)):
                _correct(i, "est", "pronom_et",
                         f"{result[i-1]} et {next_word}")
                continue

            # B6 : et + non + ADJ → est non ADJ
            if (next_lower == "non"
                    and i + 2 < n
                    and lex.is_adj_or_pp(result[i + 2])):
                _correct(i, "est", "et_non_adj",
                         f"et non {result[i+2]}")
                continue

        # ── Groupe C : leurs + verbe → leur ──

        if lower_i == "leurs" and next_word and i + 1 not in skip:
            if (lex.is_verb(next_word)
                    and not lex.is_noun(next_word)
                    and not lex.is_adj_or_pp(next_word)):
                _correct(i, "leur", "leurs_verbe", f"leurs {next_word}")
                continue

        # ── Groupe D : Confusion a/à ──

        if lower_i == "à" and i > 0 and i - 1 not in skip:
            if prev_lower in _SUBJ_3S:
                next_is_inf = (i + 1 < n and lex.is_infinitive(result[i + 1]))
                if not next_is_inf:
                    _correct(i, "a", "subj3s_a", f"{result[i-1]} à")
                    continue
            if prev_lower == "n'":
                _correct(i, "a", "n_a", "n' à")
                continue
            if prev_lower == "y":
                _correct(i, "a", "y_a", "y à")
                continue
            if (i >= 2
                    and result[i - 2].lower() in _SUBJ_3S
                    and prev_lower in _CLITIC_OBJECTS
                    and i - 2 not in skip):
                next_is_inf = (i + 1 < n and lex.is_infinitive(result[i + 1]))
                if not next_is_inf:
                    _correct(i, "a", "subj3s_clitic_a",
                             f"{result[i-2]} {result[i-1]} à")
                    continue
            if prev_lower in ("l'", "m'", "t'"):
                _correct(i, "a", "clitic_a", f"{result[i-1]} à")
                continue

        # ── Groupe E : d' + ou → d'où ──

        if lower_i == "ou" and i > 0 and prev_lower == "d'" and i - 1 not in skip:
            _correct(i, "où", "d_ou", "d' ou")
            continue

        # ── Groupe F : Accord nombre via determinant ──

        if i in modified:
            continue

        if i > 0 and i - 1 not in skip:
            if prev_lower in _PLUR_DETS:
                plural = lex.try_pluralize(result[i])
                if plural is not None:
                    _correct(i, plural, "det_plur",
                             f"{result[i-1]} {result[i]} → {plural}")
                    continue
            if prev_lower in _SING_DETS:
                singular = lex.try_singularize(result[i])
                if singular is not None:
                    _correct(i, singular, "det_sing",
                             f"{result[i-1]} {result[i]} → {singular}")
                    continue

        # ── Groupe G : avoir + infinitif -er → PP -é ──

        if i > 0 and lower_i.endswith("er") and i - 1 not in skip:
            is_after_avoir = False
            if prev_lower in _AVOIR_FORMS_SAFE:
                is_after_avoir = True
            elif prev_lower == "a" and i >= 2:
                if result[i - 2].lower() in _AVOIR_A_SUBJECTS:
                    is_after_avoir = True
            elif prev_lower == "ont" and i >= 2:
                if result[i - 2].lower() in _AVOIR_ONT_SUBJECTS:
                    is_after_avoir = True

            if is_after_avoir:
                pp = lex.infinitive_to_pp(result[i])
                if pp is not None:
                    _correct(i, pp, "avoir_inf_pp",
                             f"{result[i-1]} {result[i]} → {pp}")
                    continue

        # G2 : avoir + été + infinitif -er → PP -é
        if (i >= 2 and lower_i.endswith("er")
                and prev_lower == "été" and i - 1 not in skip):
            prev2_lower = result[i - 2].lower()
            is_avoir_ete = prev2_lower in _AVOIR_FORMS_SAFE
            if not is_avoir_ete and prev2_lower == "a" and i >= 3:
                is_avoir_ete = result[i - 3].lower() in _AVOIR_A_SUBJECTS
            if not is_avoir_ete and prev2_lower == "ont" and i >= 3:
                is_avoir_ete = result[i - 3].lower() in _AVOIR_ONT_SUBJECTS
            if is_avoir_ete:
                pp = lex.infinitive_to_pp(result[i])
                if pp is not None:
                    _correct(i, pp, "avoir_ete_inf_pp",
                             f"{result[i-2]} été {result[i]} → {pp}")
                    continue

        # ── Groupe H : possessif + fil → fils ──

        if lower_i == "fil" and i > 0 and prev_lower in _POSSESSIFS:
            _correct(i, "fils", "poss_fil_fils",
                     f"{result[i-1]} fil → fils")
            continue

        # ── Groupe I : tu + verbe → ajouter -s ──

        if i > 0 and prev_lower == "tu" and i - 1 not in skip:
            form_2s = lex.verb_add_s_2s(result[i])
            if form_2s is not None:
                _correct(i, form_2s, "tu_verb_2s",
                         f"tu {result[i]} → {form_2s}")
                continue

        # I2 : tu + [clitique] + verbe → ajouter -s
        if (i >= 2
                and result[i - 2].lower() == "tu"
                and prev_lower in _CLITIC_OBJECTS
                and i - 2 not in skip):
            form_2s = lex.verb_add_s_2s(result[i])
            if form_2s is not None:
                _correct(i, form_2s, "tu_clitic_verb_2s",
                         f"tu {result[i-1]} {result[i]} → {form_2s}")
                continue

        # A2 : on + [clitique] + verbe_3p → verbe_3s
        if (i >= 2
                and result[i - 2].lower() == "on"
                and prev_lower in _CLITIC_OBJECTS
                and i - 2 not in skip):
            form_3s = lex.verb_3p_to_3s(result[i])
            if form_3s is not None:
                _correct(i, form_3s, "on_clitic_verb_3s",
                         f"on {result[i-1]} {result[i]} → {form_3s}")
                continue

        # ── Groupe J : être + infinitif -er → PP -é ──

        if i > 0 and lower_i.endswith("er") and i - 1 not in skip:
            is_after_etre = False
            if prev_lower in _ETRE_FORMS_SAFE:
                is_after_etre = True
            elif prev_lower == "est" and i >= 2:
                if result[i - 2].lower() in _ETRE_EST_SUBJECTS:
                    is_after_etre = True
            elif prev_lower == "sont" and i >= 2:
                if result[i - 2].lower() in _ETRE_SONT_SUBJECTS:
                    is_after_etre = True

            if is_after_etre:
                pp = lex.infinitive_to_pp(result[i])
                if pp is not None:
                    _correct(i, pp, "etre_inf_pp",
                             f"{result[i-1]} {result[i]} → {pp}")
                    continue

        # ── Groupe L : preposition + PP -é → infinitif -er ──

        if (i > 0 and lower_i.endswith("é") and len(lower_i) >= 3
                and i not in modified and i - 1 not in skip):
            if prev_lower in ("à", "pour", "sans"):
                inf = lex.pp_to_infinitive(result[i])
                if inf is not None:
                    _correct(i, inf, "prep_pp_inf",
                             f"{result[i-1]} {result[i]} → {inf}")
                    continue

        # ── Groupe M : Consonne finale muette manquante ──

        if i not in modified:
            fixed = lex.try_fix_final_consonant(result[i])
            if fixed is not None:
                _correct(i, fixed, "final_consonant",
                         f"{result[i]} → {fixed}")
                continue

        # ── Groupe K : Correction accents mots inconnus ──

        if i not in modified:
            fixed = lex.try_fix_accent(result[i])
            if fixed is not None:
                _correct(i, fixed, "accent_fix",
                         f"{result[i]} → {fixed}")
                continue

    return result, corrections
