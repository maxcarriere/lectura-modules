#!/usr/bin/env python3
"""Evaluation comparative sur textes continus (redactions d'eleves).

Compare lectura-correcteur vs Grammalecte phrase par phrase sur des
textes longs et realistes, avec annotations mot-a-mot.

Usage :
    python scripts/evaluer_texte.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LEXIQUE_PATH = Path(
    "/data/work/projets/lectura/workspace/lectura-main/correcteur/donnees/lexique.db"
)


# ---------------------------------------------------------------------------
# Textes d'evaluation
# ---------------------------------------------------------------------------
@dataclass
class TexteAnnote:
    titre: str
    description: str
    phrases_erronees: list[str]
    phrases_attendues: list[str]


TEXTES: list[TexteAnnote] = [
    TexteAnnote(
        titre="Redaction CE2 — Ma journee",
        description="Texte narratif typique CE2, erreurs d'accord et conjugaison",
        phrases_erronees=[
            "ce matin je me suis levé et je suis allé a l'ecole",
            "les enfant etait content de me voir",
            "on a jouer dans la cour avec les ballon rouge",
            "la maitresse a dit que les leçon etait difficile",
            "nous avons manger a la cantine",
            "les fille jouait dans le jardin",
            "mon ami et moi on a couru très vite",
            "je suis rentré a la maison et j'ai fait mes devoir",
            "ma mere a preparé le diner",
            "je me suis couché tot parce que j'etais fatigué",
        ],
        phrases_attendues=[
            "Ce matin je me suis levé et je suis allé à l'école.",
            "Les enfants étaient contents de me voir.",
            "On a joué dans la cour avec les ballons rouges.",
            "La maîtresse a dit que les leçons étaient difficiles.",
            "Nous avons mangé à la cantine.",
            "Les filles jouaient dans le jardin.",
            "Mon ami et moi on a couru très vite.",
            "Je suis rentré à la maison et j'ai fait mes devoirs.",
            "Ma mère a préparé le dîner.",
            "Je me suis couché tôt parce que j'étais fatigué.",
        ],
    ),
    TexteAnnote(
        titre="Redaction CM1 — Mon animal prefere",
        description="Texte descriptif CM1, erreurs d'accord genre/nombre et homophones",
        phrases_erronees=[
            "j'ai un chat qui s'appelle Felix",
            "il et très grand et il a les yeux vert",
            "sa queue est long et son pelage et doux",
            "il mange beaucoup de croquette",
            "quand il dort il et très calme",
            "mais quand il se reveille il court partout",
            "les voisin dise qu'il est trop bruyant",
            "il aime pas qu'on le touche",
            "ma soeur et moi on adore jouer avec lui",
            "c'est le plus belle chat du monde",
        ],
        phrases_attendues=[
            "J'ai un chat qui s'appelle Félix.",
            "Il est très grand et il a les yeux verts.",
            "Sa queue est longue et son pelage est doux.",
            "Il mange beaucoup de croquettes.",
            "Quand il dort il est très calme.",
            "Mais quand il se réveille il court partout.",
            "Les voisins disent qu'il est trop bruyant.",
            "Il n'aime pas qu'on le touche.",
            "Ma sœur et moi on adore jouer avec lui.",
            "C'est le plus beau chat du monde.",
        ],
    ),
    TexteAnnote(
        titre="Redaction CM2 — Les vacances",
        description="Texte au passe, erreurs imparfait/passe compose et negation",
        phrases_erronees=[
            "l'ete dernier nous sommes allé en Bretagne",
            "mes parent avait loué une maison au bord de la mer",
            "les enfant jouait sur la plage tout les jours",
            "ma soeur a construire un chateau de sable",
            "les vague etait grande et on pouvait pas nager",
            "nous avons visiter des village tres jolis",
            "les gens etait gentil avec nous",
            "il faisait beau et on mangeait des crepe",
            "je voulait pas rentrer a la maison",
            "ces vacance etait les plus belle de ma vie",
        ],
        phrases_attendues=[
            "L'été dernier nous sommes allés en Bretagne.",
            "Mes parents avaient loué une maison au bord de la mer.",
            "Les enfants jouaient sur la plage tous les jours.",
            "Ma sœur a construit un château de sable.",
            "Les vagues étaient grandes et on ne pouvait pas nager.",
            "Nous avons visité des villages très jolis.",
            "Les gens étaient gentils avec nous.",
            "Il faisait beau et on mangeait des crêpes.",
            "Je ne voulais pas rentrer à la maison.",
            "Ces vacances étaient les plus belles de ma vie.",
        ],
    ),
    TexteAnnote(
        titre="Redaction 6e — La recre",
        description="Texte dialogique 6e, homophones et accords complexes",
        phrases_erronees=[
            "a la recre mes ami son venu me voir",
            "ils on dit qu'il voulait jouer au foot",
            "j'ai dit que sa me plaisait bien",
            "on a fait deux equipe avec les garcon et les fille",
            "les fille etait plus rapide que les garcon",
            "elles on marqué trois but",
            "leur victoire etait meritée",
            "les garcon etait pas content du resultat",
            "mais ils ont dit que c'etait du bon jeu",
            "on et tous rentré en classe en rigolant",
        ],
        phrases_attendues=[
            "À la récré mes amis sont venus me voir.",
            "Ils ont dit qu'ils voulaient jouer au foot.",
            "J'ai dit que ça me plaisait bien.",
            "On a fait deux équipes avec les garçons et les filles.",
            "Les filles étaient plus rapides que les garçons.",
            "Elles ont marqué trois buts.",
            "Leur victoire était méritée.",
            "Les garçons n'étaient pas contents du résultat.",
            "Mais ils ont dit que c'était du bon jeu.",
            "On est tous rentrés en classe en rigolant.",
        ],
    ),
    TexteAnnote(
        titre="Redaction 6e — Le futur",
        description="Texte au futur, erreurs de conjugaison futur et accords",
        phrases_erronees=[
            "l'annee prochaine nous irons au college",
            "les cours sera plus difficile",
            "je travaillera beaucoup pour reussir",
            "mes ami et moi on se retrouvera a la cantine",
            "les professeur nous donnera des devoir tout les soirs",
            "ma mere dit que je devra etre plus serieux",
            "elle pense que les etude sont importante",
            "je suis sur que tout ira bien",
            "les grande vacance sera geniale",
            "on pourra enfin se reposer",
        ],
        phrases_attendues=[
            "L'année prochaine nous irons au collège.",
            "Les cours seront plus difficiles.",
            "Je travaillerai beaucoup pour réussir.",
            "Mes amis et moi on se retrouvera à la cantine.",
            "Les professeurs nous donneront des devoirs tous les soirs.",
            "Ma mère dit que je devrai être plus sérieux.",
            "Elle pense que les études sont importantes.",
            "Je suis sûr que tout ira bien.",
            "Les grandes vacances seront géniales.",
            "On pourra enfin se reposer.",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Grammalecte
# ---------------------------------------------------------------------------
def _corriger_grammalecte(phrase: str) -> str:
    try:
        from pygrammalecte import grammalecte_text
        from pygrammalecte.pygrammalecte import GrammalecteGrammarMessage
    except ImportError:
        return phrase
    messages = list(grammalecte_text(phrase))
    grammar_msgs = [
        m for m in messages
        if isinstance(m, GrammalecteGrammarMessage) and m.suggestions
    ]
    grammar_msgs.sort(key=lambda m: m.start, reverse=True)
    result = phrase
    for msg in grammar_msgs:
        result = result[:msg.start] + msg.suggestions[0] + result[msg.end:]
    return result


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _normaliser(phrase: str) -> str:
    phrase = phrase.strip()
    while phrase and phrase[-1] in ".!?…":
        phrase = phrase[:-1]
    return " ".join(phrase.lower().split())


def _mots(phrase: str) -> list[str]:
    return _normaliser(phrase).split()


# ---------------------------------------------------------------------------
# Metriques mot-a-mot
# ---------------------------------------------------------------------------
@dataclass
class MetriquesMotAMot:
    """Compteurs pour l'evaluation mot-a-mot."""
    mots_errones: int = 0         # mots differents entre erronee et attendue
    mots_corriges_ok: int = 0     # mots errones corriges correctement
    mots_non_corriges: int = 0    # mots errones non corriges
    mots_sur_corriges: int = 0    # mots corrects modifies a tort
    mots_mal_corriges: int = 0    # mots errones corriges mais mal
    mots_totaux: int = 0          # total mots attendus
    mots_corrects_ok: int = 0     # mots corrects non modifies

    @property
    def precision_correction(self) -> float:
        """Parmi les mots modifies, combien sont corrects."""
        modifies = self.mots_corriges_ok + self.mots_sur_corriges + self.mots_mal_corriges
        if modifies == 0:
            return 1.0
        return self.mots_corriges_ok / modifies

    @property
    def rappel_correction(self) -> float:
        """Parmi les mots errones, combien sont corriges."""
        if self.mots_errones == 0:
            return 1.0
        return self.mots_corriges_ok / self.mots_errones

    @property
    def f1(self) -> float:
        p, r = self.precision_correction, self.rappel_correction
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        """Pourcentage de mots dans le resultat qui matchent l'attendu."""
        if self.mots_totaux == 0:
            return 1.0
        return (self.mots_corriges_ok + self.mots_corrects_ok) / self.mots_totaux


def _comparer_mot_a_mot(
    erronee: str, corrigee: str, attendue: str, metriques: MetriquesMotAMot,
) -> list[str]:
    """Compare mot a mot via SequenceMatcher pour gerer insertions/suppressions.

    Aligne corrigee vs attendue pour compter les metriques, et aligne
    erronee vs attendue pour savoir quels mots etaient errones.
    """
    from difflib import SequenceMatcher

    m_err = _mots(erronee)
    m_cor = _mots(corrigee)
    m_att = _mots(attendue)

    annotations: list[str] = []
    metriques.mots_totaux += len(m_att)

    # Construire un set de mots errones (positions dans attendue qui different)
    err_vs_att = SequenceMatcher(None, m_err, m_att)
    mots_errones_att: set[int] = set()  # indices dans m_att qui sont errones
    for tag, _i1, _i2, j1, j2 in err_vs_att.get_opcodes():
        if tag in ("replace", "insert"):
            for j in range(j1, j2):
                mots_errones_att.add(j)
        # "delete" = mots en trop dans l'erreur, pas dans l'attendu

    # Aligner corrigee vs attendue
    cor_vs_att = SequenceMatcher(None, m_cor, m_att)
    att_matched: set[int] = set()

    for tag, i1, i2, j1, j2 in cor_vs_att.get_opcodes():
        if tag == "equal":
            for k, j in enumerate(range(j1, j2)):
                att_matched.add(j)
                if j in mots_errones_att:
                    # Mot errone et corrige correctement
                    metriques.mots_errones += 1
                    metriques.mots_corriges_ok += 1
                else:
                    # Mot correct et non modifie
                    metriques.mots_corrects_ok += 1
        elif tag == "replace":
            for k_i, k_j in zip(range(i1, i2), range(j1, j2)):
                att_matched.add(k_j)
                w_cor = m_cor[k_i]
                w_att = m_att[k_j]
                if k_j in mots_errones_att:
                    metriques.mots_errones += 1
                    metriques.mots_mal_corriges += 1
                    annotations.append(
                        f"  MALCORR: → '{w_cor}' (attendu: '{w_att}')"
                    )
                else:
                    metriques.mots_sur_corriges += 1
                    annotations.append(
                        f"  SURCORR: '{w_att}' → '{w_cor}' (attendu: '{w_att}')"
                    )
            # Remaining unmatched on either side
            if i2 - i1 > j2 - j1:
                # Extra words in corrigee (insertions parasites)
                for k_i in range(i1 + (j2 - j1), i2):
                    metriques.mots_sur_corriges += 1
                    annotations.append(
                        f"  SURCORR: insertion parasite '{m_cor[k_i]}'"
                    )
            elif j2 - j1 > i2 - i1:
                # Missing words in corrigee
                for k_j in range(j1 + (i2 - i1), j2):
                    att_matched.add(k_j)
                    if k_j in mots_errones_att:
                        metriques.mots_errones += 1
                        metriques.mots_non_corriges += 1
                        annotations.append(
                            f"  MANQUE : '{m_att[k_j]}' absent"
                        )
                    else:
                        metriques.mots_sur_corriges += 1
                        annotations.append(
                            f"  SURCORR: '{m_att[k_j]}' supprime"
                        )
        elif tag == "insert":
            # Mots dans attendue absents de corrigee
            for j in range(j1, j2):
                att_matched.add(j)
                if j in mots_errones_att:
                    metriques.mots_errones += 1
                    metriques.mots_non_corriges += 1
                    annotations.append(
                        f"  MANQUE : '{m_att[j]}' absent de la sortie"
                    )
                else:
                    metriques.mots_sur_corriges += 1
                    annotations.append(
                        f"  SURCORR: '{m_att[j]}' supprime"
                    )
        elif tag == "delete":
            # Mots dans corrigee absents de attendue (insertions parasites)
            for k_i in range(i1, i2):
                metriques.mots_sur_corriges += 1
                annotations.append(
                    f"  SURCORR: insertion parasite '{m_cor[k_i]}'"
                )

    # Mots attendus non atteints (non corriges)
    for j in range(len(m_att)):
        if j not in att_matched:
            if j in mots_errones_att:
                metriques.mots_errones += 1
                metriques.mots_non_corriges += 1
                annotations.append(
                    f"  MANQUE : '{m_att[j]}' non corrige"
                )

    return annotations


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class ResultatTexte:
    texte: TexteAnnote
    n_phrases: int = 0
    lec_exact: int = 0
    gram_exact: int = 0
    lec_metriques: MetriquesMotAMot = field(default_factory=MetriquesMotAMot)
    gram_metriques: MetriquesMotAMot = field(default_factory=MetriquesMotAMot)


def evaluer(verbose: bool = False) -> None:
    print("Chargement du lexique SQLite...")
    from lectura_lexique import Lexique
    lexique = Lexique(str(LEXIQUE_PATH))
    print(f"  Lexique : {LEXIQUE_PATH.name}")

    print("Chargement de lectura-correcteur...")
    from lectura_correcteur import Correcteur
    t0 = time.perf_counter()
    correcteur = Correcteur(lexique)
    t_load = time.perf_counter() - t0
    print(f"  Correcteur chargé en {t_load:.1f}s")

    grammalecte_ok = True
    try:
        from pygrammalecte import grammalecte_text  # noqa: F401
        list(grammalecte_text("test"))
        print("  Grammalecte : disponible\n")
    except Exception as exc:
        grammalecte_ok = False
        print(f"  Grammalecte : indisponible ({exc})\n")

    # --- Par texte ---
    resultats_textes: list[ResultatTexte] = []
    total_phrases = 0
    total_lec_exact = 0
    total_gram_exact = 0
    lec_glob = MetriquesMotAMot()
    gram_glob = MetriquesMotAMot()

    for texte in TEXTES:
        rt = ResultatTexte(texte=texte, n_phrases=len(texte.phrases_erronees))
        sep = "─" * 72
        print(f"\n{sep}")
        print(f"  {texte.titre}")
        print(f"  {texte.description}")
        print(sep)

        for i, (err, att) in enumerate(
            zip(texte.phrases_erronees, texte.phrases_attendues)
        ):
            # Lectura
            t0 = time.perf_counter()
            res_lec = correcteur.corriger(err)
            t_lec = (time.perf_counter() - t0) * 1000
            cor_lec = res_lec.phrase_corrigee

            # Grammalecte
            cor_gram = _corriger_grammalecte(err) if grammalecte_ok else err

            lec_ok = _normaliser(cor_lec) == _normaliser(att)
            gram_ok = _normaliser(cor_gram) == _normaliser(att)

            if lec_ok:
                rt.lec_exact += 1
            if gram_ok:
                rt.gram_exact += 1

            # Mot-a-mot
            ann_lec = _comparer_mot_a_mot(err, cor_lec, att, rt.lec_metriques)
            ann_gram = _comparer_mot_a_mot(err, cor_gram, att, rt.gram_metriques)

            sl = "✓" if lec_ok else "✗"
            sg = "✓" if gram_ok else "✗"

            if verbose or not lec_ok or not gram_ok:
                print(f"\n  [{i+1:2d}] Erroné  : {err}")
                print(f"       Attendu : {att}")
                print(f"       Lectura : {cor_lec}  {sl}  ({t_lec:.0f}ms)")
                if grammalecte_ok:
                    print(f"       Grammal : {cor_gram}  {sg}")
                if ann_lec and not lec_ok:
                    for a in ann_lec:
                        print(f"       L {a}")
                if ann_gram and not gram_ok and grammalecte_ok:
                    for a in ann_gram:
                        print(f"       G {a}")
            elif verbose:
                print(f"  [{i+1:2d}] {sl}{sg} {err[:50]}...")

        # Totaux
        total_phrases += rt.n_phrases
        total_lec_exact += rt.lec_exact
        total_gram_exact += rt.gram_exact

        # Accumuler metriques globales
        for attr in (
            "mots_errones", "mots_corriges_ok", "mots_non_corriges",
            "mots_sur_corriges", "mots_mal_corriges", "mots_totaux",
            "mots_corrects_ok",
        ):
            setattr(lec_glob, attr, getattr(lec_glob, attr) + getattr(rt.lec_metriques, attr))
            setattr(gram_glob, attr, getattr(gram_glob, attr) + getattr(rt.gram_metriques, attr))

        resultats_textes.append(rt)

        # Resume du texte
        print(f"\n  Score phrases exactes : Lectura {rt.lec_exact}/{rt.n_phrases}"
              f"  |  Grammalecte {rt.gram_exact}/{rt.n_phrases}")
        print(f"  Score mots : Lectura F1={rt.lec_metriques.f1:.3f}"
              f" (P={rt.lec_metriques.precision_correction:.3f}"
              f" R={rt.lec_metriques.rappel_correction:.3f})"
              f"  |  Grammalecte F1={rt.gram_metriques.f1:.3f}"
              f" (P={rt.gram_metriques.precision_correction:.3f}"
              f" R={rt.gram_metriques.rappel_correction:.3f})")

    # --- Rapport global ---
    sep = "=" * 72
    print(f"\n{sep}")
    print("  BILAN GLOBAL — Evaluation sur textes continus")
    print(sep)

    print(f"\n{'Texte':<35} | {'Nb':>3} | {'Lectura':>12} | {'Grammalecte':>12}")
    print("─" * 72)
    for rt in resultats_textes:
        t = rt.texte.titre[:33]
        print(f"  {t:<33} | {rt.n_phrases:>3} |"
              f" {rt.lec_exact:>3}/{rt.n_phrases} ({100*rt.lec_exact/rt.n_phrases:4.0f}%) |"
              f" {rt.gram_exact:>3}/{rt.n_phrases} ({100*rt.gram_exact/rt.n_phrases:4.0f}%)")
    print("─" * 72)
    print(f"  {'TOTAL':<33} | {total_phrases:>3} |"
          f" {total_lec_exact:>3}/{total_phrases} ({100*total_lec_exact/total_phrases:4.0f}%) |"
          f" {total_gram_exact:>3}/{total_phrases} ({100*total_gram_exact/total_phrases:4.0f}%)")

    print(f"\n--- Metriques mot-a-mot (sur {lec_glob.mots_totaux} mots attendus,"
          f" {lec_glob.mots_errones} mots errones) ---\n")
    print(f"{'':20} | {'Lectura':>12} | {'Grammalecte':>12}")
    print("─" * 50)
    print(f"  {'Precision':18} | {lec_glob.precision_correction:>11.3f} | {gram_glob.precision_correction:>11.3f}")
    print(f"  {'Rappel':18} | {lec_glob.rappel_correction:>11.3f} | {gram_glob.rappel_correction:>11.3f}")
    print(f"  {'F1':18} | {lec_glob.f1:>11.3f} | {gram_glob.f1:>11.3f}")
    print(f"  {'Accuracy (mots)':18} | {lec_glob.accuracy:>11.3f} | {gram_glob.accuracy:>11.3f}")

    print(f"\n--- Detail des compteurs ---\n")
    print(f"{'':25} | {'Lectura':>8} | {'Grammalecte':>8}")
    print("─" * 50)
    print(f"  {'Mots errones':23} | {lec_glob.mots_errones:>8} | {gram_glob.mots_errones:>8}")
    print(f"  {'Corriges OK':23} | {lec_glob.mots_corriges_ok:>8} | {gram_glob.mots_corriges_ok:>8}")
    print(f"  {'Non corriges':23} | {lec_glob.mots_non_corriges:>8} | {gram_glob.mots_non_corriges:>8}")
    print(f"  {'Mal corriges':23} | {lec_glob.mots_mal_corriges:>8} | {gram_glob.mots_mal_corriges:>8}")
    print(f"  {'Surcorriges (FP)':23} | {lec_glob.mots_sur_corriges:>8} | {gram_glob.mots_sur_corriges:>8}")
    print(f"  {'Corrects non touches':23} | {lec_glob.mots_corrects_ok:>8} | {gram_glob.mots_corrects_ok:>8}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation comparative sur textes continus",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    evaluer(verbose=args.verbose)


if __name__ == "__main__":
    main()
