#!/usr/bin/env python3
"""Test de faux positifs sur textes propres — multi-sources.

Charge le correcteur V6 et le teste sur des phrases correctes
provenant de differentes sources pour mesurer le taux de FP.

Sources testees :
  1. Corpus benchmark OK (14 phrases dev + 8 validation)
  2. Phrases propres supplementaires (fiction, technique, dialogue)

Usage :
    python scripts/benchmark/test_fp_propre.py
    python scripts/benchmark/test_fp_propre.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))

LEXIQUE_PATH = Path(
    "/data/work/projets/lectura/workspace/Lexique/bases/lexique_lectura.db"
)


# ---------------------------------------------------------------------------
# Phrases propres par source
# ---------------------------------------------------------------------------

@dataclass
class SourcePropre:
    """Un ensemble de phrases propres pour le test FP."""
    nom: str
    description: str
    phrases: list[str]


# Phrases du benchmark (categorie OK)
def _charger_benchmark_ok() -> list[str]:
    """Charge les phrases OK du corpus benchmark dev + validation.

    Utilise la forme attendue (proprement accentuee) pour le test FP,
    car le test verifie que le correcteur ne modifie pas le texte propre.
    """
    phrases: list[str] = []
    try:
        from corpus_benchmark import CORPUS
        for cas in CORPUS:
            if cas.categorie == "OK":
                phrases.append(cas.attendue[0])
    except ImportError:
        pass
    try:
        from corpus_validation import CORPUS_VALIDATION
        for cas in CORPUS_VALIDATION:
            if cas.categorie == "OK":
                phrases.append(cas.attendue[0])
    except ImportError:
        pass
    return phrases


SOURCES: list[SourcePropre] = [
    SourcePropre(
        nom="Benchmark OK",
        description="Phrases correctes du corpus benchmark (dev + validation)",
        phrases=_charger_benchmark_ok(),
    ),
    SourcePropre(
        nom="Fiction narrative",
        description="Phrases de fiction — narration, dialogue, description",
        phrases=[
            "Le soleil se couchait lentement derrière les collines.",
            "Elle ferma les yeux et inspira profondément.",
            "Les enfants jouaient dans le jardin depuis des heures.",
            "Il ne savait pas quoi répondre à cette question.",
            "La vieille maison craquait sous le poids des années.",
            "Nous avions marché pendant toute la journée sans nous arrêter.",
            "Les oiseaux chantaient dans les arbres au-dessus de nos têtes.",
            "Il prit une grande inspiration avant de parler.",
            "La nuit était tombée et les étoiles brillaient dans le ciel.",
            "Elle le regarda avec un sourire en coin.",
            "Les rues étaient désertes à cette heure de la nuit.",
            "Il referma la porte derrière lui sans faire de bruit.",
            "Le vent soufflait fort et les arbres se courbaient.",
            "Ils marchèrent en silence pendant de longues minutes.",
            "La pluie commença à tomber doucement sur les toits.",
            "Elle posa sa main sur son épaule pour le rassurer.",
            "Les nuages sombres annonçaient un orage imminent.",
            "Il était fatigué mais il continuait à avancer.",
            "Le chat dormait paisiblement sur le rebord de la fenêtre.",
            "Ils se retrouvèrent devant le café comme d'habitude.",
        ],
    ),
    SourcePropre(
        nom="Texte technique",
        description="Phrases techniques — science, informatique, éducation",
        phrases=[
            "Le système fonctionne correctement depuis la mise à jour.",
            "Les données sont stockées dans une base de données relationnelle.",
            "Le processeur exécute les instructions une par une.",
            "Les résultats de l'expérience confirment notre hypothèse.",
            "Le programme doit être compilé avant d'être exécuté.",
            "Les élèves ont obtenu de bons résultats cette année.",
            "Le professeur explique les règles de grammaire en détail.",
            "Les tests unitaires vérifient le bon fonctionnement du code.",
            "Le serveur répond aux requêtes des utilisateurs.",
            "Les mesures de sécurité ont été renforcées récemment.",
        ],
    ),
    SourcePropre(
        nom="Dialogue courant",
        description="Phrases dialogiques — langage courant, questions, réponses",
        phrases=[
            "Tu viens avec nous ce soir ?",
            "Je ne sais pas encore si je pourrai venir.",
            "Il faut que tu te dépêches sinon on va être en retard.",
            "Elle m'a dit qu'elle viendrait demain.",
            "On peut se retrouver devant le cinéma à huit heures.",
            "Je pense que c'est une bonne idée.",
            "Est-ce que tu as fini tes devoirs ?",
            "Il ne veut pas venir avec nous au parc.",
            "Nous avons décidé de partir en vacances au mois de juillet.",
            "C'est la première fois que je viens ici.",
        ],
    ),
    SourcePropre(
        nom="Phrases complexes",
        description="Phrases longues avec structures complexes, pièges homophones",
        phrases=[
            "Les enfants qui ont terminé leurs devoirs sont allés jouer dehors.",
            "Il a acheté un livre à la librairie qui se trouve près de la gare.",
            "Son frère et sa sœur sont partis en vacances avec leurs parents.",
            "On a décidé de ne pas y aller parce qu'il pleuvait trop fort.",
            "Les élèves ont travaillé dur et ont réussi leur examen.",
            "Elle est allée à la boulangerie et a acheté du pain.",
            "Ils ont mangé à la cantine et sont rentrés en classe.",
            "Le chat de mes voisins dort sur le mur du jardin.",
            "Nous sommes allés au cinéma et nous avons vu un très bon film.",
            "Il fait beau aujourd'hui et les enfants jouent dans le parc.",
        ],
    ),
    SourcePropre(
        nom="Markdown/LaTeX",
        description="Phrases avec formatage markdown et LaTeX",
        phrases=[
            "Le **mot important** est en gras dans la phrase.",
            "Voici une formule : $E = mc^2$ qui est célèbre.",
            "```python\nprint('hello')\n```",
            "## Chapitre 2 : Les bases de la grammaire",
            "- Premier élément de la liste",
            "La fonction $f(x) = x^2 + 1$ est définie sur $\\mathbb{R}$.",
            "Utiliser `git commit` pour sauvegarder les modifications.",
            "Le théorème de \\frac{a}{b} est fondamental.",
        ],
    ),
    SourcePropre(
        nom="Pieges accents et homophones",
        description="Phrases propres ciblant les pieges des axes 1/3/4",
        phrases=[
            # Axe 1 : mots accentues qui doivent rester intacts
            "L'école est fermée pour les vacances.",
            "Il était une fois un petit garçon.",
            "Le téléphone sonne dans la cuisine.",
            "Mon âge n'a pas d'importance.",
            # Axe 3 : a/à corrects qui ne doivent PAS etre corriges
            "Il a la chance de partir en voyage.",
            "Elle a un beau jardin.",
            "On a le droit de se tromper.",
            "Qui a gagné le match hier soir ?",
            "Il n'a pas terminé son travail.",
            # Axe 3 : à correct
            "Nous allons à la piscine demain.",
            "Il pense à ses amis d'enfance.",
            # Axe 4 : accords genre corrects
            "La porte est ouverte depuis ce matin.",
            "Les fleurs blanches parfument le jardin.",
            "Une longue histoire ancienne.",
        ],
    ),
    SourcePropre(
        nom="Pieges Phase 6",
        description="Phrases propres ciblant les pieges des 10 axes Phase 6",
        phrases=[
            # Axe 1 (PP_ACCENT) : PP accentues qui doivent rester
            "Il a mangé toute la tarte.",
            "Elle a terminé son travail hier.",
            # Axe 2 (ADJ_ATTRIBUT) : attributs deja accordes
            "Elle est grande et forte.",
            "Les fleurs sont très belles.",
            # Axe 3 (ACCENT_AMBIGUS) : mots accentues corrects
            "C'est très important pour nous.",
            "Sa mère est partie en vacances.",
            # Axe 4 (ET_EST) : et correct (coordination)
            "Noir et blanc sont mes couleurs.",
            "Pierre et Marie sont partis.",
            # Axe 5 (DOUBLES_LETTRES) : mots avec doubles lettres corrects
            "Elle appelle ses amis chaque soir.",
            "La nourriture est excellente ici.",
            # Axe 6 (CONJ_SV) : conjugaisons correctes
            "Les enfants mangent à la cantine.",
            "Il mange seul dans sa chambre.",
            # Axe 7 (NEGATION) : negations completes
            "Il ne veut pas venir avec nous.",
            "Je veux plus de pain.",
            # Axe 8 (PP_ETRE) : PP accorde avec etre
            "Elle est tombée dans les escaliers.",
            "Ils sont arrivés en retard.",
            # Axe 9 (RESEGMENTATION) : mots qui ne doivent pas etre decoupes
            "Le parcours est difficile.",
            "Il a toujours faim.",
            # Axe 10 (PHONETIQUE) : mots qui ne doivent pas changer
            "La pharmacie est fermée.",
            "La philosophie est passionnante.",
        ],
    ),
    SourcePropre(
        nom="Pieges Phase 6.1-6.3",
        description="Phrases propres ciblant les pieges des corrections 6.1-6.3",
        phrases=[
            # Trait d'union : composes corrects
            "Nous partons cet après-midi.",
            "Mon grand-père habite à la campagne.",
            "Peut-être qu'il viendra demain.",
            # Capitalisation : deja capitalise
            "Le chat dort sur le canapé.",
            # son/sont : formes correctes
            "Son frère est parti en vacances.",
            "Les enfants sont fatigués ce soir.",
            # on/ont : formes correctes
            "On mange à la cantine aujourd'hui.",
            "Ils ont terminé leurs devoirs.",
            # se/ce : formes correctes
            "Il se lave les mains avant de manger.",
            "Ce garçon est très gentil.",
            # ou/où : formes correctes
            "Tu viens lundi ou mardi ?",
            "Où habites-tu maintenant ?",
            # Elision : formes correctes
            "Parce qu'il fait beau, on sort.",
            "L'école est fermée le mercredi.",
            # Fusion : formes correctes
            "Il y avait beaucoup de monde.",
            "Nous travaillons ensemble.",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Evaluation FP
# ---------------------------------------------------------------------------

@dataclass
class ResultatFP:
    """Resultat du test FP pour une source."""
    source: SourcePropre
    n_phrases: int = 0
    n_fp: int = 0
    fp_details: list[tuple[str, str]] = field(default_factory=list)

    @property
    def taux_fp(self) -> float:
        if self.n_phrases == 0:
            return 0.0
        return 100 * self.n_fp / self.n_phrases


def tester_fp(verbose: bool = False) -> None:
    """Teste le taux de FP du V6 sur les textes propres."""
    print("Chargement du lexique...")
    from lectura_lexique import Lexique
    lexique = Lexique(str(LEXIQUE_PATH))

    print("Chargement du correcteur V6...")
    from lectura_correcteur import CorrecteurV6, CorrecteurV6Config
    config = CorrecteurV6Config()
    correcteur = CorrecteurV6(lexique, config=config)
    print(f"  P2G disponible: {correcteur.p2g_disponible}\n")

    resultats: list[ResultatFP] = []
    total_phrases = 0
    total_fp = 0

    for source in SOURCES:
        if not source.phrases:
            continue

        res = ResultatFP(source=source, n_phrases=len(source.phrases))

        for phrase in source.phrases:
            result = correcteur.corriger(phrase)
            # Normaliser pour comparer
            orig = _normaliser(phrase)
            corr = _normaliser(result.phrase_corrigee)

            if orig != corr:
                res.n_fp += 1
                res.fp_details.append((phrase, result.phrase_corrigee))

        resultats.append(res)
        total_phrases += res.n_phrases
        total_fp += res.n_fp

    # --- Rapport ---
    sep = "=" * 72
    print(sep)
    print("  TEST FAUX POSITIFS — Textes propres multi-sources")
    print(sep)

    print(f"\n  {'Source':<25s} | {'Phrases':>8s} | {'FP':>4s} | {'%FP':>6s}")
    print("  " + "-" * 55)
    for res in resultats:
        marker = "  " if res.taux_fp < 1.0 else " *"
        print(f"  {res.source.nom:<25s} | {res.n_phrases:>8d} | {res.n_fp:>4d} |"
              f" {res.taux_fp:>5.1f}%{marker}")
    print("  " + "-" * 55)
    pct_total = 100 * total_fp / total_phrases if total_phrases > 0 else 0
    print(f"  {'TOTAL':<25s} | {total_phrases:>8d} | {total_fp:>4d} | {pct_total:>5.1f}%")

    # Detail des FP
    for res in resultats:
        if res.fp_details:
            print(f"\n--- FP detail : {res.source.nom} ({res.n_fp} FP) ---")
            for j, (orig, corr) in enumerate(res.fp_details[:10]):
                print(f"  [{j+1:2d}] '{orig}'")
                print(f"       -> '{corr}'")
            if len(res.fp_details) > 10:
                print(f"  ... et {len(res.fp_details) - 10} autres")

    # Verdict
    print()
    if pct_total < 1.0:
        print(f"  VERDICT : OK — taux FP global {pct_total:.1f}% < 1.0%")
    elif pct_total < 2.0:
        print(f"  VERDICT : ACCEPTABLE — taux FP global {pct_total:.1f}% < 2.0%")
    else:
        print(f"  VERDICT : ECHEC — taux FP global {pct_total:.1f}% >= 2.0%")
    print()


def _normaliser(phrase: str) -> str:
    """Normalisation legere pour comparer."""
    phrase = phrase.strip()
    while phrase and phrase[-1] in ".!?…":
        phrase = phrase[:-1]
    return " ".join(phrase.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test FP sur textes propres multi-sources",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    tester_fp(verbose=args.verbose)


if __name__ == "__main__":
    main()
