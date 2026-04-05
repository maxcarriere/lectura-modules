"""Corpus de phrases annotées pour l'évaluation comparative.

Chaque phrase est associée à sa version corrigée attendue et à une ou
plusieurs catégories d'erreur.  Le corpus couvre les règles implémentées
dans lectura-correcteur ainsi que quelques catégories non encore
implémentées (PP, NEG) afin de mesurer les lacunes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CasTest:
    id: str  # ex: "ACC_PLUR_01"
    erronee: str  # phrase avec erreur(s)
    attendue: str  # phrase corrigée attendue
    categories: list[str] = field(default_factory=list)
    description: str = ""
    implementee: bool = True  # False pour PP, NEG


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

CORPUS: list[CasTest] = [
    # ===== ACC_PLUR — Accord pluriel déterminant + nom =====
    CasTest(
        "ACC_PLUR_01",
        "les enfant jouent dans le jardin",
        "Les enfants jouent dans le jardin.",
        ["ACC_PLUR"],
        "det pluriel + nom singulier",
    ),
    CasTest(
        "ACC_PLUR_02",
        "des maison sont grandes",
        "Des maisons sont grandes.",
        ["ACC_PLUR"],
        "det pluriel + nom singulier (féminin)",
    ),
    CasTest(
        "ACC_PLUR_03",
        "les chat dorment sur le lit",
        "Les chats dorment sur le lit.",
        ["ACC_PLUR"],
        "det pluriel + nom singulier (animal)",
    ),
    CasTest(
        "ACC_PLUR_04",
        "ses livre sont sur la table",
        "Ses livres sont sur la table.",
        ["ACC_PLUR"],
        "possessif pluriel + nom singulier",
    ),
    CasTest(
        "ACC_PLUR_05",
        "ces fleur sont jolies",
        "Ces fleurs sont jolies.",
        ["ACC_PLUR"],
        "démonstratif pluriel + nom singulier",
    ),
    CasTest(
        "ACC_PLUR_06",
        "mes ami arrivent ce soir",
        "Mes amis arrivent ce soir.",
        ["ACC_PLUR"],
        "possessif pluriel + nom singulier (voyelle)",
    ),

    # ===== ACC_SV — Accord sujet-verbe =====
    CasTest(
        "ACC_SV_01",
        "les chats mange des pommes",
        "Les chats mangent des pommes.",
        ["ACC_SV"],
        "sujet pluriel + verbe singulier",
    ),
    CasTest(
        "ACC_SV_02",
        "les enfants joue dans la cour",
        "Les enfants jouent dans la cour.",
        ["ACC_SV"],
        "sujet pluriel + verbe singulier",
    ),
    CasTest(
        "ACC_SV_03",
        "les oiseaux chante le matin",
        "Les oiseaux chantent le matin.",
        ["ACC_SV"],
        "sujet pluriel + verbe singulier",
    ),
    CasTest(
        "ACC_SV_04",
        "les filles regarde le chat",
        "Les filles regardent le chat.",
        ["ACC_SV"],
        "sujet féminin pluriel + verbe singulier",
    ),
    CasTest(
        "ACC_SV_05",
        "les garçons arrive à la maison",
        "Les garçons arrivent à la maison.",
        ["ACC_SV"],
        "sujet pluriel + verbe singulier",
    ),
    CasTest(
        "ACC_SV_06",
        "les femmes parle avec le voisin",
        "Les femmes parlent avec le voisin.",
        ["ACC_SV"],
        "sujet féminin pluriel + verbe singulier",
    ),
    CasTest(
        "ACC_SV_07",
        "les hommes donne des fleurs",
        "Les hommes donnent des fleurs.",
        ["ACC_SV"],
        "sujet pluriel + verbe singulier",
    ),

    # ===== ACC_ADJ — Accord adjectif =====
    CasTest(
        "ACC_ADJ_01",
        "des maisons blanche sont dans la rue",
        "Des maisons blanches sont dans la rue.",
        ["ACC_ADJ"],
        "nom fém pluriel + adj fém singulier",
    ),
    CasTest(
        "ACC_ADJ_02",
        "les fleurs rouge sont belles",
        "Les fleurs rouges sont belles.",
        ["ACC_ADJ"],
        "nom fém pluriel + adj invariable singulier",
    ),
    CasTest(
        "ACC_ADJ_03",
        "des chats noir dorment sur le lit",
        "Des chats noirs dorment sur le lit.",
        ["ACC_ADJ"],
        "nom masc pluriel + adj masc singulier",
    ),
    CasTest(
        "ACC_ADJ_04",
        "les grandes maison sont belles",
        "Les grandes maisons sont belles.",
        ["ACC_ADJ", "ACC_PLUR"],
        "adj pluriel + nom singulier",
    ),
    CasTest(
        "ACC_ADJ_05",
        "les petit garçons jouent dans le jardin",
        "Les petits garçons jouent dans le jardin.",
        ["ACC_ADJ"],
        "det pluriel + adj singulier + nom pluriel",
    ),
    CasTest(
        "ACC_ADJ_06",
        "les voitures vert sont dans le jardin",
        "Les voitures vertes sont dans le jardin.",
        ["ACC_ADJ"],
        "nom fém pluriel + adj masc singulier",
    ),

    # ===== CONJ — Conjugaison pronom + verbe =====
    CasTest(
        "CONJ_01",
        "ils mange des pommes",
        "Ils mangent des pommes.",
        ["CONJ"],
        "ils + verbe singulier",
    ),
    CasTest(
        "CONJ_02",
        "elles arrive à la maison",
        "Elles arrivent à la maison.",
        ["CONJ"],
        "elles + verbe singulier",
    ),
    CasTest(
        "CONJ_03",
        "nous mange des pommes",
        "Nous mangeons des pommes.",
        ["CONJ"],
        "nous + verbe non conjugué à P4",
    ),
    CasTest(
        "CONJ_04",
        "vous mange des pommes",
        "Vous mangez des pommes.",
        ["CONJ"],
        "vous + verbe non conjugué à P5",
    ),
    CasTest(
        "CONJ_05",
        "je mangent des pommes",
        "Je mange des pommes.",
        ["CONJ"],
        "je + verbe pluriel",
    ),
    CasTest(
        "CONJ_06",
        "tu mangent des pommes",
        "Tu manges des pommes.",
        ["CONJ"],
        "tu + verbe pluriel",
    ),
    CasTest(
        "CONJ_07",
        "ils chante très bien",
        "Ils chantent très bien.",
        ["CONJ"],
        "ils + verbe singulier",
    ),
    CasTest(
        "CONJ_08",
        "elles parle avec le voisin",
        "Elles parlent avec le voisin.",
        ["CONJ"],
        "elles + verbe singulier",
    ),

    # ===== HOMO — Homophones =====
    CasTest(
        "HOMO_01",
        "elle et belle",
        "Elle est belle.",
        ["HOMO"],
        "et/est après pronom sujet",
    ),
    CasTest(
        "HOMO_02",
        "le chat est sur la table est il dort",
        "Le chat est sur la table et il dort.",
        ["HOMO"],
        "est/et entre deux propositions",
    ),
    CasTest(
        "HOMO_03",
        "il a un chat est un chien",
        "Il a un chat et un chien.",
        ["HOMO"],
        "est/et entre deux noms",
    ),
    CasTest(
        "HOMO_04",
        "son chat et gros",
        "Son chat est gros.",
        ["HOMO"],
        "et/est avec adjectif attribut",
    ),
    CasTest(
        "HOMO_05",
        "la maison et grande",
        "La maison est grande.",
        ["HOMO"],
        "et/est avec adjectif attribut (fém)",
    ),

    # ===== ORTH — Orthographe lexicale =====
    CasTest(
        "ORTH_01",
        "le hcat dort sur le lit",
        "Le chat dort sur le lit.",
        ["ORTH"],
        "anagramme simple",
    ),
    CasTest(
        "ORTH_02",
        "la miason est grande",
        "La maison est grande.",
        ["ORTH"],
        "transposition de lettres",
    ),
    CasTest(
        "ORTH_03",
        "le garsson mange une pomme",
        "Le garçon mange une pomme.",
        ["ORTH"],
        "substitution phonétique s/ç",
    ),
    CasTest(
        "ORTH_04",
        "la feme parle avec le voisin",
        "La femme parle avec le voisin.",
        ["ORTH"],
        "lettre manquante",
    ),
    CasTest(
        "ORTH_05",
        "les enfannts jouent dans le jardin",
        "Les enfants jouent dans le jardin.",
        ["ORTH"],
        "lettres doublées par erreur",
    ),

    # ===== GENRE — Accord en genre =====
    CasTest(
        "GENRE_01",
        "la petit fille joue dans le jardin",
        "La petite fille joue dans le jardin.",
        ["GENRE"],
        "det fém + adj masc + nom fém",
    ),
    CasTest(
        "GENRE_02",
        "une grand maison est dans la rue",
        "Une grande maison est dans la rue.",
        ["GENRE"],
        "det fém + adj masc + nom fém",
    ),
    CasTest(
        "GENRE_03",
        "le petit fille joue dans la cour",
        "La petite fille joue dans la cour.",
        ["GENRE"],
        "det masc + adj masc + nom fém",
    ),
    CasTest(
        "GENRE_04",
        "un belle maison est dans la rue",
        "Une belle maison est dans la rue.",
        ["GENRE"],
        "det masc + adj fém + nom fém",
    ),
    CasTest(
        "GENRE_05",
        "le grosse chat dort sur le lit",
        "Le gros chat dort sur le lit.",
        ["GENRE"],
        "det masc + adj fém + nom masc",
    ),

    # ===== PP — Participe passé (non implémenté) =====
    CasTest(
        "PP_01",
        "il a manger du pain",
        "Il a mangé du pain.",
        ["PP"],
        "infinitif au lieu du participe passé",
        implementee=False,
    ),
    CasTest(
        "PP_02",
        "elle a chanter toute la soirée",
        "Elle a chanté toute la soirée.",
        ["PP"],
        "infinitif au lieu du participe passé",
        implementee=False,
    ),
    CasTest(
        "PP_03",
        "ils ont jouer dans le jardin",
        "Ils ont joué dans le jardin.",
        ["PP"],
        "infinitif au lieu du participe passé",
        implementee=False,
    ),
    CasTest(
        "PP_04",
        "nous avons parler avec le voisin",
        "Nous avons parlé avec le voisin.",
        ["PP"],
        "infinitif au lieu du participe passé",
        implementee=False,
    ),
    CasTest(
        "PP_05",
        "vous avez donner des fleurs",
        "Vous avez donné des fleurs.",
        ["PP"],
        "infinitif au lieu du participe passé",
        implementee=False,
    ),

    # ===== NEG — Négation (non implémenté) =====
    CasTest(
        "NEG_01",
        "il mange pas de pommes",
        "Il ne mange pas de pommes.",
        ["NEG"],
        "négation sans ne",
        implementee=False,
    ),
    CasTest(
        "NEG_02",
        "elle dort pas bien",
        "Elle ne dort pas bien.",
        ["NEG"],
        "négation sans ne",
        implementee=False,
    ),
    CasTest(
        "NEG_03",
        "ils veulent pas jouer",
        "Ils ne veulent pas jouer.",
        ["NEG"],
        "négation sans ne",
        implementee=False,
    ),
    CasTest(
        "NEG_04",
        "je sais pas pourquoi",
        "Je ne sais pas pourquoi.",
        ["NEG"],
        "négation sans ne",
        implementee=False,
    ),

    # ===== MAJ — Majuscules / ponctuation =====
    CasTest(
        "MAJ_01",
        "le chat dort sur le lit",
        "Le chat dort sur le lit.",
        ["MAJ"],
        "majuscule en début de phrase + point final",
    ),
    CasTest(
        "MAJ_02",
        "les enfants jouent dans le jardin",
        "Les enfants jouent dans le jardin.",
        ["MAJ"],
        "majuscule en début de phrase + point final",
    ),
    CasTest(
        "MAJ_03",
        "elle parle avec son ami",
        "Elle parle avec son ami.",
        ["MAJ"],
        "majuscule en début de phrase + point final",
    ),
    CasTest(
        "MAJ_04",
        "il fait beau aujourd'hui",
        "Il fait beau aujourd'hui.",
        ["MAJ"],
        "majuscule + point final",
    ),
    CasTest(
        "MAJ_05",
        "nous aimons les fleurs rouges",
        "Nous aimons les fleurs rouges.",
        ["MAJ"],
        "majuscule + point final",
    ),

    # ===== ACC_DIST — Accord sujet-verbe à distance =====
    CasTest(
        "ACC_DIST_01",
        "les enfants de la voisine mange des pommes",
        "Les enfants de la voisine mangent des pommes.",
        ["ACC_DIST"],
        "sujet pluriel séparé du verbe par complément",
    ),
    CasTest(
        "ACC_DIST_02",
        "les chats du voisin dort sur le lit",
        "Les chats du voisin dorment sur le lit.",
        ["ACC_DIST"],
        "sujet pluriel séparé du verbe par complément",
    ),
    CasTest(
        "ACC_DIST_03",
        "les fleurs du jardin est belles",
        "Les fleurs du jardin sont belles.",
        ["ACC_DIST"],
        "sujet pluriel + être singulier à distance",
    ),
    CasTest(
        "ACC_DIST_04",
        "les livres de la fille est sur la table",
        "Les livres de la fille sont sur la table.",
        ["ACC_DIST"],
        "sujet pluriel + être singulier à distance",
    ),
    CasTest(
        "ACC_DIST_05",
        "les voitures dans la rue fait du bruit",
        "Les voitures dans la rue font du bruit.",
        ["ACC_DIST"],
        "sujet pluriel + faire singulier à distance",
    ),

    # ===== OK — Phrases correctes (détection faux positifs) =====
    CasTest(
        "OK_01",
        "Le chat dort sur le lit.",
        "Le chat dort sur le lit.",
        ["OK"],
        "phrase simple correcte",
    ),
    CasTest(
        "OK_02",
        "Les enfants jouent dans le jardin.",
        "Les enfants jouent dans le jardin.",
        ["OK"],
        "phrase plurielle correcte",
    ),
    CasTest(
        "OK_03",
        "Elle mange une pomme rouge.",
        "Elle mange une pomme rouge.",
        ["OK"],
        "phrase avec adjectif correcte",
    ),
    CasTest(
        "OK_04",
        "Nous parlons avec les voisins.",
        "Nous parlons avec les voisins.",
        ["OK"],
        "phrase avec nous correcte",
    ),
    CasTest(
        "OK_05",
        "Il fait beau aujourd'hui.",
        "Il fait beau aujourd'hui.",
        ["OK"],
        "phrase correcte avec apostrophe",
    ),
    CasTest(
        "OK_06",
        "Les grandes maisons sont belles.",
        "Les grandes maisons sont belles.",
        ["OK"],
        "phrase avec accord complexe correcte",
    ),
    CasTest(
        "OK_07",
        "Mon chat et mon chien jouent dans le jardin.",
        "Mon chat et mon chien jouent dans le jardin.",
        ["OK"],
        "phrase avec coordination correcte",
    ),
    CasTest(
        "OK_08",
        "La petite fille regarde les oiseaux.",
        "La petite fille regarde les oiseaux.",
        ["OK"],
        "phrase avec det+adj+nom fém correcte",
    ),

    # ===== MULTI — Multi-erreurs =====
    CasTest(
        "MULTI_01",
        "les enfant mange des pomme",
        "Les enfants mangent des pommes.",
        ["ACC_PLUR", "ACC_SV", "MAJ"],
        "accord pluriel + accord SV + majuscule",
    ),
    CasTest(
        "MULTI_02",
        "les chat noir dort sur le lit",
        "Les chats noirs dorment sur le lit.",
        ["ACC_PLUR", "ACC_ADJ", "ACC_SV", "MAJ"],
        "accord pluriel + adj + SV + majuscule",
    ),
    CasTest(
        "MULTI_03",
        "elle et belle et elle chante bien",
        "Elle est belle et elle chante bien.",
        ["HOMO", "MAJ"],
        "homophone et/est + majuscule",
    ),
    CasTest(
        "MULTI_04",
        "les grande maison blanche sont belles",
        "Les grandes maisons blanches sont belles.",
        ["ACC_PLUR", "ACC_ADJ", "MAJ"],
        "accord pluriel + adj + majuscule",
    ),
    CasTest(
        "MULTI_05",
        "ils mange des pomme rouge",
        "Ils mangent des pommes rouges.",
        ["CONJ", "ACC_PLUR", "ACC_ADJ", "MAJ"],
        "conjugaison + accord pluriel + adj + majuscule",
    ),
    CasTest(
        "MULTI_06",
        "les fille joue dans les jardin",
        "Les filles jouent dans les jardins.",
        ["ACC_PLUR", "ACC_SV", "MAJ"],
        "double accord pluriel + SV + majuscule",
    ),
]


def par_categorie(cat: str) -> list[CasTest]:
    """Retourne les cas de test pour une catégorie donnée."""
    return [c for c in CORPUS if cat in c.categories]


def categories() -> list[str]:
    """Retourne la liste triée des catégories uniques."""
    cats: set[str] = set()
    for c in CORPUS:
        cats.update(c.categories)
    return sorted(cats)
