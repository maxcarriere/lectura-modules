"""Corpus de phrases annotées pour l'évaluation comparative.

Chaque phrase est associée à sa version corrigée attendue et à une ou
plusieurs catégories d'erreur.  Le corpus couvre les règles implémentées
dans lectura-correcteur ainsi que quelques catégories non encore
implémentées afin de mesurer les lacunes.
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
    implementee: bool = True


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

    # ===== PP — Participe passé =====
    CasTest(
        "PP_01",
        "il a manger du pain",
        "Il a mangé du pain.",
        ["PP"],
        "infinitif au lieu du participe passé",
    ),
    CasTest(
        "PP_02",
        "elle a chanter toute la soirée",
        "Elle a chanté toute la soirée.",
        ["PP"],
        "infinitif au lieu du participe passé",
    ),
    CasTest(
        "PP_03",
        "ils ont jouer dans le jardin",
        "Ils ont joué dans le jardin.",
        ["PP"],
        "infinitif au lieu du participe passé",
    ),
    CasTest(
        "PP_04",
        "nous avons parler avec le voisin",
        "Nous avons parlé avec le voisin.",
        ["PP"],
        "infinitif au lieu du participe passé",
    ),
    CasTest(
        "PP_05",
        "vous avez donner des fleurs",
        "Vous avez donné des fleurs.",
        ["PP"],
        "infinitif au lieu du participe passé",
    ),

    # ===== NEG — Négation =====
    CasTest(
        "NEG_01",
        "il mange pas de pommes",
        "Il ne mange pas de pommes.",
        ["NEG"],
        "négation sans ne",
    ),
    CasTest(
        "NEG_02",
        "elle dort pas bien",
        "Elle ne dort pas bien.",
        ["NEG"],
        "négation sans ne",
    ),
    CasTest(
        "NEG_03",
        "ils veulent pas jouer",
        "Ils ne veulent pas jouer.",
        ["NEG"],
        "négation sans ne",
    ),
    CasTest(
        "NEG_04",
        "je sais pas pourquoi",
        "Je ne sais pas pourquoi.",
        ["NEG"],
        "négation sans ne",
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

    # ===== PP_ETRE — Accord PP avec etre =====
    CasTest(
        "PP_ETRE_01",
        "elle est allé au marché",
        "Elle est allée au marché.",
        ["PP_ETRE"],
        "PP masc -> fem sing avec etre",
    ),
    CasTest(
        "PP_ETRE_02",
        "ils sont arrivé à la maison",
        "Ils sont arrivés à la maison.",
        ["PP_ETRE"],
        "PP sing -> masc plur avec etre",
    ),
    CasTest(
        "PP_ETRE_03",
        "elles sont parti ce matin",
        "Elles sont parties ce matin.",
        ["PP_ETRE"],
        "PP masc sing -> fem plur avec etre",
    ),
    CasTest(
        "PP_ETRE_04",
        "elle est tombé dans la rue",
        "Elle est tombée dans la rue.",
        ["PP_ETRE"],
        "PP masc -> fem sing avec etre",
    ),
    CasTest(
        "PP_ETRE_05",
        "nous sommes allé au parc",
        "Nous sommes allés au parc.",
        ["PP_ETRE"],
        "PP sing -> masc plur avec etre (nous)",
    ),
    CasTest(
        "PP_ETRE_06",
        "elle est venu hier soir",
        "Elle est venue hier soir.",
        ["PP_ETRE"],
        "PP masc -> fem sing avec etre",
    ),
    CasTest(
        "PP_ETRE_07",
        "ils sont tombé dans le jardin",
        "Ils sont tombés dans le jardin.",
        ["PP_ETRE"],
        "PP sing -> masc plur avec etre",
    ),
    CasTest(
        "PP_ETRE_08",
        "elles sont venu ce matin",
        "Elles sont venues ce matin.",
        ["PP_ETRE"],
        "PP masc sing -> fem plur avec etre",
    ),

    # ===== ATTR — Accord attribut genre =====
    CasTest(
        "ATTR_01",
        "elle est petit",
        "Elle est petite.",
        ["ATTR"],
        "attribut masc -> fem apres etre",
    ),
    CasTest(
        "ATTR_02",
        "la maison est grand",
        "La maison est grande.",
        ["ATTR"],
        "attribut masc -> fem (sujet NOM fem)",
    ),
    CasTest(
        "ATTR_03",
        "elle semble content",
        "Elle semble contente.",
        ["ATTR"],
        "attribut masc -> fem apres sembler",
    ),
    CasTest(
        "ATTR_04",
        "elle devient grand",
        "Elle devient grande.",
        ["ATTR"],
        "attribut masc -> fem apres devenir",
    ),
    CasTest(
        "ATTR_05",
        "elle reste petit",
        "Elle reste petite.",
        ["ATTR"],
        "attribut masc -> fem apres rester",
    ),
    CasTest(
        "ATTR_06",
        "elles sont petit",
        "Elles sont petites.",
        ["ATTR"],
        "attribut masc sing -> fem plur (genre + nombre)",
    ),
    CasTest(
        "ATTR_07",
        "elle est grand et belle",
        "Elle est grande et belle.",
        ["ATTR"],
        "attribut masc -> fem avant coordination",
    ),
    CasTest(
        "ATTR_08",
        "la fille est content",
        "La fille est contente.",
        ["ATTR"],
        "attribut masc -> fem (sujet NOM fem)",
    ),

    # ===== CONJ_IMP — Conjugaison imparfait =====
    CasTest(
        "CONJ_IMP_01",
        "ils mangeait des pommes",
        "Ils mangeaient des pommes.",
        ["CONJ_IMP"],
        "3pl imparfait",
    ),
    CasTest(
        "CONJ_IMP_02",
        "je parlaient avec le voisin",
        "Je parlais avec le voisin.",
        ["CONJ_IMP"],
        "1s imparfait (3pl -> 1s)",
    ),
    CasTest(
        "CONJ_IMP_03",
        "tu finissait son travail",
        "Tu finissais son travail.",
        ["CONJ_IMP"],
        "2s imparfait",
    ),
    CasTest(
        "CONJ_IMP_04",
        "il mangeaient une pomme",
        "Il mangeait une pomme.",
        ["CONJ_IMP"],
        "3s imparfait (3pl -> 3s)",
    ),
    CasTest(
        "CONJ_IMP_05",
        "elle dormait bien",
        "Elle dormait bien.",
        ["CONJ_IMP", "OK"],
        "imparfait correct (pas de correction)",
    ),
    CasTest(
        "CONJ_IMP_06",
        "nous mangeait des pommes",
        "Nous mangions des pommes.",
        ["CONJ_IMP"],
        "1pl imparfait",
    ),

    # ===== CONJ_FUT — Conjugaison futur =====
    CasTest(
        "CONJ_FUT_01",
        "nous mangera des pommes",
        "Nous mangerons des pommes.",
        ["CONJ_FUT"],
        "1pl futur",
    ),
    CasTest(
        "CONJ_FUT_02",
        "ils finira le travail",
        "Ils finiront le travail.",
        ["CONJ_FUT"],
        "3pl futur",
    ),
    CasTest(
        "CONJ_FUT_03",
        "tu chantera une chanson",
        "Tu chanteras une chanson.",
        ["CONJ_FUT"],
        "2s futur",
    ),
    CasTest(
        "CONJ_FUT_04",
        "je parleront avec le voisin",
        "Je parlerai avec le voisin.",
        ["CONJ_FUT"],
        "1s futur (3pl -> 1s)",
    ),
    CasTest(
        "CONJ_FUT_05",
        "vous mangera des pommes",
        "Vous mangerez des pommes.",
        ["CONJ_FUT"],
        "2pl futur",
    ),
    CasTest(
        "CONJ_FUT_06",
        "nous mangerons des pommes",
        "Nous mangerons des pommes.",
        ["CONJ_FUT", "OK"],
        "futur correct (pas de correction)",
    ),

    # ===== HOMO_NEW — Homophones leur/leurs, ca/sa, -er/-e =====
    CasTest(
        "HOMO_NEW_01",
        "leur enfants jouent dans le jardin",
        "Leurs enfants jouent dans le jardin.",
        ["HOMO_NEW"],
        "leur -> leurs (NOM pluriel)",
    ),
    CasTest(
        "HOMO_NEW_02",
        "leurs maison est grande",
        "Leur maison est grande.",
        ["HOMO_NEW"],
        "leurs -> leur (NOM singulier)",
    ),
    CasTest(
        "HOMO_NEW_03",
        "sa mange des pommes",
        "Ça mange des pommes.",
        ["HOMO_NEW"],
        "sa -> ça (suivi d'un verbe)",
    ),
    CasTest(
        "HOMO_NEW_04",
        "ça maison est grande",
        "Sa maison est grande.",
        ["HOMO_NEW"],
        "ça -> sa (suivi d'un nom)",
    ),
    CasTest(
        "HOMO_NEW_05",
        "je vais mangé du pain",
        "Je vais manger du pain.",
        ["HOMO_NEW"],
        "PP -> infinitif apres aller",
    ),
    CasTest(
        "HOMO_NEW_06",
        "elle va chanté une chanson",
        "Elle va chanter une chanson.",
        ["HOMO_NEW"],
        "PP -> infinitif apres aller",
    ),
    CasTest(
        "HOMO_NEW_07",
        "leurs enfants jouent dans le jardin",
        "Leurs enfants jouent dans le jardin.",
        ["HOMO_NEW", "OK"],
        "leurs correct (pas de correction)",
    ),
    CasTest(
        "HOMO_NEW_08",
        "leur maison est grande",
        "Leur maison est grande.",
        ["HOMO_NEW", "OK"],
        "leur correct (pas de correction)",
    ),

    # ===== OK_NEW — Phrases correctes supplementaires =====
    CasTest(
        "OK_NEW_01",
        "Elle est partie ce matin.",
        "Elle est partie ce matin.",
        ["OK"],
        "PP etre fem correct",
    ),
    CasTest(
        "OK_NEW_02",
        "Ils mangeaient des pommes.",
        "Ils mangeaient des pommes.",
        ["OK"],
        "imparfait 3pl correct",
    ),
    CasTest(
        "OK_NEW_03",
        "Nous mangerons des pommes.",
        "Nous mangerons des pommes.",
        ["OK"],
        "futur 1pl correct",
    ),
    CasTest(
        "OK_NEW_04",
        "Leurs enfants jouent dans le jardin.",
        "Leurs enfants jouent dans le jardin.",
        ["OK"],
        "leurs + NOM pluriel correct",
    ),
    CasTest(
        "OK_NEW_05",
        "Elle est grande et belle.",
        "Elle est grande et belle.",
        ["OK"],
        "attribut fem correct",
    ),
    CasTest(
        "OK_NEW_06",
        "Il va manger du pain.",
        "Il va manger du pain.",
        ["OK"],
        "infinitif apres aller correct",
    ),
    CasTest(
        "OK_NEW_07",
        "Elles sont arrivées ce matin.",
        "Elles sont arrivées ce matin.",
        ["OK"],
        "PP etre fem plur correct",
    ),
    CasTest(
        "OK_NEW_08",
        "Leur maison est grande.",
        "Leur maison est grande.",
        ["OK"],
        "leur + NOM singulier correct",
    ),

    # ===== MULTI_NEW — Combinaisons nouvelles regles =====
    CasTest(
        "MULTI_NEW_01",
        "elle est allé et elle et contente",
        "Elle est allée et elle est contente.",
        ["PP_ETRE", "HOMO", "ATTR", "MAJ"],
        "PP etre + homophone et/est + attribut",
    ),
    CasTest(
        "MULTI_NEW_02",
        "ils mangeait des pomme rouge",
        "Ils mangeaient des pommes rouges.",
        ["CONJ_IMP", "ACC_PLUR", "ACC_ADJ", "MAJ"],
        "imparfait + accord pluriel + adj",
    ),
    CasTest(
        "MULTI_NEW_03",
        "nous mangera des pomme",
        "Nous mangerons des pommes.",
        ["CONJ_FUT", "ACC_PLUR", "MAJ"],
        "futur + accord pluriel",
    ),
    CasTest(
        "MULTI_NEW_04",
        "leur enfant mange pas de pomme",
        "Leur enfant ne mange pas de pomme.",
        ["NEG"],
        "negation + contexte leur",
    ),
    CasTest(
        "MULTI_NEW_05",
        "elles sont parti et elles sont content",
        "Elles sont parties et elles sont contentes.",
        ["PP_ETRE", "ATTR", "MAJ"],
        "PP etre + attribut genre + nombre",
    ),
    CasTest(
        "MULTI_NEW_06",
        "je vais mangé des pomme rouge",
        "Je vais manger des pommes rouges.",
        ["HOMO_NEW", "ACC_PLUR", "ACC_ADJ", "MAJ"],
        "infinitif apres aller + accord pluriel + adj",
    ),

    # ===== EDGE — Cas limites =====
    CasTest(
        "EDGE_01",
        "les enfants de la maison mange des pommes",
        "Les enfants de la maison mangent des pommes.",
        ["ACC_DIST"],
        "accord distant avec complement de + la + NOM",
    ),
    CasTest(
        "EDGE_02",
        "il a pas mangé de pommes",
        "Il n'a pas mangé de pommes.",
        ["NEG"],
        "negation avec auxiliaire avoir",
    ),
    CasTest(
        "EDGE_03",
        "les chat de la voisine dort sur le lit",
        "Les chats de la voisine dorment sur le lit.",
        ["ACC_PLUR", "ACC_DIST"],
        "accord pluriel + accord distant",
    ),
    CasTest(
        "EDGE_04",
        "Il ne mange pas de pommes.",
        "Il ne mange pas de pommes.",
        ["OK"],
        "negation complete correcte",
    ),
    CasTest(
        "EDGE_05",
        "Elle a mangé du pain.",
        "Elle a mangé du pain.",
        ["OK"],
        "PP avec avoir correct",
    ),
    CasTest(
        "EDGE_06",
        "les petit enfant mange des pomme",
        "Les petits enfants mangent des pommes.",
        ["ACC_PLUR", "ACC_ADJ", "ACC_SV", "MAJ"],
        "triple accord + majuscule",
    ),
    CasTest(
        "EDGE_07",
        "elle est pas allé au marché",
        "Elle n'est pas allée au marché.",
        ["PP_ETRE", "NEG"],
        "PP etre + negation",
    ),
    CasTest(
        "EDGE_08",
        "Il fait beau et il est content.",
        "Il fait beau et il est content.",
        ["OK"],
        "phrase complexe correcte",
    ),
    CasTest(
        "EDGE_09",
        "les femmes du voisin regarde le chat",
        "Les femmes du voisin regardent le chat.",
        ["ACC_DIST"],
        "sujet fem pluriel a distance",
    ),
    CasTest(
        "EDGE_10",
        "ils sont allé au parc et elles sont parti",
        "Ils sont allés au parc et elles sont parties.",
        ["PP_ETRE", "MAJ"],
        "double PP etre dans une phrase",
    ),
    CasTest(
        "EDGE_11",
        "elle va mangé et elle va chanté",
        "Elle va manger et elle va chanter.",
        ["HOMO_NEW", "MAJ"],
        "double infinitif apres aller",
    ),
    CasTest(
        "EDGE_12",
        "Les enfants mangent des pommes rouges.",
        "Les enfants mangent des pommes rouges.",
        ["OK"],
        "phrase plurielle complexe correcte",
    ),
    CasTest(
        "EDGE_13",
        "Elles sont grandes et belles.",
        "Elles sont grandes et belles.",
        ["OK"],
        "attribut fem plur correct",
    ),
    CasTest(
        "EDGE_14",
        "les voiture rouge sont dans le jardin",
        "Les voitures rouges sont dans le jardin.",
        ["ACC_PLUR", "ACC_ADJ", "MAJ"],
        "accord pluriel NOM + ADJ",
    ),
    CasTest(
        "EDGE_15",
        "tu mange des pommes",
        "Tu manges des pommes.",
        ["CONJ", "MAJ"],
        "conjugaison P2 + majuscule",
    ),
    CasTest(
        "EDGE_16",
        "je chante bien",
        "Je chante bien.",
        ["MAJ"],
        "majuscule + point final seulement",
    ),
    CasTest(
        "EDGE_17",
        "les enfants jouent pas dans le jardin",
        "Les enfants ne jouent pas dans le jardin.",
        ["NEG", "MAJ"],
        "negation + majuscule",
    ),
    CasTest(
        "EDGE_18",
        "Il est parti ce matin.",
        "Il est parti ce matin.",
        ["OK"],
        "PP etre masc sing correct",
    ),
    CasTest(
        "EDGE_19",
        "Ils sont arrivés à la maison.",
        "Ils sont arrivés à la maison.",
        ["OK"],
        "PP etre masc plur correct",
    ),
    CasTest(
        "EDGE_20",
        "Elle chantait une chanson.",
        "Elle chantait une chanson.",
        ["OK"],
        "imparfait 3s correct",
    ),
    CasTest(
        "EDGE_21",
        "les enfant de la femme du voisin mange des pomme",
        "Les enfants de la femme du voisin mangent des pommes.",
        ["ACC_PLUR", "ACC_DIST", "MAJ"],
        "accord distant double complement + pluriel",
    ),
    CasTest(
        "EDGE_22",
        "Sa maison est grande.",
        "Sa maison est grande.",
        ["OK"],
        "sa + NOM correct (pas de faux positif)",
    ),
    CasTest(
        "EDGE_23",
        "Ça mange des pommes.",
        "Ça mange des pommes.",
        ["OK"],
        "ça + VER correct (pas de faux positif)",
    ),
    CasTest(
        "EDGE_24",
        "nous chante des chanson",
        "Nous chantons des chansons.",
        ["CONJ", "ACC_PLUR", "MAJ"],
        "conjugaison P1p + accord pluriel",
    ),
    CasTest(
        "EDGE_25",
        "elle a pas parlé avec le voisin",
        "Elle n'a pas parlé avec le voisin.",
        ["NEG"],
        "negation avec avoir et PP",
    ),
    CasTest(
        "EDGE_26",
        "vous chante des chanson",
        "Vous chantez des chansons.",
        ["CONJ", "ACC_PLUR", "MAJ"],
        "conjugaison P2p + accord pluriel",
    ),
    CasTest(
        "EDGE_27",
        "la petit maison est belle",
        "La petite maison est belle.",
        ["GENRE"],
        "det fem + adj masc + nom fem",
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
