"""Corpus de VALIDATION GEC debiaise — jamais utilise pendant le developpement.

Distribution : ORTH 20, ACCORD 12, CONJ 8, HOMO 6, AUTRE 6, OK 8 = 60 phrases.
Formulations differentes du corpus dev, memes categories.
"""

from __future__ import annotations

from corpus_benchmark import CasBenchmark


# ═══════════════════════════════════════════════════════════════════════════
#  ORTH — Orthographe (20 phrases)
# ═══════════════════════════════════════════════════════════════════════════

ORTH_VAL = [
    # accent (6)
    CasBenchmark("V_ORTH_01", "la bibliotheque est ouverte le mercredi",
                 ["La bibliothèque est ouverte le mercredi."], "ORTH", "orth_accent", "CM1"),
    CasBenchmark("V_ORTH_02", "mon pere a achete des cereales au supermarche",
                 ["Mon père a acheté des céréales au supermarché."], "ORTH", "orth_accent", "CM1", 3),
    CasBenchmark("V_ORTH_03", "la television etait allumee toute la soiree",
                 ["La télévision était allumée toute la soirée."], "ORTH", "orth_accent", "CM2", 4),
    CasBenchmark("V_ORTH_04", "il a repete la meme erreur plusieurs fois",
                 ["Il a répété la même erreur plusieurs fois."], "ORTH", "orth_accent", "CM2", 2),
    CasBenchmark("V_ORTH_05", "le medecin a examine le bebe a l'hopital",
                 ["Le médecin a examiné le bébé à l'hôpital."], "ORTH", "orth_accent", "6e", 4),
    CasBenchmark("V_ORTH_06", "la fenetre de ma chambre donne sur la riviere",
                 ["La fenêtre de ma chambre donne sur la rivière."], "ORTH", "orth_accent", "CM2", 2),
    # phonetique (4)
    CasBenchmark("V_ORTH_07", "le rino est un animal tres gros",
                 ["Le rhino est un animal très gros."], "ORTH", "orth_phonetique", "CE2", 2),
    CasBenchmark("V_ORTH_08", "les enfants vont a la gimnas le jeudi",
                 ["Les enfants vont à la gymnas le jeudi."], "ORTH", "orth_phonetique", "CM1"),
    CasBenchmark("V_ORTH_09", "il a oublié son parapluie a l'ecole",
                 ["Il a oublié son parapluie à l'école."], "ORTH", "orth_phonetique", "CE2"),
    CasBenchmark("V_ORTH_10", "ma grand-mere fait un exelant gateau au chocolat",
                 ["Ma grand-mère fait un excellent gâteau au chocolat."], "ORTH", "orth_phonetique", "CM1", 3),
    # typo (4)
    CasBenchmark("V_ORTH_11", "le profeusser explique la lecon de grammaire",
                 ["Le professeur explique la leçon de grammaire."], "ORTH", "orth_typo", "CM1", 2),
    CasBenchmark("V_ORTH_12", "les eleves ont fait uen dictee ce matin",
                 ["Les élèves ont fait une dictée ce matin."], "ORTH", "orth_typo", "CE2", 3),
    CasBenchmark("V_ORTH_13", "nous avosn joue au football pendant la pause",
                 ["Nous avons joué au football pendant la pause."], "ORTH", "orth_typo", "CM1"),
    CasBenchmark("V_ORTH_14", "elle a recu un cadaeu pour son anniversaire",
                 ["Elle a reçu un cadeau pour son anniversaire."], "ORTH", "orth_typo", "CM2", 2),
    # double/graphie (3)
    CasBenchmark("V_ORTH_15", "la maitrese a felicite les meilleurs eleves",
                 ["La maîtresse a félicité les meilleurs élèves."], "ORTH", "orth_double", "CM2", 3),
    CasBenchmark("V_ORTH_16", "nous avons ramasé des coquilages sur la plage",
                 ["Nous avons ramassé des coquillages sur la plage."], "ORTH", "orth_double", "CM1", 2),
    CasBenchmark("V_ORTH_17", "son adrese est tres longue a retenir",
                 ["Son adresse est très longue à retenir."], "ORTH", "orth_double", "CM1", 2),
    # segmentation (3)
    CasBenchmark("V_ORTH_18", "lorsque il pleut nous restons a la maison",
                 ["Lorsqu'il pleut nous restons à la maison."], "ORTH", "orth_segmentation", "CM1", 2),
    CasBenchmark("V_ORTH_19", "il faisait peut etre trop chaud dehors",
                 ["Il faisait peut-être trop chaud dehors."], "ORTH", "orth_segmentation", "CM2"),
    CasBenchmark("V_ORTH_20", "on va tout droit puis a gauche apres le pont",
                 ["On va tout droit puis à gauche après le pont."], "ORTH", "orth_segmentation", "CM1", 2),
]


# ═══════════════════════════════════════════════════════════════════════════
#  ACCORD — Accords (12 phrases)
# ═══════════════════════════════════════════════════════════════════════════

ACCORD_VAL = [
    # det_nom (3)
    CasBenchmark("V_ACC_01", "les oiseau chantent dans les buisson",
                 ["Les oiseaux chantent dans les buissons."], "ACCORD", "accord_det_nom", "CE2", 2),
    CasBenchmark("V_ACC_02", "j'ai mange trois banane et deux orange",
                 ["J'ai mangé trois bananes et deux oranges."], "ACCORD", "accord_det_nom", "CE2", 2),
    CasBenchmark("V_ACC_03", "mes cousin sont venu pour les vacance",
                 ["Mes cousins sont venus pour les vacances."], "ACCORD", "accord_det_nom", "CM1", 2),
    # sv (3)
    CasBenchmark("V_ACC_04", "les voisins regarde le match a la television",
                 ["Les voisins regardent le match à la télévision."], "ACCORD", "accord_sv", "CM1", 2),
    CasBenchmark("V_ACC_05", "ses amies danse bien le samedi soir",
                 ["Ses amies dansent bien le samedi soir."], "ACCORD", "accord_sv", "CM1"),
    CasBenchmark("V_ACC_06", "les nuages couvre le ciel depuis ce matin",
                 ["Les nuages couvrent le ciel depuis ce matin."], "ACCORD", "accord_sv", "CM2"),
    # adj (3)
    CasBenchmark("V_ACC_07", "les roses sont tres parfume dans le jardin",
                 ["Les roses sont très parfumées dans le jardin."], "ACCORD", "accord_adj", "CM1"),
    CasBenchmark("V_ACC_08", "ces exercises sont trop long et ennuyeux",
                 ["Ces exercices sont trop longs et ennuyeux."], "ACCORD", "accord_adj", "CM2", 2),
    CasBenchmark("V_ACC_09", "les valises sont lourd et encombrant",
                 ["Les valises sont lourdes et encombrantes."], "ACCORD", "accord_adj", "6e", 2),
    # pp_etre + attr (3)
    CasBenchmark("V_ACC_10", "les garcons sont arrive les premiers",
                 ["Les garçons sont arrivés les premiers."], "ACCORD", "accord_pp_etre", "CM2", 2),
    CasBenchmark("V_ACC_11", "elle est parti chercher du pain",
                 ["Elle est partie chercher du pain."], "ACCORD", "accord_pp_etre", "CM1"),
    CasBenchmark("V_ACC_12", "les maisons sont ancien et joli dans ce village",
                 ["Les maisons sont anciennes et jolies dans ce village."], "ACCORD", "accord_attr", "6e", 2),
]


# ═══════════════════════════════════════════════════════════════════════════
#  CONJ — Conjugaison (8 phrases)
# ═══════════════════════════════════════════════════════════════════════════

CONJ_VAL = [
    # present (2)
    CasBenchmark("V_CONJ_01", "tu chante tres bien cette chanson",
                 ["Tu chantes très bien cette chanson."], "CONJ", "conj_present", "CE2"),
    CasBenchmark("V_CONJ_02", "les animaux mange leur nourriture le matin",
                 ["Les animaux mangent leur nourriture le matin."], "CONJ", "conj_present", "CM1"),
    # imparfait (2)
    CasBenchmark("V_CONJ_03", "nous jouions et nous chantions pendant la recreation",
                 ["Nous jouions et nous chantions pendant la récréation."], "CONJ", "conj_imparfait", "CM2"),
    CasBenchmark("V_CONJ_04", "les oiseaux chantait dans le jardin ce matin-la",
                 ["Les oiseaux chantaient dans le jardin ce matin-là."], "CONJ", "conj_imparfait", "CM2"),
    # futur (2)
    CasBenchmark("V_CONJ_05", "nous chanterons et nous danserons a la fete",
                 ["Nous chanterons et nous danserons à la fête."], "CONJ", "conj_futur", "CM2", 2),
    CasBenchmark("V_CONJ_06", "les invites arrivera vers huit heures ce soir",
                 ["Les invités arriveront vers huit heures ce soir."], "CONJ", "conj_futur", "CM2", 2),
    # pp_inf (1)
    CasBenchmark("V_CONJ_07", "elle a chanter toute la soiree hier",
                 ["Elle a chanté toute la soirée hier."], "CONJ", "conj_pp_inf", "CM1", 2),
    # irregulier (1)
    CasBenchmark("V_CONJ_08", "les eleves allent a la bibliotheque le mardi",
                 ["Les élèves vont à la bibliothèque le mardi."], "CONJ", "conj_irregulier", "CM1", 2),
]


# ═══════════════════════════════════════════════════════════════════════════
#  HOMO — Homophones (6 phrases)
# ═══════════════════════════════════════════════════════════════════════════

HOMO_VAL = [
    CasBenchmark("V_HOMO_01", "elle et contente de son resultat",
                 ["Elle est contente de son résultat."], "HOMO", "homo_et_est", "CE2", 2),
    CasBenchmark("V_HOMO_02", "il a un velo a la maison",
                 ["Il a un vélo à la maison."], "HOMO", "homo_a_a", "CE2"),
    CasBenchmark("V_HOMO_03", "ils son arrives en retard a l'ecole",
                 ["Ils sont arrivés en retard à l'école."], "HOMO", "homo_son_sont", "CM1", 2),
    CasBenchmark("V_HOMO_04", "on mange ce que les parents on prepare",
                 ["On mange ce que les parents ont préparé."], "HOMO", "homo_on_ont", "CM1", 2),
    CasBenchmark("V_HOMO_05", "la ville ou je suis ne est tres jolie",
                 ["La ville où je suis né est très jolie."], "HOMO", "homo_autres", "CM2"),
    CasBenchmark("V_HOMO_06", "se matin la il faisait froid dehors",
                 ["Ce matin-là il faisait froid dehors."], "HOMO", "homo_autres", "CM2"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  AUTRE — Erreurs diverses (6 phrases)
# ═══════════════════════════════════════════════════════════════════════════

AUTRE_VAL = [
    # negation (2)
    CasBenchmark("V_AUTRE_01", "elle veut pas aller au dentiste demain",
                 ["Elle ne veut pas aller au dentiste demain."], "AUTRE", "autre_negation", "CE2"),
    CasBenchmark("V_AUTRE_02", "ils savent pas ou aller pour les vacances",
                 ["Ils ne savent pas où aller pour les vacances."], "AUTRE", "autre_negation", "CM1"),
    # ordre (2)
    CasBenchmark("V_AUTRE_03", "souvent les enfants jouent dans le parc",
                 ["Les enfants jouent souvent dans le parc.",
                  "Souvent les enfants jouent dans le parc."], "AUTRE", "autre_ordre", "CM1"),
    CasBenchmark("V_AUTRE_04", "vite il court pour attraper le bus",
                 ["Il court vite pour attraper le bus."], "AUTRE", "autre_ordre", "CE2"),
    # semantique/ponctuation (2)
    CasBenchmark("V_AUTRE_05", "les pierres chantent dans la foret magique",
                 ["Les pierres chantent dans la forêt magique."], "AUTRE", "autre_semantique", "CM2"),
    CasBenchmark("V_AUTRE_06", "papa a dit on part bientot en voyage",
                 ["Papa a dit : « On part bientôt en voyage. »",
                  "Papa a dit « on part bientôt en voyage »."],
                 "AUTRE", "autre_ponctuation", "6e"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  OK — Phrases correctes / pieges (8 phrases)
# ═══════════════════════════════════════════════════════════════════════════

OK_VAL = [
    # piege homo (2)
    CasBenchmark("V_OK_01", "elle est gentille et il est content",
                 ["Elle est gentille et il est content."], "OK", "ok_piege_homo", "CE2", 0),
    CasBenchmark("V_OK_02", "on a mange et on a bu du jus",
                 ["On a mangé et on a bu du jus."], "OK", "ok_piege_homo", "CM1", 0),
    # piege accord (2)
    CasBenchmark("V_OK_03", "les enfants de ma voisine jouent dehors",
                 ["Les enfants de ma voisine jouent dehors."], "OK", "ok_piege_accord", "CM1", 0),
    CasBenchmark("V_OK_04", "le directeur des ecoles visite notre classe",
                 ["Le directeur des écoles visite notre classe."], "OK", "ok_piege_accord", "CM2", 0),
    # complexe (2)
    CasBenchmark("V_OK_05", "les eleves qui travaillent bien auront une recompense",
                 ["Les élèves qui travaillent bien auront une récompense."], "OK", "ok_complexe", "6e", 0),
    CasBenchmark("V_OK_06", "il faut que tu fasses attention en traversant la rue",
                 ["Il faut que tu fasses attention en traversant la rue."], "OK", "ok_complexe", "6e", 0),
    # simple (2)
    CasBenchmark("V_OK_07", "nous allons au cinema ce soir",
                 ["Nous allons au cinéma ce soir."], "OK", "ok_simple", "CE2", 0),
    CasBenchmark("V_OK_08", "le soleil brille et les oiseaux chantent",
                 ["Le soleil brille et les oiseaux chantent."], "OK", "ok_simple", "CE2", 0),
]


# ═══════════════════════════════════════════════════════════════════════════
#  CORPUS COMPLET VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

CORPUS_VALIDATION: list[CasBenchmark] = (
    ORTH_VAL + ACCORD_VAL + CONJ_VAL + HOMO_VAL + AUTRE_VAL + OK_VAL
)

CATEGORIES_VALIDATION = ["ORTH", "ACCORD", "CONJ", "HOMO", "AUTRE", "OK"]


def _verifier_corpus_validation() -> None:
    """Verification de coherence du corpus de validation."""
    ids = [c.id for c in CORPUS_VALIDATION]
    assert len(ids) == len(set(ids)), f"IDs dupliques : {[x for x in ids if ids.count(x) > 1]}"
    assert len(CORPUS_VALIDATION) == 60, f"Attendu 60 phrases, obtenu {len(CORPUS_VALIDATION)}"
    for c in CORPUS_VALIDATION:
        assert c.categorie in CATEGORIES_VALIDATION, f"{c.id}: categorie inconnue '{c.categorie}'"
        assert isinstance(c.attendue, list) and len(c.attendue) >= 1, f"{c.id}: attendue doit etre une liste non vide"


_verifier_corpus_validation()
