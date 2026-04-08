"""Corpus de benchmark GEC debiaise pour l'evaluation multi-outils.

Distribution basee sur les etudes empiriques des erreurs CE2-6e :
- ORTH  35%  (42 phrases) : accents, phonetique, typos, doubles, invariables
- ACCORD 22% (26 phrases) : det-nom, sujet-verbe, adjectif, PP etre, attribut
- CONJ  13%  (16 phrases) : present, imparfait, futur, PP/infinitif, irregulier
- HOMO   9%  (11 phrases) : et/est, a/a, son/sont, on/ont, autres
- AUTRE  9%  (11 phrases) : negation, ordre, semantique, ponctuation
- OK    12%  (14 phrases) : pieges homophones, pieges accords, complexes, simples

Total : 120 phrases, dont ~20% multi-erreurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CasBenchmark:
    """Un cas de test pour le benchmark GEC."""

    id: str
    erronee: str
    attendue: list[str]
    categorie: str
    sous_categorie: str
    niveau: str
    nb_erreurs: int = 1


# ═══════════════════════════════════════════════════════════════════════════
#  ORTH — Orthographe (42 phrases)
# ═══════════════════════════════════════════════════════════════════════════

# --- orth_accent (10) ---
ORTH_ACCENT = [
    CasBenchmark("ORTH_01", "l'ecole est fermee aujourd'hui",
                 ["L'école est fermée aujourd'hui."], "ORTH", "orth_accent", "CE2"),
    CasBenchmark("ORTH_02", "ma mere a prepare le diner",
                 ["Ma mère a préparé le dîner."], "ORTH", "orth_accent", "CE2", 3),
    CasBenchmark("ORTH_03", "il etait tres fatigue apres la recreation",
                 ["Il était très fatigué après la récréation."], "ORTH", "orth_accent", "CM1", 4),
    CasBenchmark("ORTH_04", "nous preferons les crepes au sucre",
                 ["Nous préférons les crêpes au sucre."], "ORTH", "orth_accent", "CM1", 2),
    CasBenchmark("ORTH_05", "le telephone a sonne pendant la lecon",
                 ["Le téléphone a sonné pendant la leçon."], "ORTH", "orth_accent", "CM2", 3),
    CasBenchmark("ORTH_06", "la foret etait mysterieuse et sombre",
                 ["La forêt était mystérieuse et sombre."], "ORTH", "orth_accent", "CM2", 3),
    CasBenchmark("ORTH_07", "le dejeuner sera pret a midi",
                 ["Le déjeuner sera prêt à midi."], "ORTH", "orth_accent", "CM1", 3),
    CasBenchmark("ORTH_08", "j'ai achete un gateau a la boulangerie",
                 ["J'ai acheté un gâteau à la boulangerie."], "ORTH", "orth_accent", "CE2", 3),
    CasBenchmark("ORTH_09", "l'ete dernier nous avons visite un chateau",
                 ["L'été dernier nous avons visité un château."], "ORTH", "orth_accent", "CM2", 3),
    CasBenchmark("ORTH_10", "l'eleve a reussi son controle de francais",
                 ["L'élève a réussi son contrôle de français."], "ORTH", "orth_accent", "CM2", 4),
]

# --- orth_phonetique (8) ---
ORTH_PHONETIQUE = [
    CasBenchmark("ORTH_11", "je vais a la pissine avec mes copains",
                 ["Je vais à la piscine avec mes copains."], "ORTH", "orth_phonetique", "CE2"),
    CasBenchmark("ORTH_12", "le maittre a ecrit au tablo",
                 ["Le maître a écrit au tableau."], "ORTH", "orth_phonetique", "CE2", 2),
    CasBenchmark("ORTH_13", "le cocolat chaud est mon boisson preferee",
                 ["Le chocolat chaud est ma boisson préférée."], "ORTH", "orth_phonetique", "CE2", 2),
    CasBenchmark("ORTH_14", "les zanimo du zoo sont magnifiques",
                 ["Les animaux du zoo sont magnifiques."], "ORTH", "orth_phonetique", "CE2"),
    CasBenchmark("ORTH_15", "on a feter mon aniversaire samedi",
                 ["On a fêté mon anniversaire samedi."], "ORTH", "orth_phonetique", "CM1", 2),
    CasBenchmark("ORTH_16", "le dinausor etait un animal enormme",
                 ["Le dinosaure était un animal énorme."], "ORTH", "orth_phonetique", "CM1", 3),
    CasBenchmark("ORTH_17", "la geografie et les matematic sont difficiles",
                 ["La géographie et les mathématiques sont difficiles."], "ORTH", "orth_phonetique", "CM2", 2),
    CasBenchmark("ORTH_18", "le farmacien a donné les medicamants",
                 ["Le pharmacien a donné les médicaments."], "ORTH", "orth_phonetique", "CM2", 2),
]

# --- orth_typo (6) ---
ORTH_TYPO = [
    CasBenchmark("ORTH_19", "je suispas content de mon resultat",
                 ["Je ne suis pas content de mon résultat."], "ORTH", "orth_typo", "CM1", 2),
    CasBenchmark("ORTH_20", "le chat drot sur le canape",
                 ["Le chat dort sur le canapé."], "ORTH", "orth_typo", "CE2", 2),
    CasBenchmark("ORTH_21", "ils jounet dans la cour de l'ecole",
                 ["Ils jouent dans la cour de l'école."], "ORTH", "orth_typo", "CE2", 2),
    CasBenchmark("ORTH_22", "le solei brille dans le ciel bleu",
                 ["Le soleil brille dans le ciel bleu."], "ORTH", "orth_typo", "CE2"),
    CasBenchmark("ORTH_23", "nous avosn mange a la cantine",
                 ["Nous avons mangé à la cantine."], "ORTH", "orth_typo", "CM1", 2),
    CasBenchmark("ORTH_24", "elle regarde la televisoin chaque soir",
                 ["Elle regarde la télévision chaque soir."], "ORTH", "orth_typo", "CM1", 2),
]

# --- orth_double (5) ---
ORTH_DOUBLE = [
    CasBenchmark("ORTH_25", "nous aportons notre gouter a l'ecole",
                 ["Nous apportons notre goûter à l'école."], "ORTH", "orth_double", "CM1", 2),
    CasBenchmark("ORTH_26", "les pommes sont excelentes cette annee",
                 ["Les pommes sont excellentes cette année."], "ORTH", "orth_double", "CM2", 2),
    CasBenchmark("ORTH_27", "mon adresse est dificile a retenir",
                 ["Mon adresse est difficile à retenir."], "ORTH", "orth_double", "CM1"),
    CasBenchmark("ORTH_28", "la maitresse a corrigé nos redactions rapidemment",
                 ["La maîtresse a corrigé nos rédactions rapidement."], "ORTH", "orth_double", "CM2", 2),
    CasBenchmark("ORTH_29", "le balon a atteri dans le jardin",
                 ["Le ballon a atterri dans le jardin."], "ORTH", "orth_double", "CM1", 2),
]

# --- orth_invariable (5) ---
ORTH_INVARIABLE = [
    CasBenchmark("ORTH_30", "il marche toutjours tres vite",
                 ["Il marche toujours très vite."], "ORTH", "orth_invariable", "CE2", 2),
    CasBenchmark("ORTH_31", "nous sommes partis enssemble au parc",
                 ["Nous sommes partis ensemble au parc."], "ORTH", "orth_invariable", "CM1"),
    CasBenchmark("ORTH_32", "maintenent je sais lire et ecrire",
                 ["Maintenant je sais lire et écrire."], "ORTH", "orth_invariable", "CE2", 2),
    CasBenchmark("ORTH_33", "je viendrai peut etre demain",
                 ["Je viendrai peut-être demain."], "ORTH", "orth_invariable", "CM2"),
    CasBenchmark("ORTH_34", "il fait beacoup de sport le mercredi",
                 ["Il fait beaucoup de sport le mercredi."], "ORTH", "orth_invariable", "CE2"),
]

# --- orth_graphie (4) ---
ORTH_GRAPHIE = [
    CasBenchmark("ORTH_35", "ma seur joue du piano le mercredi",
                 ["Ma sœur joue du piano le mercredi."], "ORTH", "orth_graphie", "CM1"),
    CasBenchmark("ORTH_36", "les oeux de mon chat sont verts",
                 ["Les yeux de mon chat sont verts."], "ORTH", "orth_graphie", "CE2"),
    CasBenchmark("ORTH_37", "nous avons appris les siences naturelles",
                 ["Nous avons appris les sciences naturelles."], "ORTH", "orth_graphie", "CM2"),
    CasBenchmark("ORTH_38", "le kirurgien a opere mon grand-pere",
                 ["Le chirurgien a opéré mon grand-père."], "ORTH", "orth_graphie", "6e", 2),
]

# --- orth_segmentation (4) ---
ORTH_SEGMENTATION = [
    CasBenchmark("ORTH_39", "parce que il fait beau on va dehors",
                 ["Parce qu'il fait beau on va dehors."], "ORTH", "orth_segmentation", "CE2"),
    CasBenchmark("ORTH_40", "il y avait beau coup de monde au marche",
                 ["Il y avait beaucoup de monde au marché."], "ORTH", "orth_segmentation", "CM1", 2),
    CasBenchmark("ORTH_41", "aujourdhui nous allons au cinema",
                 ["Aujourd'hui nous allons au cinéma."], "ORTH", "orth_segmentation", "CM1", 2),
    CasBenchmark("ORTH_42", "la maitresse nous a dit de travailler en semble",
                 ["La maîtresse nous a dit de travailler ensemble."], "ORTH", "orth_segmentation", "CM2", 2),
]


# ═══════════════════════════════════════════════════════════════════════════
#  ACCORD — Accords (26 phrases)
# ═══════════════════════════════════════════════════════════════════════════

# --- accord_det_nom (6) ---
ACCORD_DET_NOM = [
    CasBenchmark("ACC_01", "les chien courent dans le jardin",
                 ["Les chiens courent dans le jardin."], "ACCORD", "accord_det_nom", "CE2"),
    CasBenchmark("ACC_02", "j'ai mange trois pomme et deux banane",
                 ["J'ai mangé trois pommes et deux bananes."], "ACCORD", "accord_det_nom", "CE2", 2),
    CasBenchmark("ACC_03", "les oiseaux chantent dans les arbre",
                 ["Les oiseaux chantent dans les arbres."], "ACCORD", "accord_det_nom", "CE2"),
    CasBenchmark("ACC_04", "nous avons lu des livre passionnant",
                 ["Nous avons lu des livres passionnants."], "ACCORD", "accord_det_nom", "CM1", 2),
    CasBenchmark("ACC_05", "les feuille des arbre tombent en automne",
                 ["Les feuilles des arbres tombent en automne."], "ACCORD", "accord_det_nom", "CM1", 2),
    CasBenchmark("ACC_06", "mes parent sont parti en vacance",
                 ["Mes parents sont partis en vacances."], "ACCORD", "accord_det_nom", "CM2", 3),
]

# --- accord_sv (6) ---
ACCORD_SV = [
    CasBenchmark("ACC_07", "les enfants mange a la cantine",
                 ["Les enfants mangent à la cantine."], "ACCORD", "accord_sv", "CE2"),
    CasBenchmark("ACC_08", "mes amis joue dans la cour",
                 ["Mes amis jouent dans la cour."], "ACCORD", "accord_sv", "CE2"),
    CasBenchmark("ACC_09", "les oiseaux chante quand le soleil se leve",
                 ["Les oiseaux chantent quand le soleil se lève."], "ACCORD", "accord_sv", "CM1", 2),
    CasBenchmark("ACC_10", "les voitures roule vite sur l'autoroute",
                 ["Les voitures roulent vite sur l'autoroute."], "ACCORD", "accord_sv", "CM1"),
    CasBenchmark("ACC_11", "nous joue au ballon dans le parc",
                 ["Nous jouons au ballon dans le parc."], "ACCORD", "accord_sv", "CE2"),
    CasBenchmark("ACC_12", "les eleves travaille bien cette annee",
                 ["Les élèves travaillent bien cette année."], "ACCORD", "accord_sv", "CM2", 2),
]

# --- accord_adj (5) ---
ACCORD_ADJ = [
    CasBenchmark("ACC_13", "les fleurs sont tres belle dans le jardin",
                 ["Les fleurs sont très belles dans le jardin."], "ACCORD", "accord_adj", "CM1"),
    CasBenchmark("ACC_14", "ces histoires sont passionnant et drole",
                 ["Ces histoires sont passionnantes et drôles."], "ACCORD", "accord_adj", "CM2", 3),
    CasBenchmark("ACC_15", "les garcons sont content de leur resultat",
                 ["Les garçons sont contents de leur résultat."], "ACCORD", "accord_adj", "CM1", 2),
    CasBenchmark("ACC_16", "j'ai vu des etoiles brillant dans le ciel",
                 ["J'ai vu des étoiles brillantes dans le ciel."], "ACCORD", "accord_adj", "CM2", 2),
    CasBenchmark("ACC_17", "les maisons sont tout blanc avec des volets bleu",
                 ["Les maisons sont toutes blanches avec des volets bleus."], "ACCORD", "accord_adj", "6e", 3),
]

# --- accord_pp_etre (5) ---
ACCORD_PP_ETRE = [
    CasBenchmark("ACC_18", "les filles sont arrive en retard",
                 ["Les filles sont arrivées en retard."], "ACCORD", "accord_pp_etre", "CM2"),
    CasBenchmark("ACC_19", "ma soeur est parti a l'ecole ce matin",
                 ["Ma sœur est partie à l'école ce matin."], "ACCORD", "accord_pp_etre", "CM2", 3),
    CasBenchmark("ACC_20", "elles sont tombe dans la cour",
                 ["Elles sont tombées dans la cour."], "ACCORD", "accord_pp_etre", "CM1"),
    CasBenchmark("ACC_21", "les enfants sont sorti jouer dehors",
                 ["Les enfants sont sortis jouer dehors."], "ACCORD", "accord_pp_etre", "CM2"),
    CasBenchmark("ACC_22", "nous sommes alle au cinema hier soir",
                 ["Nous sommes allés au cinéma hier soir."], "ACCORD", "accord_pp_etre", "CM2", 2),
]

# --- accord_attr (4) ---
ACCORD_ATTR = [
    CasBenchmark("ACC_23", "les filles sont intelligent et serieuse",
                 ["Les filles sont intelligentes et sérieuses."], "ACCORD", "accord_attr", "CM2", 2),
    CasBenchmark("ACC_24", "cette robe est trop petit pour moi",
                 ["Cette robe est trop petite pour moi."], "ACCORD", "accord_attr", "CM1"),
    CasBenchmark("ACC_25", "les gateaux sont delicieux et bien cuit",
                 ["Les gâteaux sont délicieux et bien cuits."], "ACCORD", "accord_attr", "6e", 2),
    CasBenchmark("ACC_26", "la mer est calme et bleu aujourd'hui",
                 ["La mer est calme et bleue aujourd'hui."], "ACCORD", "accord_attr", "CM2"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  CONJ — Conjugaison (16 phrases)
# ═══════════════════════════════════════════════════════════════════════════

# --- conj_present (4) ---
CONJ_PRESENT = [
    CasBenchmark("CONJ_01", "je mange et je bois du jus",
                 ["Je mange et je bois du jus."], "CONJ", "conj_present", "CE2"),
    CasBenchmark("CONJ_02", "les enfants prend le bus chaque matin",
                 ["Les enfants prennent le bus chaque matin."], "CONJ", "conj_present", "CM1"),
    CasBenchmark("CONJ_03", "nous fesons nos devoirs apres l'ecole",
                 ["Nous faisons nos devoirs après l'école."], "CONJ", "conj_present", "CM1", 2),
    CasBenchmark("CONJ_04", "les voisins disent qu'ils veulent demenager",
                 ["Les voisins disent qu'ils veulent déménager."], "CONJ", "conj_present", "CM2"),
]

# --- conj_imparfait (4) ---
CONJ_IMPARFAIT = [
    CasBenchmark("CONJ_05", "quand j'etais petit je jouait tout le temps",
                 ["Quand j'étais petit je jouais tout le temps."], "CONJ", "conj_imparfait", "CM1", 2),
    CasBenchmark("CONJ_06", "les animaux dormaient tranquillement",
                 ["Les animaux dormaient tranquillement."], "CONJ", "conj_imparfait", "CM1"),
    CasBenchmark("CONJ_07", "nous chantions et nous dansion pendant la fete",
                 ["Nous chantions et nous dansions pendant la fête."], "CONJ", "conj_imparfait", "CM2", 2),
    CasBenchmark("CONJ_08", "il faisait beau et les gens se promenais dans le parc",
                 ["Il faisait beau et les gens se promenaient dans le parc."], "CONJ", "conj_imparfait", "CM2"),
]

# --- conj_futur (3) ---
CONJ_FUTUR = [
    CasBenchmark("CONJ_09", "demain je travaillera toute la journee",
                 ["Demain je travaillerai toute la journée."], "CONJ", "conj_futur", "CM2", 2),
    CasBenchmark("CONJ_10", "les eleves passerons un examen vendredi",
                 ["Les élèves passeront un examen vendredi."], "CONJ", "conj_futur", "CM2", 2),
    CasBenchmark("CONJ_11", "nous partiron en vacances au mois de juillet",
                 ["Nous partirons en vacances au mois de juillet."], "CONJ", "conj_futur", "CM2"),
]

# --- conj_pp_inf (3) ---
CONJ_PP_INF = [
    CasBenchmark("CONJ_12", "j'ai manger une pomme apres l'ecole",
                 ["J'ai mangé une pomme après l'école."], "CONJ", "conj_pp_inf", "CM1", 2),
    CasBenchmark("CONJ_13", "il faut ecouté le professeur en classe",
                 ["Il faut écouter le professeur en classe."], "CONJ", "conj_pp_inf", "CM2", 2),
    CasBenchmark("CONJ_14", "les enfants veulent jouer mais ils doivent travaillé",
                 ["Les enfants veulent jouer mais ils doivent travailler."], "CONJ", "conj_pp_inf", "CM2"),
]

# --- conj_irregulier (2) ---
CONJ_IRREGULIER = [
    CasBenchmark("CONJ_15", "ils allent a la plage tous les etes",
                 ["Ils vont à la plage tous les étés."], "CONJ", "conj_irregulier", "CM1", 2),
    CasBenchmark("CONJ_16", "les enfants faisent trop de bruit en classe",
                 ["Les enfants font trop de bruit en classe."], "CONJ", "conj_irregulier", "CM1"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  HOMO — Homophones (11 phrases)
# ═══════════════════════════════════════════════════════════════════════════

# --- homo_et_est (3) ---
HOMO_ET_EST = [
    CasBenchmark("HOMO_01", "le chat et noir est blanc",
                 ["Le chat est noir et blanc."], "HOMO", "homo_et_est", "CE2"),
    CasBenchmark("HOMO_02", "ma soeur est grande et elle et gentille",
                 ["Ma sœur est grande et elle est gentille."], "HOMO", "homo_et_est", "CE2", 2),
    CasBenchmark("HOMO_03", "il et parti est il reviendra demain",
                 ["Il est parti et il reviendra demain."], "HOMO", "homo_et_est", "CM1", 2),
]

# --- homo_a_a (2) ---
HOMO_A_A = [
    CasBenchmark("HOMO_04", "il a un chat a la maison",
                 ["Il a un chat à la maison."], "HOMO", "homo_a_a", "CE2"),
    CasBenchmark("HOMO_05", "elle à mange a la cantine avec ses amies",
                 ["Elle a mangé à la cantine avec ses amies."], "HOMO", "homo_a_a", "CM1", 2),
]

# --- homo_son_sont (2) ---
HOMO_SON_SONT = [
    CasBenchmark("HOMO_06", "les enfants sont fatigues et son frere aussi",
                 ["Les enfants sont fatigués et son frère aussi."], "HOMO", "homo_son_sont", "CM1"),
    CasBenchmark("HOMO_07", "ils son partis avec sont velo",
                 ["Ils sont partis avec son vélo."], "HOMO", "homo_son_sont", "CM2", 2),
]

# --- homo_on_ont (2) ---
HOMO_ON_ONT = [
    CasBenchmark("HOMO_08", "on mange ce que les cuisiniers on prepare",
                 ["On mange ce que les cuisiniers ont préparé."], "HOMO", "homo_on_ont", "CM1", 2),
    CasBenchmark("HOMO_09", "ils ont joue et ont a gagne le match",
                 ["Ils ont joué et on a gagné le match."], "HOMO", "homo_on_ont", "CM2", 2),
]

# --- homo_autres (2) ---
HOMO_AUTRES = [
    CasBenchmark("HOMO_10", "se garcon la mange sa pomme",
                 ["Ce garçon-là mange sa pomme."], "HOMO", "homo_autres", "CM2", 2),
    CasBenchmark("HOMO_11", "ou vas-tu cet apres-midi ou demain",
                 ["Où vas-tu cet après-midi ou demain."],
                 "HOMO", "homo_autres", "6e"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  AUTRE — Erreurs diverses (11 phrases)
# ═══════════════════════════════════════════════════════════════════════════

# --- autre_negation (3) ---
AUTRE_NEGATION = [
    CasBenchmark("AUTRE_01", "il veut pas venir avec nous au parc",
                 ["Il ne veut pas venir avec nous au parc."], "AUTRE", "autre_negation", "CE2"),
    CasBenchmark("AUTRE_02", "je sais pas ou il habite",
                 ["Je ne sais pas où il habite."], "AUTRE", "autre_negation", "CM1"),
    CasBenchmark("AUTRE_03", "elle mange jamais de legumes a la cantine",
                 ["Elle ne mange jamais de légumes à la cantine."], "AUTRE", "autre_negation", "CM1", 2),
]

# --- autre_ordre (3) ---
AUTRE_ORDRE = [
    CasBenchmark("AUTRE_04", "bien il chante tres pour son age",
                 ["Il chante très bien pour son âge."], "AUTRE", "autre_ordre", "CE2", 2),
    CasBenchmark("AUTRE_05", "toujours elle arrive en retard a l'ecole",
                 ["Elle arrive toujours en retard à l'école."], "AUTRE", "autre_ordre", "CM1", 2),
    CasBenchmark("AUTRE_06", "rapide tres le chien court dans le jardin",
                 ["Le chien court très rapide dans le jardin.",
                  "Le chien court très rapidement dans le jardin."],
                 "AUTRE", "autre_ordre", "CM2", 2),
]

# --- autre_semantique (2) ---
AUTRE_SEMANTIQUE = [
    CasBenchmark("AUTRE_07", "le chien a mange une chaise au petit dejeuner",
                 ["Le chien a mangé une chaise au petit déjeuner."],
                 "AUTRE", "autre_semantique", "CM1"),
    CasBenchmark("AUTRE_08", "les poissons volent dans le ciel bleu",
                 ["Les poissons volent dans le ciel bleu."],
                 "AUTRE", "autre_semantique", "CM2"),
]

# --- autre_ponctuation (3) ---
AUTRE_PONCTUATION = [
    CasBenchmark("AUTRE_09", "bonjour comment vas tu aujourd'hui",
                 ["Bonjour, comment vas-tu aujourd'hui ?",
                  "Bonjour comment vas-tu aujourd'hui ?"],
                 "AUTRE", "autre_ponctuation", "CM1"),
    CasBenchmark("AUTRE_10", "j'aime les pommes les poires et les bananes",
                 ["J'aime les pommes, les poires et les bananes."],
                 "AUTRE", "autre_ponctuation", "CM2"),
    CasBenchmark("AUTRE_11", "elle a dit je viendrai demain",
                 ["Elle a dit : « Je viendrai demain. »",
                  "Elle a dit « je viendrai demain »."],
                 "AUTRE", "autre_ponctuation", "6e"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  OK — Phrases correctes / pieges (14 phrases)
# ═══════════════════════════════════════════════════════════════════════════

# --- ok_piege_homo (4) ---
OK_PIEGE_HOMO = [
    CasBenchmark("OK_01", "il est grand et elle est petite",
                 ["Il est grand et elle est petite."], "OK", "ok_piege_homo", "CE2", 0),
    CasBenchmark("OK_02", "son frere a un chat a la maison",
                 ["Son frère a un chat à la maison."], "OK", "ok_piege_homo", "CE2", 0),
    CasBenchmark("OK_03", "on mange ce qu'on veut a la maison",
                 ["On mange ce qu'on veut à la maison."], "OK", "ok_piege_homo", "CM1", 0),
    CasBenchmark("OK_04", "ou vas-tu ce matin ou cet apres-midi",
                 ["Où vas-tu ce matin ou cet après-midi ?"], "OK", "ok_piege_homo", "CM2", 0),
]

# --- ok_piege_accord (4) ---
OK_PIEGE_ACCORD = [
    CasBenchmark("OK_05", "les enfants mangent des pommes rouges",
                 ["Les enfants mangent des pommes rouges."], "OK", "ok_piege_accord", "CE2", 0),
    CasBenchmark("OK_06", "le chat de mes voisins dort sur le mur",
                 ["Le chat de mes voisins dort sur le mur."], "OK", "ok_piege_accord", "CM1", 0),
    CasBenchmark("OK_07", "la maitresse nous a donné beaucoup de devoirs",
                 ["La maîtresse nous a donné beaucoup de devoirs."], "OK", "ok_piege_accord", "CM2", 0),
    CasBenchmark("OK_08", "les filles sont parties avant les garcons",
                 ["Les filles sont parties avant les garçons."], "OK", "ok_piege_accord", "CM2", 0),
]

# --- ok_complexe (3) ---
OK_COMPLEXE = [
    CasBenchmark("OK_09", "nous avons visité le château de Versailles pendant les vacances",
                 ["Nous avons visité le château de Versailles pendant les vacances."],
                 "OK", "ok_complexe", "6e", 0),
    CasBenchmark("OK_10", "les élèves qui ont réussi leur examen sont contents",
                 ["Les élèves qui ont réussi leur examen sont contents."],
                 "OK", "ok_complexe", "6e", 0),
    CasBenchmark("OK_11", "il faut que nous partions avant qu'il ne pleuve",
                 ["Il faut que nous partions avant qu'il ne pleuve."],
                 "OK", "ok_complexe", "6e", 0),
]

# --- ok_simple (3) ---
OK_SIMPLE = [
    CasBenchmark("OK_12", "le chat dort sur le canapé",
                 ["Le chat dort sur le canapé."], "OK", "ok_simple", "CE2", 0),
    CasBenchmark("OK_13", "nous allons à la plage demain",
                 ["Nous allons à la plage demain."], "OK", "ok_simple", "CE2", 0),
    CasBenchmark("OK_14", "ma mère prépare le repas du soir",
                 ["Ma mère prépare le repas du soir."], "OK", "ok_simple", "CM1", 0),
]


# ═══════════════════════════════════════════════════════════════════════════
#  CORPUS COMPLET
# ═══════════════════════════════════════════════════════════════════════════

CORPUS: list[CasBenchmark] = (
    ORTH_ACCENT + ORTH_PHONETIQUE + ORTH_TYPO + ORTH_DOUBLE
    + ORTH_INVARIABLE + ORTH_GRAPHIE + ORTH_SEGMENTATION
    + ACCORD_DET_NOM + ACCORD_SV + ACCORD_ADJ + ACCORD_PP_ETRE + ACCORD_ATTR
    + CONJ_PRESENT + CONJ_IMPARFAIT + CONJ_FUTUR + CONJ_PP_INF + CONJ_IRREGULIER
    + HOMO_ET_EST + HOMO_A_A + HOMO_SON_SONT + HOMO_ON_ONT + HOMO_AUTRES
    + AUTRE_NEGATION + AUTRE_ORDRE + AUTRE_SEMANTIQUE + AUTRE_PONCTUATION
    + OK_PIEGE_HOMO + OK_PIEGE_ACCORD + OK_COMPLEXE + OK_SIMPLE
)

CATEGORIES = ["ORTH", "ACCORD", "CONJ", "HOMO", "AUTRE", "OK"]

SOUS_CATEGORIES = sorted({cas.sous_categorie for cas in CORPUS})


def _verifier_corpus() -> None:
    """Verification de coherence du corpus."""
    ids = [c.id for c in CORPUS]
    assert len(ids) == len(set(ids)), f"IDs dupliques : {[x for x in ids if ids.count(x) > 1]}"
    assert len(CORPUS) == 120, f"Attendu 120 phrases, obtenu {len(CORPUS)}"
    for c in CORPUS:
        assert c.categorie in CATEGORIES, f"{c.id}: categorie inconnue '{c.categorie}'"
        assert isinstance(c.attendue, list) and len(c.attendue) >= 1, f"{c.id}: attendue doit etre une liste non vide"


_verifier_corpus()
