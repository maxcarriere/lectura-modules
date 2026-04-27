"""Constantes pour les regles grammaticales."""

from __future__ import annotations

# Determinants pluriels / singuliers
PLUR_DET = frozenset({
    "les", "des", "ces", "ses", "mes", "tes",
    "nos", "vos", "leurs", "aux", "plusieurs", "quelques",
    "certains", "certaines", "tous", "toutes",
    # Adjectifs indefinis pluriels
    "nombreux", "nombreuses", "différents", "différentes",
    "divers", "diverses",
    # Nombres ecrits (declenchent l'accord pluriel)
    "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
    "neuf", "dix", "onze", "douze", "treize", "quatorze",
    "quinze", "seize", "vingt", "trente", "quarante",
    "cinquante", "soixante", "cent", "mille",
    # Nombres composes courants
    "dix-sept", "dix-huit", "dix-neuf",
    "vingt-et-un", "vingt-deux", "vingt-trois",
    "trente-et-un", "trente-deux",
})

SING_FEM_DET = frozenset({
    "la", "une", "cette", "sa", "ma", "ta", "aucune",
})

SING_MASC_DET = frozenset({
    "le", "un", "ce", "cet", "son", "mon", "ton", "aucun",
})

SING_DET = SING_MASC_DET | SING_FEM_DET | frozenset({
    "l'", "l", "leur", "chaque", "du", "au",
})

DET_GENRE_MAP: dict[str, str] = {
    "le": "la", "la": "le",
    "un": "une", "une": "un",
    "ce": "cette", "cette": "ce",
    "mon": "ma", "ma": "mon",
    "ton": "ta", "ta": "ton",
    "son": "sa", "sa": "son",
}

# Prepositions (pour sauter les complements dans l'accord sujet-verbe)
PREPOSITIONS = frozenset({
    "de", "du", "à", "au", "pour", "par", "avec",
    "dans", "en", "sur", "sous", "chez", "sans",
    "entre", "vers", "contre", "lors",
})

# Pronoms sujets -> personne attendue
PRONOM_PERSONNE: dict[str, tuple[str, str]] = {
    # pronom -> (personne, nombre)
    "je": ("1", ""), "j'": ("1", ""),
    "tu": ("2", ""),
    "il": ("3", "s"), "elle": ("3", "s"), "on": ("3", "s"),
    "nous": ("1", "p"),
    "vous": ("2", "p"),
    "ils": ("3", "p"), "elles": ("3", "p"),
}

# Pronoms sujets 3e personne du pluriel
SUJETS_3PL = frozenset({"ils", "elles"})

# Mots invariables
INVARIABLES = frozenset({
    "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
    "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "vingt", "trente", "quarante", "cinquante", "soixante",
    "cent", "mille", "chose",
    # Mots terminant en s/x/z au singulier (pas de pluriel distinct)
    "bras", "corps", "temps", "poids", "voix", "choix",
    "noix", "croix", "fois", "bois", "mois", "fils",
    "pays", "souris", "avis", "repas", "tapis", "propos",
    "discours", "cours", "concours", "parcours", "secours", "recours",
    "sens", "flux", "embarras", "laps", "abus", "acces", "exces",
    "proces", "progres", "succes", "brebis", "perdrix", "index",
    "prix", "tas", "dos", "ras", "cas", "gaz", "riz", "nez",
    "bas", "gras", "biais", "relais", "palais", "marais",
    "taux", "faux", "houx",
    # Mots etrangers invariables en français
    "kanji", "hurricane", "tsunami", "bonus", "campus", "consensus",
    "terminus", "prospectus", "virus", "oasis", "atlas",
    # Mots deja au pluriel (latin, etc.)
    "minima", "maxima", "media",
    # Adverbes et mots invariables courants
    "plus", "moins", "très", "assez", "trop",
    # Mois (quasi-jamais pluralises)
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
    # PP en -is/-us invariables au singulier (mis, pris, etc.)
    "mis", "remis", "admis", "permis", "promis", "soumis", "commis",
    "pris", "compris", "appris", "repris", "surpris",
    "acquis", "requis", "inclus", "exclus",
})

# Copules conjuguees au singulier (pour depluralization attribut du sujet)
COPULES_SINGULIER = frozenset({
    "est", "était", "sera", "serait", "fut",
    "semble", "paraît", "parait", "devient", "reste", "demeure",
})

# Copules conjuguees au pluriel (pour accord attribut du sujet)
COPULES_PLURIEL = frozenset({
    "sont", "étaient", "seront", "seraient", "furent",
    "semblent", "paraissent", "deviennent", "restent", "demeurent",
})

# Verbes irreguliers : forme 3e sg -> forme 3e pl
# Formes fausses courantes de verbes irreguliers -> correction
IRREGULIERS_FORMES_FAUSSES: dict[str, str] = {
    "allent": "vont", "allet": "allez", "alle": "va",
    "faisent": "font", "faist": "fait",
    "prenent": "prennent",
    "peuve": "peuvent", "peuent": "peuvent",
    "voivent": "voient",
    "savont": "savent",
    "vienent": "viennent",
    # Formes correctes d'autres personnes, fausses en contexte 3pl
    "ai": "ont", "as": "ont",           # avoir P1s/P2s → P3p
    "es": "sont", "suis": "sont",       # etre P2s/P1s → P3p
    "sommes": "sont",                    # etre P1p → P3p
    "avons": "ont", "avez": "ont",      # avoir P1p/P2p → P3p
}

# Verbes irreguliers : forme 3e sg -> forme 3e pl
IRREGULIERS_3PL: dict[str, str] = {
    # P3s → P3p
    "va": "vont", "fait": "font", "a": "ont", "dit": "disent",
    "est": "sont", "veut": "veulent", "peut": "peuvent",
    "doit": "doivent", "sait": "savent", "vient": "viennent",
    "tient": "tiennent", "voit": "voient", "croit": "croient",
    "écrit": "écrivent", "lit": "lisent", "conduit": "conduisent",
    "boit": "boivent", "connaît": "connaissent", "naît": "naissent",
    "plaît": "plaisent",
    "dort": "dorment", "court": "courent", "meurt": "meurent",
    "sort": "sortent", "sent": "sentent", "part": "partent",
    "sert": "servent", "met": "mettent", "bat": "battent",
    "suit": "suivent", "vit": "vivent", "rit": "rient",
    "résout": "résolvent", "ouvre": "ouvrent",
    "offre": "offrent", "souffre": "souffrent",
    "couvre": "couvrent", "cueille": "cueillent",
    # P1s/P2s/P1p/P2p → P3p (irreguliers etre/avoir)
    "suis": "sont", "es": "sont",
    "sommes": "sont", "êtes": "sont",
    "avons": "ont", "avez": "ont",
    "ai": "ont", "as": "ont",
    # P1p/P2p → P3p (irreguliers modaux et 3e groupe)
    "devons": "doivent", "devez": "doivent",
    "pouvons": "peuvent", "pouvez": "peuvent",
    "voulons": "veulent", "voulez": "veulent",
    "savons": "savent", "savez": "savent",
    "tenons": "tiennent", "tenez": "tiennent",
    "venons": "viennent", "venez": "viennent",
    "voyons": "voient", "voyez": "voient",
    "croyons": "croient", "croyez": "croient",
    "buvons": "boivent", "buvez": "boivent",
    "mettons": "mettent", "mettez": "mettent",
    "battons": "battent", "battez": "battent",
    "suivons": "suivent", "suivez": "suivent",
    "vivons": "vivent", "vivez": "vivent",
    "prenons": "prennent", "prenez": "prennent",
    "maintenons": "maintiennent", "maintenez": "maintiennent",
    "obtenons": "obtiennent", "obtenez": "obtiennent",
    "devenons": "deviennent", "devenez": "deviennent",
    "revenons": "reviennent", "revenez": "reviennent",
    "provenons": "proviennent", "provenez": "proviennent",
    "convenons": "conviennent", "convenez": "conviennent",
    # P1s/P2s → P3p (irreguliers)
    "peux": "peuvent", "dois": "doivent", "veux": "veulent",
    "vaux": "valent", "sais": "savent",
    "deviens": "deviennent", "proviens": "proviennent",
    "suis": "sont",
}


def generer_candidats_3pl(mot: str) -> list[str]:
    """Genere des candidats 3e personne du pluriel a partir d'une forme verbale."""
    low = mot.lower()
    # Irreguliers en priorite absolue
    if low in IRREGULIERS_3PL:
        return [IRREGULIERS_3PL[low]]

    candidats: list[str] = []

    # --- A. Depuis P3s (radical identique, ajout -nt/-ent) ---
    # 1er groupe : -e -> -ent
    if low.endswith("e") and not low.endswith(("ent", "nt")):
        candidats.append(mot + "nt")
    # 2e groupe : -it -> -issent
    if low.endswith("it"):
        candidats.append(mot[:-2] + "issent")
    # 3e groupe dormir/sortir : -rt -> -rment (retirer le dernier char)
    if low.endswith("rt"):
        candidats.append(mot[:-1] + "ment")
    # 3e groupe prendre : -nd -> -nnent
    if low.endswith("nd"):
        candidats.append(mot[:-2] + "nnent")
    # 3e groupe battre/mettre : -t (apres voyelle) -> -tent
    if low.endswith("t") and not low.endswith(("nt", "rt", "it")):
        candidats.append(mot + "ent")
    # 3e groupe ouvrir/offrir : -re -> -rent
    if low.endswith("re"):
        candidats.append(mot[:-1] + "ent")

    # --- B. Depuis P1p (-ons → -ent) ---
    # "finissons" → "finissent", "arrivons" → "arrivent"
    if low.endswith("ons") and len(low) > 4:
        candidats.append(low[:-3] + "ent")

    # --- C. Depuis P2s (-es → -ent) ---
    # "utilises" → "utilisent", "offres" → "offrent"
    # Note: no "-tes" exclusion — lexique.existe() filters invalid candidates
    # (acceptes→acceptent ✓, faites→faitent ✗ lexique)
    if low.endswith("es") and len(low) > 3:
        candidats.append(low[:-2] + "ent")

    # --- D. Depuis P2p (-ez → -ent) ---
    # "adoptez" → "adoptent"
    if low.endswith("ez") and len(low) > 3:
        candidats.append(low[:-2] + "ent")

    return candidats


AUXILIAIRES = frozenset({
    "a", "ai", "as", "avons", "avez", "ont",
    "avait", "avais", "avaient", "avions", "aviez",
    "aura", "auras", "auront", "aurons", "aurez",
    "aurait", "aurais", "auraient", "aurions", "auriez",
    "est", "suis", "es", "sommes", "êtes", "sont",
    "était", "étais", "étaient", "étions", "étiez",
    "sera", "seras", "seront", "serons", "serez",
    "serait", "serais", "seraient", "serions", "seriez",
    "fut", "fût", "fus", "furent", "fûmes", "fûtes",
    # Subjonctif (passif : "qu'il soit reconnu", "soient attribuées")
    "soit", "soient", "sois", "soyons", "soyez",
    "ait", "aie", "aies", "aient", "ayons", "ayez",
    # Participe present (ayant publié, étant donné)
    "ayant", "étant",
    # Infinitifs (passif : "pour être expulsé", "peut être accompagné")
    "être", "avoir",
})

# Genre/nombre des pronoms sujets
PRONOM_GENRE: dict[str, tuple[str, str]] = {
    "je": ("Masc", "Sing"), "j'": ("Masc", "Sing"),
    "tu": ("Masc", "Sing"),
    "il": ("Masc", "Sing"), "on": ("Masc", "Sing"),
    "elle": ("Fem", "Sing"),
    "nous": ("Masc", "Plur"),
    "vous": ("Masc", "Plur"),
    "ils": ("Masc", "Plur"),
    "elles": ("Fem", "Plur"),
}

# Formes conjuguees d'etre
ETRE_FORMES = frozenset({
    "suis", "es", "est", "sommes", "êtes", "sont",
    "etais", "etait", "etions", "etiez", "etaient",
    "étais", "était", "étions", "étiez", "étaient",
    "serai", "seras", "sera", "serons", "serez", "seront",
    "serais", "serait", "serions", "seriez", "seraient",
    "fus", "fut", "fût", "fûmes", "fûtes", "furent",
    "soit", "soient", "sois", "soyons", "soyez",
    "être",  # infinitif (passif: "peut être pratiquée")
})

# Copules : etre + verbes d'etat
COPULES_ALL = ETRE_FORMES | frozenset({
    "semble", "sembles", "semblent", "semblait", "semblaient",
    "devient", "deviens", "deviennent", "devenait", "devenaient",
    "reste", "restes", "restent", "restait", "restaient",
    "parait", "paraît", "paraissent", "paraissait", "paraissaient",
    "demeure", "demeures", "demeurent", "demeurait", "demeuraient",
})

# Verbes modaux (requierent un infinitif apres)
MODAUX_FORMES = frozenset({
    "faut", "doit", "doivent", "devait", "devaient",
    "peut", "peuvent", "pouvait", "pouvaient",
    "veut", "veulent", "voulait", "voulaient",
    "sait", "savent", "savait", "savaient",
})

# Formes conjuguees d'aller (pour homophone -er/-e)
ALLER_FORMES = frozenset({
    "vais", "vas", "va", "allons", "allez", "vont",
    "irai", "iras", "ira", "irons", "irez", "iront",
    "allais", "allait", "allions", "alliez", "allaient",
})


def generer_candidats_1pl(mot: str) -> list[str]:
    """Genere des candidats 1re personne du pluriel.

    mange→mangeons, dort→dormons, finit→finissons.
    """
    low = mot.lower()
    candidats: list[str] = []
    # 1er groupe : -e → -eons
    if low.endswith("e") and not low.endswith("ons"):
        candidats.append(low[:-1] + "eons")
    # -ent → -ons (3pl→1pl)
    if low.endswith("ent") and len(low) > 3:
        candidats.append(low[:-3] + "ons")
    # -it → -issons (2e groupe)
    if low.endswith("it"):
        candidats.append(low[:-2] + "issons")
    # -t/-d → -ons (3e groupe dormir, prendre)
    if low.endswith(("t", "d")) and not low.endswith(("ent", "it")):
        candidats.append(low[:-1] + "ons")
    return candidats


def generer_candidats_2pl(mot: str) -> list[str]:
    """Genere des candidats 2e personne du pluriel.

    mange→mangez, dort→dormez, finit→finissez.
    """
    low = mot.lower()
    candidats: list[str] = []
    # 1er groupe : -e → -ez
    if low.endswith("e") and not low.endswith("ez"):
        candidats.append(low[:-1] + "ez")
    # -ent → -ez (3pl→2pl)
    if low.endswith("ent") and len(low) > 3:
        candidats.append(low[:-3] + "ez")
    # -it → -issez (2e groupe)
    if low.endswith("it"):
        candidats.append(low[:-2] + "issez")
    # -t/-d → -ez (3e groupe)
    if low.endswith(("t", "d")) and not low.endswith(("ent", "it")):
        candidats.append(low[:-1] + "ez")
    return candidats


def generer_candidats_singulier(mot: str, personne: str) -> list[str]:
    """Genere des candidats singulier a partir d'une forme pluriel.

    mangent→mange (P3), mangent→manges (P2), mangent→mange (P1).
    Gere aussi les verbes a changement de radical.
    """
    low = mot.lower()
    candidats: list[str] = []
    if low.endswith("ent") and len(low) > 3:
        radical = low[:-3]
        # Stem-changing verbs en priorite (plus specifiques)
        # 2e groupe : finissent → finit
        if radical.endswith("iss"):
            candidats.append(radical[:-2] + "t")
        # 3e groupe -iennent → -ient : appartiennent → appartient
        if radical.endswith("ienn"):
            candidats.append(radical[:-1] + "t")
        # 3e groupe -ennent → -end : comprennent → comprend
        if radical.endswith("enn"):
            candidats.append(radical[:-1] + "d")
        # 3e groupe -uisent → -uit : construisent → construit
        if radical.endswith("uis"):
            candidats.append(radical[:-1] + "t")
        # 3e groupe -aissent → -aît : disparaissent → disparaît
        if radical.endswith("aiss"):
            candidats.append(radical[:-3] + "ît")
        # 3e groupe -oivent → -oit : reçoivent → reçoit
        if radical.endswith("oiv"):
            candidats.append(radical[:-1] + "t")
        # Generiques
        if personne == "2":
            candidats.append(radical + "es")
        candidats.append(radical + "e")
        # 3e groupe : dorment → dort
        candidats.append(radical + "t")
    if low.endswith("es") and personne == "1":
        candidats.append(low[:-1])
    return candidats


PARTICIPES_IRREGULIERS: dict[str, str] = {
    "faire": "fait", "dire": "dit", "écrire": "écrit",
    "prendre": "pris", "mettre": "mis", "suivre": "suivi",
    "vivre": "vécu", "boire": "bu", "lire": "lu",
    "voir": "vu", "savoir": "su", "pouvoir": "pu",
    "vouloir": "voulu", "devoir": "dû", "recevoir": "reçu",
    "croire": "cru", "connaître": "connu", "naître": "né",
    "être": "été", "avoir": "eu", "tenir": "tenu",
    "venir": "venu", "mourir": "mort", "courir": "couru",
    "ouvrir": "ouvert", "offrir": "offert", "souffrir": "souffert",
    "couvrir": "couvert", "peindre": "peint", "craindre": "craint",
    "joindre": "joint", "conduire": "conduit", "construire": "construit",
    "produire": "produit", "rire": "ri", "plaire": "plu",
    "asseoir": "assis", "atteindre": "atteint",
}


def generer_candidats_participe(mot: str) -> list[str]:
    """Genere des candidats participe passe a partir d'un infinitif.

    manger→mangé, finir→fini, rendre→rendu, faire→fait.
    """
    low = mot.lower()
    # Irreguliers en priorite
    if low in PARTICIPES_IRREGULIERS:
        return [PARTICIPES_IRREGULIERS[low]]
    # Reguliers
    candidats: list[str] = []
    if low.endswith("er"):
        candidats.append(low[:-2] + "é")
    if low.endswith("ir"):
        candidats.append(low[:-2] + "i")
    if low.endswith("re"):
        candidats.append(low[:-2] + "u")
    return candidats


def generer_candidats_pluriel(mot: str) -> list[str]:
    """Genere les formes plurielles d'un mot (pas seulement +s)."""
    low = mot.lower()
    candidats: list[str] = []
    # -al → -aux (cheval→chevaux, journal→journaux)
    if low.endswith("al"):
        candidats.append(low[:-2] + "aux")
    # -eau → -eaux (gâteau→gâteaux, chapeau→chapeaux)
    if low.endswith("eau"):
        candidats.append(low + "x")
    # -au → -aux (noyau→noyaux, tuyau→tuyaux)
    if low.endswith("au") and not low.endswith("eau"):
        candidats.append(low + "x")
    # -eu → -eux (jeu→jeux) — sauf pneu, bleu
    if low.endswith("eu") and low not in ("pneu", "bleu"):
        candidats.append(low + "x")
    # Defaut : +s
    candidats.append(low + "s")
    return candidats


def generer_candidats_singulier_nom(mot: str) -> list[str]:
    """Genere des candidats singuliers a partir d'un NOM/ADJ pluriel.

    rencontres→rencontre, généraux→général, chapeaux→chapeau.
    """
    low = mot.lower()
    candidats: list[str] = []
    # -aux → -al (chevaux→cheval, généraux→général)
    if low.endswith("aux") and len(low) > 3:
        candidats.append(low[:-3] + "al")
    # -eaux → -eau (chapeaux→chapeau, gâteaux→gâteau)
    if low.endswith("eaux") and len(low) > 4:
        candidats.append(low[:-1])  # remove trailing x
    # -eux → -eu (jeux→jeu)
    if low.endswith("eux") and len(low) > 3:
        candidats.append(low[:-1])  # remove trailing x
    # Default: remove trailing -s
    if low.endswith("s") and not low.endswith("ss") and len(low) > 2:
        candidats.append(low[:-1])
    # Remove trailing -x (for -oux etc.)
    if low.endswith("x") and not low.endswith(("aux", "eux")) and len(low) > 2:
        candidats.append(low[:-1])
    return candidats


def generer_candidats_feminin(mot: str) -> list[str]:
    """Genere des candidats feminins a partir d'une forme masculine."""
    low = mot.lower()
    candidats: list[str] = []
    if not low.endswith("e"):
        candidats.append(low + "e")         # grand → grande
    if low.endswith("eux"):
        candidats.append(low[:-3] + "euse")  # heureux → heureuse
    if low.endswith("er"):
        candidats.append(low[:-2] + "ère")   # premier → première
    if low.endswith("f"):
        candidats.append(low[:-1] + "ve")    # actif → active
    if low.endswith("el"):
        candidats.append(low + "le")         # bel → belle
    if low.endswith("en"):
        candidats.append(low + "ne")         # ancien → ancienne
    if low.endswith("on"):
        candidats.append(low + "ne")         # bon → bonne
    if low.endswith("et") and not low.endswith("elet"):
        candidats.append(low + "te")         # net → nette, muet → muette
    if low.endswith("teur"):
        candidats.append(low[:-4] + "trice") # directeur → directrice
    if low.endswith("eur") and not low.endswith(("eux", "teur")):
        candidats.append(low[:-3] + "euse")  # danseur → danseuse
    if low.endswith("oux"):
        candidats.append(low[:-3] + "ousse")  # roux → rousse
        candidats.append(low[:-3] + "ouce")   # doux → douce
    if low.endswith("aux"):
        candidats.append(low[:-3] + "ausse")  # faux → fausse
    return candidats


def generer_candidats_masculin(mot: str) -> list[str]:
    """Genere des candidats masculins a partir d'une forme feminine.

    petite→petit, grosse→gros, verte→vert, grande→grand,
    heureuse→heureux, active→actif, belle→bel.
    """
    low = mot.lower()
    candidats: list[str] = []
    if low.endswith("sse"):
        candidats.append(low[:-2])           # grosse → gros
    if low.endswith("euse"):
        candidats.append(low[:-4] + "eux")   # heureuse → heureux
        candidats.append(low[:-4] + "eur")   # danseuse → danseur
    if low.endswith("ve"):
        candidats.append(low[:-2] + "f")     # active → actif
    if low.endswith("ère"):
        candidats.append(low[:-3] + "er")    # première → premier
    if low.endswith("lle"):
        candidats.append(low[:-2])           # belle → bel
    if low.endswith("nne"):
        candidats.append(low[:-2])           # bonne → bon
    if low.endswith("ette"):
        candidats.append(low[:-4] + "et")    # nette → net, muette → muet
    if low.endswith("trice"):
        candidats.append(low[:-5] + "teur")  # directrice → directeur
    if low.endswith("ente"):
        candidats.append(low[:-4] + "ent")   # absente → absent
    if low.endswith("ousse"):
        candidats.append(low[:-5] + "oux")    # rousse → roux
    if low.endswith("ouce"):
        candidats.append(low[:-4] + "oux")    # douce → doux
    if low.endswith("ausse"):
        candidats.append(low[:-5] + "aux")    # fausse → faux
    if low.endswith("e") and not low.endswith(("sse", "euse", "ve", "ère", "lle", "nne", "trice", "ousse", "ouce", "ausse")):
        candidats.append(low[:-1])           # petite → petit, verte → vert
    return candidats


# --- Mots transparents (ne changent pas l'analyse sujet-verbe) ---
_TRANSPARENTS = frozenset({
    "ne", "n'", "pas", "plus", "jamais", "rien",
    "point", "y", "en",
    # Pronoms reflexifs (entre sujet et verbe)
    "se", "s'", "me", "m'", "te", "t'",
})


def trouver_sujet_genre_nombre(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    idx_verbe: int,
    lexique,
    pos_nominaux: tuple[str, ...] = ("NOM", "NOM PROPRE"),
) -> tuple[str, str] | None:
    """Trouve le genre et nombre du sujet en scannant a gauche du verbe.

    Retourne (genre, nombre) via : pronom > NOM (lexique.info) > DET.
    genre: "Masc" ou "Fem", nombre: "Sing" ou "Plur".
    ``pos_nominaux`` permet d'elargir la recherche (e.g. inclure ADJ).
    """
    for j in range(idx_verbe - 1, max(-1, idx_verbe - 5), -1):
        w = mots[j].lower()
        if w in _TRANSPARENTS:
            continue
        # Pronom sujet
        if w in PRONOM_GENRE:
            return PRONOM_GENRE[w]
        # NOM → utiliser lexique
        pos_j = pos_tags[j] if j < len(pos_tags) else ""
        if pos_j in pos_nominaux and lexique is not None:
            infos = lexique.info(mots[j])
            if infos:
                best = max(infos, key=lambda e: float(e.get("freq") or 0))
                genre = best.get("genre", "")
                nombre = best.get("nombre", "")
                # Ambiguite : si les entrees ont les deux genres, skip
                _genres = {e.get("genre") for e in infos if e.get("genre")}
                if len(_genres) > 1:
                    break  # genre ambigu, pas de detection fiable
                g = "Fem" if genre == "f" else "Masc"
                n = "Plur" if nombre == "p" else "Sing"
                return (g, n)
            # Fallback CRF morpho : le tagger predit genre/nombre meme si
            # le mot n'est pas dans le lexique
            genre_list = morpho.get("genre", [])
            nombre_list = morpho.get("nombre", [])
            if j < len(genre_list):
                g_crf = genre_list[j]
                n_crf = nombre_list[j] if j < len(nombre_list) else "Sing"
                if g_crf in ("Fem", "Masc"):
                    return (g_crf, n_crf if n_crf in ("Sing", "Plur") else "Sing")
        break
    return None


def generer_candidats_pp_accorde(pp: str, genre: str, nombre: str) -> list[str]:
    """Genere les formes accordees d'un participe passe.

    alle + Fem+Sing → allee
    alle + Masc+Plur → alles
    alle + Fem+Plur → allees
    """
    low = pp.lower()
    candidats: list[str] = []

    if genre == "Masc" and nombre == "Sing":
        # Forme de base: retirer les marques fem/plur
        # signalées → signalé, situés → situé, agrandie → agrandi
        if low.endswith("\xe9es"):
            candidats.append(low[:-3] + "\xe9")     # signalées → signalé
        elif low.endswith(("ies", "ues")):
            candidats.append(low[:-2])               # démolies → démoli
        elif low.endswith("\xe9e"):
            candidats.append(low[:-2] + "\xe9")      # lancée → lancé
        elif low.endswith("\xe9s"):
            candidats.append(low[:-1])               # situés → situé
        elif low.endswith(("is", "us", "ts")):
            candidats.append(low[:-1])               # construits → construit
        elif low.endswith("es"):
            candidats.append(low[:-2])               # faites → fait
        elif low.endswith("ie"):
            candidats.append(low[:-1])               # agrandie → agrandi
        elif low.endswith("ue"):
            candidats.append(low[:-1])               # battue → battu
        elif low.endswith("s"):
            candidats.append(low[:-1])               # generic pluriel
        elif low.endswith("e"):
            candidats.append(low[:-1])               # generic feminin
    elif genre == "Fem" and nombre == "Sing":
        if low.endswith("\xe9es"):
            candidats.append(low[:-1])               # signalées → signalée
        elif low.endswith("\xe9s"):
            candidats.append(low[:-1] + "e")         # situés → située
        elif low.endswith("s") and not low.endswith("es"):
            candidats.append(low[:-1] + "e")         # construits → construite
        elif low.endswith("s"):
            candidats.append(low[:-1])               # alles → allee
        elif low.endswith("e"):
            pass  # deja feminin singulier
        else:
            candidats.append(low + "e")
    elif genre == "Masc" and nombre == "Plur":
        if not low.endswith("s"):
            candidats.append(low + "s")
    elif genre == "Fem" and nombre == "Plur":
        if low.endswith("e"):
            candidats.append(low + "s")
        elif low.endswith("s"):
            # alles → allees
            candidats.append(low[:-1] + "es")
        else:
            candidats.append(low + "es")

    return candidats
