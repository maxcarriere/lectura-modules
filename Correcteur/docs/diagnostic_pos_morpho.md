# Diagnostic POS/MORPHO — Mai 2026

Evaluation de la qualite de l'etiquetage POS et MORPHO dans le pipeline de
correction, sur les 272 cas du benchmark (152 evaluation + 120 benchmark).

Score actuel du correcteur : **210/272 (77.2%)**.

## 1. Infrastructure disponible

| Composant | Statut | POS | Morpho | Contextuel |
|-----------|--------|-----|--------|------------|
| LexiqueTagger | Actif (defaut) | Oui (par freq) | genre/nombre | Non |
| G2P Unifie V2 | Disponible | Oui (19 cats, top-K) | 6 traits (UD) | Oui (ONNX) |
| POS n-gram | Actif | Tri/bigrammes | PM trigrammes | Oui (sequence) |
| MorphoTagger CRF | Disponible | Oui | 6 traits | Oui |
| Viterbi POS (amont) | OFF par defaut | Trigramme | Non | Oui |
| Viterbi PM (aval) | OFF par defaut | PM trigramme | genre/nombre | Oui |

**Constat cle** : le pipeline utilise par defaut le LexiqueTagger (zero modele,
par frequence) au lieu du G2P Unifie V2 qui est un etiqueteur contextuel complet.

## 2. Qualite POS sur les phrases attendues

### 2.1 Mots-ancres (784 occurrences)

Les mots-ancres sont les pronoms, articles, prepositions, conjonctions et
auxiliaires dont le POS est quasiment non-ambigu.

| Tagger | Ancres correctes | Taux |
|--------|-----------------|------|
| LexiqueTagger | 776/784 | **99.0%** |
| G2P Unifie V2 | 760/784 | **96.9%** |

**Resultat contre-intuitif** : le LexiqueTagger est meilleur sur les ancres.

Explication : le LexiqueTagger a des overrides codes en dur (`_FUNCTION_WORD_POS`)
pour "est"=AUX, "on"=PRO:per, etc. Le G2P n'a pas ces corrections et fait des
erreurs systematiques :

- **"sont" = VER au lieu de AUX** (5 occurrences) — le G2P ne distingue pas AUX/VER
- **"au" = NOM ou ADV au lieu de PRE** (14 occurrences) — erreur systematique
- **"a" = VER au lieu de AUX** (3 occurrences)
- **"de" = ART:ind au lieu de PRE** (1 occurrence)

Les erreurs du LexiqueTagger sont :
- **"J'" = vide** (5 cas) — l'elision n'est pas reconnue
- **"que" = PRO:int au lieu de CON** (2 cas) — ambigu legitime

### 2.2 Accord LexiqueTagger / G2P

| | Valeur |
|---|---|
| Tokens analyses | 1674 |
| Accord Lex/G2P | 1236/1656 (74.6%) |
| Desaccords | 420 (25.4%) |

Le taux de desaccord de 25% est eleve. Principaux desaccords :

| Mot | LexiqueTagger | G2P | Occurrences | Commentaire |
|-----|--------------|-----|-------------|-------------|
| les | ART | ART:def | 79 | Sous-typage (ok) |
| le | ART | ART:def | 78 | Sous-typage (ok) |
| la | ART | ART:def | 54 | Sous-typage (ok) |
| des | ART | ART:ind | 38 | Sous-typage (ok) |
| du | ART | ART:ind | 13 | Sous-typage (ok) |
| ce | PRO:dem | ADJ:dem | 9 | G2P souvent correct |
| petite | NOM | ADJ | 8 | G2P correct |
| ecole | VER | NOM | 7 | **G2P correct, Lex faux** |
| soir | VER | NOM | 5 | **G2P correct, Lex faux** |
| belle | NOM | ADJ | 6 | G2P correct |
| va | ADV | VER | 5 | **G2P correct, Lex faux** |
| sont | AUX | VER | 5 | Lex correct (override) |

La majorite des desaccords (262/420) sont du sous-typage d'articles (ART vs
ART:def/ART:ind) — non impactant. Les vrais desaccords montrent que le G2P
est meilleur sur les mots de contenu (ecole, soir, belle, va) et le
LexiqueTagger meilleur sur les mots-outils (sont, a, au).

### 2.3 Morpho du G2P

La confiance POS du G2P est stable :
- Ancres : 0.905 en moyenne
- Contenu : 0.898 en moyenne

Le G2P fournit aussi genre, nombre, personne — informations non disponibles
de maniere fiable avec le LexiqueTagger.

## 3. Violations de contraintes sequentielles

Seulement 2 violations detectees apres un pronom personnel sujet
(PRO:per + NOM/ADJ/ART) :
- "Il marche" → marche=NOM (devrait etre VER)
- "je bois" → bois=NOM (devrait etre VER)

Ces cas sont des homographes NOM/VER ou le LexiqueTagger choisit NOM par
frequence. Le G2P les etiquette correctement (VER) grace au contexte.

## 4. Segmentation par ponctuation

| Longueur segment | Nombre | % |
|-----------------|--------|---|
| 1-3 mots | 18 | 6% |
| 4-6 mots | 157 | 56% |
| 7-10 mots | 105 | 37% |
| 11+ mots | 1 | 0% |

Longueur moyenne : 6.0 mots, max : 11.

Les segments sont courts. Un Viterbi sur des segments de 6 mots avec ~18 etats
POS est instantane (~18^3 x 6 = 35K operations).

## 5. Analyse des 63 cas en erreur

### 5.1 Erreurs de POS impactant la correction

| Erreur POS | Cas | Impact |
|-----------|-----|--------|
| soir=VER au lieu de NOM | "ce soir" | LexiqueTagger, G2P correct |
| ecole=VER au lieu de NOM | "l'ecole" | LexiqueTagger, G2P correct |
| parti=NOM au lieu de VER | "est parti" | LexiqueTagger, G2P correct |
| belle=NOM au lieu de ADJ | "est belle" | LexiqueTagger, G2P correct |
| visite=NOM au lieu de VER | "avons visite" | LexiqueTagger, G2P correct |
| a=AUX au lieu de PRE(a) | "a retenir" | Les deux faux (= "a" ambigu) |
| contente=VER au lieu de ADJ | "est contente" | Les deux faux |
| ete=SIGLE au lieu de NOM | "l'ete" | LexiqueTagger OOV |

### 5.2 Patterns d'erreurs recurrents

**Homographes NOM/VER** : le LexiqueTagger choisit systematiquement NOM pour
les mots ambigus (soir, marche, bois, parti, ecole, visite). Le G2P les
resout correctement dans presque tous les cas.

**Mots OOV** : les mots sans accent (ecole, ete, eleve, controle) n'ont pas
d'entree dans le lexique et recoivent un POS vide ou incorrect. Le G2P
les etiquette correctement car il opere au niveau caractere.

**Homophones grammaticaux** : "a" (AUX) vs "a" (PRE→a) reste difficile
pour les deux taggers.

## 6. Recommandations

### 6.1 Gain immediat : utiliser le G2P comme tagger principal

Le G2P Unifie V2 est meilleur que le LexiqueTagger sur :
- Homographes NOM/VER (soir, marche, ecole, parti, visite) : ~10 cas
- Mots OOV (sans accent) : etiquettes correctement
- Morpho (genre, nombre, personne) : disponible

Mais il est moins bon sur :
- AUX vs VER (sont, a) : pas de distinction
- "au" : systematiquement faux

**Strategie hybride recommandee** : utiliser le G2P comme base, avec les
overrides du LexiqueTagger (_FUNCTION_WORD_POS) appliques par-dessus pour
les mots-ancres. Cela combine le meilleur des deux.

### 6.2 Architecture a passes iteratives

```
Passe 0 — Ancrage
  - Identifier les mots-ancres (pronoms, articles, prep) via _FUNCTION_WORD_POS
  - Fixer leur POS/MORPHO avec confiance=1.0
  - Segmenter par la ponctuation

Passe 1 — Etiquetage contextuel
  - G2P Unifie V2 sur chaque segment
  - Injecter les contraintes des ancres via lex_features
  - Produire top-K POS + morpho + confiance

Passe 2 — Desambiguation homophones
  - Pour les mots-ancres ambigus (son/sont, on/ont, a/a)
  - POS n-gram tranche sur la base de la sequence
  - Fixer le resultat avec confiance elevee

Passe 3 — Correction orthographique
  - Candidats accent/homophones pour les mots OOV ou basse freq
  - Mettre a jour POS/MORPHO apres chaque correction validee
  - Relancer l'etiquetage des voisins si le POS change

Passe 4 — Correction grammaticale
  - Regles d'accord avec les POS/MORPHO stabilises
  - Viterbi PM pour valider la coherence finale
```

### 6.3 Contraintes n-gram a implementer

Sequences impossibles a penaliser dans le POS n-gram :

| Apres | Interdit | Raison |
|-------|----------|--------|
| PRO:per sujet | NOM, ADJ, ART | "je mange" pas "je NOM" |
| ART:def | VER, AUX, PRE, CON | "le chat" pas "le mange" |
| ART:ind | VER, AUX, PRE, CON | "un chat" pas "un mange" |
| PRE | PRE, CON, AUX | "dans le" pas "dans dans" |

Ces contraintes sont deja largement couvertes par les trigrammes POS n-gram
mais meritent une verification explicite comme garde-fou.

### 6.4 Potentiel du G2P iteratif

Le G2P V2 accepte des `lex_features` (vecteur 24D par mot) qui incluent :
- One-hot des POS candidats du lexique
- Flag connu/inconnu
- Flag non-ambigu

**Idee** : apres la passe 1, fixer les POS des mots resolus en tant que
contraintes lex_features, puis relancer le G2P. Les positions voisines
beneficieront de cette information fixee.

Cela necessite de modifier `G2PUnifieAdapter._analyser_cached()` pour
accepter des contraintes POS par position. Le modele ONNX supporte deja
cette interface via `lex_features`.

### 6.5 Chiffrage du gain potentiel

| Source de gain | Cas concernes | Gain estime |
|----------------|--------------|-------------|
| G2P au lieu de LexiqueTagger (NOM/VER) | ~10 | +3 a +5 |
| Mise a jour POS apres correction accent | ~5 | +2 a +3 |
| Desambiguation homophones par n-gram | ~5 | +1 a +3 |
| Contraintes ancres + n-gram | ~3 | +1 a +2 |
| **Total estime** | | **+7 a +13** |

Objectif : 210 + 10 = **220/272 (~81%)** en premiere iteration.

## 7. Fichiers de reference

- Script diagnostic : `scripts/diagnostic_pos_morpho.py`
- LexiqueTagger : `src/lectura_correcteur/_tagger_lexique.py`
- G2P Unifie V2 : `src/lectura_correcteur/_adapter_g2p_unifie.py`
- Engine ONNX V2 : `src/lectura_correcteur/data/g2p_v2/inference_onnx_v2.py`
- POS n-gram : `src/lectura_correcteur/_pos_ngram.py`
- Viterbi POS : `src/lectura_correcteur/_analyse_viterbi.py`
- Viterbi PM : `src/lectura_correcteur/_viterbi_morpho.py`
- Config : `src/lectura_correcteur/_config.py`
