# Plan V5 — Correcteur a base P2G

## Vision

Remplacer l'approche V1 (etiquetage lexique fragile + regles amendees
phrase par phrase) par une approche propre :

1. **Etiquetage P2G** : POS/Morpho predit par le modele P2G sans
   ortho_words (independant de l'orthographe fautive)
2. **Regles claires** : regles de grammaire classiques du francais,
   ecrites pour fonctionner avec un etiquetage fiable
3. **Iteration data-driven** : amelioration continue par test sur
   corpus, analyse des erreurs, correction des regles

Le P2G s'ameliore independamment (autre chantier). Le correcteur
beneficie automatiquement de chaque amelioration du P2G.

## Architecture

```
Phrase
  |
  v
1. Tokenisation + Syntaxe + Resegmentation      [V1 existant]
  |
  v
2. G2P -> phonemisation de chaque mot            [P2G pipeline]
  |
  v
3. P2G sans ortho_words -> POS + Morpho (UD)     [P2G pipeline]
  |
  v
4. Conversion UD -> short-form + fusion lexique   [_morpho_fusion.py]
  |
  v
5. Orthographe (VerificateurOrthographe)          [V1 existant]
  |
  v
6. Re-tag : POS P2G (non corriges) / lexique (corriges)
  |
  v
7. Regles de grammaire V5                         [A REECRIRE]
  |
  v
8. Reconstruction                                 [V1 existant]
```

## Etat actuel (baseline)

Resultats sur corpus FLE (299 phrases) + 100 negatives :

| Metrique      | V1-lexique | V5-P2G  | Delta |
|---------------|------------|---------|-------|
| Accord        | 60/96      | 67/96   | +7    |
| Conjugaison   | 74/102     | 77/102  | +3    |
| Homophone     | 48/61      | 46/61   | -2    |
| Participe     | 36/40      | 36/40   |  0    |
| Negatif (FP)  | 68/100     | 69/100  | +1    |
| **Positifs**  | 218/299    | 226/299 | **+8**|

Le P2G apporte des gains nets sur accords (genre) et conjugaison.
Regressions sur homophones (et/est, a/a) car les regles V1 sont
calibrees sur POS lexique — a reecrire.

## Corpus disponibles

| Fichier                        | Phrases | Type              |
|--------------------------------|---------|-------------------|
| `grammaire_fle.tsv`           |     299 | FLE, courtes      |
| `grammaire_wicopaco.tsv`      |   6 167 | Wikipedia, longues |
| `negatif_wicopaco.tsv`        |   1 000 | Correctes (guard) |
| `fr_gec_akufeldt.tsv`         |  66 500 | Synthetique, categories |
| `fr_gec_multilingual.tsv`     |  67 157 | Synthetique, categories |

Chemin : `/home/moi/Documents/work/projets/lectura/workspace/Corpus/Correcteur/`

Categories dans fr_gec_akufeldt.tsv :
- PluralSingularNounDestroyer (7468)
- ApostropheChangerDestroyer (7408)
- AdjectiveMisplacerDestroyer (6974)
- ContractionsDestroyer (6541)
- GenderDeterminerDestroyer (5884)
- PluralDeterminerDestroyer (5204)
- PrepositionReplacerDestroyer (4342)
- PersonVerbDisagreement (4184)
- RemoveAccentsDestroyer (3912)
- FrRemovePunctuationDestroyer (3466)
- FrVerbNumberChanger (3231)
- PossessiveReplacerDestroyer (2183)
- ReflexivePronounRemover (1649)
- PasRemoverDestroyer (1369)
- AdjectiveGenderChangeDestroyer (1050)
- RandomTypoDestroyOperation (814)
- RandomWordRemoveOperation (413)
- CapitalsDestroyer (168)
- SimpleQuestionDestroyer (154)

## Pipeline iteratif autonome

### Principe

```
1. Piocher 200 phrases aleatoires (mix FLE + negatives + GEC)
2. Evaluer V5 sur ces 200 phrases
3. Analyser les erreurs (FP, FN, WR)
4. Modifier les regles pour corriger les patterns d'erreur
5. Re-evaluer sur les memes 200 phrases
6. Si amelioration : valider sur 200 NOUVELLES phrases (anti-overfit)
7. Si validation OK : commit les changements
8. Repiocher 200 phrases et recommencer
9. Arreter quand F1 plafonne sur 3 iterations consecutives
```

### Metriques suivies

- **Precision** : corrections correctes / total corrections faites
- **Recall** : corrections faites / corrections attendues
- **F1** : 2 * P * R / (P + R)
- **F0.5** : pondere precision (penalise FP)
- **Taux FP** : phrases correctes modifiees a tort
- **Exact match** : phrases parfaitement corrigees

### Composition du batch de 200 phrases

- 60 phrases FLE (grammaire_fle.tsv, piochees aleatoirement)
- 40 phrases negatives courtes (negatif_wicopaco.tsv, < 120 chars)
- 100 phrases GEC (fr_gec_akufeldt.tsv, < 150 chars, mix categories)

### Regles de conduite

1. Ne modifier QUE les fichiers sous `src/lectura_correcteur/`
2. Ne PAS modifier les corpus
3. Ne PAS modifier les autres modules (Lexique, Phonemiseur, etc.)
4. Garder `correcteur.py` (V1) intact comme reference
5. Logger chaque iteration dans ce fichier (resultats, changements)
6. Faire un backup avant chaque modification majeure
7. Si F1 baisse de plus de 5% sur la validation : rollback

## Fichiers du module Correcteur

### Modifiables
- `src/lectura_correcteur/correcteur_v5.py` — pipeline principal
- `src/lectura_correcteur/_morpho_fusion.py` — conversion P2G -> V1
- `src/lectura_correcteur/grammaire/` — regles de grammaire
- Tout fichier sous `src/lectura_correcteur/` si necessaire
- Creation de nouveaux fichiers de regles autorisee

### A garder intacts (reference)
- `src/lectura_correcteur/correcteur.py` — V1 original
- Les corpus dans le dossier Corpus/

## Journal des iterations

### Iteration 0 — Baseline (2025-05-21)

V5 avec POS P2G dans la grammaire, regles V1 inchangees.

FLE (299) + negatives (100) :
- Positifs : 226/299 (75.6%)
- Negatifs : 69/100 (69.0%), FP=31
- Accord : 67/96 (69.8%)
- Conjugaison : 77/102 (75.5%)
- Homophone : 46/61 (75.4%)
- Participe : 36/40 (90.0%)

Gains V5 vs V1 (14 phrases) :
- Accords genre (une bon idee, un bonne livre, etc.)
- Conjugaison modale (je pouvoir venir, nous pouvoir jouer)
- Accords pluriel nom apres determinant (les belle fleur, les nouveau eleve)

Regressions V5 vs V1 (5 phrases) :
- Homophones et/est : P2G dit AUX pour "et" (POS intentionnel),
  les regles cherchent CON → ne se declenchent pas
- Homophones a/a : meme probleme
- Accord attribut apres etre : "la soupe est chaud" non corrige

→ Les regles homophones doivent etre reecrites pour le paradigme P2G.
  Avec P2G : comparer POS_P2G vs POS_LEXIQUE(forme_ecrite).
  Si POS_P2G != POS_LEXIQUE → homophone potentiel.

### Iteration 1 — Homophones P2G + guards (2025-05-21)

Ajout de `detecter_homophones_p2g()`, guards, desactivation LM/BiLSTM.

| Metrique | Iter 0 | Iter 1 | Delta |
|----------|--------|--------|-------|
| FLE OK   | 228    | 233    | +5    |
| FLE FP   | 2      | 0      | -2    |
| NEG OK   | 73/100 | 74/100 | +1    |
| F1       | 0.865  | 0.876  | +0.011|

### Iteration 2 — Guards homophones + negation (2025-05-21)

Renforcement des guards P2G homophones (est→et, et→est, A→À),
correction negation (scan clitiques pour "ne" existant).

| Metrique | Iter 1 | Iter 2 | Delta |
|----------|--------|--------|-------|
| FLE OK   | 233    | 233    | 0     |
| NEG FP   | 26     | 20     | -6    |
| Total F1 | 0.876  | 0.879  | +0.003|

Changements :
- `_homophones.py` : supprime ou/où du P2G, guard majuscule isolee,
  guard est→et (copule/auxiliaire), guard et→est (infinitif), locutions adverbiales
- `_negation.py` : scan 3 positions en arriere a travers clitiques

### Iteration 3 — Conjugaison PRO + infinitif (2025-05-21)

Ajout Regle 4b : PRO sujet + infinitif → conjuguer au present.
Bug fix : `_conjuguer_via_lexique` compatible LexiqueNormalise
(mode="ind"/"indicatif", temps="pre"/"present").

| Metrique  | Iter 2 | Iter 3 | Delta  |
|-----------|--------|--------|--------|
| FLE OK    | 233    | 248    | **+15**|
| FLE FN    | 40     | 28     | -12    |
| FLE WR    | 26     | 23     | -3     |
| FLE F1    | 0.876  | 0.907  | +0.031 |
| NEG FP    | 20     | 22     | +2     |
| Total F1  | 0.879  | 0.899  | +0.020 |
| Total F0.5| 0.875  | 0.887  | +0.012 |

2 NEG FP supplementaires de `conjugaison.accord_morpho` (interaction
guards homophones), pas de Regle 4b.

FN conjugaison restants : "je/tu/vous/nous va" (aller wrong person,
pas infinitif → Regle 5 a ameliorer).

FN restants par categorie :
- homophone (10) : et/est ×4, ce/se ×3, ta/t'a, ma/m'a, c'est/ce
- accord (9) : genre attribut ×3, genre DET ×3, pluriel verbe ×2, des→de
- conjugaison (5) : aller wrong person ×4, cours→courent

### Iteration 4 — FN restants (genre attribut, conjugaison, homophones)

6 categories de correctifs :

1. **Conjugaison aller (Regle 5 lemme fallback)** :
   `_conjuguer_via_lemme()` — remonte au lemme via info(), puis conjugue.
   je/tu/vous/nous va → vais/vas/allez/allons.

2. **_already_ok exclut imperatif** :
   "va" est P2 imperatif dans le lexique → le guard `_already_ok`
   bloquait la correction "tu va→vas". Ajout filtre `mode != imp`.

3. **ce→se (POS ADJ:dem)** :
   Condition elargie : `pos.startswith(("DET", "ADJ:dem"))`.

4. **ta→t'a / ma→m'a** :
   Nouvelle regle dans `_homophones.py`. Guard dans Regle 5 :
   auxiliaires contractes (`"'" in w and w.endswith("a")`) pour
   eviter la corruption du PP (appelé→appelle).

5. **Genre attribut (Regle 9)** :
   - `formes_de` lookup pour irreguliers (blanc→blanche, ouvert→ouverte)
   - Expansion NOM→ADJ quand ADJ freq > NOM freq (chaud tague NOM)
   - Guard : require NOM entries exist (evite FP contente)
   - `trouver_sujet_genre_nombre` : filtre cgram avant ambiguite,
     skip si genre vide (NOM PROPRE sans genre → pas de defaut Masc)

6. **generer_candidats_feminin** : +patterns anc→anche, ec→eche, c→che

| Metrique  | Iter 3 | Iter 4 | Delta   |
|-----------|--------|--------|---------|
| FLE OK    | 248    | 261    | **+13** |
| FLE FN    | 28     | 15     | -13     |
| FLE WR    | 23     | 23     |  0      |
| FLE FP    | 0      | 0      |  0      |
| FLE F1    | 0.907  | 0.932  | +0.025  |
| NEG FP    | 22     | 20     | -2      |
| Total F1  | 0.899  | 0.922  | +0.023  |
| Total F0.5| 0.887  | 0.901  | +0.014  |

FN restants (15) par categorie :
- homophone et→est (4) : P2G limitation (tag NOM au lieu de CON)
- homophone a→a (2) : P2G limitation (tag PRE correct pour "a")
- homophone c'est→ce (1) : tokenisation complexe
- accord DET genre (3) : le table, le voiture, la soleil (regle desactivee)
- accord pluriel verbe (3) : les pommes est, les chats dort, les garcons cours
- accord des→de (1) : des nouvelles → de nouvelles (regle absente)
- accord genre ADJ (1) : cette beau journee → cette belle journee

### Iteration 5 — FN restants : DET genre, et→est, ADJ pluriel, invariables

7 categories de correctifs :

1. **Regle 3 guard : est+ADJ singulier ≠ coordination** :
   Raffine `_skip_coord` : quand next est ADJ singulier (sans s/x/z)
   et pas de PLUR_DET en amont → ne pas skipper (pas coordination).
   Corrige "les pommes est rouge" → "sont rouges".

2. **Cascade ADJ pluriel apres copule** :
   Apres est→sont dans Regle 3, si le mot suivant est ADJ singulier,
   le pluraliser automatiquement (rouge→rouges, content→contents).

3. **Regle 8 DET genre fallback (le/la + NOM)** :
   Quand adj_idx est None (pas d'ADJ intercale), verifier le genre
   du NOM directement. Si NOM univoque (un seul genre, freq > 10)
   et DET ne correspond pas → corriger le/la.
   Corrige "le table" → "la table", "la soleil" → "le soleil".

4. **et→est contextuel (DET/poss + NOM + et + ADJ)** :
   Nouvelle regle independante du POS P2G : si pattern
   possessif+NOM+"et"+ADJ_lexique → remplacer "et" par "est".
   Guards : skip PP (du/des), skip PRE, skip coordination (NOM/et a i+2).
   Corrige "sa voiture et rouge" → "est rouge".

5. **Regle 6 formes_de fallback (irreguliers)** :
   Pour les ADJ irreguliers (beau→belle, nouveau→nouvelle),
   utiliser `lexique.formes_de(lemme)` quand les patterns suffixaux
   ne trouvent pas de candidat.
   Corrige "cette beau journee" → "cette belle journee".

6. **Regle 20 : des + ADJ + NOM pluriel → de** :
   Nouvelle regle : "des bons gateaux" → "de bons gateaux".
   Guard : NOM doit etre pluriel (termine en s/x/z, len > 2).

7. **Guard invariables homophones (heureux, doux, etc.)** :
   La regle est→et (NOM_plur + est + ADJ_plur) ne doit pas
   considerer les ADJ invariables (meme forme sing/plur) comme
   preuve de coordination. Check lexique: si ADJ a singulier ET
   pluriel → skip. Compatible valeurs normalisees ("s"/"p").

| Metrique  | Iter 4 | Iter 5 | Delta   |
|-----------|--------|--------|---------|
| FLE OK    | 261    | 274    | **+13** |
| FLE FN    | 15     | 5      | -10     |
| FLE WR    | 23     | 20     | -3      |
| FLE FP    | 0      | 0      |  0      |
| FLE F1    | 0.932  | 0.956  | +0.024  |
| NEG FP    | 20     | 20     |  0      |
| Total F1  | 0.922  | 0.940  | +0.018  |
| Total F0.5| 0.901  | 0.915  | +0.014  |

FN restants (5) — tous hard/P2G limitations :
- homophone a→a (2) : "ma mere a raison", "ton pere a une voiture"
  (P2G tag PRE pour "a", correct phonetiquement)
- homophone c'est→ce (1) : tokenisation c'est = un seul token
- accord pluriel verbe (2) : "les chats dort", "les garcons cours"
  (P2G tag NOM/ADJ au lieu de VER → Regle 3 ne fire pas)

WR restants (20) — categories principales :
- elision manquante (2) : "je avoir" → "je ai" (pas "j'ai")
- temps incorrect (2) : "il manger hier" → present au lieu d'imparfait
- cascade genre+nombre incomplete (8) : "les vieux maison" (ADJ+NOM)
- reordonnancement ADJ (2) : "les bleu voiture" → pas de reorder
- homophones complexes (3) : "il on mange", "ou est la gare", "il ni a"
- auxiliaire etre vs avoir (2) : "ils ont partir" (pas "sont partis")
- negation parasite (1) : "sans parle" → insere "ne" a tort

### Iteration 6 — ADJ antepose + genre NOM + elision

3 categories de correctifs :

1. **Regle 2b : PLUR_DET + ADJ_lexique + NOM → accord cascade** :
   Nouvelle regle apres Regle 2. Detecte le pattern ADJ antepose
   meme quand P2G tag l'ADJ comme NOM (lookup lexique pour ADJ entries).
   Utilise `formes_de(lemme)` pour trouver la forme cible (genre+plur),
   avec fallback generer_candidats_feminin/masculin + pluraliser.
   Corrige: "les vieux maison"→"vieilles maisons", "les blanc chemise"→
   "blanches chemises", "les long journee"→"longues journees", etc.
   Guard: NOM gender doit etre univoque (skip "enfant" m+f).
   Guard: utilise la forme ORIGINALE pour le genre (evite que la forme
   pluralisee par Regle 2 ait des entrees differentes, ex: "vagues"
   n'a que m mais "vague" a m+f).

2. **Fix 3a : genre NOM univoque** :
   Corrige le bug ou "les petit enfant" etait feminise a tort.
   Change `any(genre == "f")` → `all(genre == "f" for NOM entries)`.
   "enfant" a m+f → Fix 3a ne feminise plus.

3. **Elision post-grammaire** :
   Nouvelle passe finale dans `appliquer_grammaire` qui detecte
   pronom/article + voyelle et elide: "je"→"j'", "le"→"l'", etc.
   Guards: seulement si le mot suivant existe dans le lexique (evite
   elision devant loanwords), et seulement si le mot suivant a ete
   modifie par les regles (evite les FP sur phrases correctes).

4. **Regle 20 : extension POS** :
   "des" + NOM(ADJ_lexique) + NOM → "de" : accepte NOM-tague
   avec entrees ADJ dans le lexique (P2G mistagging).

| Metrique  | Iter 5 | Iter 6 | Delta   |
|-----------|--------|--------|---------|
| FLE OK    | 274    | 283    | **+9**  |
| FLE FN    | 5      | 5      |  0      |
| FLE WR    | 20     | 11     | -9      |
| FLE FP    | 0      | 0      |  0      |
| FLE F1    | 0.956  | 0.973  | +0.017  |
| NEG FP    | 20     | 20     |  0      |
| Total F1  | 0.940  | 0.953  | +0.013  |
| Total F0.5| 0.915  | 0.934  | +0.019  |

Validation seed=123 : FLE 283 OK, NEG FP 18, Total F1=0.955 (stable).

FN restants (5) — inchanges, tous hard/P2G limitations.

WR restants (11) :
- temps incorrect (2) : "il manger hier" → present au lieu d'imparfait
- reordonnancement ADJ (2) : "les bleu voiture" → pas de reorder
- homophones complexes (3) : "il on mange", "ou est la gare", "il ni a"
- auxiliaire etre vs avoir (1) : "ils ont partir" (pas "sont partis")
- negation parasite (1) : "sans parle" → insere "ne" a tort
- souriant post-nominal (1) : "les fille souriant" → souriant pas accorde
- genre cascade partielle (1) : "les grosse vague" → OK maintenant

### Iteration 7 — Possessifs + typographie francaise

3 changements, ciblant le corpus GEC :

1. **Regle 11d : Accord genre possessif** (`_accord.py`) :
   Corrige son/sa, mon/ma, ton/ta selon le genre du NOM qui suit.
   Trois cas : (1) euphonie sa+voyelle→son, ma+voyelle→mon, ta+voyelle→ton ;
   (2) son/mon/ton + NOM_fem_consonant → sa/ma/ta ;
   (3) sa/ma/ta + NOM_masc_consonant → son/mon/ton.
   Gere ADJ intercale ("mon propre maison" → "ma propre maison").
   Guard: genre NOM doit etre univoque (skip "enfant" m+f).
   Impact GEC: PossessiveReplacerDestroyer passe de 0% → ~50% OK.

2. **Regle 11e : RETIREE** (`_accord.py`) :
   Tentative DET_sing + NOM_plur → pluraliser DET (inverse de Regle 1).
   Retiree car conflit fondamental avec la strategie existante
   (Regle 1 singularise le NOM pour matcher le DET).
   Regressions : 6 FLE (le chats→Les chats au lieu de Le chat),
   5 NEG FP (C'est une des→C'est des des, l'un→l'des).

3. **Typographie francaise : espace avant ? et !** (`_utils.py`) :
   Retire `?` et `!` du set `_NO_SPACE_BEFORE`, ajoute un set
   `_FRENCH_SPACE_BEFORE` pour forcer un espace avant ces signes.
   En francais, l'espace avant ? et ! est obligatoire.
   Impact GEC : +19 OK (phrases ou le seul delta etait l'espace),
   FP de 17 → 6 (-11), precision 0.286 → 0.382.
   Pas de regression FLE ni NEG.

| Metrique  | Iter 6 | Iter 7 | Delta    |
|-----------|--------|--------|----------|
| FLE OK    | 283    | 283    |  0       |
| FLE FN    | 5      | 5      |  0       |
| FLE WR    | 11     | 11     |  0       |
| NEG FP    | 20     | 20     |  0       |
| Total F1  | 0.953  | 0.953  |  0       |
| GEC OK    | —      | 117    | (baseline)|
| GEC WR    | —      | 183    |          |
| GEC FN    | —      | 194    |          |
| GEC FP    | —      | 6      |          |
| GEC F1    | —      | 0.379  |          |

Analyse GEC (500 phrases, seed=42, 20 categories × 25 phrases) :

Categories a bonne performance (>40% OK) :
- AdjectiveGenderChangeDestroyer : 17/25 (68%)
- CapitalsDestroyer : 15/25 (60%)
- PossessiveReplacerDestroyer : 13/25 (52%) [ameliore par 11d]
- GenderDeterminerDestroyer : 12/25 (48%)
- PluralSingularNounDestroyer : 12/25 (48%)
- RemoveAccentsDestroyer : 10/25 (40%)

Categories partielles (>0% OK) :
- PersonVerbDisagreement : 9/25 (36%)
- FrVerbNumberChanger : 5/25 (20%)
- RandomTypoDestroyOperation : 3/25 (12%)
- None (FP control) : 21/25 (84%)

Categories hors-perimetre (0% OK, necessitent des capacites absentes) :
- ReflexivePronounRemover : insertion de pronoms reflexifs
- RandomWordRemoveOperation : reinsertion de mots
- ApostropheChangerDestroyer : gestion des apostrophes
- ContractionsDestroyer : expansion/contraction (du/de le)
- PrepositionReplacerDestroyer : choix de preposition
- PasRemoverDestroyer : reinsertion de "pas"
- SimpleQuestionDestroyer : restructuration interrogative
- AdjectiveMisplacerDestroyer : reordonnancement de mots
- FrRemovePunctuationDestroyer : reinsertion de ponctuation
- PluralDeterminerDestroyer : conflit DET vs NOM (strategie inverse)
