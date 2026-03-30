# Synthèse — Projet G2P Multi-tâche avec Liaisons

## Contexte

Après la création réussie de MORPHO_CRF (88,47% tag complet, 97,47% POS, zéro dépendance, 2,9 Mo), réflexion sur une nouvelle direction stratégique : un modèle unifié qui prédit **POS + Morpho + Phonétique + Liaisons** en une seule passe.

## État des lieux — Pipeline actuel

| Module | Architecture | Taille | Accuracy |
|--------|-------------|--------|----------|
| MORPHO_CRF | CRF composite, Viterbi pur Python | 2,9 Mo | 88,47% tag, 97,47% POS |
| MORPHO_BiLSTM | BiLSTM ONNX INT8 | ~10 Mo | 87,71% tag, 96,13% POS |
| G2P_CRF | CRF char-level + `_CONT` labels | 747 Ko + 278 Ko corrections | 83,2% intrinsèque, ~100% avec corrections |
| G2P_BiLSTM | BiLSTM ONNX INT8 | 614 Ko + 142 Ko corrections | 90,8% intrinsèque, ~100% |
| G2P_Seq2Seq | Enc-Dec + attention ONNX INT8 | ~2,1 Mo + 89 Ko corrections | 96,7% intrinsèque, ~100% |
| Liaisons | Algorithmique (863 h aspiré, POS-aware) | Pur Python | N/A |

Le pipeline existant est **déjà POS-aware** : le G2P accepte un paramètre `pos=` pour la désambiguïsation d'homographes (table `g2p_pos` de 402 entrées).

---

## Décisions de design prises

### Labels composites : OUI pour POS+Morpho, NON pour phonèmes

- Les labels composites (`VER|Ind|Pres|3|Plur`) fonctionnent très bien pour POS+Morpho (144 tags, CRF bat le BiLSTM)
- Ajouter les phonèmes aux labels composites = **explosion combinatoire** (144 × ~40 = milliers de labels)
- Le G2P reste une tâche séparée (char-level, pas word-level)

### Granularité : mot par mot avec marqueurs de liaison

Le problème central : POS/Morpho sont par **mot**, mais la phonétique avec liaisons est par **groupe de lecture** (les‿anciens‿amis = un seul bloc phonétique).

**Solution retenue** : rester au niveau mot, mais ajouter des **marqueurs de liaison** aux labels morpho.

```
Mot :       les           anciens         amis
Morpho :    ART:def|Plur|Lz  ADJ|Masc|Plur|Lz  NOM|Masc|Plur
Phone :     le            ɑ̃sjɛ̃             ami
Résultat :  le + z + ɑ̃sjɛ̃ + z + ami = lezɑ̃sjɛ̃zami
```

Les marqueurs de liaison : `Lz`, `Lt`, `Ln`, `Lr`, `Lv` (ou rien = pas de liaison).

### Gestion de tous les types de jonctions

| Jonction | Phénomène | Gestion |
|----------|-----------|---------|
| **Liaison** | Phonème latent ajouté (les‿enfants) | Marqueur `Lz/Lt/Ln/Lr` dans le label morpho |
| **Élision** | Concaténation (l'enfant) | Déjà token unique dans UD, G2P de `l'` = `/l/`, concaténation |
| **Enchaînement** | Re-syllabation (avec‿elle) | Simple concaténation phonétique |
| **Composé** | Liaison forcée (peut-être) | Déjà token unique dans UD, G2P du composé entier |
| **Dénasalisation** | bon‿ami → bɔnami | Post-traitement : si `Ln` + voyelle nasale finale → dénasaliser |

### Corpus UD : comment il gère ces cas

| Phénomène | Traitement UD | Nb occurrences (train) |
|-----------|--------------|----------------------|
| Contractions (du, des, au) | **Éclatées** (MWT : `des` → `de` + `les`) | 11 299 |
| Élisions (l', d', n') | **Token unique** (non séparées) | ~18 000 |
| Composés à trait d'union | **Token unique** (peut-être, vis-à-vis) | ~4 778 |
| Inversion (-t-il) | **Token séparé** (`-t-il` = un token) | ~100+ |

**Décision** : reconstruire les formes de surface pour les MWT (`des` au lieu de `de` + `les`) car c'est ce que l'utilisateur fournit en entrée.

---

## Pipeline proposé

### Architecture : 2 CRFs + post-traitement (zéro dépendance)

```
Texte brut
    │
    ▼
Tokenisation (regex)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  CRF Morpho-Liaison (word-level, Viterbi)       │
│  Labels : POS|Morpho|Liaison (~200-250 tags)    │
│  Ex: "les" → ART:def|Plur|Lz                   │
│      "amis" → NOM|Masc|Plur                     │
│  Taille estimée : ~3-4 Mo JSON                  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  CRF G2P POS-aware (char-level, Viterbi)        │
│  Entrée : caractères + POS comme feature        │
│  Sortie : phonèmes IPA par mot                  │
│  Ex: "les" [ART:def] → "le"                    │
│  Taille estimée : ~800 Ko - 1 Mo JSON           │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Table de corrections G2P (JSON, optionnel)     │
│  g2p + g2p_pos (homographes)                    │
│  Taille : ~300 Ko                               │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Reconstruction des groupes (pur Python)        │
│  - Liaison : insérer le phonème latent          │
│  - Dénasalisation : bɔ̃ + Ln → bɔn              │
│  - Élision : concaténer (l + ɑ̃fɑ̃ → lɑ̃fɑ̃)      │
│  - Enchaînement : concaténer                    │
└─────────────────────────────────────────────────┘
    │
    ▼
Sortie : {mot, pos, morpho, phonème, liaison, groupe_phone}
```

**Total estimé : ~5 Mo, zéro dépendance Python.**

---

## Préparation du corpus d'entraînement

### Sources de données

| Source | Volume | Apporte |
|--------|--------|---------|
| Corpus UD French (train) | 17 968 phrases, ~375K tokens | POS, morpho, contexte phrastique |
| Dictionnaire GLAFF/Lexique | ~80 000 mots | Couverture lexicale G2P |
| `lectura_liaisons` | Algorithmique | Détermination des liaisons (POS-aware) |
| G2P existant (Seq2Seq) | Modèle entraîné | Phonémisation des mots du corpus |
| `aligneur.py` (lectura-main) | DFS français-spécialisé | Alignement graphème↔phonème |

### Pipeline de génération

```
Étape 1 : Charger corpus UD → reconstruire formes de surface (des, au, du)
Étape 2 : Pour chaque phrase → appliquer lectura_liaisons.classify()
          → déterminer les marqueurs Lz/Lt/Ln/Lr par mot
          → construire labels : POS|Morpho|Liaison
Étape 3 : Pour chaque mot → phonémiser (G2P Seq2Seq existant)
Étape 4 : Pour chaque mot → aligner graphèmes↔phonèmes (aligneur.py maison)
          → produire les labels char-level pour le CRF G2P

Résultat : corpus avec (mot, label_morpho_liaison, alignement_g2p)
```

### Entraînement mixte phrases + mots

Le CRF G2P peut être entraîné sur **deux sources combinées** :
- **Dictionnaire** (80K mots isolés) : couverture lexicale, cas rares
- **Phrases** (375K tokens en contexte) : POS réel, désambiguïsation homographes

Le CRF sklearn-crfsuite accepte une liste de séquences, on concatène les deux listes.

---

## Amélioration du CRF G2P (sans dépendance)

Le CRF G2P actuel est à 83,2%. Pistes d'amélioration sans ajouter de dépendance :

| Amélioration | Gain estimé | Effort |
|-------------|-------------|--------|
| Utiliser l'aligneur maison (DFS) au lieu de l'alignement naïf | +2-4% | Moyen |
| Features enrichies (digraphes, nasales, position fine, POS) | +1-3% | Faible |
| Plus de données (GLAFF complet + Lexique 3) | +1-2% | Faible |
| Entraînement mixte phrases + mots | +1-2% | Moyen |
| Plus de règles post-traitement (R14, R15...) | +1-2% | Faible |
| **Total estimé** | **~88-92%** | - |

Avec corrections : toujours ~100% sur le vocabulaire fréquent.

### Si on accepte des dépendances

| Niveau | Dépendance | Accuracy G2P | Taille runtime |
|--------|-----------|-------------|---------------|
| 0 — CRF amélioré | Aucune | ~88-92% | 0 Mo |
| 1 — BiLSTM numpy | numpy (~25 Mo) | ~90% | 25 Mo |
| 2 — BiLSTM/Seq2Seq ONNX | onnxruntime (~50 Mo) | 90-96,7% | 50 Mo |

---

## Compatibilité mobile

| Backend | Dépendance mobile | Taille runtime |
|---------|-------------------|---------------|
| CRF natif Swift/Kotlin | **Rien** (JSON + Viterbi ~80 lignes) | **0 Mo** |
| CRF → ONNX | onnxruntime-mobile | ~5 Mo |
| BiLSTM/Seq2Seq ONNX | onnxruntime-mobile | ~5 Mo |

Le format JSON du CRF est un atout : lisible par tous les langages. Le Viterbi est un algo simple (~80 lignes) portable en Swift/Kotlin/C++/Rust.

**Stratégie recommandée** : développer en Python (itération rapide), puis porter le Viterbi en natif ou exporter en ONNX pour mobile.

---

## Estimation des chances de réussite

| Composant | Accuracy estimée | Confiance |
|-----------|-----------------|-----------|
| Morpho (POS+traits) | ~88% tag complet, ~97% POS | Haute (prouvé) |
| Prédiction liaison | ~95-97% | Haute (quasi-déterministe depuis POS) |
| G2P CRF amélioré | ~88-92% intrinsèque | Moyenne-haute |
| G2P + corrections | ~99-100% | Haute |
| Reconstruction groupes | ~100% | Très haute (algorithmique) |

---

## Ressources clés dans le codebase

| Fichier | Rôle |
|---------|------|
| `MORPHO/MORPHO_CRF/lectura_morpho.py` | CRF Morpho existant (à étendre avec liaisons) |
| `MORPHO/MORPHO_CRF/entrainement/entrainer_crf.py` | Entraînement CRF Morpho (à étendre) |
| `G2P/G2P_CRF/lectura_g2p.py` | CRF G2P existant (à enrichir avec features POS) |
| `G2P/lectura_g2p.py` | Façade G2P multi-backend (référence API) |
| `Liaisons/lectura_liaisons.py` | Module liaisons (classify, merge, apply_jonctions) |
| `lectura-main/.../aligneur.py` | Aligneur graphème↔phonème DFS (363 lignes) |
| `lectura-main/.../phone_to_graphemes.csv` | Table phonème→graphèmes (65 entrées) |
| `POS/POS_CRF/entrainement/donnees/pos_train_merged.conllu` | Corpus UD train (17 968 phrases) |
| `POS/POS_CRF/modele/mini_lexique.json` | Mini-lexique correction POS mots-outils |

---

## Prochaines étapes

1. **Préparer le corpus** : script qui combine UD + G2P + Liaisons → données d'entraînement avec labels morpho-liaison
2. **Entraîner le CRF Morpho-Liaison** : étendre entrainer_crf.py avec les marqueurs Lz/Lt/Ln/Lr
3. **Améliorer le CRF G2P** : intégrer l'aligneur maison + features POS + entraînement mixte
4. **Implémenter la reconstruction des groupes** : fonction post-traitement (~50 lignes)
5. **Évaluer** : accuracy morpho, accuracy liaison, accuracy G2P, qualité des groupes reconstruits
6. **Itérer** : table de corrections, règles post-traitement, ajustements

---

## Questions ouvertes

- **Liaisons facultatives** : quelle politique ? Toujours appliquer, jamais, ou prédire ? (recommandation : choisir une politique cohérente dans le corpus)
- **Entraînement end-to-end (Solution C)** : un Seq2Seq phrase-level qui apprend les liaisons implicitement reste la cible à long terme, mais plus complexe
- **Quel nom pour le nouveau module ?** (suggestion : `G2P_Unifie` ou `Lectura_Phone`)

---

*Synthèse rédigée le 25 mars 2026 — Session de travail MORPHO_CRF + réflexion G2P multi-tâche*
