# MEMO — Audit Architecture & Commercial (Session du 12 mai 2026)

## Contexte

Audit global des modules Lectura : architecture, publication, presentation site, strategie commerciale.
Objectif : passer de la dimension MVP a la dimension Produit commercialisable.

---

## Architecture retenue

### Deux couches
- **Couche 1 — Briques atomiques** : Tokeniseur, Phonemiseur, Graphemiseur, Aligneur, Formules, Correcteur, Lexique
- **Couche 2 — Pipelines** : assemblent les briques, enrichissements optionnels via extras `[...]`
  - `lectura-g2p` (tokeniseur + formules + phonemiseur, extras: `[aligneur]`, futur `[lexique]`)
  - `lectura-tts` (futur)

### Modele economique
- Code AGPL-3.0 (copyleft, pas NC — le copyleft force l'achat de licence commerciale par les entreprises)
- Modeles prives : jamais publies, accessibles via API ou licence commerciale
- Export public (GitHub/PyPI) : code sans modeles → fallback API automatique
- Export prive (VPS/client) : code + modeles chiffres (.enc) + serveur

---

## Decisions prises / a prendre

### 1. Tokeniseur ↔ Formules : DECOUPLAGE REALISE ✅ → Publie Tokeniseur 2.3.0 + Formules 3.1.0

**Etat initial :** Formules dependait de Tokeniseur pour `maths.py` (tokenize_maths, MathToken, UNIT_NAMES_LOWER).

**Refactoring effectue (12 mai 2026) :**
1. `maths.py` copie dans `Formules/src/lectura_formules/_maths.py` (source de verite)
2. `maths.py` supprime du Tokeniseur
3. Tokeniseur importe depuis `lectura_formules._maths` si Formules installe
4. Sans Formules : mode degrade (unites non reconnues, enfants MATHS vides)
5. Dependance `lectura-tokeniseur` retiree du `pyproject.toml` de Formules

**Resultat : Les deux modules sont maintenant independants (zero-dep chacun).**
- 630 tests passent (134 Tokeniseur + 496 Formules)
- 51 tests G2P Pipeline passent
- Tokenisation complete fonctionne (formules, dates, maths, unites)

**Fichiers modifies :**
- `Formules/src/lectura_formules/_maths.py` — NOUVEAU (copie de maths.py)
- `Formules/src/lectura_formules/lecture_formules.py` — import depuis `._maths`
- `Formules/pyproject.toml` — `dependencies = []`
- `Tokeniseur/src/lectura_tokeniseur/maths.py` — SUPPRIME
- `Tokeniseur/src/lectura_tokeniseur/__init__.py` — try/except avec fallback None
- `Tokeniseur/src/lectura_tokeniseur/classification.py` — try/except avec fallback set()/[]

**Architecture resultante :**
```
Tokeniseur (zero-dep)           Formules (zero-dep)
├── Detecte les formules         ├── _maths.py (source de verite)
├── Classifie (DATE, TEL...)     ├── Lit/vocalise les formules
├── [si Formules] unites, MATHS  └── enrichir_formules()
└── [sinon] mode degrade
```

### 2. Meta-package `lectura`
- Actuellement v3.0.0, installe 7 modules a plat
- **Decision : le supprimer a court terme.** Chaque pipeline (lectura-g2p, lectura-tts)
  est un meilleur point d'entree. Pourra etre recree plus tard si besoin.

### 3. Licence AGPL vs NC
- **Decision : AGPL-3.0 confirme.** Standard industriel pour dual-licensing.
- Ajuster le discours : "open-source AGPL + licence commerciale" (pas "libre mais NC")

### 4. Chiffrement des modeles
- 5 modules ont _crypto.py : Phonemiseur, TTS-Mono, TTS-Multi, TTS-Diphone, VC
- Chiffrement XOR (obfuscation) dans les wheels prives uniquement
- Export PUBLIC exclut tous les modeles (ni .onnx ni .enc)
- **Graphemiseur : ajouter _crypto.py** (meme pattern que Phonemiseur) — a faire

### 5. Integration Lexique dans G2P Pipeline — Strategie Lookup

**Principe :** Le G2P est autonome pour ~99% de la langue francaise.
Le Lexique est un enrichissement optionnel qui permet :
- Lookup direct pour mots connus (bypass du reseau neural)
- Traitement des sigles qui ne se lisent pas lettre par lettre
- Extension a tous les mots via un lexique generaliste
- Possibilite de fournir un lexique technique/metier personnalise

**Architecture retenue (Niveau 1 — post-traitement, sans retrain) :**
- Avec `[lexique]` : le pipeline fait un lookup d'abord, le modele n'intervient qu'en backend
  pour les mots inconnus ou ambigus
- Sans `[lexique]` : comportement actuel (100% neural + homographes.json)
- Le Lexique mini ne contient que les colonnes essentielles : ortho / phone / POS (+ morpho ?)
- Note : certains phones du lexique ont ete completes via le phonemiseur

**Deux modes de livraison :**
- Version publique (PyPI) : `lectura-g2p[lexique]` → appel API au lexique complet
- Version privee : embarque un lexique-mini.db (ortho/phone/POS, taille reduite)

**API Lexique :**
- L'API sert le lexique complet (toutes colonnes) — la taille n'a pas d'importance en reseau
- Chaque module prive embarque son propre sous-lexique optimise pour son usage
- Endpoint type : `/lexique/lookup?mots=chat,UNESCO&champs=phone,pos`

**A concevoir :**
- [ ] Schema du lexique-mini (quelles colonnes, quelle taille cible)
- [ ] Endpoint API pour le lexique
- [ ] Mecanisme de lookup dans le pipeline G2P (avant/apres inference neurale)
- [ ] Support d'un lexique technique utilisateur en parametre

### 6. Correcteur
- Pret a publier/mettre a jour
- Un fix necessaire : glob pattern dans pyproject.toml pour data/g2p_v2/
- 497 tests passent, zero-dep, architecture propre

### 7. Graphemiseur
- Brique autonome, destinee au STT futur
- Ajouter _crypto.py pour chiffrer les modeles (meme pattern que Phonemiseur)

### 8. TTS/VC
- Travailles dans une autre instance (hors scope ici)

---

## Presentation site — Structure affinee

### Trois grandes sections

```
Solutions (B2B)
├── Briques de developpement         # ex "Modules Metiers" : G2P, Correction, TTS
│   ├── Phonetisation (G2P Pipeline)
│   ├── Correction orthographique
│   ├── Synthese vocale (TTS)
│   └── ...
├── Logiciels                        # Lectura Edition, Lectura Lexique, etc.
│   ├── Lectura Edition
│   └── Lectura Lexique
└── Services
    ├── Integration
    └── Editorial

Produits (B2C)
├── Applications
│   └── App Lecture, etc.
├── Livres
└── Videos / Medias

Technique (R&D / Open Source)
├── Modules (briques atomiques)
│   ├── Tokeniseur
│   ├── Phonemiseur
│   ├── Graphemiseur
│   ├── Aligneur
│   ├── Formules
│   ├── Correcteur
│   └── Lexique
├── Pipelines (modules metiers — doc technique)
│   ├── G2P Pipeline
│   ├── TTS Pipeline (futur)
│   └── ...
├── Lexique / API                   # acces API au lexique
└── Recherche                       # publications, articles techniques
```

### Articulation Solutions ↔ Technique
- **Solutions > Briques de developpement** = presentation orientee probleme/valeur
  ("Vous avez besoin de phonetiser du francais ?"), pricing, CTA, cas d'usage
- **Technique > Pipelines** = documentation technique detaillee du meme module
  (API reference, exemples de code, architecture, benchmarks)
- **Technique > Modules** = documentation des briques atomiques (pour devs avances)
- Lien croise : chaque page Solution renvoie vers la doc technique correspondante,
  et chaque page technique mentionne la Solution associee

### Noms des sections (a affiner)
- "Solutions" OK (standard B2B)
- "Produits" OK (standard B2C)
- "Technique" ou "R&D" ou "Developpeurs" ou "Open Source" — a choisir

---

## Prochaines etapes identifiees

### Corrections immediates
- [x] Decoupler Tokeniseur ↔ Formules (maths.py deplace, deps retirees) ✅ → Tokeniseur 2.3.0 + Formules 3.1.0
- [ ] Fix pyproject.toml Correcteur (glob g2p_v2)
- [ ] Mettre a jour texte "Six packages" sur le site

### Architecture
- [ ] Ajouter [lexique] comme extra dans lectura-g2p
- [ ] Concevoir le schema lexique-mini (ortho/phone/POS/morpho)
- [ ] Concevoir le mecanisme de lookup dans le pipeline G2P
- [ ] Ajouter _crypto.py au Graphemiseur
- [ ] Decider du sort du meta-package (suppression recommandee)

### Site / Commercial
- [ ] Restructurer le site en 3 sections (Solutions / Produits / Technique)
- [ ] Creer pages "Briques de developpement" orientees valeur/probleme
- [ ] Garder pages modules dans section Technique (doc detaillee)
- [ ] Creer page pricing
- [ ] Documenter l'API publique (endpoints, quotas, auth)
- [ ] Corriger terminologie licence (pas NC, c'est copyleft AGPL)
- [ ] Ajouter CTA commerciaux (contact, demo, essai)
