# Architecture : Modules vs Modeles

Ce document explique la separation entre les deux produits Lectura.

## Deux produits distincts

```
workspace/
├── Modules/     ← Code source (AGPL, PyPI)
└── Modèles/     ← Fichiers modeles + outillage (commercial, archive)
```

### Modules/ — Le logiciel

Bibliotheques Python publiees sur PyPI sous licence AGPL-3.0.
Le developpeur peut utiliser le code gratuitement sous AGPL, ou obtenir une
licence commerciale pour l'integrer dans un produit proprietaire.

Chaque module contient un sous-dossier `modeles/` avec une **copie de
reference** du fichier modele necessaire au fonctionnement local :

```
Modules/Phonemiseur/src/lectura_phonemiseur/modeles/unifie_v2_int8.onnx
Modules/TTS-Diphone/src/lectura_tts_diphone/modeles/diphones.dpk.gz
```

Ces fichiers sont :
- **Exclus** de l'export public (PyPI, GitHub) via le pattern `"modeles/"`
- **Inclus** dans l'export prive (VPS, client commercial)

L'utilisateur gratuit utilise l'API. Le client payant recoit les modeles.

### Modeles/ — La ressource

Les fichiers modeles eux-memes + l'outillage complet pour les reproduire
ou les ameliorer. Vendus sous licence commerciale comme une archive autonome.

Structure type :
```
Modèles/TTS_Diphone/
├── modele/              ← fichier(s) de poids (source de verite)
│   └── diphones.dpk.gz
├── entrainement/        ← scripts de generation/entrainement
├── evaluation/          ← benchmarks, metriques
├── exemples/            ← usage standalone
└── README.md
```

Le client recoit l'archive et peut :
- Utiliser le modele tel quel
- Re-entrainer sur ses propres donnees
- Adapter la voix (voice conversion), les poids, etc.

## Flux des fichiers modeles

```
Modèles/<Module>/modele/<fichier>          ← source de verite (dev, training)
        │
        ▼  (copie manuelle apres generation)
Modules/<Module>/src/<pkg>/modeles/<fichier>  ← copie de reference (runtime)
        │
        ├── export public  → EXCLU (pattern "modeles/")
        └── export prive   → INCLUS → VPS + clients
```

## Export (exporter.py)

| Mode | Modeles inclus ? | Destination |
|------|-----------------|-------------|
| `--mode public` | Non | `output/Modules/` → GitHub + PyPI |
| `--mode private` | Oui | `output/Modules-private/` → VPS |

Le filtrage repose sur le pattern `"modeles/"` dans `PUBLIC_EXCLUDE_PATTERNS`.
Cela exclut tout sous-dossier `modeles/` quel que soit le module.

## Ajouter un modele a un nouveau module

1. Generer/entrainer le modele dans `Modèles/<Module>/`
2. Copier le fichier de poids dans `Modules/<Module>/src/<pkg>/modeles/`
3. `git add` le fichier dans le workspace (l'exporter utilise `git ls-files`)
4. Verifier : `python exporter.py --mode public --dry-run` (modele absent)
5. Verifier : `python exporter.py --mode private --dry-run` (modele present)
