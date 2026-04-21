# Lectura Modules — Guide de developpement

## Workflow de publication

```
1. Developper dans workspace/Modules/
2. Exporter vers output/Modules/     (python exporter.py)
3. Pusher sur GitHub                  (git push dans output/Modules/)
4. Publier sur PyPI                   (twine upload dist/*)
5. Mettre a jour le site              (lectura-site, GitHub Pages)
```

### Etape 1 — Developpement (workspace)

Le code source est dans `/data/work/projets/lectura/workspace/Modules/`.
Chaque module est un package Python autonome dans son propre dossier :

```
Modules/
├── Tokeniseur/    pip install lectura-tokeniseur
├── G2P/           pip install lectura-g2p
├── P2G/           pip install lectura-p2g
├── Aligneur/      pip install lectura-aligneur
├── Formules/      pip install lectura-formules
├── Correcteur/    pip install lectura-correcteur  (en cours)
├── Lexique/       donnees linguistiques partagees
├── pyproject.toml meta-package "lectura" (installe tout)
└── exporter.py    script d'export vers output/
```

Installation en mode dev (editable) :
```bash
pip install -e Tokeniseur/
pip install -e G2P/[onnx]
```

Tests :
```bash
cd Tokeniseur && python -m pytest tests/
cd G2P && python -m pytest tests/
```

### Etape 2 — Export (workspace → output)

```bash
python exporter.py                        # export public + prive
python exporter.py --mode public          # public uniquement
python exporter.py --mode private         # prive uniquement
python exporter.py --dry-run              # apercu sans copie
```

Deux exports :

| Mode | Destination | Contenu |
|------|-------------|---------|
| public | `output/Modules/` | Code AGPL, sans modeles ni serveur → GitHub + PyPI |
| private | `output/Modules-private/` | Tout : code + modeles + serveur → VPS / client |

`output/Modules/` a son propre repo git (remote : `lectura-modules` sur GitHub).
`output/Modules-private/` n'a pas de repo git (copie autonome).

### Etape 3 — Push GitHub

```bash
cd /data/work/projets/lectura/workspace/output/Modules/
git add -A && git commit -m "Description des changements"
git push origin main
```

Le repo GitHub : `github.com/maxcarriere/lectura-modules`

### Etape 4 — Publier sur PyPI

```bash
cd output/Modules/<Module>/
python -m build
twine upload dist/*
```

Packages PyPI :
- `lectura` (meta-package)
- `lectura-tokeniseur`, `lectura-g2p`, `lectura-p2g`
- `lectura-aligneur`, `lectura-formules`

### Etape 5 — Mettre a jour le site

Le site est dans `/data/work/projets/lectura/site/` (repo `lectura-site`).
Jekyll + GitHub Pages, deploye automatiquement sur push.

Pages modules : `solutions/outils/modules/<module>.md`
Page listing : `solutions/outils/modules.md`

```bash
cd /data/work/projets/lectura/site/
git add -A && git commit -m "MAJ module X"
git push origin main
```

## Architecture des modules

### Backends d'inference (G2P, P2G)

Quatre backends disponibles :

| Backend | Dependance | Vitesse | Fichier poids |
|---------|-----------|---------|---------------|
| API | aucune | ~100 ms/phrase | aucun (serveur Lectura) |
| ONNX Runtime | `onnxruntime` | ~2 ms/phrase | `*_int8.onnx` (serveur) |
| NumPy | `numpy` | ~50 ms/phrase | `*_weights.json` (modeles_numpy/) |
| Pure Python | aucune | ~200 ms/phrase | `*_weights.json` (modeles_numpy/) |

Factory `creer_engine(mode, models_dir)` : cascade de detection des modeles
(parametre > env var > ~/.lectura/models/ > site-packages > API).
Les modeles ONNX ne sont PAS dans le wheel PyPI — produit payant a part.
L'utilisateur installe les modeles dans `~/.lectura/models/` ou utilise l'API.

### Zero dependance

Tokeniseur, Formules et Aligneur n'ont aucune dependance Python.
G2P et P2G fonctionnent sans dependance via le backend pur Python.

### Conventions

- Python 3.10+ avec type hints complets (PEP-561)
- Pas d'accents dans les noms de fichiers et le code source
- Double licence : AGPL-3.0 (libre) + Licence Commerciale (payante)
- Modeles pre-entraines sous MODEL_LICENCE.md

## Ajouter un nouveau module

1. Creer le dossier `Modules/<NomModule>/` avec structure standard :
   ```
   NomModule/
   ├── pyproject.toml
   ├── src/lectura_nommodule/
   │   ├── __init__.py
   │   └── ...
   ├── tests/
   ├── README.md
   ├── LICENCE.txt
   └── LICENCE-COMMERCIALE.md
   ```
2. Ajouter le module dans `Modules/pyproject.toml` (meta-package)
3. Ajouter le module dans `Modules/README.md`
4. Exporter, pusher GitHub, publier PyPI
5. Creer la page module sur le site (`solutions/outils/modules/<module>.md`)
6. Ajouter la carte dans `solutions/outils/modules.md`

## Repos Git du projet

| Composant | Chemin workspace | Repo GitHub |
|-----------|-----------------|-------------|
| Modules | `workspace/Modules/` | `lectura-modules` |
| Site | `site/` | `lectura-site` |
| Corpus | `workspace/Corpus/` | `lectura-corpus` |
| Lexique | `workspace/Lexique/` | `lectura-lexique` |
