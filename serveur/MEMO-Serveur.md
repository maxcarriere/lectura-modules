# MEMO Serveur — Lectura API

## Infos de connexion

| Info | Valeur |
|------|--------|
| Fournisseur | OVH — VPS-1 |
| Hostname | vps-57c3ed4e.vps.ovh.ca |
| IP | 192.99.247.55 |
| OS | Debian 12 (reinstalle le 2 mai 2026) |
| Region | Beauharnois (BHS), Canada |
| vCores | 4 |
| RAM | 8 Go |
| Stockage | 75 Go (sda1, monte sur /) |
| Domaine | api.lec-tu-ra.com |
| HTTPS | certbot, renouvellement auto |
| Panel OVH | https://manager.eu.ovhcloud.com/#/dedicated/vps/vps-57c3ed4e.vps.ovh.ca/dashboard |
| Date engagement | 15 avril 2026 |

## Connexion SSH

```bash
# Utilisateur applicatif (celui qu'on utilise)
ssh lectura@192.99.247.55

# Ou avec le raccourci (si configure dans ~/.ssh/config)
ssh lectura-vps

# Cle SSH locale
~/.ssh/id_ed25519  (commentaire: claude-code@lectura)
```

**3 comptes configurables** :

| Utilisateur | SSH key | Mot de passe | sudo |
|-------------|---------|-------------|------|
| `lectura` | oui | meme que les autres | NOPASSWD |
| `debian` | oui | meme que les autres | oui (avec mdp) |
| `root` | oui | meme que les autres | - |

Le mot de passe est le meme pour les 3 comptes (voir gestionnaire de mots de passe).

**Config SSH locale recommandee** (`~/.ssh/config`) :
```
Host lectura-vps
  HostName 192.99.247.55
  User lectura
  IdentityFile ~/.ssh/id_ed25519
```

## Architecture sur le VPS

```
/opt/lectura/              # Racine applicative
├── venv/                  # Python 3.11 virtualenv
└── serveur/               # FastAPI app + routers
    ├── app.py             # Point d'entree
    ├── auth.py            # Auth + rate limiting
    └── routers/           # g2p, p2g, aligneur, formules, tts, tts_diphone, correcteur

/home/lectura/.lectura/models/   # Modeles ONNX
├── g2p/                   # unifie_v2_int8.onnx, vocab, etc.
├── p2g/                   # unifie_p2g_v3_int8.onnx, vocab
├── tts_mono/              # fastpitch + hifigan (6 fichiers ONNX)
└── tts_diphone/           # diphones.dpk.gz, diphone_statistics.pkl
```

Les modules Python sont installes dans le venv depuis PyPI (pas de code source).

## Commandes courantes

```bash
# Verifier que le serveur tourne
curl http://localhost:8000/health

# Statut du service
sudo systemctl status lectura-api

# Voir les logs (temps reel)
sudo journalctl -u lectura-api -f

# Redemarrer le serveur
sudo systemctl restart lectura-api
# Ou : kill $(pgrep -f uvicorn)  → systemd relance automatiquement

# Installer/mettre a jour un module
/opt/lectura/venv/bin/pip install lectura-phonemiseur[onnx] --upgrade
sudo systemctl restart lectura-api

# Espace disque
df -h /
```

## Deploiement d'un nouveau module

```bash
# 1. Publier sur PyPI d'abord (depuis la machine locale)
# 2. Puis sur le VPS :
/opt/lectura/venv/bin/pip install lectura-NOM-MODULE --upgrade

# 3. Si le module a des modeles ONNX :
mkdir -p ~/.lectura/models/NOM_MODULE/
# Copier les fichiers ONNX depuis local :
# scp -i ~/.ssh/id_ed25519 fichier.onnx lectura@192.99.247.55:~/.lectura/models/NOM_MODULE/

# 4. Redemarrer le serveur
sudo systemctl restart lectura-api

# 5. Verifier
curl http://localhost:8000/health
```

## Endpoints API

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/health` | GET | Statut du serveur |
| `/g2p/analyser` | POST | Grapheme vers phoneme |
| `/p2g/analyser` | POST | Phoneme vers grapheme |
| `/aligneur/analyze` | POST | Alignement grapheme-phoneme + syllabation |
| `/formules/lire` | POST | Lecture de formules (nombres, dates, etc.) |
| `/correcteur/corriger` | POST | Correction orthographique et grammaticale |
| `/tts/synthesize` | POST | Synthese vocale monospeaker (audio base64) |
| `/tts-diphone/synthesize` | POST | Synthese vocale diphone WORLD (audio base64) |

## Packages installes sur le VPS

```
lectura-aligneur          4.0.0
lectura-correcteur        1.0.1
lectura-formules          3.0.1
lectura-g2p               4.1.0
lectura-phonemiseur       4.0.0
lectura-lexique           1.3.0
lectura-graphemiseur      4.0.0
lectura-tokeniseur        2.2.1
lectura-tts-monospeaker   1.4.0
lectura-tts-multispeaker  1.2.0
lectura-tts-diphone       1.4.0
```

---

## CE QU'IL NE FAUT PAS FAIRE

### Panel OVH — Pieges a eviter

1. **NE PAS cliquer "Reinstaller"** sur l'OS/Distribution
   - Ca EFFACE TOUT le serveur et reinstalle un OS vierge
   - Toutes les donnees, modules, modeles, config sont perdus
   - C'est IRREVERSIBLE

2. **Mode Rescue — Precautions**
   - Le mode rescue est un systeme temporaire qui demarre A COTE du systeme normal
   - Les donnees du disque principal ne sont PAS effacees
   - MAIS le serveur API ne tourne plus tant qu'on est en rescue
   - Pour sortir : Boot → `...` → "Booter sur le disque dur" → Redemarrer
   - Le mot de passe rescue est envoye par EMAIL (peut prendre du temps)
   - La cle SSH change entre rescue et mode normal → `ssh-keygen -R 192.99.247.55`

3. **NE PAS utiliser l'API OVH pour rebuild** sans precaution
   - Une tache echouee peut BLOQUER toutes les operations pendant des heures
   - Seul le support OVH peut purger une tache coincee
   - Toujours faire un snapshot AVANT

4. **Snapshot** : penser a en faire un AVANT toute operation risquee
   (Panel → Sauvegarde → Snapshot → Creer)

### SSH — Problemes courants

1. **"REMOTE HOST IDENTIFICATION HAS CHANGED"**
   - Normal apres un reboot rescue/normal ou une reinstallation
   - Solution : `ssh-keygen -f ~/.ssh/known_hosts -R 192.99.247.55`
   - Puis se reconnecter normalement

2. **"Permission denied (publickey)"**
   - La cle SSH n'est plus autorisee sur le serveur
   - Soit le serveur a ete reinstalle (→ reconfigurer authorized_keys)
   - Soit l'utilisateur n'existe plus

### Serveur API

1. **Ne pas modifier les fichiers sur le VPS directement**
   - Modifier dans workspace/Modules/serveur/, exporter, puis deployer

2. **Le serveur se relance automatiquement** si on kill uvicorn
   (systemd Restart=always, RestartSec=5)

3. **Docker/n8n** est installe mais desactive (ports 80/443 liberes pour nginx)

---

## Checklist apres reinstallation

Si le VPS a ete reinstalle (volontairement ou par erreur) :

- [ ] Se connecter (`ssh debian@IP` puis changer le mot de passe expire)
- [ ] Creer l'utilisateur lectura : `sudo useradd -m -s /bin/bash lectura`
- [ ] Configurer le mot de passe : `echo "lectura:MOT_DE_PASSE" | sudo chpasswd`
- [ ] Configurer le meme pour root : `echo "root:MOT_DE_PASSE" | sudo chpasswd`
- [ ] NOPASSWD sudo : `echo "lectura ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/lectura`
- [ ] Ajouter la cle SSH pour lectura, debian et root
- [ ] Desactiver Docker : `sudo systemctl stop docker docker.socket && sudo systemctl disable docker docker.socket`
- [ ] Installer deps : `sudo apt install python3-pip python3-venv nginx certbot python3-certbot-nginx espeak-ng`
- [ ] Creer le venv : `sudo mkdir -p /opt/lectura && sudo chown lectura:lectura /opt/lectura && python3 -m venv /opt/lectura/venv`
- [ ] Installer les modules : `pip install lectura-tokeniseur lectura-phonemiseur[onnx] lectura-graphemiseur[onnx] lectura-aligneur lectura-formules lectura-lexique lectura-correcteur lectura-tts-monospeaker[onnx,g2p] lectura-tts-diphone[all] uvicorn[standard] fastapi python-multipart slowapi`
- [ ] Copier les modeles dans `~/.lectura/models/` (g2p/, p2g/, tts_mono/, tts_diphone/)
- [ ] Copier le code serveur : `scp -r serveur/ lectura@IP:/opt/lectura/`
- [ ] Creer le service systemd (voir `/etc/systemd/system/lectura-api.service`)
- [ ] Configurer nginx (reverse proxy port 8000) + certbot pour api.lec-tu-ra.com
- [ ] Tester : `curl https://api.lec-tu-ra.com/health`
- [ ] Faire un snapshot OVH
