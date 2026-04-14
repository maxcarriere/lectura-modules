#!/bin/bash
# Setup script pour le VPS Lectura API
# A executer en root sur un Ubuntu 24.04 LTS
set -e

echo "=== Installation Lectura API ==="

# 1. Utilisateur
if ! id -u lectura &>/dev/null; then
    useradd -m -s /bin/bash lectura
    echo "Utilisateur lectura cree"
fi

# 2. Paquets systeme
apt update
apt install -y python3.11 python3.11-venv nginx certbot python3-certbot-nginx ufw

# 3. Pare-feu
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

# 4. Dossier applicatif
mkdir -p /opt/lectura
chown lectura:lectura /opt/lectura

# 5. Venv Python
su - lectura -c "
cd /opt/lectura
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Installer les modules Lectura en mode local complet
# pip install ./lectura_g2p_complet-*.whl
# pip install ./lectura_p2g_complet-*.whl
# pip install ./lectura_aligneur_complet-*.whl
# pip install lectura-formules lectura-tokeniseur lectura-lexique
"

# 6. Nginx
cp /opt/lectura/serveur/deploy/nginx.conf /etc/nginx/sites-available/lectura-api
ln -sf /etc/nginx/sites-available/lectura-api /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# 7. SSL (apres configuration DNS)
# certbot --nginx -d api.lec-tu-ra.com

# 8. Systemd
cp /opt/lectura/serveur/deploy/lectura-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable lectura-api
systemctl start lectura-api

echo "=== Installation terminee ==="
echo "Verifier : curl http://localhost:8000/health"
