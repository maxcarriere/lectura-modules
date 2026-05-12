#!/bin/bash
# Script de montage du disque de donnees (sda1) sur le VPS OVH
# A executer en ROOT via la console KVM du panel OVH
#
# Ce script :
# 1. Monte /dev/sda1 sur /data
# 2. Migre /opt/lectura vers /data/lectura (symlink)
# 3. Ajoute lectura au groupe sudo
# 4. Rend le montage permanent (fstab)
# 5. Redemarre le serveur API

set -e

echo "=== Montage du disque de donnees ==="

# Verifier qu'on est root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERREUR: ce script doit etre execute en root"
    exit 1
fi

# 1. Verifier le filesystem sur sda1
echo "1. Verification du filesystem sur /dev/sda1 ..."
FS_TYPE=$(blkid -o value -s TYPE /dev/sda1 2>/dev/null || echo "")
if [ -z "$FS_TYPE" ]; then
    echo "   Pas de filesystem detecte. Creation ext4..."
    mkfs.ext4 /dev/sda1
else
    echo "   Filesystem detecte: $FS_TYPE"
fi

# 2. Creer le point de montage et monter
echo "2. Montage de /dev/sda1 sur /data ..."
mkdir -p /data
mount /dev/sda1 /data
df -h /data
echo "   OK"

# 3. Migrer /opt/lectura vers /data
echo "3. Migration de /opt/lectura vers /data/lectura ..."
if [ -L /opt/lectura ]; then
    echo "   /opt/lectura est deja un symlink, rien a faire"
elif [ -d /opt/lectura ]; then
    cp -a /opt/lectura /data/lectura
    rm -rf /opt/lectura
    ln -s /data/lectura /opt/lectura
    echo "   Migration terminee"
else
    mkdir -p /data/lectura
    ln -s /data/lectura /opt/lectura
    echo "   Dossier cree"
fi
chown -R lectura:lectura /data/lectura

# 4. Ajouter au fstab pour montage permanent
echo "4. Ajout dans /etc/fstab ..."
if grep -q '/dev/sda1' /etc/fstab; then
    echo "   Deja present dans fstab"
else
    echo '/dev/sda1 /data ext4 defaults 0 2' >> /etc/fstab
    echo "   Ajoute"
fi

# 5. Ajouter lectura au groupe sudo
echo "5. Ajout de lectura au groupe sudo ..."
usermod -aG sudo lectura
echo "   OK"

# 6. Ajouter la cle SSH dans authorized_keys de root (optionnel)
echo "6. Copie cle SSH de lectura vers root ..."
mkdir -p /root/.ssh
if [ -f /home/lectura/.ssh/authorized_keys ]; then
    cp /home/lectura/.ssh/authorized_keys /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "   OK"
else
    echo "   Pas de cle trouvee pour lectura"
fi

# 7. Redemarrer le serveur API
echo "7. Redemarrage du serveur API ..."
if systemctl is-active --quiet lectura-api; then
    systemctl restart lectura-api
    echo "   Redemarrage OK"
else
    echo "   Service lectura-api non trouve, demarrage manuel :"
    echo "   su - lectura -c 'cd /opt/lectura && /opt/lectura/venv/bin/uvicorn serveur.app:app --host 0.0.0.0 --port 8000 --workers 2 &'"
fi

# 8. Verification finale
echo ""
echo "=== Verification ==="
echo "Disque:"
df -h /data
echo ""
echo "Symlink:"
ls -la /opt/lectura
echo ""
echo "Groupes lectura:"
groups lectura
echo ""
echo "Serveur:"
curl -s http://localhost:8000/health || echo "Serveur pas encore pret"
echo ""
echo "=== Termine ==="
echo "Espace disponible : $(df -h /data | tail -1 | awk '{print $4}')"
