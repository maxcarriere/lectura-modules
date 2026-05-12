# Évaluation — Lectura Liaisons v1.0

## Protocole

- **Module** : Moteur de liaisons et jonctions purement algorithmique (règles grammaticales + phonétiques)
- **Données de test** : Jeu de test intégré — 33 paires de mots annotées avec type de jonction attendu
- **Métriques** : Classification correcte (kind + type + phonème latent)
- **API évaluée** : `LecturaLiaisons.classify(w1, w2)` → `LiaisonDecision`

## Résultats

| Métrique | Correct | Total | Score |
|----------|---------|-------|-------|
| Classification complète (kind+typ+latent) | 33 | 33 | **100%** |
| Kind correct (gram/ench/none) | 33 | 33 | **100%** |
| Type correct (oblig/fac/int/none) | 33 | 33 | **100%** |

## Répartition du jeu de test

| Type de jonction | Nb paires | Exemples |
|------------------|-----------|----------|
| Liaison obligatoire | 23 | les‿enfants, un‿ami, très‿important |
| Liaison facultative | 1 | soldats‿anglais |
| Liaison interdite | 2 | et + alors, les + onze |
| Enchaînement | 3 | avec + elle, il + arrive |
| Pas de jonction | 4 | les + chats, les + haricots (h aspiré) |

## Cas couverts

### Liaisons obligatoires
- **ART + NOM/ADJ** : les‿enfants, des‿amis, les‿anciens
- **PRO:per + VER/AUX** : nous‿avons, vous‿êtes
- **PRE + mot** : dans‿un, sans‿effort, chez‿elle, en‿avance
- **ADV + ADJ** : très‿important, plus‿utile, moins‿important
- **ADJ + NOM** : petit‿enfant, grand‿ami, gros‿ours
- **AUX "est"** : est‿arrivé, est‿important
- **VER/AUX + ADJ** : sont‿arrivés

### Phonèmes latents
- **/z/** depuis `s`, `x`, `z` finals (les, plus, chez)
- **/t/** depuis `t`, `d` finals (petit, grand, est)
- **/n/** depuis morphologie nasale (un, en, bon)

### Blocages
- **"et"** : liaison toujours interdite
- **"onze"** : liaison bloquée
- **h aspiré** : haricots, héros → pas de liaison
- **Mot2 à initiale consonantique** : les chats → pas de liaison

### Enchaînements
- Consonne finale prononcée + voyelle : avec‿elle, il‿arrive

### Cas particulier
- **Dénasalisation** : bon‿ami → /n/ latent avec patch phonétique /ɔ̃/ → /ɔ/

## Points forts

- **Couverture grammaticale** : toutes les règles obligatoires du français standard
- **h aspiré** : liste embarquée de 863 mots (compressée zlib)
- **Zéro dépendance** : fonctionne en Python pur
- **API à deux niveaux** : par paires (`classify`) et par tokens (`apply_jonctions`)

## Limites connues

- **Pronoms avec consonne finale prononcée** : `ils`, `elles` → l'enchaînement est détecté plutôt que la liaison grammaticale (la consonne /l/ est déjà réalisée)
- **Liaisons après verbe inversé** : non couvert dans le jeu de test actuel
- **Registre** : les règles correspondent au français standard — le registre familier peut différer
- **Contexte prosodique** : pas de prise en compte du débit ou de l'emphase

## Caractéristiques techniques

| Propriété | Valeur |
|-----------|--------|
| Architecture | Règles grammaticales + phonétiques |
| Dépendances runtime | Aucune (Python pur) |
| Liste h aspiré | 863 mots (embarquée, compressée) |
| Python minimum | 3.10 |
| Entrée requise | Orthographe + IPA + POS tags |

## Lancer l'évaluation

```bash
python evaluer.py           # résumé
python evaluer.py --verbose  # détail des erreurs
```
