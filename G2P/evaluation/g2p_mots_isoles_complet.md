# Évaluation G2P — Modèle unifié — Mots isolés (rapport complet)

- **Modèle** : unifié BiLSTM multi-tâche (onnx)
- **Mode** : mot isolé (sans contexte phrastique)
- **Fréquences** : Lexique 3.83 (freqlivres)
- **Gold** : dico.csv (match si pred ∈ prononciations valides)
- **Filtres** : composés (apostrophe/tiret) exclus, homographes inclus

## Tranche 0→50% (couverture cumulée → 50%)

- Formes Lexique : 75
- Évaluées : 75
- **Word Acc** : 100.00% | o/ɔ-tol : 100.00% | o/ɔ+e/ɛ-tol : 100.00%
- **Erreurs** : 0

## Tranche 50→80% (couverture cumulée → 80%)

- Formes Lexique : 2,092
- Évaluées : 2,092
- **Word Acc** : 99.62% | o/ɔ-tol : 99.71% | o/ɔ+e/ɛ-tol : 100.00%
- **Erreurs** : 8

**Répartition des erreurs :**

| Catégorie | Nb | % |
|-----------|-----|---|
| voyelle mi-ouverte (ɛ/e, ɔ/o) | 8 | 100% |

| Mot | Fréq | Prédiction | Gold | Cat. |
|-----|------|------------|------|------|
| es | 334.3 | e | ɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ait | 148.7 | ɛ | e | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| jolie | 51.8 | ʒɔli | ʒoli | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayé | 45.7 | eseje | esɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| approcha | 42.7 | apʁɔʃa | apʁoʃa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essaie | 39.6 | esɛ | ɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| arrêtait | 37.1 | aʁetɛ | aʁɛtɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayait | 35.8 | esɛjɛ | esejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |

## Tranche 80→90% (couverture cumulée → 90%)

- Formes Lexique : 5,511
- Évaluées : 5,510
- **Word Acc** : 99.26% | o/ɔ-tol : 99.53% | o/ɔ+e/ɛ-tol : 99.98%
- **Erreurs** : 41

**Répartition des erreurs :**

| Catégorie | Nb | % |
|-----------|-----|---|
| voyelle mi-ouverte (ɛ/e, ɔ/o) | 40 | 98% |
| autre | 1 | 2% |

| Mot | Fréq | Prédiction | Gold | Cat. |
|-----|------|------------|------|------|
| pot | 32.3 | pɔ | po | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gaieté | 27.1 | ɡɛte | ɡete | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gêné | 25.9 | ʒene | ʒɛne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ignore | 23.0 | iɲɔʁ | iɲoʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayant | 22.6 | esɛjɑ̃ | esejɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| serrés | 19.9 | sɛʁe | seʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| flottait | 19.8 | flɔtɛ | flotɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pression | 19.1 | pʁesjɔ̃ | pʁɛsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| serrées | 19.1 | sɛʁe | seʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| quais | 18.9 | kɛ | ke | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empêchait | 18.6 | ɑ̃peʃɛ | ɑ̃pɛʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aimez | 18.4 | eme | ɛme | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cessait | 18.2 | sesɛ | sɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gaiement | 16.2 | ɡɛmɑ̃ | ɡemɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pots | 15.7 | pɔ | po | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| jolies | 15.7 | ʒɔli | ʒoli | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayais | 15.3 | esɛjɛ | esejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prof | 15.1 | pʁof | pʁɔf | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aigu | 14.5 | ɛɡy | eɡy | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| adressait | 14.1 | adʁesɛ | adʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| hocha | 14.1 | ɔʃa | oʃa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| vêtus | 14.1 | vɛty | vety | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| agonie | 13.5 | aɡɔni | aɡoni | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| grogna | 12.0 | ɡʁɔɲa | ɡʁoɲa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| devriez | 11.9 | dəvʁije | dəvʁje | autre |
| traitait | 10.3 | tʁetɛ | tʁɛtɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| desquels | 10.1 | dɛkɛl | dekɛl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| arrêtèrent | 10.0 | aʁetɛʁ | aʁɛtɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| guetter | 10.0 | ɡɛte | ɡete | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mollets | 9.9 | mɔlɛ | molɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigoler | 9.8 | ʁiɡɔle | ʁiɡole | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayai | 9.7 | esɛjɛ | esejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| arrêtaient | 9.6 | aʁetɛ | aʁɛtɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| desquelles | 9.4 | dɛkɛl | dekɛl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| jolis | 8.8 | ʒɔli | ʒoli | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| obsession | 8.8 | opsesjɔ̃ | ɔpsesjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| vieillir | 8.6 | vjɛjiʁ | vjejiʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| effrayé | 8.5 | efʁeje | efʁɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| frottait | 8.4 | fʁɔtɛ | fʁotɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigolade | 8.4 | ʁiɡɔlad | ʁiɡolad | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| laissai | 8.1 | lesɛ | lɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |

## Tranche 90→95% (couverture cumulée → 95%)

- Formes Lexique : 9,329
- Évaluées : 9,329
- **Word Acc** : 98.78% | o/ɔ-tol : 99.28% | o/ɔ+e/ɛ-tol : 100.00%
- **Erreurs** : 114

**Répartition des erreurs :**

| Catégorie | Nb | % |
|-----------|-----|---|
| voyelle mi-ouverte (ɛ/e, ɔ/o) | 113 | 99% |
| consonne finale muette | 1 | 1% |

| Mot | Fréq | Prédiction | Gold | Cat. |
|-----|------|------------|------|------|
| asseyez | 7.8 | aseje | asɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| frotta | 7.7 | fʁɔta | fʁota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| verrons | 7.6 | vɛʁɔ̃ | veʁɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bled | 7.2 | bled | blɛd | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| effrayait | 7.2 | efʁɛjɛ | efʁejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| chuchota | 7.1 | ʃyʃɔta | ʃyʃota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gémellaire | 7.1 | ʒemɛlɛʁ | ʒemelɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| conseillers | 6.9 | kɔ̃seje | kɔ̃sɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| agressif | 6.8 | aɡʁesif | aɡʁɛsif | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rayures | 6.8 | ʁɛjyʁ | ʁejyʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| arrêtez | 6.7 | aʁete | aʁɛte | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| progression | 6.6 | pʁoɡʁesjɔ̃ | pʁoɡʁɛsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rapprocha | 6.6 | ʁapʁɔʃa | ʁapʁoʃa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| agressive | 6.2 | aɡʁesiv | aɡʁɛsiv | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| arrêtai | 6.2 | aʁetɛ | aʁɛtɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayaient | 6.1 | esɛjɛ | esejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ensoleillée | 5.9 | ɑ̃soleje | ɑ̃solɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| renseigné | 5.8 | ʁɑ̃sɛɲe | ʁɑ̃seɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| accrocha | 5.7 | akʁɔʃa | akʁoʃa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aigus | 5.7 | ɛɡy | eɡy | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| effrayée | 5.5 | efʁeje | efʁɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| vêtues | 5.5 | vety | vɛty | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coke | 5.5 | kɔk | kok | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dressaient | 5.5 | dʁesɛ | dʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| décrocha | 5.5 | dekʁɔʃa | dekʁoʃa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aisselles | 5.3 | esɛl | ɛsɛl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| polie | 5.3 | pɔli | poli | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ignorent | 5.1 | iɲɔʁ | iɲoʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| raccrocha | 5.1 | ʁakʁɔʃa | ʁakʁoʃa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coma | 4.9 | kɔma | koma | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| frayer | 4.9 | fʁɛje | fʁeje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| effrayer | 4.7 | efʁeje | efʁɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ho | 4.7 | ɔ | o | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| laissèrent | 4.7 | lesɛʁ | lɛsɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mesdames | 4.7 | medam | mɛdam | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| arrêtais | 4.7 | aʁetɛ | aʁɛtɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| colla | 4.7 | kɔla | kola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empêchaient | 4.6 | ɑ̃peʃɛ | ɑ̃pɛʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| flottante | 4.6 | flɔtɑ̃t | flotɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| agression | 4.5 | aɡʁesjɔ̃ | aɡʁɛsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cessaient | 4.5 | sesɛ | sɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| téléphona | 4.5 | telefɔna | telefona | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| clochards | 4.5 | klɔʃaʁ | kloʃaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| colossal | 4.5 | kolɔsal | kɔlɔsal | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| exceptions | 4.5 | ɛksɛpsjɔ̃ | eksɛpsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pernod | 4.5 | pɛʁnɔ | pɛʁno | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enveloppa | 4.4 | ɑ̃vəlɔpa | ɑ̃vəlopa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| verrous | 4.4 | vɛʁu | veʁu | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| godet | 4.3 | ɡɔdɛ | ɡodɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| accommoder | 4.3 | akomɔde | akomode | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bottines | 4.3 | bɔtin | botin | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pressent | 4.3 | pʁɛs | pʁes | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| promena | 4.3 | pʁɔməna | pʁoməna | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cessèrent | 4.2 | sesɛʁ | sɛsɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| logé | 4.2 | lɔʒe | loʒe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescendit | 4.2 | ʁədesɑ̃di | ʁədɛsɑ̃di | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gais | 4.1 | ɡɛ | ɡe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| incessant | 4.1 | ɛ̃sɛsɑ̃ | ɛ̃sesɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| treillis | 4.1 | tʁɛji | tʁeji | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigolo | 4.1 | ʁiɡɔlo | ʁiɡolo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| collections | 4.0 | kolɛksjɔ̃ | kɔlɛksjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coiffer | 4.0 | kwafɛ | kwafe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enseigné | 3.9 | ɑ̃sɛɲe | ɑ̃seɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ensoleillé | 3.9 | ɑ̃soleje | ɑ̃solɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| poker | 3.8 | pɔkeʁ | pɔke | consonne finale muette |
| aisselle | 3.7 | esɛl | ɛsɛl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| baigné | 3.7 | bɛɲe | beɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| blessait | 3.6 | blesɛ | blɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essaient | 3.6 | esɛ | ɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressaillit | 3.6 | tʁɛsaji | tʁesaji | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| desserrer | 3.6 | desɛʁe | dɛsɛʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| incessamment | 3.6 | ɛ̃sɛsamɑ̃ | ɛ̃sesamɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| obsédante | 3.6 | ɔpsedɑ̃t | opsedɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| notions | 3.5 | nosjɔ̃ | nɔsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| colombe | 3.5 | kɔlɔ̃b | kolɔ̃b | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| diagnostic | 3.5 | djaɡnɔstik | djaɡnostik | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cantonade | 3.5 | kɑ̃tɔnad | kɑ̃tonad | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| celluloïd | 3.4 | sɛlylɔid | selylɔid | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| embêté | 3.4 | ɑ̃bete | ɑ̃bɛte | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayons | 3.3 | esɛjɔ̃ | esejɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| effrayés | 3.3 | efʁeje | efʁɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| excellents | 3.2 | eksɛlɑ̃ | ɛksɛlɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| apaisait | 3.2 | apezɛ | apɛzɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| atrocement | 3.2 | atʁosəmɑ̃ | atʁɔsəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| brochet | 3.2 | bʁɔʃɛ | bʁoʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| godets | 3.2 | ɡɔdɛ | ɡodɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| orangers | 3.2 | ɔʁɑ̃ʒe | oʁɑ̃ʒe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| serrures | 3.2 | sɛʁyʁ | seʁyʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tapota | 3.2 | tapɔta | tapota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| odorante | 3.1 | odɔʁɑ̃t | odoʁɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| adressaient | 3.0 | adʁesɛ | adʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dorures | 3.0 | doʁyʁ | dɔʁyʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| déchaîné | 3.0 | deʃene | deʃɛne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescendu | 3.0 | ʁədesɑ̃dy | ʁədɛsɑ̃dy | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cahots | 3.0 | kaɔ | kao | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dégringoler | 3.0 | deɡʁɛ̃ɡɔle | deɡʁɛ̃ɡole | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| incessantes | 3.0 | ɛ̃sɛsɑ̃t | ɛ̃sesɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rêvée | 3.0 | ʁɛve | ʁeve | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| convoqua | 2.9 | kɔ̃vɔka | kɔ̃voka | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| excellentes | 2.9 | ɛksɛlɑ̃t | ɛkselɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| formations | 2.8 | fɔʁmasjɔ̃ | foʁmasjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| impressionnait | 2.8 | ɛ̃pʁesjonɛ | ɛ̃pʁɛsjonɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescend | 2.8 | ʁədesɑ̃ | ʁədɛsɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| émeraudes | 2.8 | eməʁod | ɛməʁod | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| blottie | 2.8 | blɔti | bloti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aidez | 2.8 | ede | ɛde | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| allégrement | 2.8 | aleɡʁəmɑ̃ | alɛɡʁəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dominos | 2.8 | dɔmino | domino | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dépêchez | 2.8 | depeʃe | depɛʃe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| faiblir | 2.8 | fɛbliʁ | febliʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| godillots | 2.8 | ɡɔdijo | ɡodijo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| holà | 2.8 | ɔla | ola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| quolibets | 2.8 | kɔlibɛ | kolibɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| retraités | 2.8 | ʁətʁete | ʁətʁɛte | voyelle mi-ouverte (ɛ/e, ɔ/o) |

## Tranche 95→99% (couverture cumulée → 99%)

- Formes Lexique : 30,805
- Évaluées : 30,802
- **Word Acc** : 98.07% | o/ɔ-tol : 98.94% | o/ɔ+e/ɛ-tol : 100.00%
- **Erreurs** : 595

**Répartition des erreurs :**

| Catégorie | Nb | % |
|-----------|-----|---|
| voyelle mi-ouverte (ɛ/e, ɔ/o) | 586 | 98% |
| schwa parasite | 5 | 1% |
| autre | 4 | 1% |

| Mot | Fréq | Prédiction | Gold | Cat. |
|-----|------|------------|------|------|
| aboli | 2.7 | abɔli | aboli | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| arrêtons | 2.7 | aʁetɔ̃ | aʁɛtɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| intéressais | 2.7 | ɛ̃teʁesɛ | ɛ̃teʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rayée | 2.7 | ʁɛje | ʁeje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| verront | 2.7 | vɛʁɔ̃ | veʁɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| conter | 2.6 | kɔ̃tɛ | kɔ̃te | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| impeccablement | 2.6 | ɛ̃pɛkabləmɑ̃ | ɛ̃pekabləmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigolant | 2.6 | ʁiɡɔlɑ̃ | ʁiɡolɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| agresseur | 2.6 | aɡʁesœʁ | aɡʁɛsœʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rayés | 2.6 | ʁɛje | ʁeje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| déchaînée | 2.6 | deʃene | deʃɛne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| adressai | 2.6 | adʁesɛ | adʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| erré | 2.6 | eʁe | ɛʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| promenés | 2.6 | pʁɔməne | pʁoməne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dockers | 2.5 | dɔkɛʁ | dokɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| guetté | 2.5 | ɡɛte | ɡete | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| payaient | 2.5 | pɛjɛ | pejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| vieillissement | 2.5 | vjɛjisəmɑ̃ | vjejisəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| boniments | 2.4 | bɔnimɑ̃ | bonimɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| envola | 2.4 | ɑ̃vɔla | ɑ̃vola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| homogène | 2.4 | omɔʒɛn | omoʒɛn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tomes | 2.4 | tom | tɔm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| toussota | 2.4 | tusɔta | tusota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| zen | 2.4 | zen | zɛn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| économe | 2.4 | ekonom | ekonɔm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| obsessions | 2.4 | opsesjɔ̃ | ɔpsesjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pilori | 2.4 | pilɔʁi | piloʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| stoppa | 2.4 | stɔpa | stopa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressaillement | 2.4 | tʁɛsajəmɑ̃ | tʁesajəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| vieillis | 2.4 | vjɛji | vjeji | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| confettis | 2.3 | kɔ̃fɛti | kɔ̃feti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gaies | 2.3 | ɡɛ | ɡe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| potins | 2.3 | pɔtɛ̃ | potɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescendait | 2.3 | ʁədesɑ̃dɛ | ʁədɛsɑ̃dɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| reprocha | 2.3 | ʁəpʁɔʃa | ʁəpʁoʃa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ferrées | 2.2 | fɛʁe | feʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| flottants | 2.2 | flɔtɑ̃ | flotɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| maigrir | 2.2 | mɛɡʁiʁ | meɡʁiʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| monotones | 2.2 | mɔnotɔn | monotɔn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| affola | 2.2 | afɔla | afola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cessai | 2.2 | sesɛ | sɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| libellule | 2.2 | libɛlyl | libelyl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| renseignés | 2.2 | ʁɑ̃sɛɲe | ʁɑ̃seɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rétrospective | 2.2 | ʁetʁɔspɛktiv | ʁetʁospɛktiv | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tessons | 2.2 | tɛsɔ̃ | tesɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| blottit | 2.1 | blɔti | bloti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| chopine | 2.1 | ʃɔpin | ʃopin | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cogna | 2.1 | kɔɲa | koɲa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gober | 2.1 | ɡɔbe | ɡobe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| moellons | 2.1 | mwɛlɔ̃ | mwelɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pivota | 2.1 | pivɔta | pivota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rainures | 2.1 | ʁɛnyʁ | ʁenyʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| saigné | 2.1 | sɛɲe | seɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coquins | 2.0 | kɔkɛ̃ | kokɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| délaissé | 2.0 | delese | delɛse | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| caressais | 2.0 | kaʁesɛ | kaʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| atterrit | 2.0 | ateʁi | atɛʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| baignée | 2.0 | bɛɲe | beɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| boa | 2.0 | bɔa | boa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ferrés | 2.0 | fɛʁe | feʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| home | 2.0 | ɔm | om | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigolard | 2.0 | ʁiɡɔlaʁ | ʁiɡolaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| vola | 2.0 | vɔla | vola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| oppressant | 1.9 | opʁɛsɑ̃ | opʁesɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ensommeillée | 1.9 | ɑ̃someje | ɑ̃somɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| apostolique | 1.9 | apɔstolik | apostolik | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| atterri | 1.9 | ateʁi | atɛʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bulldozer | 1.9 | byldɔze | byldoze | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dévorante | 1.9 | devɔʁɑ̃t | devoʁɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| vénitienne | 1.8 | venitjɛn | venisjɛn | autre |
| cessez | 1.8 | sese | sɛse | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dégringolade | 1.8 | deɡʁɛ̃ɡɔlad | deɡʁɛ̃ɡolad | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| homards | 1.8 | ɔmaʁ | omaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| impeccables | 1.8 | ɛ̃pɛkabl | ɛ̃pekabl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| incessants | 1.8 | ɛ̃sɛsɑ̃ | ɛ̃sesɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| lettré | 1.8 | lɛtʁe | letʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| obsédait | 1.8 | ɔpsedɛ | opsedɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pollen | 1.8 | pɔlen | pɔlɛn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| stones | 1.8 | stɔn | ston | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bégayait | 1.8 | beɡɛjɛ | beɡejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dressage | 1.8 | dʁesaʒ | dʁɛsaʒ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essouffle | 1.8 | esufl | ɛsufl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| hennissement | 1.8 | ɛnisəmɑ̃ | enisəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rayer | 1.8 | ʁɛje | ʁeje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| socquettes | 1.8 | sɔkɛt | sokɛt | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| zonard | 1.8 | zɔnaʁ | zonaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| asseyons | 1.7 | asejɔ̃ | asɛjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| baignés | 1.7 | bɛɲe | beɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| desserré | 1.7 | desɛʁe | deseʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayent | 1.7 | esɛj | esej | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gigolo | 1.7 | ʒiɡɔlo | ʒiɡolo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gloriole | 1.7 | ɡlɔʁjɔl | ɡloʁjɔl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mollard | 1.7 | mɔlaʁ | molaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| opine | 1.7 | ɔpin | opin | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescends | 1.7 | ʁədesɑ̃ | ʁədɛsɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigolos | 1.7 | ʁiɡɔlo | ʁiɡolo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sanglota | 1.7 | sɑ̃ɡlɔta | sɑ̃ɡlota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| téléphonez | 1.7 | telefɔne | telefone | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| peigné | 1.6 | pɛɲe | peɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aiguiser | 1.6 | ɛɡize | eɡize | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| amaigri | 1.6 | amɛɡʁi | ameɡʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| charognards | 1.6 | ʃaʁɔɲaʁ | ʃaʁoɲaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ortolans | 1.6 | ɔʁtɔlɑ̃ | ɔʁtolɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| renseignée | 1.6 | ʁɑ̃sɛɲe | ʁɑ̃seɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| veiné | 1.6 | vɛne | vene | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| resserré | 1.6 | ʁəsɛʁe | ʁəseʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| apostrophe | 1.6 | apɔstʁɔf | apostʁɔf | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bobards | 1.6 | bɔbaʁ | bobaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bobo | 1.6 | bɔbo | bɔbɔ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| magnolias | 1.6 | maɲɔlja | maɲolja | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| oppressante | 1.6 | opʁɛsɑ̃t | opʁesɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| papeteries | 1.6 | papɛtəʁi | papetəʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| peignés | 1.6 | pɛɲe | peɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| potin | 1.6 | pɔtɛ̃ | potɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prêchait | 1.6 | pʁeʃɛ | pʁɛʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescendant | 1.6 | ʁədesɑ̃dɑ̃ | ʁədɛsɑ̃dɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressaillait | 1.6 | tʁɛsajɛ | tʁesajɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| adressais | 1.5 | adʁesɛ | adʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| blottis | 1.5 | blɔti | bloti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| borborygmes | 1.5 | bɔʁbɔʁiɡm | bɔʁboʁiɡm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bovins | 1.5 | bɔvɛ̃ | bovɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| confetti | 1.5 | kɔ̃fɛti | kɔ̃feti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| florins | 1.5 | flɔʁɛ̃ | floʁɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ignora | 1.5 | iɲɔʁa | iɲoʁa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| impressionna | 1.5 | ɛ̃pʁesjona | ɛ̃pʁɛsjona | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mijotait | 1.5 | miʒɔtɛ | miʒotɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressaillements | 1.5 | tʁɛsajəmɑ̃ | tʁesajəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| déterré | 1.4 | detɛʁe | deteʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aigri | 1.4 | ɛɡʁi | eɡʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aiguillage | 1.4 | ɛɡɥijaʒ | eɡɥijaʒ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| arrêtât | 1.4 | aʁeta | aʁɛta | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cahot | 1.4 | kaɔ | kao | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| consola | 1.4 | kɔ̃sɔla | kɔ̃sola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| décocha | 1.4 | dekɔʃa | dekoʃa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| frottaient | 1.4 | fʁɔtɛ | fʁotɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| jockeys | 1.4 | ʒɔkɛ | ʒokɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| macaronis | 1.4 | makaʁɔni | makaʁoni | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| plaisez | 1.4 | pleze | plɛze | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rainure | 1.4 | ʁɛnyʁ | ʁenyʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescendus | 1.4 | ʁədesɑ̃dy | ʁədɛsɑ̃dy | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigola | 1.4 | ʁiɡɔla | ʁiɡola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enneigées | 1.4 | ɑ̃nɛʒe | ɑ̃neʒe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bégayant | 1.4 | beɡɛjɑ̃ | beɡejɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| acolytes | 1.4 | akɔlit | akolit | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| daigné | 1.4 | dɛɲe | deɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| décolla | 1.4 | dekɔla | dekola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| effrayaient | 1.4 | efʁɛjɛ | efʁejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empiétements | 1.4 | ɑ̃pjetəmɑ̃ | ɑ̃pjɛtəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| endosses | 1.4 | ɑ̃dɔs | ɑ̃dos | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essaims | 1.4 | esɛ̃ | ɛsɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| frayait | 1.4 | fʁɛjɛ | fʁejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| neigeuse | 1.4 | nɛʒøz | neʒøz | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| odorantes | 1.4 | odɔʁɑ̃t | odoʁɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| promenèrent | 1.4 | pʁɔmənɛʁ | pʁomənɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prélever | 1.4 | pʁɛləve | pʁeləve | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rafraîchit | 1.4 | ʁafʁɛʃi | ʁafʁeʃi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| raisiné | 1.4 | ʁezine | ʁɛzine | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| verrues | 1.4 | vɛʁy | veʁy | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pessimistes | 1.3 | pesimist | pɛsimist | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ensommeillé | 1.3 | ɑ̃someje | ɑ̃somɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| obsédée | 1.3 | ɔpsede | ɔbsede | autre |
| bedonnant | 1.3 | bədonɑ̃ | bødonɑ̃ | schwa parasite |
| bloqua | 1.3 | blɔka | bloka | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cessais | 1.3 | sesɛ | sɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cessât | 1.3 | sesa | sɛsa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empêchèrent | 1.3 | ɑ̃peʃɛʁ | ɑ̃pɛʃɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| excellait | 1.3 | ɛkselɛ | ɛksɛlɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| hiéroglyphes | 1.3 | jeʁɔɡlif | jeʁoɡlif | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| isola | 1.3 | izɔla | izola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| logée | 1.3 | lɔʒe | loʒe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| maigrichon | 1.3 | mɛɡʁiʃɔ̃ | meɡʁiʃɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| majordome | 1.3 | maʒɔʁdom | maʒɔʁdɔm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| métronome | 1.3 | metʁonom | metʁonɔm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| omelettes | 1.3 | ɔməlɛt | oməlɛt | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| onomatopées | 1.3 | ɔnɔmatope | onomatope | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| overdose | 1.3 | ovɛʁdoz | ɔvɛʁdoz | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| papeterie | 1.3 | papɛtəʁi | papɛtʁi | schwa parasite |
| projeta | 1.3 | pʁɔʒəta | pʁoʒəta | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| suraigu | 1.3 | syʁɛɡy | syʁeɡy | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| accession | 1.2 | aksesjɔ̃ | aksɛsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| botter | 1.2 | bɔte | bote | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cacophonie | 1.2 | kakofɔni | kakɔfɔni | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| colombelle | 1.2 | kɔlɔ̃bɛl | kolɔ̃bɛl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| corons | 1.2 | kɔʁɔ̃ | koʁɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| hochets | 1.2 | ɔʃɛ | oʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| maîtrisant | 1.2 | metʁizɑ̃ | mɛtʁizɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mercurochrome | 1.2 | mɛʁkyʁɔkʁom | mɛʁkyʁokʁom | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mêlèrent | 1.2 | melɛʁ | mɛlɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| peignée | 1.2 | pɛɲe | peɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| potelée | 1.2 | potəle | pɔtəle | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| relayaient | 1.2 | ʁəlɛjɛ | ʁəlejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| toboggan | 1.2 | tɔbɔɡɑ̃ | toboɡɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressailli | 1.2 | tʁɛsaji | tʁesaji | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| verrouillée | 1.2 | vɛʁuje | veʁuje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| étonnements | 1.2 | etɔnəmɑ̃ | ɛtɔnəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| fainéants | 1.2 | fɛneɑ̃ | feneɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| belligérants | 1.2 | bɛliʒeʁɑ̃ | beliʒeʁɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| blêmit | 1.1 | blemi | blɛmi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cajoler | 1.1 | kaʒɔle | kaʒole | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dodelinant | 1.1 | dɔdəlinɑ̃ | dɔdlinɑ̃ | schwa parasite |
| dolmen | 1.1 | dɔlmen | dɔlmɛn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dressèrent | 1.1 | dʁesɛʁ | dʁɛsɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dédaigné | 1.1 | dedɛɲe | dedeɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dépêchait | 1.1 | depeʃɛ | depɛʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enneigée | 1.1 | ɑ̃nɛʒe | ɑ̃neʒe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| flageolets | 1.1 | flaʒɔlɛ | flaʒolɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| galopante | 1.1 | ɡalɔpɑ̃t | ɡalopɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| hennissements | 1.1 | ɛnisəmɑ̃ | enisəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| homo | 1.1 | ɔmo | omo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| interpellait | 1.1 | ɛ̃tɛʁpelɛ | ɛ̃tɛʁpɛlɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pressens | 1.1 | pʁɛsɑ̃ | pʁesɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prosterné | 1.1 | pʁɔstɛʁne | pʁostɛʁne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| revêtit | 1.1 | ʁəveti | ʁəvɛti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| roblot | 1.1 | ʁoblo | ʁɔblo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sifflota | 1.1 | siflɔta | siflota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| surélevé | 1.1 | syʁɛləve | syʁeləve | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| acquiescé | 1.1 | akjese | akjɛse | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| apostolat | 1.1 | apɔstola | apostola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| autodidacte | 1.1 | otɔdidakt | otodidakt | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| blessaient | 1.1 | blesɛ | blɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| blessante | 1.1 | blesɑ̃t | blɛsɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dépareillés | 1.1 | depaʁeje | depaʁɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empocha | 1.1 | ɑ̃pɔʃa | ɑ̃poʃa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| freiné | 1.1 | fʁɛne | fʁene | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| implora | 1.1 | ɛ̃plɔʁa | ɛ̃ploʁa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| impressionnistes | 1.1 | ɛ̃pʁesjonist | ɛ̃pʁɛsjonist | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| lessivé | 1.1 | lesive | lɛsive | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| monosyllabes | 1.1 | mɔnɔsilab | monosilab | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| neurones | 1.1 | nøʁɔn | nøʁon | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| polar | 1.1 | pɔlaʁ | polaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| proclama | 1.1 | pʁɔklama | pʁoklama | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rafraîchissait | 1.1 | ʁafʁɛʃisɛ | ʁafʁeʃisɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rafraîchissements | 1.1 | ʁafʁɛʃisəmɑ̃ | ʁafʁeʃisəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redressai | 1.1 | ʁədʁesɛ | ʁədʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rétrospectivement | 1.1 | ʁetʁɔspɛktivəmɑ̃ | ʁetʁɔspɛktivmɑ̃ | schwa parasite |
| spaghettis | 1.1 | spaɡɛti | spaɡeti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| transgression | 1.1 | tʁɑ̃sɡʁesjɔ̃ | tʁɑ̃sɡʁɛsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressaille | 1.1 | tʁɛsaj | tʁesaj | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| trolleybus | 1.1 | tʁɔlɛbys | tʁolɛbys | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| veinules | 1.1 | vɛnyl | venyl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| yoga | 1.1 | jɔɡa | joɡa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| écheveaux | 1.1 | eʃəvo | ɛʃəvo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dépareillées | 1.0 | depaʁeje | depaʁɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gainées | 1.0 | ɡene | ɡɛne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| chopé | 1.0 | ʃɔpe | ʃope | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dégringola | 1.0 | deɡʁɛ̃ɡɔla | deɡʁɛ̃ɡola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empêtrait | 1.0 | ɑ̃petʁɛ | ɑ̃pɛtʁɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| exhibitionniste | 1.0 | ɛzibisjonist | ɛzibisjɔnist | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| frottais | 1.0 | fʁɔtɛ | fʁotɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| globules | 1.0 | ɡlobyl | ɡlɔbyl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gomina | 1.0 | ɡɔmina | ɡomina | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| interjections | 1.0 | ɛ̃tɛʁʒɛksjɔ̃ | ɛ̃tɛʁʒeksjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ironisa | 1.0 | iʁɔniza | iʁoniza | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| options | 1.0 | ɔpsjɔ̃ | opsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prescience | 1.0 | pʁɛsjɑ̃s | pʁesjɑ̃s | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rafraîchi | 1.0 | ʁafʁɛʃi | ʁafʁeʃi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rayées | 1.0 | ʁɛje | ʁeje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescendirent | 1.0 | ʁədesɑ̃diʁ | ʁədɛsɑ̃diʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| soda | 1.0 | sɔda | soda | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| verroterie | 1.0 | vɛʁɔtəʁi | vɛʁɔtʁi | schwa parasite |
| maigrichonne | 1.0 | mɛɡʁiʃɔn | meɡʁiʃɔn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cessation | 0.9 | sesasjɔ̃ | sɛsasjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| compresseur | 0.9 | kɔ̃pʁesœʁ | kɔ̃pʁɛsœʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cresson | 0.9 | kʁɛsɔ̃ | kʁesɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dorure | 0.9 | doʁyʁ | dɔʁyʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| embêtait | 0.9 | ɑ̃betɛ | ɑ̃bɛtɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essieux | 0.9 | esjø | ɛsjø | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| holocauste | 0.9 | ɔlokost | olokost | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| intellect | 0.9 | ɛ̃tɛlɛkt | ɛ̃telɛkt | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| malaisément | 0.9 | malɛzemɑ̃ | malezemɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| maraîchers | 0.9 | maʁɛʃe | maʁeʃe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| meccano | 0.9 | mɛkano | mekano | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| obnubilé | 0.9 | obnybile | ɔbnybile | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| opina | 0.9 | ɔpina | opina | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| origan | 0.9 | ɔʁiɡɑ̃ | oʁiɡɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pêchait | 0.9 | peʃɛ | pɛʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ronchonna | 0.9 | ʁɔ̃ʃɔna | ʁɔ̃ʃona | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| réveillez | 0.9 | ʁeveje | ʁevɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sanguinolent | 0.9 | sɑ̃ɡinɔlɑ̃ | sɑ̃ɡinolɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sophistiquée | 0.9 | sɔfistike | sofistike | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressautait | 0.9 | tʁesotɛ | tʁɛsotɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressaute | 0.9 | tʁesot | tʁɛsot | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| têtus | 0.9 | tɛty | tety | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| émeri | 0.9 | eməʁi | ɛməʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| angora | 0.9 | ɑ̃ɡɔʁa | ɑ̃ɡoʁa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| embêtée | 0.9 | ɑ̃bete | ɑ̃bɛte | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| adosse | 0.9 | adɔs | ados | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| astronomes | 0.9 | astʁonom | astʁonɔm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| barmen | 0.9 | baʁmen | baʁmɛn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cessions | 0.9 | sesjɔ̃ | sɛsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| chevrotines | 0.9 | ʃəvʁɔtin | ʃəvʁotin | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dévora | 0.9 | devɔʁa | devoʁa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ecchymoses | 0.9 | ɛkimoz | ekimoz | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empressèrent | 0.9 | ɑ̃pʁesɛʁ | ɑ̃pʁɛsɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enseignée | 0.9 | ɑ̃sɛɲe | ɑ̃seɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| entremêlaient | 0.9 | ɑ̃tʁəmelɛ | ɑ̃tʁəmɛlɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essoufflait | 0.9 | esuflɛ | ɛsuflɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| moqua | 0.9 | mɔka | moka | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mêlez | 0.9 | mele | mɛle | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| oméga | 0.9 | ɔmeɡa | omeɡa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| panoramas | 0.9 | panɔʁama | panoʁama | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| polaroïd | 0.9 | pɔlaʁoid | pɔlaʁɔid | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescendaient | 0.9 | ʁədesɑ̃dɛ | ʁədɛsɑ̃dɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rétrospectif | 0.9 | ʁetʁɔspɛktif | ʁetʁospɛktif | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sexagénaire | 0.9 | sezaʒenɛʁ | sɛzaʒenɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| traînèrent | 0.9 | tʁenɛʁ | tʁɛnɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressautent | 0.9 | tʁesot | tʁɛsot | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| amaigrie | 0.8 | amɛɡʁi | ameɡʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| belliqueuse | 0.8 | bɛlikøz | belikøz | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cessons | 0.8 | sesɔ̃ | sɛsɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| colora | 0.8 | kolɔʁa | koloʁa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| despotisme | 0.8 | dɛspotizm | despotizm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| décevait | 0.8 | dɛsəvɛ | desəvɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| développa | 0.8 | devəlɔpa | devəlopa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ensoleillées | 0.8 | ɑ̃soleje | ɑ̃solɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ensoleillés | 0.8 | ɑ̃soleje | ɑ̃solɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayions | 0.8 | esɛjjɔ̃ | esejjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayèrent | 0.8 | esɛjɛʁ | esejɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| flageolantes | 0.8 | flaʒɔlɑ̃t | flaʒolɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| flotta | 0.8 | flɔta | flota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gloria | 0.8 | ɡlɔʁja | ɡloʁja | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| maîtrisa | 0.8 | metʁiza | mɛtʁiza | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| nettoient | 0.8 | nɛtwa | netwa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| néophyte | 0.8 | neɔfit | neofit | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| orienta | 0.8 | ɔʁjɑ̃ta | oʁjɑ̃ta | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| plaidé | 0.8 | plɛde | plede | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prélevé | 0.8 | pʁɛləve | pʁeləve | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| péquenots | 0.8 | pekəno | pɛkəno | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| remémore | 0.8 | ʁəmemɔʁ | ʁəmemoʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigolades | 0.8 | ʁiɡɔlad | ʁiɡolad | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| romanichels | 0.8 | ʁɔmaniʃɛl | ʁomaniʃɛl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| somnolents | 0.8 | sɔmnɔlɑ̃ | sɔmnolɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| verrouillé | 0.8 | vɛʁuje | veʁuje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| vieillissante | 0.8 | vjɛjisɑ̃t | vjejisɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mordorée | 0.8 | mɔʁdɔʁe | mɔʁdoʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| polyglotte | 0.8 | pɔliɡlɔt | poliɡlɔt | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tressautant | 0.8 | tʁesotɑ̃ | tʁɛsotɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aiguillages | 0.7 | ɛɡɥijaʒ | eɡɥijaʒ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| auditions | 0.7 | odisjɔ̃ | ɔdisjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bottait | 0.7 | bɔtɛ | botɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bulldozers | 0.7 | byldɔzɛʁ | byldozɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| caissières | 0.7 | kɛsjɛʁ | kesjɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| caressai | 0.7 | kaʁesɛ | kaʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dégringolant | 0.7 | deɡʁɛ̃ɡɔlɑ̃ | deɡʁɛ̃ɡolɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dévêtue | 0.7 | devety | devɛty | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| excelle | 0.7 | eksɛl | ɛksɛl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gainé | 0.7 | ɡene | ɡɛne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| hennit | 0.7 | ɛni | eni | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mollissait | 0.7 | mɔlisɛ | molisɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| onomatopée | 0.7 | ɔnɔmatope | ɔnɔmatɔpe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| peccadilles | 0.7 | pɛkadij | pekadij | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| popotin | 0.7 | popɔtɛ̃ | popotɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pressing | 0.7 | pʁɛsiŋ | pʁesiŋ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prélevée | 0.7 | pʁɛləve | pʁeləve | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prêtez | 0.7 | pʁete | pʁɛte | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pêchaient | 0.7 | peʃɛ | pɛʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redressaient | 0.7 | ʁədʁesɛ | ʁədʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| renfrogna | 0.7 | ʁɑ̃fʁɔɲa | ʁɑ̃fʁoɲa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rêvés | 0.7 | ʁɛve | ʁeve | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| souhaitai | 0.7 | swetɛ | swɛtɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| tempêtait | 0.7 | tɑ̃petɛ | tɑ̃pɛtɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| traitez | 0.7 | tʁete | tʁɛte | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| éconocroques | 0.7 | ekonokʁɔk | ekɔnɔkʁɔk | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| abhorrait | 0.7 | aboʁɛ | abɔʁɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| accommoda | 0.7 | akɔmɔda | akomoda | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| adressât | 0.7 | adʁesa | adʁɛsa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| agora | 0.7 | aɡɔʁa | aɡoʁa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| amollie | 0.7 | amɔli | amoli | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| apaisèrent | 0.7 | apezɛʁ | apɛzɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| apostropha | 0.7 | apɔstʁɔfa | apostʁofa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bottillons | 0.7 | bɔtijɔ̃ | botijɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bottins | 0.7 | bɔtɛ̃ | botɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bovine | 0.7 | bɔvin | bovin | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bunker | 0.7 | bunkeʁ | bunkɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cabochons | 0.7 | kabɔʃɔ̃ | kaboʃɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cahotante | 0.7 | kaɔtɑ̃t | kaotɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| colombophile | 0.7 | kolɔ̃bɔfil | kɔlɔ̃bɔfil | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cormorans | 0.7 | kɔʁmɔʁɑ̃ | kɔʁmoʁɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| crémerie | 0.7 | kʁeməʁi | kʁɛməʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dodelinait | 0.7 | dɔdəlinɛ | dodəlinɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dépaysée | 0.7 | depeize | depɛize | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dépaysés | 0.7 | depeize | depɛize | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dévêtit | 0.7 | deveti | devɛti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enchaînant | 0.7 | ɑ̃ʃenɑ̃ | ɑ̃ʃɛnɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essuient | 0.7 | esɥi | ɛsɥi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gainée | 0.7 | ɡene | ɡɛne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| grelottante | 0.7 | ɡʁəlɔtɑ̃t | ɡʁəlotɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| grignota | 0.7 | ɡʁiɲɔta | ɡʁiɲota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| guêpier | 0.7 | ɡɛpje | ɡepje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| monacale | 0.7 | mɔnakal | monakal | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pergola | 0.7 | pɛʁɡɔla | pɛʁɡola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pogroms | 0.7 | pɔɡʁɔm | poɡʁɔm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prosterna | 0.7 | pʁɔstɛʁna | pʁostɛʁna | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rafraîchis | 0.7 | ʁafʁɛʃi | ʁafʁeʃi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ramollis | 0.7 | ʁamɔli | ʁamoli | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigolez | 0.7 | ʁiɡɔle | ʁiɡole | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| seller | 0.7 | sɛle | sele | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| surveillez | 0.7 | syʁveje | syʁvɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| veillez | 0.7 | veje | vɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| économes | 0.7 | ekonom | ekonɔm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| apostrophes | 0.7 | apɔstʁɔf | apostʁɔf | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| endosse | 0.7 | ɑ̃dɔs | ɑ̃dos | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| illettré | 0.7 | ilɛtʁe | iletʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| paresseuses | 0.7 | paʁɛsøz | paʁesøz | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| graissée | 0.7 | ɡʁese | ɡʁɛse | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| belligérante | 0.6 | bɛliʒeʁɑ̃t | beliʒeʁɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| horlogers | 0.6 | ɔʁlɔʒe | ɔʁloʒe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sectaires | 0.6 | sɛktɛʁ | sektɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| affaiblissant | 0.6 | afɛblisɑ̃ | afeblisɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| apostrophait | 0.6 | apɔstʁofɛ | apostʁofɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| brocards | 0.6 | bʁɔkaʁ | bʁokaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| caressants | 0.6 | kaʁɛsɑ̃ | kaʁesɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| chopines | 0.6 | ʃɔpin | ʃopin | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| choqua | 0.6 | ʃɔka | ʃoka | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| claironner | 0.6 | kleʁone | klɛʁone | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coffio | 0.6 | kofjo | kɔfjo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| conseillez | 0.6 | kɔ̃seje | kɔ̃sɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coquillettes | 0.6 | kɔkijɛt | kokijɛt | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dogmatique | 0.6 | doɡmatik | dɔɡmatik | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dressai | 0.6 | dʁesɛ | dʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| embraye | 0.6 | ɑ̃bʁɛj | ɑ̃bʁej | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empiétement | 0.6 | ɑ̃pjetəmɑ̃ | ɑ̃pjɛtəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empêchai | 0.6 | ɑ̃peʃɛ | ɑ̃pɛʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enneigés | 0.6 | ɑ̃nɛʒe | ɑ̃neʒe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| fessées | 0.6 | fese | fɛse | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| galopa | 0.6 | ɡalɔpa | ɡalopa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gigolos | 0.6 | ʒiɡɔlo | ʒiɡolo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| godemiché | 0.6 | ɡɔdəmiʃe | ɡodəmiʃe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| goder | 0.6 | ɡɔde | ɡode | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| golden | 0.6 | ɡɔlden | ɡɔldɛn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gominé | 0.6 | ɡɔmine | ɡomine | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| graissé | 0.6 | ɡʁese | ɡʁɛse | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| guettés | 0.6 | ɡɛte | ɡete | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gyrophares | 0.6 | ʒiʁɔfaʁ | ʒiʁofaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| illettrée | 0.6 | ilɛtʁe | iletʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| impressionnaient | 0.6 | ɛ̃pʁesjonɛ | ɛ̃pʁɛsjonɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| malodorante | 0.6 | malodɔʁɑ̃t | malodoʁɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| modifia | 0.6 | mɔdifja | modifja | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| oracles | 0.6 | ɔʁakl | oʁakl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| oscilla | 0.6 | ɔsija | osija | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ovoïdal | 0.6 | ovɔidal | ovoidal | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| promenez | 0.6 | pʁɔməne | pʁoməne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| propulsa | 0.6 | pʁɔpylsa | pʁopylsa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rayait | 0.6 | ʁɛjɛ | ʁejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigodon | 0.6 | ʁiɡɔdɔ̃ | ʁiɡodɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rodée | 0.6 | ʁɔde | ʁode | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| smokings | 0.6 | smokiŋ | smɔkiŋ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| spermatozoïdes | 0.6 | spɛʁmatɔzoid | spɛʁmatozoid | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| suraigus | 0.6 | syʁɛɡy | syʁeɡy | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| verrouilla | 0.6 | vɛʁuja | veʁuja | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| vénerie | 0.6 | venəʁi | vɛnəʁi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sophistiquées | 0.6 | sɔfistike | sofistike | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| accommodé | 0.5 | akomɔde | akomode | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| adorons | 0.5 | adɔʁɔ̃ | adoʁɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| amaigrissement | 0.5 | amɛɡʁisəmɑ̃ | ameɡʁisəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| assiettées | 0.5 | asjɛte | asjete | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ballottait | 0.5 | balɔtɛ | balotɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| blotties | 0.5 | blɔti | bloti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| caressantes | 0.5 | kaʁɛsɑ̃t | kaʁesɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coron | 0.5 | kɔʁɔ̃ | koʁɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coïncida | 0.5 | kɔɛ̃sida | koɛ̃sida | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| crématorium | 0.5 | kʁematoʁjɔm | kʁematɔʁjɔm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| djellabas | 0.5 | dʒelaba | dʒɛlaba | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dressez | 0.5 | dʁese | dʁɛse | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| déblayaient | 0.5 | deblɛjɛ | deblejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| décevants | 0.5 | dɛsəvɑ̃ | desəvɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enserré | 0.5 | ɑ̃seʁe | ɑ̃sɛʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ensommeillés | 0.5 | ɑ̃someje | ɑ̃somɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayerai | 0.5 | esɛjəʁɛ | esejəʁɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| fouettées | 0.5 | fwɛte | fwete | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gainés | 0.5 | ɡene | ɡɛne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| groggy | 0.5 | ɡʁɔɡi | ɡʁoɡi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ineffables | 0.5 | inefabl | inɛfabl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| laitage | 0.5 | letaʒ | lɛtaʒ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| lessivés | 0.5 | lesive | lɛsive | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| maîtrisait | 0.5 | metʁizɛ | mɛtʁizɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mêlai | 0.5 | melɛ | mɛlɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| obsédantes | 0.5 | ɔpsedɑ̃t | opsedɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| parabellum | 0.5 | paʁabɛlɔm | paʁabelɔm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| polytechniciens | 0.5 | pɔlitɛknisjɛ̃ | politɛknisjɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| poneys | 0.5 | pɔnɛ | ponɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| potelé | 0.5 | potəle | pɔtəle | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| potines | 0.5 | pɔtin | potin | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pressez | 0.5 | pʁese | pʁɛse | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prélevées | 0.5 | pʁɛləve | pʁeləve | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prélevés | 0.5 | pʁɛləve | pʁeləve | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rafraîchissement | 0.5 | ʁafʁɛʃisəmɑ̃ | ʁafʁeʃisəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| relayant | 0.5 | ʁəlɛjɑ̃ | ʁəlejɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sirops | 0.5 | siʁɔ | siʁo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| thermostat | 0.5 | tɛʁmɔsta | tɛʁmosta | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| traitât | 0.5 | tʁeta | tʁɛta | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| troïka | 0.5 | tʁɔika | tʁɔjka | autre |
| voguant | 0.5 | vɔɡɑ̃ | voɡɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| embêtés | 0.5 | ɑ̃bete | ɑ̃bɛte | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ensommeillées | 0.5 | ɑ̃someje | ɑ̃somɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| fêlés | 0.5 | fele | fɛle | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prospections | 0.5 | pʁɔspɛksjɔ̃ | pʁospɛksjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rigolards | 0.5 | ʁiɡɔlaʁ | ʁiɡolaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| abhorré | 0.5 | aboʁe | abɔʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bottées | 0.5 | bɔte | bote | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bricoleurs | 0.5 | bʁikɔlœʁ | bʁikolœʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| adressez | 0.5 | adʁese | adʁɛse | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aficionados | 0.5 | afisjɔnado | afisjonado | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aimèrent | 0.5 | emɛʁ | ɛmɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| allégement | 0.5 | aleʒəmɑ̃ | alɛʒəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| assujettit | 0.5 | asyʒeti | asyʒɛti | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| autodestruction | 0.5 | otodestʁyksjɔ̃ | otodɛstʁyksjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| aînesse | 0.5 | enɛs | ɛnɛs | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| baisez | 0.5 | beze | bɛze | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| biffer | 0.5 | bifɛ | bife | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| borborygme | 0.5 | bɔʁbɔʁiɡm | bɔʁboʁiɡm | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coccinelles | 0.5 | kɔksinɛl | koksinɛl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| colombins | 0.5 | kɔlɔ̃bɛ̃ | kolɔ̃bɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| compression | 0.5 | kɔ̃pʁesjɔ̃ | kɔ̃pʁɛsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| convalescents | 0.5 | kɔ̃valesɑ̃ | kɔ̃valɛsɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| coïncident | 0.5 | koɛ̃sidɑ̃ | kɔɛ̃sidɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| descellé | 0.5 | desele | desɛle | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dossard | 0.5 | dɔsaʁ | dosaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dressais | 0.5 | dʁesɛ | dʁɛsɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| déblayait | 0.5 | deblɛjɛ | deblejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dénivellations | 0.5 | denivelasjɔ̃ | denivɛlasjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dépaysé | 0.5 | depeize | depɛize | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dévorèrent | 0.5 | devoʁɛʁ | devɔʁɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enlaidir | 0.5 | ɑ̃lɛdiʁ | ɑ̃lediʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essaimé | 0.5 | eseme | ɛsɛme | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| essayés | 0.5 | eseje | esɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| florentins | 0.5 | flɔʁɑ̃tɛ̃ | floʁɑ̃tɛ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| fraisiers | 0.5 | fʁɛzje | fʁezje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| frayaient | 0.5 | fʁɛjɛ | fʁejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| freinée | 0.5 | fʁɛne | fʁene | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| gaîment | 0.5 | ɡɛmɑ̃ | ɡemɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| godillot | 0.5 | ɡɔdijo | ɡodijo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| grognards | 0.5 | ɡʁɔɲaʁ | ɡʁoɲaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| hagiographe | 0.5 | aʒjɔɡʁaf | aʒjoɡʁaf | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| impressionnent | 0.5 | ɛ̃pʁesjɔn | ɛ̃pʁɛsjɔn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| laconisme | 0.5 | lakɔnizm | lakɔnism | autre |
| malaisée | 0.5 | malɛze | maleze | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| marmotta | 0.5 | maʁmɔta | maʁmota | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| morilles | 0.5 | mɔʁij | moʁij | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| mêlât | 0.5 | mela | mɛla | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| obsessionnelle | 0.5 | opsesjonɛl | ɔpsesjonɛl | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| obsédants | 0.5 | ɔpsedɑ̃ | opsedɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| oued | 0.5 | wed | wɛd | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pierreuses | 0.5 | pjɛʁøz | pjeʁøz | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| pierrots | 0.5 | pjɛʁo | pjeʁo | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| piétement | 0.5 | pjetəmɑ̃ | pjɛtəmɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| polygone | 0.5 | pɔliɡɔn | poliɡon | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prélevait | 0.5 | pʁɛləvɛ | pʁeləvɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| prévenante | 0.5 | pʁevənɑ̃t | pʁɛvənɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rafraîchie | 0.5 | ʁafʁɛʃi | ʁafʁeʃi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rafraîchissante | 0.5 | ʁafʁɛʃisɑ̃t | ʁafʁeʃisɑ̃t | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redescendis | 0.5 | ʁədesɑ̃di | ʁədɛsɑ̃di | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| redressèrent | 0.5 | ʁədʁesɛʁ | ʁədʁɛsɛʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| relayait | 0.5 | ʁəlɛjɛ | ʁəlejɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| remémora | 0.5 | ʁəmemɔʁa | ʁəmemoʁa | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rodomontades | 0.5 | ʁodɔmɔ̃tad | ʁodomɔ̃tad | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| rosbif | 0.5 | ʁosbif | ʁɔsbif | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| saignées | 0.5 | sɛɲe | seɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sanguinolents | 0.5 | sɑ̃ɡinɔlɑ̃ | sɑ̃ɡinolɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| scellait | 0.5 | selɛ | sɛlɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| scolie | 0.5 | skɔli | skoli | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sellier | 0.5 | sɛlje | selje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sodomie | 0.5 | sodɔmi | sɔdɔmi | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sollicita | 0.5 | sɔlisita | solisita | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| sollicitant | 0.5 | sɔlisitɑ̃ | solisitɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| suffoqua | 0.5 | syfɔka | syfoka | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| surélevée | 0.5 | syʁɛləve | syʁeləve | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| synchrones | 0.5 | sɛ̃kʁɔn | sɛ̃kʁon | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| toboggans | 0.5 | tɔbɔɡɑ̃ | toboɡɑ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| traitai | 0.5 | tʁetɛ | tʁɛtɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| traînez | 0.5 | tʁene | tʁɛne | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| empêtrés | 0.4 | ɑ̃pɛtʁe | ɑ̃petʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| enseignés | 0.4 | ɑ̃sɛɲe | ɑ̃seɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| illettrés | 0.4 | ilɛtʁe | iletʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| incommodée | 0.4 | ɛ̃kɔmode | ɛ̃komode | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| jobards | 0.4 | ʒɔbaʁ | ʒobaʁ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| adressions | 0.4 | adʁesjɔ̃ | adʁɛsjɔ̃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| amollies | 0.4 | amɔli | amoli | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| appareillé | 0.4 | apaʁeje | apaʁɛje | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| assiettée | 0.4 | asjɛte | asjete | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| associa | 0.4 | asɔsja | asosja | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| atterrissent | 0.4 | ateʁis | atɛʁis | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| baignées | 0.4 | bɛɲe | beɲe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| ballottaient | 0.4 | balɔtɛ | balotɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| belladone | 0.4 | bɛladɔn | beladɔn | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bobino | 0.4 | bɔbino | bobino | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bolets | 0.4 | bɔlɛ | bolɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| boniches | 0.4 | bɔniʃ | boniʃ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| bêchait | 0.4 | beʃɛ | bɛʃɛ | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| cajola | 0.4 | kaʒɔla | kaʒola | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| corroborer | 0.4 | kɔʁoboʁe | koʁoboʁe | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| damned | 0.4 | damned | damnɛd | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| despotique | 0.4 | dɛspotik | despotik | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| desseller | 0.4 | desɛle | dɛsɛle | voyelle mi-ouverte (ɛ/e, ɔ/o) |
| dogmatiques | 0.4 | doɡmatik | dɔɡmatik | voyelle mi-ouverte (ɛ/e, ɔ/o) |

---

## Récapitulatif

| Tranche | Couv. | Évaluées | Acc (exacte) | Acc (o/ɔ) | Acc (o/ɔ+e/ɛ) | Erreurs |
|---------|-------|----------|-------------|-----------|---------------|--------|
| 0→50% | 50% | 75 | 100.00% | 100.00% | 100.00% | 0 |
| 50→80% | 80% | 2,092 | 99.62% | 99.71% | 100.00% | 8 |
| 80→90% | 90% | 5,510 | 99.26% | 99.53% | 99.98% | 41 |
| 90→95% | 95% | 9,329 | 98.78% | 99.28% | 100.00% | 114 |
| 95→99% | 99% | 30,802 | 98.07% | 98.94% | 100.00% | 595 |

## Répartition globale des erreurs

| Catégorie | Nb | % |
|-----------|-----|---|
| voyelle mi-ouverte (ɛ/e, ɔ/o) | 747 | 98.5% |
| autre | 5 | 0.7% |
| schwa parasite | 5 | 0.7% |
| consonne finale muette | 1 | 0.1% |

**Total erreurs : 758**

---

## Élisions

Test des formes élidées (j', l', d', etc.) en mode isolé :

| Forme | Gold | Prédiction | Correct |
|-------|------|------------|--------|
| c' | s | s | OK |
| d' | d | d | OK |
| j' | ʒ | ʒ | OK |
| jusqu' | ʒysk | ʒysk | OK |
| l' | l | l | OK |
| lorsqu' | lɔʁsk | lɔʁsk | OK |
| m' | m | m | OK |
| n' | n | n | OK |
| presqu' | pʁɛsk | pʁɛsk | OK |
| puisqu' | pɥisk | pɥisk | OK |
| qu' | k | k | OK |
| quelqu' | kɛlk | kɛlk | OK |
| quoiqu' | kwak | kwak | OK |
| s' | s | s | OK |
| t' | t | t | OK |

**Élisions : 15/15 correctes**


*Temps total : 47.9s*
