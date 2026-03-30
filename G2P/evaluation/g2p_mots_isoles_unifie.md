# Évaluation G2P — Modèle unifié — Mots isolés

- **Modèle** : unifié BiLSTM multi-tâche (onnx)
- **Mode** : mot isolé (sans contexte phrastique)
- **Fréquences** : Lexique 3.83 (freqlivres)
- **Gold** : dico.csv (match si pred in prononciations valides)
- **Filtres** : composés (apostrophe/tiret) exclus
- **Homographes** : 1,124 mots identifiés

---

## 1. Tous les mots (homographes inclus)

| Tranche | Couv. | Formes | Évaluées | Acc (exacte) | Acc (o/ɔ) | Acc (o/ɔ+e/ɛ) | Erreurs |
|---------|-------|--------|----------|-------------|-----------|---------------|--------|
| 0→50% | 50% | 75 | 75 | 100.00% | 100.00% | 100.00% | 0 |
| 50→80% | 80% | 2,092 | 2,092 | 97.13% | 97.23% | 97.51% | 60 |
| 80→90% | 90% | 5,511 | 5,510 | 95.46% | 95.74% | 96.19% | 250 |
| 90→95% | 95% | 9,329 | 9,329 | 93.98% | 94.51% | 95.24% | 562 |
| 95→99% | 99% | 30,805 | 30,802 | 92.09% | 92.98% | 94.07% | 2,437 |

**Accuracy cumulée :**

| Couverture | Mots | Acc (exacte) | Acc (o/ɔ) | Acc (o/ɔ+e/ɛ) |
|------------|------|-------------|-----------|---------------|
| 50% | 75 | 100.00% | 100.00% | 100.00% |
| 80% | 2,167 | 97.23% | 97.32% | 97.60% |
| 90% | 7,677 | 95.96% | 96.18% | 96.59% |
| 95% | 17,006 | 94.87% | 95.27% | 95.85% |
| 99% | 47,808 | 93.08% | 93.80% | 94.70% |

---

## 2. Sans homographes (mots non-ambigus uniquement)

| Tranche | Couv. | Formes | Homo exclus | Évaluées | Acc (exacte) | Acc (o/ɔ) | Acc (o/ɔ+e/ɛ) | Erreurs |
|---------|-------|--------|-------------|----------|-------------|-----------|---------------|--------|
| 0→50% | 50% | 75 | 4 | 71 | 100.00% | 100.00% | 100.00% | 0 |
| 50→80% | 80% | 2,092 | 24 | 2,068 | 97.34% | 97.44% | 97.68% | 55 |
| 80→90% | 90% | 5,511 | 41 | 5,469 | 95.45% | 95.72% | 96.18% | 249 |
| 90→95% | 95% | 9,329 | 61 | 9,268 | 94.03% | 94.55% | 95.26% | 553 |
| 95→99% | 99% | 30,805 | 270 | 30,532 | 92.10% | 92.98% | 94.06% | 2,413 |

**Accuracy cumulée (sans homographes) :**

| Couverture | Mots | Acc (exacte) | Acc (o/ɔ) | Acc (o/ɔ+e/ɛ) |
|------------|------|-------------|-----------|---------------|
| 50% | 71 | 100.00% | 100.00% | 100.00% |
| 80% | 2,139 | 97.43% | 97.52% | 97.76% |
| 90% | 7,608 | 96.00% | 96.23% | 96.62% |
| 95% | 16,876 | 94.92% | 95.31% | 95.88% |
| 99% | 47,408 | 93.10% | 93.81% | 94.71% |

---

## 3. Erreurs les plus fréquentes (tous mots, top 100)

| # | Mot | Fréq | Prédiction | Gold (proche) | Toutes prononc. |
|---|-----|------|------------|---------------|----------------|
| 1 | te * | 774.3 | t | te | te, tə |
| 2 | coeur | 380.1 | kuœʁ | kœʁ | kœʁ |
| 3 | es * | 334.3 | e | ɛ | ɛ, ɛs |
| 4 | oeil | 278.5 | uj | œj | œj |
| 5 | eût | 235.3 | ø | y | y |
| 6 | ait | 148.7 | ɛ | e | e |
| 7 | tes | 145.0 | t | te | te |
| 8 | étions | 137.5 | esjɔ̃ | etjɔ̃ | etjɔ̃ |
| 9 | exemple | 119.2 | ɛkɑ̃pl | ɛɡzɑ̃pl | ɛɡzɑ̃pl |
| 10 | soeur | 116.5 | suœʁ | sœʁ | sœʁ |
| 11 | existence | 93.8 | ezistɑ̃s | ɛɡzistɑ̃s | ɛɡzistɑ̃s |
| 12 | exactement | 92.4 | ɛkaktəmɑ̃ | ɛɡzaktəmɑ̃ | ɛɡzaktəmɑ̃ |
| 13 | oublié | 87.7 | ublje | ublije | ublije |
| 14 | dessus | 84.8 | desy | dəsy | dəsy |
| 15 | expliquer | 76.5 | ɛkplike | ɛsplike | ɛksplike, ɛsplike |
| 16 | faim | 74.9 | fɛ̃m | fɛ̃ | fe, fɛŋ, fɛ̃ |
| 17 | oublier | 74.2 | ublje | ublije | ublije |
| 18 | expression | 69.3 | ɛkpʁɛsjɔ̃ | ɛkspʁɛsjɔ̃ | ɛkspʁesjɔ̃, ɛkspʁɛsjɔ̃ |
| 19 | oeuvre | 69.0 | wœvʁ | œvʁ | œvʁ |
| 20 | cul | 64.5 | kyl | ky | ky |
| 21 | gars | 59.3 | ɡaʁ | ɡa | ɡa |
| 22 | existe | 58.0 | ezist | ɛɡzist | ɛɡzist |
| 23 | cria | 55.7 | kʁja | kʁija | kʁija |
| 24 | etc | 55.0 | ek | ɛtseteʁa | ɛtseteʁa |
| 25 | parfum | 52.4 | paʁfɔm | paʁfœ̃ | paʁfœ̃ |
| 26 | jolie | 51.8 | ʒɔli | ʒoli | ʒoli |
| 27 | os | 49.7 | o | os | os, ɔs |
| 28 | put * | 49.7 | pyt | put | put, py |
| 29 | extrême | 49.3 | ɛktʁɛm | ɛkstʁɛm | ɛkstʁɛm |
| 30 | eus | 49.0 | ø | y | eus, y |
| 31 | expérience | 48.1 | ɛkpeʁjɑ̃s | ɛkspeʁjɑ̃s | ɛkspeʁjɑ̃s, ɛkspɛʁjɑ̃s |
| 32 | crier | 47.3 | kʁje | kʁije | kʁije |
| 33 | essayé | 45.7 | eseje | esɛje | esɛje |
| 34 | soeurs | 44.5 | soœʁ | sœʁ | sœʁ |
| 35 | approcha | 42.7 | apʁɔʃa | apʁoʃa | apʁoʃa |
| 36 | promenade | 42.4 | pʁɔmənad | pʁɔmnad | pʁɔmnad |
| 37 | relations * | 42.2 | ʁəlasjɔ̃ | ʁølasjɔ̃ | ʁølasjɔ̃, ʁølatjɔ̃ |
| 38 | clients | 41.6 | kljɑ̃ | klijɑ̃ | klijɑ̃ |
| 39 | tabac | 41.3 | tabak | taba | taba |
| 40 | taxi | 41.2 | tazi | taksi | taksi |
| 41 | draps | 41.0 | dʁap | dʁa | dʁa |
| 42 | alcool | 39.7 | alkuɔl | alkɔɔl | alkɔl, alkɔɔl |
| 43 | essaie | 39.6 | esɛ | ɛsɛ | ɛsɛ |
| 44 | fusil | 39.3 | fyzil | fyzi | fyzi |
| 45 | sourcils | 39.0 | suʁsil | suʁsi | suʁsi |
| 46 | extraordinaire | 38.8 | ɛktʁaɔʁdinɛʁ | ɛkstʁaɔʁdinɛʁ | ɛkstʁaɔʁdinɛʁ, ɛkstʁɔʁdinɛʁ |
| 47 | gentil | 38.5 | ʒɑ̃til | ʒɑ̃ti | ʒɑ̃ti |
| 48 | explique | 38.2 | ɛkplik | ɛksplik | ɛksplik |
| 49 | drifter * | 37.5 | dʁiftɛʁ | dʁiftœʁ | dʁifte, dʁiftœʁ |
| 50 | prétexte | 37.5 | pʁetɛkt | pʁetɛkst | pʁetɛkst |
| 51 | arrêtait | 37.1 | aʁetɛ | aʁɛtɛ | aʁɛtɛ |
| 52 | bibliothèque | 36.8 | bibljotɛk | biblijotɛk | biblijotɛk, biblijɔtɛk |
| 53 | ouvriers | 36.7 | uvʁje | uvʁije | uvʁije |
| 54 | expliqua | 36.4 | ɛkplika | ɛksplika | ɛksplika |
| 55 | extérieur | 36.0 | ɛkteʁjœʁ | ɛksteʁjœʁ | ɛksteʁjœʁ |
| 56 | essayait | 35.8 | esɛjɛ | esejɛ | esejɛ |
| 57 | clef | 35.6 | klɛf | kle | kle |
| 58 | hélas | 35.5 | ela | elas | elas |
| 59 | mme | 33.2 | mm | madam | madam |
| 60 | orgueil | 33.2 | ɔʁɡɛj | ɔʁɡœj | ɔʁɡœj |
| 61 | exprès | 32.8 | ɛkpʁɛ | ɛkspʁɛ | ɛkspʁɛ, ɛkspʁɛs, ɛspʁɛs |
| 62 | existait | 32.6 | ezistɛ | ɛɡzistɛ | ɛɡzistɛ |
| 63 | écho | 32.5 | eʃo | eko | eko |
| 64 | index | 32.4 | ɛ̃de | ɛ̃dɛks | ɛ̃dɛks |
| 65 | pot | 32.3 | pɔ | po | po |
| 66 | texte | 31.4 | tɛkt | tɛkst | tɛkst |
| 67 | instinct | 31.0 | ɛ̃stɛ̃kt | ɛ̃stɛ̃ | ɛ̃stɛ̃ |
| 68 | estomac | 30.1 | ɛstomak | ɛstoma | ɛstoma, ɛstɔma |
| 69 | triomphe | 30.1 | tʁjɔ̃f | tʁiɔ̃f | tʁijɔ̃f, tʁiɔ̃f |
| 70 | prière | 30.0 | pʁjɛʁ | pʁijɛʁ | pʁijɛʁ |
| 71 | oeufs | 29.8 | oœf | ø | ø |
| 72 | e | 29.1 | ə | ø | ø |
| 73 | explication | 28.9 | ɛkplikasjɔ̃ | ɛksplikasjɔ̃ | ɛksplikasjɔ̃ |
| 74 | client | 28.8 | kljɑ̃ | klijɑ̃ | klijɑ̃ |
| 75 | s | 28.5 |  | ɛs | ɛs |
| 76 | quatrième | 28.5 | katʁjɛm | katʁijɛm | katʁijɛm |
| 77 | poings | 28.1 | pwa | pwɛ̃ | pwɛ̃ |
| 78 | expliqué | 27.6 | ɛkplike | ɛksplike | ɛksplike |
| 79 | fixé | 27.4 | fize | fikse | fikse |
| 80 | tablier | 27.2 | tablje | tablije | tablije |
| 81 | gaieté | 27.1 | ɡɛte | ɡete | ɡete |
| 82 | aiment | 27.0 | ɛmɑ̃ | ɛm | ɛm |
| 83 | nerfs | 26.6 | nɛʁf | nɛʁ | nɛʁ |
| 84 | pleurait | 26.5 | plœʁɛ | pløʁɛ | pløʁɛ |
| 85 | criait | 26.4 | kʁjɛ | kʁijɛ | kʁijɛ |
| 86 | excuse | 26.1 | ɛkkyz | ɛkskyz | ɛkskyz |
| 87 | baisers | 25.9 | bezeʁ | beze | beze |
| 88 | exprimer | 25.9 | ɛkpʁime | ɛkspʁime | ɛkspʁime |
| 89 | gêné | 25.9 | ʒene | ʒɛne | ʒɛne |
| 90 | soixante | 25.9 | swazɑ̃t | swasɑ̃t | swasɑ̃t |
| 91 | écria | 25.7 | ekʁja | ekʁija | ekʁija |
| 92 | restent | 25.2 | ʁɛstt | ʁɛst | ʁɛst |
| 93 | choeur | 24.9 | ʃuœʁ | kœʁ | kœʁ |
| 94 | fixer | 24.8 | fikɛʁ | fikse | fikse, fiksœʁ |
| 95 | villa | 24.8 | vila | villa | villa |
| 96 | appuyé | 24.8 | apɥje | apɥije | apɥije |
| 97 | oeuvres | 24.1 | wœvʁ | œvʁ | œvʁ |
| 98 | criant | 23.6 | kʁjɑ̃ | kʁijɑ̃ | kʁijɑ̃ |
| 99 | propriétaire * | 23.5 | pʁopʁjetɛʁ | pʁopʁijetɛʁ | pʁopʁijetɛʁ, pʁɔpʁijetɛʁ |
| 100 | eurent | 23.4 | œʁ | yʁ | yʁ |

*Les mots marqués \* sont des homographes.*

---

## 4. Erreurs les plus fréquentes (hors homographes, top 100)

| # | Mot | Fréq | Prédiction | Gold (proche) | Toutes prononc. |
|---|-----|------|------------|---------------|----------------|
| 1 | coeur | 380.1 | kuœʁ | kœʁ | kœʁ |
| 2 | oeil | 278.5 | uj | œj | œj |
| 3 | eût | 235.3 | ø | y | y |
| 4 | ait | 148.7 | ɛ | e | e |
| 5 | tes | 145.0 | t | te | te |
| 6 | étions | 137.5 | esjɔ̃ | etjɔ̃ | etjɔ̃ |
| 7 | exemple | 119.2 | ɛkɑ̃pl | ɛɡzɑ̃pl | ɛɡzɑ̃pl |
| 8 | soeur | 116.5 | suœʁ | sœʁ | sœʁ |
| 9 | existence | 93.8 | ezistɑ̃s | ɛɡzistɑ̃s | ɛɡzistɑ̃s |
| 10 | exactement | 92.4 | ɛkaktəmɑ̃ | ɛɡzaktəmɑ̃ | ɛɡzaktəmɑ̃ |
| 11 | oublié | 87.7 | ublje | ublije | ublije |
| 12 | dessus | 84.8 | desy | dəsy | dəsy |
| 13 | expliquer | 76.5 | ɛkplike | ɛsplike | ɛksplike, ɛsplike |
| 14 | faim | 74.9 | fɛ̃m | fɛ̃ | fe, fɛŋ, fɛ̃ |
| 15 | oublier | 74.2 | ublje | ublije | ublije |
| 16 | expression | 69.3 | ɛkpʁɛsjɔ̃ | ɛkspʁɛsjɔ̃ | ɛkspʁesjɔ̃, ɛkspʁɛsjɔ̃ |
| 17 | oeuvre | 69.0 | wœvʁ | œvʁ | œvʁ |
| 18 | cul | 64.5 | kyl | ky | ky |
| 19 | gars | 59.3 | ɡaʁ | ɡa | ɡa |
| 20 | existe | 58.0 | ezist | ɛɡzist | ɛɡzist |
| 21 | cria | 55.7 | kʁja | kʁija | kʁija |
| 22 | etc | 55.0 | ek | ɛtseteʁa | ɛtseteʁa |
| 23 | parfum | 52.4 | paʁfɔm | paʁfœ̃ | paʁfœ̃ |
| 24 | jolie | 51.8 | ʒɔli | ʒoli | ʒoli |
| 25 | os | 49.7 | o | os | os, ɔs |
| 26 | extrême | 49.3 | ɛktʁɛm | ɛkstʁɛm | ɛkstʁɛm |
| 27 | eus | 49.0 | ø | y | eus, y |
| 28 | expérience | 48.1 | ɛkpeʁjɑ̃s | ɛkspeʁjɑ̃s | ɛkspeʁjɑ̃s, ɛkspɛʁjɑ̃s |
| 29 | crier | 47.3 | kʁje | kʁije | kʁije |
| 30 | essayé | 45.7 | eseje | esɛje | esɛje |
| 31 | soeurs | 44.5 | soœʁ | sœʁ | sœʁ |
| 32 | approcha | 42.7 | apʁɔʃa | apʁoʃa | apʁoʃa |
| 33 | promenade | 42.4 | pʁɔmənad | pʁɔmnad | pʁɔmnad |
| 34 | clients | 41.6 | kljɑ̃ | klijɑ̃ | klijɑ̃ |
| 35 | tabac | 41.3 | tabak | taba | taba |
| 36 | taxi | 41.2 | tazi | taksi | taksi |
| 37 | draps | 41.0 | dʁap | dʁa | dʁa |
| 38 | alcool | 39.7 | alkuɔl | alkɔɔl | alkɔl, alkɔɔl |
| 39 | essaie | 39.6 | esɛ | ɛsɛ | ɛsɛ |
| 40 | fusil | 39.3 | fyzil | fyzi | fyzi |
| 41 | sourcils | 39.0 | suʁsil | suʁsi | suʁsi |
| 42 | extraordinaire | 38.8 | ɛktʁaɔʁdinɛʁ | ɛkstʁaɔʁdinɛʁ | ɛkstʁaɔʁdinɛʁ, ɛkstʁɔʁdinɛʁ |
| 43 | gentil | 38.5 | ʒɑ̃til | ʒɑ̃ti | ʒɑ̃ti |
| 44 | explique | 38.2 | ɛkplik | ɛksplik | ɛksplik |
| 45 | prétexte | 37.5 | pʁetɛkt | pʁetɛkst | pʁetɛkst |
| 46 | arrêtait | 37.1 | aʁetɛ | aʁɛtɛ | aʁɛtɛ |
| 47 | bibliothèque | 36.8 | bibljotɛk | biblijotɛk | biblijotɛk, biblijɔtɛk |
| 48 | ouvriers | 36.7 | uvʁje | uvʁije | uvʁije |
| 49 | expliqua | 36.4 | ɛkplika | ɛksplika | ɛksplika |
| 50 | extérieur | 36.0 | ɛkteʁjœʁ | ɛksteʁjœʁ | ɛksteʁjœʁ |
| 51 | essayait | 35.8 | esɛjɛ | esejɛ | esejɛ |
| 52 | clef | 35.6 | klɛf | kle | kle |
| 53 | hélas | 35.5 | ela | elas | elas |
| 54 | mme | 33.2 | mm | madam | madam |
| 55 | orgueil | 33.2 | ɔʁɡɛj | ɔʁɡœj | ɔʁɡœj |
| 56 | exprès | 32.8 | ɛkpʁɛ | ɛkspʁɛ | ɛkspʁɛ, ɛkspʁɛs, ɛspʁɛs |
| 57 | existait | 32.6 | ezistɛ | ɛɡzistɛ | ɛɡzistɛ |
| 58 | écho | 32.5 | eʃo | eko | eko |
| 59 | index | 32.4 | ɛ̃de | ɛ̃dɛks | ɛ̃dɛks |
| 60 | pot | 32.3 | pɔ | po | po |
| 61 | texte | 31.4 | tɛkt | tɛkst | tɛkst |
| 62 | instinct | 31.0 | ɛ̃stɛ̃kt | ɛ̃stɛ̃ | ɛ̃stɛ̃ |
| 63 | estomac | 30.1 | ɛstomak | ɛstoma | ɛstoma, ɛstɔma |
| 64 | triomphe | 30.1 | tʁjɔ̃f | tʁiɔ̃f | tʁijɔ̃f, tʁiɔ̃f |
| 65 | prière | 30.0 | pʁjɛʁ | pʁijɛʁ | pʁijɛʁ |
| 66 | oeufs | 29.8 | oœf | ø | ø |
| 67 | e | 29.1 | ə | ø | ø |
| 68 | explication | 28.9 | ɛkplikasjɔ̃ | ɛksplikasjɔ̃ | ɛksplikasjɔ̃ |
| 69 | client | 28.8 | kljɑ̃ | klijɑ̃ | klijɑ̃ |
| 70 | s | 28.5 |  | ɛs | ɛs |
| 71 | quatrième | 28.5 | katʁjɛm | katʁijɛm | katʁijɛm |
| 72 | poings | 28.1 | pwa | pwɛ̃ | pwɛ̃ |
| 73 | expliqué | 27.6 | ɛkplike | ɛksplike | ɛksplike |
| 74 | fixé | 27.4 | fize | fikse | fikse |
| 75 | tablier | 27.2 | tablje | tablije | tablije |
| 76 | gaieté | 27.1 | ɡɛte | ɡete | ɡete |
| 77 | aiment | 27.0 | ɛmɑ̃ | ɛm | ɛm |
| 78 | nerfs | 26.6 | nɛʁf | nɛʁ | nɛʁ |
| 79 | pleurait | 26.5 | plœʁɛ | pløʁɛ | pløʁɛ |
| 80 | criait | 26.4 | kʁjɛ | kʁijɛ | kʁijɛ |
| 81 | excuse | 26.1 | ɛkkyz | ɛkskyz | ɛkskyz |
| 82 | baisers | 25.9 | bezeʁ | beze | beze |
| 83 | exprimer | 25.9 | ɛkpʁime | ɛkspʁime | ɛkspʁime |
| 84 | gêné | 25.9 | ʒene | ʒɛne | ʒɛne |
| 85 | soixante | 25.9 | swazɑ̃t | swasɑ̃t | swasɑ̃t |
| 86 | écria | 25.7 | ekʁja | ekʁija | ekʁija |
| 87 | restent | 25.2 | ʁɛstt | ʁɛst | ʁɛst |
| 88 | choeur | 24.9 | ʃuœʁ | kœʁ | kœʁ |
| 89 | fixer | 24.8 | fikɛʁ | fikse | fikse, fiksœʁ |
| 90 | villa | 24.8 | vila | villa | villa |
| 91 | appuyé | 24.8 | apɥje | apɥije | apɥije |
| 92 | oeuvres | 24.1 | wœvʁ | œvʁ | œvʁ |
| 93 | criant | 23.6 | kʁjɑ̃ | kʁijɑ̃ | kʁijɑ̃ |
| 94 | eurent | 23.4 | œʁ | yʁ | yʁ |
| 95 | eussent | 23.4 | øs | ys | ys |
| 96 | réflexion | 23.4 | ʁeflɛzjɔ̃ | ʁeflɛksjɔ̃ | ʁeflɛksjɔ̃ |
| 97 | revolver | 23.3 | ʁəvɔlvɛʁ | ʁevɔlvɛʁ | ʁevɔlvɛʁ |
| 98 | crié | 23.0 | kʁje | kʁije | kʁije |
| 99 | ignore | 23.0 | iɲɔʁ | iɲoʁ | iɲoʁ |
| 100 | essayant | 22.6 | esɛjɑ̃ | esejɑ̃ | esejɑ̃ |

---

## Résumé

| Condition | Mots évalués | Word Acc (exacte) | Word Acc (o/ɔ+e/ɛ tol.) |
|-----------|-------------|-------------------|------------------------|
| Tous les mots (→99%) | 47,808 | 93.08% | 94.70% |
| Sans homographes (→99%) | 47,408 | 93.10% | 94.71% |
| Homographes exclus | 400 | — | — |

*Backend : onnx — Temps : 68.9s*
