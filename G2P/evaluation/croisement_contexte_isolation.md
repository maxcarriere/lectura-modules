# Croisement G2P : Contexte vs Isolation

- **Modèle** : unifié BiLSTM multi-tâche (ONNX INT8)
- **Données** : test CoNLL-U (1712 phrases)
- **Mots évalués** : 27,684 (skippés : 191)
- **Temps** : 43.3s

## Matrice de concordance (match exact)

|  | Isolé OK | Isolé KO | Total |
|--|----------|----------|-------|
| **Contexte OK** | 24,715 (89.3%) | 67 (0.2%) | 24,782 (89.5%) |
| **Contexte KO** | 50 (0.2%) | 2,852 (10.3%) | 2,902 (10.5%) |
| **Total** | 24,765 (89.5%) | 2,919 (10.5%) | 27,684 |

## Matrice de concordance (tolérance o/ɔ + e/ɛ)

|  | Isolé OK | Isolé KO | Total |
|--|----------|----------|-------|
| **Contexte OK** | 24,846 (89.7%) | 49 (0.2%) | 24,895 (89.9%) |
| **Contexte KO** | 48 (0.2%) | 2,741 (9.9%) | 2,789 (10.1%) |
| **Total** | 24,894 | 2,790 | 27,684 |

## Interprétation

- **Le contexte aide** (ctx OK, iso KO) : **67** mots (0.2%)
  → Post-traitement en isolation **risque de casser** ces prédictions en contexte
- **Erreur structurelle** (les deux KO) : **2,852** mots (10.3%)
  → Post-traitement sûr (le modèle se trompe dans tous les cas)
- **Contexte nuit** (ctx KO, iso OK) : **50** mots (0.2%)
  → Le modèle prédit mieux en isolation pour ces mots
- **Tout correct** : **24,715** mots (89.3%)

## Catégories des erreurs d'isolation (contexte OK, iso KO)

| Catégorie | Nombre | % |
|-----------|--------|---|
| autre | 27 | 40.3% |
| voyelle mi-ouverte (ɛ/e, ɔ/o) | 23 | 34.3% |
| consonne finale muette | 15 | 22.4% |
| schwa manquant | 2 | 3.0% |

## Exemples : le contexte aide (ctx OK, iso KO) — top 50

| Mot | Gold | Prédiction contexte | Prédiction isolée |
|-----|------|--------------------|-----------------|
| Ahmed | amed | amed | amɛd |
| Gdem | ɡdem | ɡdem | ɡdɛm |
| hamed | amed | amed | amɛd |
| pression | pʁɛsjɔ̃ | pʁɛsjɔ̃ | pʁesjɔ̃ |
| Veto | veto | veto | vəto |
| Andrea | ɑ̃dʁea | ɑ̃dʁea | ɑ̃dʁe |
| Campuchea | kɑ̃pyʃa | kɑ̃pyʃa | kɑ̃pyʃia |
| bobo | bobo | bobo | bɔbo |
| Yoga | joɡa | joɡa | jɔɡa |
| impeccable | ɛ̃pekabl | ɛ̃pekabl | ɛ̃pɛkabl |
| US | ys | ys | y |
| anobli | anobli | anobli | anɔbli |
| RD | ʁ | ʁ | ʁd |
| sông | soŋ | soŋ | sɔ̃ |
| Madonna | madɔna | madɔna | madona |
| pot | po | po | pɔ |
| Jésus | ʒezy | ʒezy | ʒezys |
| ait | e | e | ɛ |
| CO2 | kɔ | kɔ | kɔʁ |
| Schaer | ʃae | ʃae | ʃa |
| m2 | m | m | mʁ |
| PARTICULIERES | paʁtikyliʁ | paʁtikyliʁ | paʁtikyljʁ |
| os | ɔs | ɔs | o |
| vraisemblablement | vʁɛsɑ̃blabləmɑ̃ | vʁɛsɑ̃blabləmɑ̃ | vʁɛzɑ̃blabləmɑ̃ |
| Laisné | lɛne | lɛne | lɛsne |
| abus | aby | aby | abys |
| Essonne | ɛsɔn | ɛsɔn | esɔn |
| Drinking | dʁinkiŋ | dʁinkiŋ | dʁɛ̃nkiŋ |
| Colin | kolɛ̃ | kolɛ̃ | kɔlɛ̃ |
| c~ | k | k | kb |
| Blumenfeld | blymɑ̃fɛld | blymɑ̃fɛld | blymɑ̃nfɛld |
| vingt-six | vɛ̃tsis | vɛ̃tsis | vɛ̃tsi |
| OMS | ɔm | ɔm | ɔ |
| vigilan~ | viʒilɑ̃ | viʒilɑ̃ | viʒilɑ̃b |
| te | tə | tə | t |
| jolie | ʒoli | ʒoli | ʒɔli |

*36 mots uniques concernés (sur 67 occurrences)*

## Exemples : erreur structurelle (ctx KO, iso KO) — top 50

| Mot | Gold | Prédiction contexte | Prédiction isolée |
|-----|------|--------------------|-----------------|
| sens | sɑ̃ | sɑ̃s | sɑ̃s |
| qu' | k | kə | kə |
| l' | l | lə | lə |
| d' | d | də | də |
| Averroès | avɛʁɔɛs | avɛʁoɛ | avɛʁoɛ |
| décentrement | desɑ̃tʁømɑ̃ | desɑ̃tʁəmɑ̃ | desɑ̃tʁəmɑ̃ |
| ONG | ɔ̃ŋ | ɔ̃ɡ | ɔ̃ |
| 35 | e | f | f |
| 1er | e | lɛʁ | leʁ |
| lituanienne | lityanjɛn | litɥanjɛn | litɥanjɛn |
| 31 | e | f | f |
| 2012 | tʃ | s | s |
| 2013 | tʃ | sf | ff |
| subsaharienne | sybzaaʁjɛn | sypsaaʁjɛn | sybsaaʁjɛn |
| 2007 | tʃ | f | s |
| paroxysme | paʁɔksizm | paʁɔzism | paʁozism |
| n' | n | nə | nə |
| c' | s | sø | sø |
| s' | s | sə | sə |
| coeur | kœʁ | koœʁ | kuœʁ |
| 100 | tʃ | s | s |
| Moyen-âge | mwajɛnaʒ | mwajɑ̃taʒ | mwajɑ̃taʒ |
| l'on | lɔ̃ | leɔ̃ | ləɔ̃ |
| externes | ɛkstɛʁn | ɛktɛʁn | ɛktɛʁn |
| relations | ʁølasjɔ̃ | ʁəlasjɔ̃ | ʁəlasjɔ̃ |
| 20 | e | f | f |
| textes | tɛkst | tɛkt | tɛkt |
| el-Kadhafi | ɛlpkadafi | ɛltkadafi | ɛltkadafi |
| somalisation | sɔmazasjɔ̃ | somalizasjɔ̃ | somalizasjɔ̃ |
| anti-hérétiques | ɑ̃tiseʁetik | ɑ̃titeʁetik | ɑ̃titeʁetik |
| Wade | wɛd | wad | wad |
| texte | tɛkst | tɛkt | tɛkt |
| N' | n | nə | nə |
| -pas | spa | tpa | tpa |
| iii | jei | iii | iii |
| 45 | e | f | f |
| extérieure | ɛksteʁjœʁ | ɛkteʁjœʁ | ɛkteʁjœʁ |
| -il | il | til | til |
| C' | s | sø | sø |
| fixera | fiksəʁa | fikəʁa | fikəʁa |
| agenda | aʒɛ̃da | aʒɑ̃da | aʒɑ̃da |
| probable | pʁobablə | pʁobabl | pʁobabl |
| -nous | nu | tnu | tnu |
| 174 | tʃ | s | s |
| fondamentalisme | fɔ̃damɑ̃talism | fɔ̃damɑ̃talizm | fɔ̃damɑ̃talizm |
| -t-il | ti | tttil | tttil |
| 30 | e | f | f |
| publié | pyblije | pyblje | pyblje |
| Berlinale | bɛʁlɛinal | bɛʁlinal | bɛʁlinal |
| expérimentale | ɛkspeʁimɑ̃tal | ɛkpeʁimɑ̃tal | ɛkpeʁimɑ̃tal |

*869 mots uniques concernés (sur 2852 occurrences)*

## Exemples : le contexte nuit (ctx KO, iso OK) — top 30

| Mot | Gold | Prédiction contexte | Prédiction isolée |
|-----|------|--------------------|-----------------|
| jusqu' | ʒysk | ʒyskə | ʒysk |
| Jusqu' | ʒysk | ʒyskə | ʒysk |
| Moody's | mudi | mudj | mudi |
| net | nɛt | nɛ | nɛt |
| Soumaila | sumɛla | sumɛa | sumɛla |
| essayez | eseje | esɛje | eseje |
| Pub | pœb | pyb | pœb |
| peut-être | pøtɛtʁ | pøɛtʁ | pøtɛtʁ |
| écozones | ekozon | ekozɔn | ekozon |
| bin | bin | bɛ̃n | bin |
| raz | ʁa | ʁaz | ʁa |
| pédophile | pedofil | pedɔfil | pedofil |
| au-delà | odəla | otdəla | odəla |
| rejetées | ʁəʒte | ʁəʒəte | ʁəʒte |
| EU | y | ø | y |
| Nobel | nɔbɛl | nobɛl | nɔbɛl |
| subsidiaire | sybzidjɛʁ | sypzidjɛʁ | sybzidjɛʁ |
| infarctus | ɛ̃faʁktys | ɛ̃faʁkty | ɛ̃faʁktys |
| adéquate | adekwat | adekat | adekwat |
| pharmacodynamiques | faʁmakɔdinamik | faʁmakodinamik | faʁmakɔdinamik |
| Jacqueline | ʒakəlin | ʒaklin | ʒakəlin |
| B. | b | bʁ | b |
| peur | pøʁ | pœʁ | pøʁ |
| opus | opys | opy | opys |
| Flahault | flao | flaol | flao |
| est | e | est | e |
| requiert | ʁəkjɛʁ | ʁəkjɛ | ʁəkjɛʁ |

*27 mots uniques concernés (sur 50 occurrences)*

