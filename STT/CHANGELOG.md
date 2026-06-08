# Changelog — lectura-stt

## v3.0.0 (2026-06-08)

- Pipeline optimal avec postprocessing CTC :
  - parse_ctc_v2 (segments enrichis : liaisons, composes, elisions)
  - strip_liaisons (suppression liaisons erronees via PhoneLexicon)
  - split_elisions (separation clitiques elides)
  - split_merged_words (decoupe mots sur-segmentes)
  - merge_and_rescore (fusion + rescoring lexical)
  - try_elision_merges (clitiques elides adjacents)
  - rejoin_elisions (reconstruction avec apostrophes et tirets)
- Dependance CTC >= 2.0 (modele medium 10.6M params)
- WER benchmark : ~23.5% (all) / ~19.7% (parole courante)
- Integration P2G V7 (analyser_v2 avec lex_select)
- PhoneLexicon : detection automatique depuis le graphemiseur

## v2.0.0 (2026-06-04)

- Pipeline CTC + P2G complet
- Cascade P2G : lectura-p2g → lectura-graphemiseur → aucun
- Support formules via lectura-p2g (nombres, sigles, noms propres)
- Elisions et ponctuation automatiques

## v1.0.0 (2026-05-11)

- Version initiale
- Pipeline audio → texte simple (CTC + P2G)
