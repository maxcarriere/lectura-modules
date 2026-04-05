"""Exemple 5 — Pipeline complet : texte → tokens → phonetique → syllabes.

Cet exemple montre comment enchainer les modules Lectura pour un
traitement complet du francais.

pip install lectura-tokeniseur lectura-formules lectura-g2p[onnx] lectura-aligneur
"""

from pathlib import Path

# --- Etape 1 : Tokenisation ---
from lectura_tokeniseur import LecturaTokeniseur, Formule

texte = "Le 1er janvier 2025, les 42 eleves ont lu 3/4 du livre."

tk = LecturaTokeniseur()
resultat_tk = tk.analyze(texte)

print("=" * 60)
print("ETAPE 1 — Tokenisation")
print("=" * 60)
print(f"Texte original  : {texte}")
print(f"Texte normalise : {resultat_tk.normalise}")
print(f"Nb mots         : {resultat_tk.nb_mots}")
print(f"Nb formules     : {len(resultat_tk.formules)}")
print()

# --- Etape 2 : Lecture des formules ---
from lectura_formules import enrichir_formules

enrichir_formules(resultat_tk.formules)

print("=" * 60)
print("ETAPE 2 — Lecture des formules")
print("=" * 60)
for f in resultat_tk.formules:
    print(f"  {f.formule_type.value:12s} {f.text!r:20s} → {f.display_fr!r}")
print()

# --- Etape 3 : G2P (phonetisation) ---
MODELES = Path(__file__).resolve().parent.parent / "G2P" / "modeles"

print("=" * 60)
print("ETAPE 3 — Phonetisation G2P")
print("=" * 60)

try:
    from lectura_nlp.inference_onnx import OnnxInferenceEngine
    from lectura_nlp.tokeniseur import tokeniser

    engine = OnnxInferenceEngine(
        str(MODELES / "unifie_int8.onnx"),
        str(MODELES / "unifie_vocab.json"),
    )

    # Phonetiser les mots (pas les formules)
    mots_texte = [t.text for t in resultat_tk.tokens if hasattr(t, "ortho")]
    tokens_g2p = tokeniser(" ".join(mots_texte))
    resultat_g2p = engine.analyser(tokens_g2p)

    for mot in resultat_g2p["mots"]:
        print(f"  {mot['ortho']:15s} → /{mot['phonemes']:15s}/  POS={mot['pos']}")

except ImportError:
    print("  (onnxruntime non installe — etape sautee)")
    print("  pip install lectura-g2p[onnx]")

print()
print("=" * 60)
print("Pipeline complet termine.")
print("=" * 60)
