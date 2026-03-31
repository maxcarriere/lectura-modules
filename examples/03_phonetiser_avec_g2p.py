"""Exemple 3 — Convertir du texte en phonetique IPA (G2P).

pip install lectura-g2p[onnx]

Trois backends disponibles :
  - OnnxInferenceEngine  (~2ms/phrase, necessite onnxruntime)
  - NumpyInferenceEngine (~50ms/phrase, necessite numpy)
  - PureInferenceEngine  (~200ms/phrase, zero dependance)
"""

from pathlib import Path

# Adapter le chemin vers les modeles
MODELES = Path(__file__).resolve().parent.parent / "G2P" / "modeles"

# --- Backend ONNX (le plus rapide) ---
try:
    from lectura_nlp.inference_onnx import OnnxInferenceEngine
    from lectura_nlp.tokeniseur import tokeniser

    engine = OnnxInferenceEngine(
        str(MODELES / "unifie_int8.onnx"),
        str(MODELES / "unifie_vocab.json"),
    )

    phrase = "Les enfants jouent dans le jardin."
    tokens = tokeniser(phrase)
    resultat = engine.analyser(tokens)

    print("Phrase :", phrase)
    print()
    for mot in resultat["mots"]:
        print(f"  {mot['ortho']:15s} → /{mot['phonemes']}/  (POS: {mot['pos']})")

except ImportError:
    print("onnxruntime non installe. Essayez : pip install lectura-g2p[onnx]")

# --- Backend Pure Python (zero dependance) ---
print("\n--- Backend Pure Python ---")
from lectura_nlp.inference_pure import PureInferenceEngine
from lectura_nlp.tokeniseur import tokeniser

engine_pure = PureInferenceEngine(
    str(MODELES / "unifie_weights.json"),
    str(MODELES / "unifie_vocab.json"),
)

phrase = "Bonjour, comment allez-vous ?"
tokens = tokeniser(phrase)
resultat = engine_pure.analyser(tokens)

print("Phrase :", phrase)
for mot in resultat["mots"]:
    print(f"  {mot['ortho']:15s} → /{mot['phonemes']}/")
