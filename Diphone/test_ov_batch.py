"""Batch : OpenVoice avec nadine/ezwa + regeneration en prosodie corpus.

Genere :
  /tmp/tts_prosody_v2/openvoice_nadine/  (20 fichiers, ref nadine)
  /tmp/tts_prosody_v2/openvoice_ezwa/    (20 fichiers, ref ezwa)
  /tmp/tts_prosody_v2/corpus/            (20 fichiers, prosodie corpus)
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# ── Fix numpy compat ──
import numpy.core.numeric
if not hasattr(np, '_core'):
    class _core: pass
    nc = _core()
    nc.numeric = numpy.core.numeric
    sys.modules['numpy._core'] = nc
    sys.modules['numpy._core.numeric'] = numpy.core.numeric

from lectura_vc._openvoice import OpenVoiceConverter
from lectura_tts_diphone.engine import DiphoneEngine, SynthMode

# ── Config ──
OPENVOICE_MODELS = Path("/data/work/projets/lectura/workspace/Modules/VC/src/lectura_vc/modeles")
EXTRAITS = Path("/home/moi/Documents/work/projets/lectura/workspace/Modules/VC/extraits")
INPUT_DIR = Path("/tmp/tts_prosody_v2")
OUT_NADINE = INPUT_DIR / "openvoice_nadine"
OUT_EZWA = INPUT_DIR / "openvoice_ezwa"
OUT_CORPUS = INPUT_DIR / "corpus"

PHRASES = {
    "d_court1":    ("Le chat dort.", "period"),
    "d_court2":    ("Bonjour.", "period"),
    "d_simple":    ("Le petit chat est mort.", "period"),
    "d_moyen":     ("La pluie tombe sur les toits de la ville.", "period"),
    "d_long":      ("Les enfants jouent dans le jardin pendant que les parents discutent.", "period"),
    "v_simple":    ("Le matin, je prends un café.", "period"),
    "v_double":    ("Il ouvrit la porte, regarda dehors, et ne vit personne.", "period"),
    "v_longue":    ("Après avoir longuement réfléchi, il décida de partir, malgré la pluie.", "period"),
    "v_enum":      ("Il acheta du pain, du beurre, du lait, et du fromage.", "period"),
    "q_court":     ("Comment allez-vous ?", "question"),
    "q_moyen":     ("Est-ce que vous avez vu le dernier film ?", "question"),
    "q_long":      ("Pourquoi les oiseaux chantent-ils au lever du soleil ?", "question"),
    "e_court":     ("Quel bonheur !", "exclamation"),
    "e_moyen":     ("Il fait vraiment beau aujourd'hui !", "exclamation"),
    "e_long":      ("Je n'aurais jamais imaginé une chose pareille !", "exclamation"),
    "s_court":     ("Je ne sais pas...", "period"),
    "s_moyen":     ("Il y avait quelque chose de bizarre...", "period"),
    "n_dialogue":  ("Oui, c'est une bonne idée.", "period"),
    "n_complexe":  ("Le professeur, qui avait beaucoup voyagé, racontait ses aventures.", "period"),
    "n_lecture":   ("La nuit était calme, les étoiles brillaient dans le ciel.", "period"),
}


def batch_openvoice(inputs, ref_path, output_dir, label):
    """Passe tous les fichiers dans OpenVoice avec une reference donnee."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ov = OpenVoiceConverter(OPENVOICE_MODELS)
    tgt_se = ov.extract_se(ref_path)
    print(f"\n{'='*60}")
    print(f"  OpenVoice → {label} (ref: {ref_path.name})")
    print(f"{'='*60}")

    for wav_path in sorted(inputs):
        src_se = ov.extract_se(wav_path)
        audio_out, sr_out = ov.convert(wav_path, src_se, tgt_se, tau=0.0)
        peak = np.max(np.abs(audio_out))
        if peak > 0:
            audio_out = audio_out * 0.9 / peak
        out_name = wav_path.name.replace("orig_", f"ov_{label}_")
        out_path = output_dir / out_name
        sf.write(out_path, audio_out, sr_out)
        print(f"  {wav_path.name} -> {out_name}")

    print(f"  Output: {output_dir}")


def batch_corpus_prosody():
    """Regenere les 20 phrases avec prosody_style='corpus'."""
    OUT_CORPUS.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Prosodie corpus (style='corpus')")
    print(f"{'='*60}")

    engine = DiphoneEngine()
    engine.load()

    # On a besoin de phonemiser les textes
    # Utiliser le pipeline G2P si disponible
    try:
        from lectura_g2p import creer_engine as creer_g2p
        g2p = creer_g2p()
        has_g2p = True
    except ImportError:
        has_g2p = False
        print("  WARN: lectura_g2p non disponible, skip")
        return

    for name, (text, boundary) in PHRASES.items():
        # Phonemiser
        result = g2p.phonemize(text)
        groups = result if isinstance(result, list) else [result]

        # S'assurer que chaque groupe a le bon format
        formatted_groups = []
        for g in groups:
            if isinstance(g, dict):
                formatted_groups.append(g)
            else:
                formatted_groups.append({
                    "phones": list(g) if isinstance(g, str) else g,
                    "boundary": boundary,
                    "word_boundaries": [],
                })

        # Forcer le boundary du dernier groupe
        if formatted_groups:
            formatted_groups[-1]["boundary"] = boundary

        audio = engine.synthesize_groups(
            formatted_groups,
            mode=SynthMode.FLUIDE,
            prosody_style="corpus",
            duration_scale=1.0,
            pause_scale=1.0,
            macro_expressivity=1.0,
            micro_expressivity=1.0,
            seed=42,
            spectral_contrast=1.5,
            ap_cleanup=1.5,
            formant_sharpening=1.3,
            base_f0=200.0,
        )

        out_path = OUT_CORPUS / f"corpus_{name}.wav"
        sf.write(str(out_path), audio, 44100)
        print(f"  {name}: {text[:40]:40s} -> {len(audio)/44100:.2f}s")

    print(f"  Output: {OUT_CORPUS}")


# ── Main ──
if __name__ == "__main__":
    inputs = sorted(INPUT_DIR.glob("orig_*.wav"))
    print(f"{len(inputs)} fichiers source")

    # 1. OpenVoice nadine
    batch_openvoice(inputs, EXTRAITS / "nadine.wav", OUT_NADINE, "nadine")

    # 2. OpenVoice ezwa
    batch_openvoice(inputs, EXTRAITS / "ezwa.wav", OUT_EZWA, "ezwa")

    # 3. Prosodie corpus
    batch_corpus_prosody()

    print(f"\n{'='*60}")
    print("  Tout est genere.")
    print(f"{'='*60}")
