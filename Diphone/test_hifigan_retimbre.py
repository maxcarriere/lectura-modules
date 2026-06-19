"""Test : re-timbrage d'audio diphone via HiFi-GAN SIWIS.

Compare 3 approches :
  A. Audio diphone brut (pyworld.synthesize — timbre moche)
  B. Audio diphone → mel → HiFi-GAN SIWIS (re-timbrage leger, 14 MB)
  C. Audio diphone → OpenVoice (re-timbrage actuel, 126 MB)

Genere 3 fichiers WAV pour comparaison auditive.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# ── Config ──────────────────────────────────────────────────────────────

HIFIGAN_PATH = Path(
    "/data/work/projets/lectura/workspace/Modules/TTS-Monospeaker"
    "/src/lectura_tts_monospeaker/modeles/hifigan.onnx"
)
OPENVOICE_MODELS = Path(
    "/data/work/projets/lectura/workspace/Modules/VC/src/lectura_vc/modeles"
)
OUTPUT_DIR = Path("/data/work/projets/lectura/workspace/Modules/TTS-Diphone/samples_retimbre")

# Mel-spectrogram params (must match HiFi-GAN training)
MEL_SR = 22050
MEL_N_FFT = 1024
MEL_HOP = 256
MEL_N_MELS = 80
MEL_FMIN = 0
MEL_FMAX = 8000


# ── Fonctions utilitaires ───────────────────────────────────────────────

def audio_to_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convertir audio en mel-spectrogramme (format HiFi-GAN).

    Returns:
        np.ndarray shape (80, T) en log scale.
    """
    # Resample si necessaire
    if sr != MEL_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=MEL_SR)

    # Mel-spectrogramme (meme params que l'entrainement FastPitch/HiFi-GAN)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=MEL_SR,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP,
        n_mels=MEL_N_MELS,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        power=1.0,  # amplitude (pas puissance)
    )
    # Log mel (clip pour eviter log(0))
    log_mel = np.log(np.maximum(mel, 1e-5)).astype(np.float32)
    return log_mel


def hifigan_synthesize(mel: np.ndarray) -> np.ndarray:
    """Passer un mel-spectrogramme dans HiFi-GAN ONNX.

    Args:
        mel: shape (80, T) float32 log mel-spectrogram.

    Returns:
        audio float32 a 22050 Hz.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        str(HIFIGAN_PATH), opts,
        providers=["CPUExecutionProvider"],
    )

    mel_input = mel[np.newaxis].astype(np.float32)  # (1, 80, T)
    audio = sess.run(None, {"mel": mel_input})[0]  # (1, 1, T_audio)
    return audio.squeeze().astype(np.float32)


def openvoice_convert(audio: np.ndarray, sr: int) -> np.ndarray:
    """Passer l'audio dans OpenVoice avec la voix SIWIS comme cible.

    Utilise la source elle-meme comme reference SE (auto-timbrage).
    En pratique on utiliserait un SE SIWIS pre-calcule.
    """
    from lectura_vc._openvoice import OpenVoiceConverter

    ov = OpenVoiceConverter(OPENVOICE_MODELS)

    # Extraire SE de la source (= la voix diphone moyennee)
    src_se = ov.extract_se(audio, sr=sr)

    # Pour la cible, on utilise un extrait SIWIS si disponible,
    # sinon on utilise src_se (ce qui fait un "auto-cleanup")
    siwis_ref = Path(
        "/home/moi/Documents/work/projets/lectura/workspace/Modules/VC/extraits/siwis.wav"
    )
    if siwis_ref.exists():
        tgt_se = ov.extract_se(siwis_ref)
        print(f"  OpenVoice reference: {siwis_ref.name}")
    else:
        print("  OpenVoice: reference SIWIS introuvable, auto-timbrage")
        tgt_se = src_se

    out, out_sr = ov.convert(audio, src_se, tgt_se, sr=sr, tau=0.0)
    return out


# ── Synthese diphone de test ────────────────────────────────────────────

def generate_diphone_audio() -> tuple[np.ndarray, int]:
    """Genere un audio diphone de test.

    Retourne (audio float32, sample_rate).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode

    engine = DiphoneEngine()
    engine.load()

    # Phrase test : "Bonjour comment allez-vous"
    # Un seul groupe prosodique (phrase simple sans virgule)
    groups = [
        {
            "phones": ["b", "ɔ̃", "ʒ", "u", "ʁ",
                       "k", "ɔ", "m", "ɑ̃",
                       "a", "l", "e",
                       "v", "u"],
            "boundary": "period",
            "word_boundaries": [5, 9, 12],
        },
    ]

    audio = engine.synthesize_groups(
        groups,
        mode=SynthMode.FLUIDE,
        prosody_style="regles",
        base_f0=200.0,
        macro_expressivity=1.0,
        micro_expressivity=1.0,
        seed=42,
    )
    return audio, 44100


# ── Main ────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generer l'audio diphone brut
    print("1. Synthese diphone...")
    audio_diphone, sr_diphone = generate_diphone_audio()
    out_a = OUTPUT_DIR / "A_diphone_brut.wav"
    sf.write(out_a, audio_diphone, sr_diphone)
    print(f"   -> {out_a} ({len(audio_diphone)/sr_diphone:.2f}s)")

    # 2. Re-timbrage via HiFi-GAN
    print("2. Re-timbrage HiFi-GAN SIWIS (14 MB)...")
    mel = audio_to_mel(audio_diphone, sr_diphone)
    print(f"   Mel shape: {mel.shape} (min={mel.min():.1f}, max={mel.max():.1f})")
    audio_hifigan = hifigan_synthesize(mel)
    # Normaliser
    peak = np.max(np.abs(audio_hifigan))
    if peak > 0:
        audio_hifigan = audio_hifigan * 0.9 / peak
    out_b = OUTPUT_DIR / "B_hifigan_retimbre.wav"
    sf.write(out_b, audio_hifigan, MEL_SR)
    print(f"   -> {out_b} ({len(audio_hifigan)/MEL_SR:.2f}s)")

    # 3. Re-timbrage via OpenVoice
    print("3. Re-timbrage OpenVoice (126 MB)...")
    try:
        audio_ov = openvoice_convert(audio_diphone, sr_diphone)
        peak = np.max(np.abs(audio_ov))
        if peak > 0:
            audio_ov = audio_ov * 0.9 / peak
        out_c = OUTPUT_DIR / "C_openvoice_retimbre.wav"
        sf.write(out_c, audio_ov, 22050)
        print(f"   -> {out_c} ({len(audio_ov)/22050:.2f}s)")
    except Exception as e:
        print(f"   OpenVoice echoue: {e}")
        print("   (Ignoré — comparez A vs B)")

    print()
    print("Comparaison:")
    print(f"  A (brut):     {out_a}")
    print(f"  B (HiFi-GAN): {out_b}")
    if 'out_c' in dir():
        print(f"  C (OpenVoice): {out_c}")
    print()
    print("Ecoutez A vs B : est-ce que le timbre est meilleur en B ?")


if __name__ == "__main__":
    main()
