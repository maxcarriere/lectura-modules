"""Moteur d'inference ONNX pour FormulaCTC.

Chargement du modele ONNX (INT8 prefere, fallback FP32), decodage CTC
greedy, conversion tokens → noms semantiques.

Pattern identique aux autres modules Lectura (Phonemiseur, TTS) :
factory creer_engine(), decouverte de modeles, lazy import onnxruntime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from lectura_stt_formules._mel import compute_log_mel, load_audio
from lectura_stt_formules._vocab import NUM_TOKENS, VOCAB

# Noms des fichiers modeles
_MODEL_INT8 = "formula_ctc_int8.onnx"
_MODEL_FP32 = "formula_ctc.onnx"


# ──────────────────────────────────────────────
# Decouverte des modeles
# ──────────────────────────────────────────────

def _resoudre_modeles_dir(models_dir: str | Path | None = None) -> Path | None:
    """Cascade de resolution du dossier modeles.

    1. Parametre explicite ``models_dir``
    2. Variable d'environnement ``LECTURA_MODELS_DIR/stt_formules/``
    3. ``~/.lectura/models/stt_formules/``
    4. Site-packages ``lectura_stt_formules/modeles/``

    Returns:
        Le dossier si un fichier ONNX (INT8 ou FP32) y existe, sinon None.
    """
    candidats: list[Path] = []

    # 1. Parametre explicite
    if models_dir is not None:
        candidats.append(Path(models_dir))

    # 2. Env var
    env_dir = os.environ.get("LECTURA_MODELS_DIR")
    if env_dir:
        candidats.append(Path(env_dir) / "stt_formules")

    # 3. Home
    candidats.append(Path.home() / ".lectura" / "models" / "stt_formules")

    # 4. Site-packages (a cote de ce fichier)
    candidats.append(Path(__file__).resolve().parent / "modeles")

    for d in candidats:
        if d.is_dir() and (
            (d / _MODEL_INT8).is_file() or (d / _MODEL_FP32).is_file()
        ):
            return d

    return None


# ──────────────────────────────────────────────
# Decodage CTC greedy (numpy)
# ──────────────────────────────────────────────

def _ctc_greedy_decode(logits: np.ndarray, blank_id: int = 0) -> list[int]:
    """Decodage CTC greedy : argmax + suppression repetitions/blanks.

    Args:
        logits: (T, V) log-probabilites pour une seule sequence

    Returns:
        Liste de token IDs decodes (sans blanks ni repetitions).
    """
    pred_ids = np.argmax(logits, axis=-1)  # (T,)
    decoded: list[int] = []
    prev = None
    for idx in pred_ids.tolist():
        if idx != blank_id and idx != prev:
            decoded.append(idx)
        prev = idx
    return decoded


# ──────────────────────────────────────────────
# Moteur d'inference ONNX
# ──────────────────────────────────────────────

class OnnxFormulaEngine:
    """Moteur d'inference FormulaCTC via ONNX Runtime.

    Charge le modele ONNX (INT8 prefere, fallback FP32), extrait le mel
    spectrogram, fait l'inference et decode le resultat CTC.
    """

    def __init__(self, models_dir: str | Path | None = None) -> None:
        """Charge le modele ONNX.

        Args:
            models_dir: chemin vers le dossier contenant formula_ctc*.onnx.
                Si None, utilise la cascade de resolution standard.

        Raises:
            FileNotFoundError: si aucun modele ONNX n'est trouve.
            ImportError: si onnxruntime n'est pas installe.
        """
        resolved = _resoudre_modeles_dir(models_dir)
        if resolved is None:
            raise FileNotFoundError(
                "Aucun modele ONNX trouve. Installez les modeles dans "
                "~/.lectura/models/stt_formules/ ou passez models_dir= "
                "explicitement."
            )

        # Preferer INT8, fallback FP32
        int8_path = resolved / _MODEL_INT8
        fp32_path = resolved / _MODEL_FP32
        if int8_path.is_file():
            self._model_path = int8_path
        elif fp32_path.is_file():
            self._model_path = fp32_path
        else:
            raise FileNotFoundError(
                f"Ni {_MODEL_INT8} ni {_MODEL_FP32} dans {resolved}"
            )

        # Lazy import onnxruntime
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime est requis pour l'inference. "
                "Installez-le : pip install lectura-stt-formules[inference]"
            )

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )

    @property
    def model_path(self) -> Path:
        """Chemin du modele ONNX charge."""
        return self._model_path

    def transcrire(self, audio_path: str | Path) -> dict:
        """Transcrit un fichier audio en tokens semantiques.

        Args:
            audio_path: chemin vers le fichier audio (WAV, FLAC, etc.)

        Returns:
            dict avec :
                - ``tokens``: list[int] — token IDs decodes
                - ``names``: list[str] — noms semantiques
                - ``logits``: ndarray (T', 87) — logits bruts
        """
        waveform = load_audio(audio_path)
        mel = compute_log_mel(waveform)
        return self.transcrire_mel(mel)

    def transcrire_mel(self, mel: np.ndarray) -> dict:
        """Transcrit un mel spectrogram pre-calcule.

        Args:
            mel: ndarray float32 de shape (1, n_mels, T) ou (1, 1, n_mels, T)

        Returns:
            dict avec tokens, names, logits (cf. transcrire).
        """
        # Le modele attend (B, 1, 80, T)
        if mel.ndim == 3:
            mel = mel[:, np.newaxis, :, :]  # (1, 80, T) -> (1, 1, 80, T)

        mel = mel.astype(np.float32)

        # Inference ONNX
        logits = self._session.run(None, {"mel": mel})[0]  # (B, T', V)

        # Decodage CTC greedy (premiere sequence du batch)
        tokens = _ctc_greedy_decode(logits[0])

        # Conversion tokens -> noms
        names = [VOCAB.get(t, f"?{t}") for t in tokens]

        return {
            "tokens": tokens,
            "names": names,
            "logits": logits[0],
        }
