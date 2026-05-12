"""Utilitaires ffmpeg pour l'export vidéo — pure Python, zéro Qt."""

from __future__ import annotations

import subprocess
import tempfile

from ..models import ExportTimingData, VideoExportOptions


def is_ffmpeg_available() -> bool:
    """Vérifie si imageio-ffmpeg est installé et fonctionnel."""
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:
        return False


def get_ffmpeg_path() -> str:
    """Retourne le chemin du binaire ffmpeg via imageio-ffmpeg."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_ffmpeg_command(
    options: VideoExportOptions,
    width: int,
    height: int,
    fps: int,
    audio_path: str,
) -> list[str]:
    """Construit la commande ffmpeg pour l'encodage vidéo."""
    ffmpeg = get_ffmpeg_path()

    cmd = [
        ffmpeg,
        "-y",  # overwrite
        # Input vidéo : raw RGBA depuis stdin
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        # Input audio
        "-i", audio_path,
    ]

    fmt = options.format

    if fmt == "mp4":
        cmd += [
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
        ]
    elif fmt == "webm":
        cmd += [
            "-c:v", "libvpx-vp9",
            "-crf", "20",
            "-b:v", "0",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
            "-c:a", "libopus",
            "-b:a", "128k",
        ]
    elif fmt == "mov":
        cmd += [
            "-c:v", "prores_ks",
            "-profile:v", "4",
            "-pix_fmt", "yuva444p10le",
            "-c:a", "pcm_s16le",
        ]

    cmd += ["-shortest", options.output_path]
    return cmd


def export_temp_audio(segments, timing_data: ExportTimingData) -> str:
    """Exporte l'audio en WAV temporaire pour le muxage vidéo.

    Parameters
    ----------
    segments : list[AudioSegment]
        Segments audio au format exporter.
    timing_data : ExportTimingData
        Données de timing.

    Returns
    -------
    str
        Chemin du fichier WAV temporaire.
    """
    from ..models import AudioExportOptions
    from ..audio import export_audio

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    options = AudioExportOptions(
        format="wav",
        include_timing=False,
        normalize=True,
    )

    export_audio(segments, tmp_path, timing_data, options)
    return tmp_path
