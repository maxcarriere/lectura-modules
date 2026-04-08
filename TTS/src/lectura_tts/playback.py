"""Lecture audio — backend sounddevice ou pygame.

Fournit AudioPlayer pour jouer des buffers numpy float32 mono.
"""

from __future__ import annotations

import threading

import numpy as np


class AudioPlayer:
    """Lecteur audio simple pour buffers numpy float32 mono."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._playing = False
        self._thread: threading.Thread | None = None

    def play_buffer(self, samples: np.ndarray, sample_rate: int) -> None:
        """Joue un buffer audio (bloquant). Appeler depuis un thread séparé."""
        if len(samples) == 0:
            return

        self._stop_event.clear()
        self._playing = True

        try:
            self._play_sounddevice(samples, sample_rate)
        except Exception:
            self._play_pygame(samples, sample_rate)
        finally:
            self._playing = False

    def _play_sounddevice(self, samples: np.ndarray, sample_rate: int) -> None:
        """Backend sounddevice (préféré)."""
        import sounddevice as sd

        # Jouer en chunks pour pouvoir interrompre
        chunk_size = int(sample_rate * 0.05)  # 50ms
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
        stream.start()

        try:
            for i in range(0, len(samples), chunk_size):
                if self._stop_event.is_set():
                    break
                chunk = samples[i:i + chunk_size]
                stream.write(chunk.reshape(-1, 1))
        finally:
            stream.stop()
            stream.close()

    def _play_pygame(self, samples: np.ndarray, sample_rate: int) -> None:
        """Backend pygame (fallback)."""
        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)

        # Convertir float32 → int16
        samples_int16 = (samples * 32767).astype(np.int16)
        sound = pygame.mixer.Sound(buffer=samples_int16.tobytes())
        channel = sound.play()

        if channel is not None:
            while channel.get_busy():
                if self._stop_event.is_set():
                    channel.stop()
                    break
                self._stop_event.wait(0.015)

    def stop(self) -> None:
        """Arrête la lecture en cours."""
        self._stop_event.set()

    def is_playing(self) -> bool:
        """Retourne True si une lecture est en cours."""
        return self._playing
