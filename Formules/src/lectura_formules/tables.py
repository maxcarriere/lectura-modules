"""TablesStore — chargement et stockage des tables CSV externalisées.

Charge les 9 CSV embarqués via importlib.resources (Python 3.10+)
et expose un singleton paresseux via get_store().

Licence : CC-BY-SA-4.0
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass UniteDef
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UniteDef:
    """Définition d'une unité lue depuis unites.csv."""
    label: str           # "[10z]"
    base: str            # "[10]"
    variante: str        # "avec son z"
    value: int | None    # 10
    texte: str           # "dix"
    phone: str           # "diz"
    sound: str           # "num015"
    display_rom: str     # "X"
    display_num: str     # ""
    display_fr: str      # "dix"


# ══════════════════════════════════════════════════════════════════════════════
# TablesStore
# ══════════════════════════════════════════════════════════════════════════════

class TablesStore:
    """Charge et stocke les tables CSV."""

    def __init__(self) -> None:
        self.unite_by_label: dict[str, UniteDef] = {}
        self.numeral_to_unit: dict[str, str] = {}       # "zero" → "[0]"
        self.symbol_to_unit: dict[str, str] = {}         # "I" → "[1]"
        self.roman_block_map: dict[str, list[str]] = {}  # "[1]" → ["[1]"]
        self.ordinal_map: dict[str, tuple[str, str, str]] = {}  # "un" → ("unième", phone, sound)
        self.letter_map: dict[str, tuple[str, str, str]] = {}   # "A" → ("a", phone, sound)
        self.symbol_sound_map: dict[str, tuple[str, str, str, str]] = {}  # "€" → (texte, phone, sound, cat)
        self.month_map: dict[int, tuple[str, str, str]] = {}    # 1 → ("janvier", phone, sound)
        self.day_map: dict[str, tuple[str, str, str]] = {}      # "lundi" → (texte, phone, sound)
        self.greek_map: dict[str, tuple[str, str, str]] = {}    # "α" → ("alpha", phone, sound)

    @classmethod
    def load(cls) -> TablesStore:
        """Charge depuis les CSV embarqués (importlib.resources)."""
        store = cls()
        store._load_unites()
        store._load_numeraux()
        store._load_symboles()
        store._load_romains()
        store._load_ordinaux()
        store._load_lettres()
        store._load_symboles_sons()
        store._load_calendrier()
        store._load_grec()
        return store

    # -- Chargeurs internes ---------------------------------------------------

    def _read_csv(self, filename: str) -> list[dict[str, str]]:
        """Lit un CSV depuis le package data/tables/."""
        ref = resources.files("lectura_formules.data.tables").joinpath(filename)
        raw = ref.read_text(encoding="utf-8")
        # Supprimer BOM éventuel
        if raw.startswith("\ufeff"):
            raw = raw[1:]
        reader = csv.DictReader(io.StringIO(raw))
        return list(reader)

    def _load_unites(self) -> None:
        for row in self._read_csv("unites.csv"):
            label = row["unite"]
            val_str = row.get("value", "")
            value: int | None = None
            if val_str:
                try:
                    value = int(val_str)
                except ValueError:
                    pass
            ud = UniteDef(
                label=label,
                base=row.get("base", ""),
                variante=row.get("variante", ""),
                value=value,
                texte=row.get("texte", ""),
                phone=row.get("phone", ""),
                sound=row.get("sound", ""),
                display_rom=row.get("display_rom", ""),
                display_num=row.get("display_num", ""),
                display_fr=row.get("display_fr", ""),
            )
            self.unite_by_label[label] = ud

    def _load_numeraux(self) -> None:
        for row in self._read_csv("numeraux.csv"):
            numeral = row["numeraux"]
            unit_label = row["unites"]
            self.numeral_to_unit[numeral] = unit_label

    def _load_symboles(self) -> None:
        for row in self._read_csv("symboles.csv"):
            symbol = row["symboles"]
            unit_label = row["unites"]
            self.symbol_to_unit[symbol] = unit_label

    def _load_romains(self) -> None:
        for row in self._read_csv("romains.csv"):
            nombre = row["nombre"]
            romain = row["romain"]
            # Parser la composition : "[1][5]" → ["[1]", "[5]"]
            import re
            parts = re.findall(r"\[[^\]]+\]", romain)
            self.roman_block_map[nombre] = parts

    def _load_ordinaux(self) -> None:
        for row in self._read_csv("ordinaux_sons.csv"):
            cardinal = row.get("cardinal", "")
            texte = row.get("texte", "")
            phone = row.get("phone", "")
            sound = row.get("sound", "")
            if cardinal:
                self.ordinal_map[cardinal] = (texte, phone, sound)

    def _load_lettres(self) -> None:
        for row in self._read_csv("lettres_sons.csv"):
            lettre = row["lettre"]
            self.letter_map[lettre] = (
                row.get("texte", ""),
                row.get("phone", ""),
                row.get("sound", ""),
            )

    def _load_symboles_sons(self) -> None:
        for row in self._read_csv("symboles_sons.csv"):
            symbole = row["symbole"]
            self.symbol_sound_map[symbole] = (
                row.get("texte", ""),
                row.get("phone", ""),
                row.get("sound", ""),
                row.get("categorie", ""),
            )

    def _load_calendrier(self) -> None:
        month_idx = 0
        for row in self._read_csv("calendrier_sons.csv"):
            cat = row.get("categorie", "")
            texte = row.get("texte", "")
            phone = row.get("phone", "")
            sound = row.get("sound", "")
            if cat == "mois":
                month_idx += 1
                self.month_map[month_idx] = (texte, phone, sound)
            elif cat == "jour":
                self.day_map[texte] = (texte, phone, sound)

    def _load_grec(self) -> None:
        for row in self._read_csv("grec_sons.csv"):
            caractere = row["caractere"]
            self.greek_map[caractere] = (
                row.get("texte", ""),
                row.get("phone", ""),
                row.get("sound", ""),
            )


# ══════════════════════════════════════════════════════════════════════════════
# Singleton paresseux
# ══════════════════════════════════════════════════════════════════════════════

_store: TablesStore | None = None


def get_store() -> TablesStore:
    """Retourne le singleton TablesStore (chargement paresseux)."""
    global _store
    if _store is None:
        _store = TablesStore.load()
    return _store


# ══════════════════════════════════════════════════════════════════════════════
# Accès aux fichiers sons (chemin configurable)
# ══════════════════════════════════════════════════════════════════════════════

_sounds_dir: Path | None = None


def set_sounds_dir(path: str | Path) -> None:
    """Configure le chemin du dossier sons/fr/wav/.

    Les WAV ne sont pas embarqués dans le package pip (~12 Mo).
    Le programme appelant doit configurer ce chemin s'il veut
    utiliser les sons.

    >>> from lectura_formules.tables import set_sounds_dir
    >>> set_sounds_dir("/chemin/vers/data/sons/fr/wav")
    """
    global _sounds_dir
    _sounds_dir = Path(path)


def get_sounds_dir() -> Path | None:
    """Retourne le chemin du dossier sons, ou None si non configuré.

    Tente dans l'ordre :
    1. Le chemin configuré via set_sounds_dir()
    2. Le dossier embarqué dans le package (si présent)
    """
    if _sounds_dir is not None:
        return _sounds_dir

    # Fallback : chercher dans le package (développement local)
    try:
        ref = resources.files("lectura_formules.data.sons.fr.wav")
        p = Path(str(ref))
        if p.is_dir():
            return p
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    return None


def get_sound_path(sound_id: str) -> Path | None:
    """Retourne le chemin WAV pour un sound_id, ou None si inexistant."""
    sounds = get_sounds_dir()
    if sounds is None:
        return None
    wav = sounds / f"{sound_id}.wav"
    if wav.exists():
        return wav
    return None
