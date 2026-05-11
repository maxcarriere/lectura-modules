"""Backend CSV autonome pour le Correcteur.

Charge ``lexique_correcteur.csv.gz`` en memoire et expose la meme
interface que ``lectura_lexique.Lexique`` (methodes utilisees par le
Correcteur : existe, info, frequence, phone_de, homophones, formes_de,
lemme_de, conjuguer).

Zero dependance externe — uniquement la stdlib Python (gzip + csv).

Optimisation memoire : les entrees sont stockees comme tuples compacts.
Les dicts sont reconstruits a la demande dans les methodes publiques.
Les chaines frequentes sont internees pour eviter les doublons.
"""

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path
from typing import Any

from lectura_correcteur._multext import decoder_multext

# ── Layout des tuples ─────────────────────────────────────────────────
# Chaque entree est un tuple :
#   (ortho, phone, cgram, genre, nombre, freq, lemme, multext,
#    mode, temps, personne)

_I_ORTHO = 0
_I_PHONE = 1
_I_CGRAM = 2
_I_GENRE = 3
_I_NOMBRE = 4
_I_FREQ = 5
_I_LEMME = 6
_I_MULTEXT = 7
_I_MODE = 8
_I_TEMPS = 9
_I_PERSONNE = 10


def _to_dict(t: tuple) -> dict[str, Any]:
    """Convertit un tuple compact en dict compatible LexiqueProtocol."""
    return {
        "ortho": t[_I_ORTHO],
        "phone": t[_I_PHONE],
        "cgram": t[_I_CGRAM],
        "genre": t[_I_GENRE],
        "nombre": t[_I_NOMBRE],
        "freq": t[_I_FREQ],
        "lemme": t[_I_LEMME],
        "multext": t[_I_MULTEXT],
        "mode": t[_I_MODE],
        "temps": t[_I_TEMPS],
        "personne": t[_I_PERSONNE],
    }


class LexiqueCSV:
    """Lexique en memoire charge depuis un CSV compresse.

    Fournit les methodes attendues par le Correcteur :
    existe, info, frequence, phone_de, homophones, formes_de,
    lemme_de, conjuguer, close.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._path = Path(csv_path)

        # Index primaire : ortho (lowercase) → list[tuple]
        self._data: dict[str, list[tuple]] = {}

        # Set de formes (lowercase) pour existe() O(1)
        self._formes: frozenset[str] = frozenset()

        # Index secondaires (lazy)
        self._phone_index: dict[str, list[tuple]] | None = None
        self._lemme_index: dict[str, list[tuple]] | None = None

        self._charger()

    # ── Chargement ────────────────────────────────────────────────────

    def _charger(self) -> None:
        """Charge le CSV gzip et construit l'index primaire."""
        data: dict[str, list[tuple]] = {}

        # Pool d'internement pour chaines repetees
        _intern: dict[str, str] = {}

        def intern(s: str) -> str:
            """Interne une chaine pour partager l'objet."""
            t = _intern.get(s)
            if t is None:
                _intern[s] = s
                return s
            return t

        with gzip.open(str(self._path), "rt", encoding="utf-8") as fin:
            reader = csv.reader(fin, delimiter=";")
            header = next(reader)

            # Indices des colonnes
            idx = {col: i for i, col in enumerate(header)}

            i_ortho = idx["ortho"]
            i_phone = idx["phone"]
            i_cgram = idx["cgram"]
            i_genre = idx["genre"]
            i_nombre = idx["nombre"]
            i_freq = idx["freq"]
            i_lemme = idx["lemme"]
            i_multext = idx["multext"]

            for row in reader:
                if len(row) < len(header):
                    continue

                ortho = row[i_ortho]
                phone = intern(row[i_phone])
                cgram = intern(row[i_cgram])
                genre_raw = row[i_genre]
                nombre_raw = row[i_nombre]
                multext = intern(row[i_multext])
                lemme = row[i_lemme]
                freq_raw = row[i_freq]

                try:
                    freq = float(freq_raw) if freq_raw else 0.0
                except (ValueError, TypeError):
                    freq = 0.0

                # Decoder multext pour les traits
                traits = decoder_multext(multext) if multext else {}

                genre = intern(genre_raw or traits.get("genre", ""))
                nombre = intern(traits.get("nombre", nombre_raw))
                mode = intern(traits.get("mode", ""))
                temps = intern(traits.get("temps", ""))
                personne = intern(traits.get("personne", ""))

                entry = (
                    ortho, phone, cgram, genre, nombre,
                    freq, lemme, multext,
                    mode, temps, personne,
                )

                key = ortho.lower()
                bucket = data.get(key)
                if bucket is None:
                    data[key] = [entry]
                else:
                    bucket.append(entry)

        self._data = data
        self._formes = frozenset(data.keys())

    def _get_phone_index(self) -> dict[str, list[tuple]]:
        """Construit et retourne l'index phone → entries (lazy)."""
        if self._phone_index is not None:
            return self._phone_index

        phone_index: dict[str, list[tuple]] = {}
        for entries in self._data.values():
            for entry in entries:
                phone = entry[_I_PHONE]
                if phone:
                    bucket = phone_index.get(phone)
                    if bucket is None:
                        phone_index[phone] = [entry]
                    else:
                        bucket.append(entry)

        self._phone_index = phone_index
        return phone_index

    def _get_lemme_index(self) -> dict[str, list[tuple]]:
        """Construit et retourne l'index lemme → entries (lazy)."""
        if self._lemme_index is not None:
            return self._lemme_index

        lemme_index: dict[str, list[tuple]] = {}
        for entries in self._data.values():
            for entry in entries:
                lemme = entry[_I_LEMME]
                if lemme:
                    key = lemme.lower()
                    bucket = lemme_index.get(key)
                    if bucket is None:
                        lemme_index[key] = [entry]
                    else:
                        bucket.append(entry)

        self._lemme_index = lemme_index
        return lemme_index

    # ── API publique (LexiqueProtocol) ────────────────────────────────

    def existe(self, mot: str) -> bool:
        """Test d'appartenance O(1)."""
        return mot.lower() in self._formes

    def info(self, mot: str) -> list[dict[str, Any]]:
        """Entrees lexicales completes pour un mot."""
        entries = self._data.get(mot.lower())
        if not entries:
            return []
        return [_to_dict(e) for e in entries]

    def frequence(self, mot: str) -> float:
        """Frequence maximale parmi toutes les entrees du mot."""
        entries = self._data.get(mot.lower())
        if not entries:
            return 0.0
        return max(e[_I_FREQ] for e in entries)

    def phone_de(self, mot: str) -> str | None:
        """Prononciation la plus frequente."""
        entries = self._data.get(mot.lower())
        if not entries:
            return None

        best_phone = ""
        best_freq = -1.0
        for e in entries:
            phone = e[_I_PHONE]
            if phone and e[_I_FREQ] > best_freq:
                best_freq = e[_I_FREQ]
                best_phone = phone

        return best_phone or None

    def homophones(self, phone: str) -> list[dict[str, Any]]:
        """Tous les mots ayant cette prononciation."""
        entries = self._get_phone_index().get(phone)
        if not entries:
            return []
        return [_to_dict(e) for e in entries]

    def formes_de(self, lemme: str, cgram: str | None = None) -> list[dict[str, Any]]:
        """Toutes les formes flechies d'un lemme, avec filtre POS optionnel."""
        entries = self._get_lemme_index().get(lemme.lower(), [])

        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        for e in entries:
            if cgram is not None:
                e_cgram = e[_I_CGRAM]
                if not e_cgram.startswith(cgram):
                    continue

            ortho = e[_I_ORTHO]
            key = ortho.lower()
            if ortho and key not in seen:
                seen.add(key)
                result.append(_to_dict(e))

        return result

    def lemme_de(self, mot: str) -> str | None:
        """Lemme le plus frequent parmi les entrees d'un mot."""
        entries = self._data.get(mot.lower())
        if not entries:
            return None

        freq_par_lemme: dict[str, float] = {}
        for e in entries:
            lemme = e[_I_LEMME]
            if not lemme:
                continue
            freq_par_lemme[lemme] = freq_par_lemme.get(lemme, 0.0) + e[_I_FREQ]

        if not freq_par_lemme:
            return None

        return max(freq_par_lemme, key=lambda k: freq_par_lemme[k])

    def conjuguer(self, verbe: str) -> dict[str, dict[str, dict[str, str]]]:
        """Table de conjugaison d'un verbe.

        Retour : {mode: {temps: {"1s": forme, "2s": forme, ...}}}
        """
        entries = self._get_lemme_index().get(verbe.lower(), [])

        table: dict[str, dict[str, dict[str, str]]] = {}

        for e in entries:
            cgram = e[_I_CGRAM]
            if not cgram.startswith(("VER", "AUX")):
                continue

            mode = e[_I_MODE]
            temps = e[_I_TEMPS]
            ortho = e[_I_ORTHO]

            if not mode or not ortho:
                continue

            personne = e[_I_PERSONNE]
            nombre = e[_I_NOMBRE]
            # Normaliser nombre long
            if isinstance(nombre, str) and len(nombre) > 1:
                nombre = nombre[0]

            # Construire la cle
            if personne and nombre:
                cle = f"{personne}{nombre}"
            elif personne:
                cle = personne
            else:
                cle = ""

            if mode not in table:
                table[mode] = {}
            if temps not in table[mode]:
                table[mode][temps] = {}
            if cle not in table[mode][temps]:
                table[mode][temps][cle] = ortho

        return table

    def close(self) -> None:
        """Libere les ressources."""
        self._data.clear()
        self._formes = frozenset()
        self._phone_index = None
        self._lemme_index = None
