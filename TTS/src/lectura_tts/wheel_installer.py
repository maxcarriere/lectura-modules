"""Installeur de wheels PyPI pour le mode frozen (PyInstaller).

Télécharge les wheels depuis PyPI, résout les dépendances transitives,
et les extrait dans un dossier plugins local. Aucune dépendance GUI.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import struct
import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# ── Dossier plugins ──────────────────────────────────────────────────────

def get_plugins_dir() -> Path:
    """Retourne le dossier plugins propre à la plateforme."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
        return base / "Lectura" / "plugins"
    elif system == "Darwin":
        return Path.home() / "Library/Application Support/Lectura/plugins"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "")
        base = Path(xdg) if xdg else Path.home() / ".local/share"
        return base / "lectura" / "plugins"


def ensure_plugins_on_path() -> None:
    """Ajoute le dossier plugins à sys.path si pas déjà présent."""
    plugins = str(get_plugins_dir())
    if plugins not in sys.path:
        sys.path.insert(0, plugins)


# ── Manifest (installed.json) ────────────────────────────────────────────

_MANIFEST_NAME = "installed.json"


def _manifest_path() -> Path:
    return get_plugins_dir() / _MANIFEST_NAME


def load_manifest() -> dict[str, Any]:
    """Charge le manifest {engine_key: {packages: {name: version}, directories: [...]}}."""
    p = _manifest_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(data: dict[str, Any]) -> None:
    """Écrit le manifest."""
    p = _manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")


# ── Helpers PyPI ─────────────────────────────────────────────────────────

def _pypi_json(package: str) -> dict[str, Any]:
    """Fetch les métadonnées d'un package depuis l'API JSON PyPI."""
    url = f"https://pypi.org/pypi/{package}/json"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _normalize_name(name: str) -> str:
    """Normalise un nom de package PEP 503 (lowercase, tirets → underscores)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _is_importable(package_name: str) -> bool:
    """Vérifie si un package est déjà importable."""
    import_name = package_name.replace("-", "_").lower()
    # Cas spéciaux connus
    import_map = {
        "pillow": "PIL",
        "pyyaml": "yaml",
        "scikit-learn": "sklearn",
        "beautifulsoup4": "bs4",
        "piper-tts": "piper",
        "gtts": "gtts",
        "kokoro-onnx": "kokoro_onnx",
        "edge-tts": "edge_tts",
        "piper-phonemize": "piper_phonemize",
    }
    import_name = import_map.get(package_name.lower(), import_name)
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


# ── Évaluation des markers PEP 508 ──────────────────────────────────────

def _evaluate_marker(marker_str: str) -> bool:
    """Évalue un environment marker PEP 508."""
    try:
        from packaging.markers import Marker
        m = Marker(marker_str)
        return m.evaluate()
    except Exception:
        # Si on ne peut pas évaluer, on suppose que c'est requis
        return True


_REQ_RE = re.compile(
    r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)"  # nom du package
    r"(\[.*?\])?"                                       # extras optionnels
    r"([^;]*)"                                          # version specifier
    r"(;\s*(.+))?"                                      # marker optionnel
    r"$"
)


def _parse_requires_dist(req_str: str) -> tuple[str, str] | None:
    """Parse une entrée requires_dist. Retourne (name, marker) ou None si extra-only."""
    req_str = req_str.strip()
    m = _REQ_RE.match(req_str)
    if not m:
        return None
    name = m.group(1)
    marker = (m.group(6) or "").strip()
    return name, marker


# ── Résolution de dépendances ────────────────────────────────────────────

def resolve_dependencies(packages: list[str]) -> list[str]:
    """Résout les dépendances récursivement via PyPI JSON API.

    Retourne la liste des packages à installer, en ordre topologique
    (dépendances avant dépendants). Saute les packages déjà importables.
    """
    order: list[str] = []
    seen: set[str] = set()

    def _resolve(pkg_name: str) -> None:
        normalized = _normalize_name(pkg_name)
        if normalized in seen:
            return
        seen.add(normalized)

        if _is_importable(pkg_name):
            log.debug("Skipping %s (already importable)", pkg_name)
            return

        # Fetch metadata
        try:
            meta = _pypi_json(pkg_name)
        except Exception as exc:
            log.warning("Cannot fetch PyPI metadata for %s: %s", pkg_name, exc)
            # On l'ajoute quand même — l'erreur sera visible au téléchargement
            order.append(pkg_name)
            return

        # Parse requires_dist
        requires = meta.get("info", {}).get("requires_dist") or []
        for req_str in requires:
            parsed = _parse_requires_dist(req_str)
            if parsed is None:
                continue
            dep_name, marker = parsed

            # Ignorer les extras
            if "extra ==" in marker or "extra ==" in req_str:
                continue

            # Évaluer les markers d'environnement
            if marker and not _evaluate_marker(marker):
                continue

            _resolve(dep_name)

        order.append(pkg_name)

    for pkg in packages:
        _resolve(pkg)

    return order


# ── Sélection du wheel compatible ────────────────────────────────────────

def _get_sys_tags() -> list[tuple[str, str, str]]:
    """Retourne les tags système compatibles (interpreter, abi, platform)."""
    try:
        from packaging.tags import sys_tags
        return [(str(t.interpreter), str(t.abi), str(t.platform)) for t in sys_tags()]
    except ImportError:
        # Fallback minimal
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        bits = struct.calcsize("P") * 8
        if sys.platform == "linux":
            machine = platform.machine()
            plat = f"manylinux_2_17_{machine}"
        elif sys.platform == "win32":
            plat = f"win_amd64" if bits == 64 else "win32"
        elif sys.platform == "darwin":
            plat = "macosx_10_9_x86_64"
        else:
            plat = "any"
        return [
            (py, f"{py}", plat),
            (py, "abi3", plat),
            (py, "none", plat),
            ("py3", "none", "any"),
        ]


def _parse_wheel_filename(filename: str) -> list[tuple[str, str, str]] | None:
    """Parse un nom de wheel PEP 427, retourne les tags ou None."""
    if not filename.endswith(".whl"):
        return None
    parts = filename[:-4].split("-")
    if len(parts) < 5:
        return None
    # Les 3 derniers segments sont pytag-abitag-plattag (chacun peut être composé avec .)
    py_tags = parts[-3].split(".")
    abi_tags = parts[-2].split(".")
    plat_tags = parts[-1].split(".")
    result = []
    for py in py_tags:
        for abi in abi_tags:
            for plat in plat_tags:
                result.append((py, abi, plat))
    return result


def select_compatible_wheel(files: list[dict[str, Any]]) -> str | None:
    """Sélectionne l'URL du wheel le plus compatible parmi les fichiers d'une release.

    Retourne l'URL de téléchargement ou None.
    """
    sys_tags = _get_sys_tags()
    sys_tags_set = set(sys_tags)

    # Priorité : on veut le premier wheel qui matche par ordre de préférence des tags
    best_url: str | None = None
    best_priority = len(sys_tags) + 1  # plus petit = meilleur

    for f in files:
        fn = f.get("filename", "")
        url = f.get("url", "")
        if not fn.endswith(".whl"):
            continue
        wheel_tags = _parse_wheel_filename(fn)
        if wheel_tags is None:
            continue
        for wt in wheel_tags:
            if wt in sys_tags_set:
                try:
                    prio = sys_tags.index(wt)
                except ValueError:
                    continue
                if prio < best_priority:
                    best_priority = prio
                    best_url = url
                    break  # Ce wheel a matché, pas besoin de tester les autres tags

    return best_url


# ── Installation ─────────────────────────────────────────────────────────

def install_packages(
    top_packages: list[str],
    engine_key: str,
    progress_cb: Callable[[int], None] | None = None,
) -> None:
    """Résout les dépendances, télécharge les wheels et les extrait dans plugins/.

    Raises:
        RuntimeError: si un wheel compatible n'est pas trouvé ou le téléchargement échoue.
    """
    plugins = get_plugins_dir()
    plugins.mkdir(parents=True, exist_ok=True)

    # 1. Résolution
    all_packages = resolve_dependencies(top_packages)
    if not all_packages:
        log.info("All packages already installed for %s", engine_key)
        return

    total = len(all_packages)
    manifest = load_manifest()
    engine_entry = manifest.setdefault(engine_key, {"packages": {}, "directories": []})
    all_directories: list[str] = list(engine_entry.get("directories", []))

    for i, pkg_name in enumerate(all_packages):
        if progress_cb:
            progress_cb(int(i / total * 100))

        log.info("Installing %s (%d/%d)", pkg_name, i + 1, total)

        # Fetch release info
        try:
            meta = _pypi_json(pkg_name)
        except Exception as exc:
            raise RuntimeError(f"Impossible de récupérer les infos PyPI pour {pkg_name}: {exc}")

        version = meta["info"]["version"]
        release_files = meta.get("urls", [])
        if not release_files:
            # Essayer dans releases
            release_files = meta.get("releases", {}).get(version, [])

        # Sélectionner un wheel compatible
        url = select_compatible_wheel(release_files)
        if url is None:
            raise RuntimeError(
                f"Aucun wheel compatible trouvé pour {pkg_name} {version} "
                f"(Python {sys.version_info.major}.{sys.version_info.minor}, "
                f"{platform.system()} {platform.machine()})"
            )

        # Télécharger le wheel
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                wheel_data = resp.read()
        except Exception as exc:
            raise RuntimeError(f"Échec téléchargement de {pkg_name}: {exc}")

        # Extraire dans plugins/
        extracted_dirs: set[str] = set()
        with zipfile.ZipFile(BytesIO(wheel_data)) as zf:
            for member in zf.namelist():
                # Extraire tous les fichiers sauf les .dist-info RECORD
                zf.extract(member, plugins)
                top_dir = member.split("/")[0]
                extracted_dirs.add(top_dir)

        engine_entry["packages"][pkg_name] = version
        for d in extracted_dirs:
            if d not in all_directories:
                all_directories.append(d)

    engine_entry["directories"] = all_directories
    manifest[engine_key] = engine_entry
    save_manifest(manifest)

    if progress_cb:
        progress_cb(100)

    log.info("Installation terminée pour %s", engine_key)


# ── Désinstallation ──────────────────────────────────────────────────────

def uninstall_engine(engine_key: str) -> None:
    """Désinstalle un moteur en supprimant ses répertoires exclusifs.

    Les répertoires partagés avec d'autres moteurs ne sont pas supprimés.
    """
    import shutil

    plugins = get_plugins_dir()
    manifest = load_manifest()
    entry = manifest.get(engine_key)
    if not entry:
        return

    # Collecter les répertoires utilisés par d'autres moteurs
    shared: set[str] = set()
    for other_key, other_entry in manifest.items():
        if other_key == engine_key:
            continue
        for d in other_entry.get("directories", []):
            shared.add(d)

    # Supprimer les répertoires exclusifs
    for d in entry.get("directories", []):
        if d in shared:
            log.debug("Keeping shared directory: %s", d)
            continue
        target = plugins / d
        if target.is_dir():
            shutil.rmtree(target)
            log.info("Removed directory: %s", target)
        elif target.is_file():
            target.unlink()
            log.info("Removed file: %s", target)

    del manifest[engine_key]
    save_manifest(manifest)
    log.info("Désinstallation terminée pour %s", engine_key)


# ── Requête ──────────────────────────────────────────────────────────────

def is_engine_installed(engine_key: str) -> bool:
    """Vérifie si un moteur est installé via le manifest."""
    manifest = load_manifest()
    return engine_key in manifest
