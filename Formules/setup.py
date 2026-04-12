"""Build binaire Cython — compile les .py en .so/.pyd."""
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

# --- Detection automatique du package ---
src = Path("src")
packages = [p.name for p in src.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
if not packages:
    sys.exit("Aucun package trouve dans src/")
pkg_name = packages[0]

# --- Cythonize les .py (sauf __init__.py) ---
py_files = [
    str(p) for p in (src / pkg_name).rglob("*.py")
    if p.name != "__init__.py"
]

ext_modules = cythonize(
    py_files,
    language_level="3",
    compiler_directives={"embedsignature": True},
    nthreads=4,
)

# --- Exclure les .py compiles et les .c generes du wheel ---
compiled_stems = {Path(f.sources[0]).stem for f in ext_modules}


class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [(pkg, mod, path) for pkg, mod, path in modules
                if mod == "__init__" or mod not in compiled_stems]

    def find_data_files(self, package, src_dir):
        """Exclure les .c generes par Cython des data files."""
        files = super().find_data_files(package, src_dir)
        return [f for f in files if not f.endswith(".c")]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_py": build_py},
)
