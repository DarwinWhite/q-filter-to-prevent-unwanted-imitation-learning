"""
Utility package for the project.

This module purposefully avoids importing heavy or MPI/TF-dependent submodules
at import time (to prevent SLURM/mpi4py/TF initialization errors).  It
provides lazy-loading helpers for the heavier modules and exposes lightweight
utilities in a safe way.
"""

import importlib
import types
import warnings
from typing import Optional

__all__ = [
    "util",
    "load_normalizer",
    "load_generate_demos",
    "safe_import",
]

def safe_import(module_path: str) -> Optional[types.ModuleType]:
    """
    Attempt to import a module and return it. If the import fails, return None
    and issue a warning. This avoids hard crashes at package import time.
    """
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        warnings.warn(f"Optional import failed for '{module_path}': {e}")
        return None


# Lightweight (commonly-used) utilities: import on demand via attribute access.
# This avoids importing TF/mpi4py when the package is imported.
class _LazyModuleProxy:
    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = safe_import(self._module_name)
            if self._module is None:
                raise ImportError(f"Could not import module '{self._module_name}'. "
                                  "Ensure dependencies are installed and the environment is compatible.")
        return self._module

    def __getattr__(self, name):
        mod = self._load()
        return getattr(mod, name)

    def __dir__(self):
        mod = safe_import(self._module_name)
        return dir(mod) if mod is not None else [self._module_name]


# Expose util via a lazy proxy (safe, lightweight)
util = _LazyModuleProxy("src.utils.util")


def load_normalizer():
    """
    Lazy-load and return the normalizer module.

    Usage:
        normalizer = load_normalizer()
        Normalizer = normalizer.Normalizer
    """
    mod = safe_import("src.utils.normalizer")
    if mod is None:
        raise ImportError("src.utils.normalizer could not be imported. "
                          "This module requires TensorFlow/mpi4py or other optional deps.")
    return mod


def load_generate_demos():
    """
    Lazy-load and return the generate_demos module.

    Usage:
        gen = load_generate_demos()
        gen.collect_demonstrations_with_params(...)
    """
    mod = safe_import("src.utils.generate_demos")
    if mod is None:
        raise ImportError("src.utils.generate_demos could not be imported.")
    return mod
