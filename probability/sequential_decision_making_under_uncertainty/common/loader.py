"""
Import a solver from another problem folder.

Problem folders are named by tag (e.g. `01_Id-Ud-Bk-Ec-Oe-Kk`), which is not a valid
Python package name. Later problems reuse earlier methods, so load them by path:

    enum = load("problems/01_Id-Ud-Bk-Ec-Oe-Kk/enumeration/solve.py")
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relpath: str):
    path = ROOT / relpath
    name = "sdmu_" + "_".join(path.relative_to(ROOT).with_suffix("").parts).replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
