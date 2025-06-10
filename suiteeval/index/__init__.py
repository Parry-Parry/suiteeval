from suiteeval._optional import (
    pyterrier_dr_available,
    pyterrier_pisa_available,
    pyterrier_available,
)
import importlib
from typing import Any, Dict

_available_indices: Dict[str, Any] = {}

if pyterrier_dr_available():
    _mod = importlib.import_module("suiteeval.index.flex")
    TemporaryFlexIndex = _mod.TemporaryFlexIndex
    _available_indices["TemporaryFlexIndex"] = TemporaryFlexIndex

if pyterrier_pisa_available():
    _mod = importlib.import_module("suiteeval.index.pisa")
    TemporaryPISAIndex = _mod.TemporaryPISAIndex
    _available_indices["TemporaryPISAIndex"] = TemporaryPISAIndex

if pyterrier_available():
    _mod = importlib.import_module("suiteeval.index.terrier")
    TemporaryTerrierIndex = _mod.TemporaryTerrierIndex
    _available_indices["TemporaryTerrierIndex"] = TemporaryTerrierIndex

__all__ = [
    name
    for name in ("TemporaryFlexIndex", "TemporaryPISAIndex", "TemporaryTerrierIndex")
    if name in globals()
]

# Expose registry for introspection or factory use
available_indices = _available_indices.copy()
