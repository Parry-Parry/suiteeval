"""Top-level package for SuiteEval."""

__version__ = "0.1.3"

from suiteeval.suite import (
    Suite,
    BEIR,
    Lotte,
    MSMARCODocument,
    MSMARCOPassage,
    NanoBEIR,
)
from suiteeval.context import DatasetContext

__all__ = [
    "Suite",
    "BEIR",
    "Lotte",
    "MSMARCODocument",
    "MSMARCOPassage",
    "NanoBEIR",
    "DatasetContext",
]
