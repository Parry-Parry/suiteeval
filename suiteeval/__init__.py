"""Top-level package for SuiteEval."""

__version__ = "0.1.8"

from suiteeval.suite import (
    Suite,
    BEIR,
    BRIGHT,
    Lotte,
    MSMARCODocument,
    MSMARCOPassage,
    NanoBEIR,
)
from suiteeval.suite.base import RunConfig
from suiteeval.context import DatasetContext

__all__ = [
    "Suite",
    "RunConfig",
    "BEIR",
    "BRIGHT",
    "Lotte",
    "MSMARCODocument",
    "MSMARCOPassage",
    "NanoBEIR",
    "DatasetContext",
]
