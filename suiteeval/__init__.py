"""Top-level package for SuiteEval."""
__version__ = "0.1.0"

from suiteeval.suite import Suite, BEIR, Lotte, MSMARCODocument, MSMARCOPassage, NanoBEIR
from suiteeval import index as index

__all__ = [
    "Suite",
    "BEIR",
    "Lotte",
    "MSMARCODocument",
    "MSMARCOPassage",
    "NanoBEIR",
    "index",
]
