"""Top-level package for SuiteEval."""
__version__ = "0.1.0"

from suiteeval.suite import Suite, BEIR, Lotte, MSMARCODocument, MSMARCOPassage, NanoBEIR
from suiteeval.index import Temporary as Temporary
from suiteeval import reranking as reranking

__all__ = [
    "Suite",
    "BEIR",
    "Lotte",
    "MSMARCODocument",
    "MSMARCOPassage",
    "NanoBEIR",
    "Temporary",
    "reranking",
]
