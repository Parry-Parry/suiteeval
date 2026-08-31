import pandas as pd
import pyterrier as pt
from ir_measures import nDCG
from pyterrier import Transformer

from suiteeval.context import DatasetContext
from suiteeval.suite.base import Suite

datasets = [
    "bright/aops",
    "bright/biology",
    "bright/earth-science",
    "bright/economics",
    "bright/leetcode",
    "bright/pony",
    "bright/psychology",
    "bright/robotics",
    "bright/stackoverflow",
    "bright/sustainable-living",
    "bright/theoremqa-questions",
    "bright/theoremqa-theorems",
]

measures = [nDCG @ 10]

FILTER_VALUE = -100


class DocumentFilter(pt.Transformer):
    """Drop ``(qid, docno)`` pairs flagged as excluded in the qrels."""

    def __init__(self, qrels: pd.DataFrame, filter_value: int = FILTER_VALUE):
        super().__init__()
        self._flagged = qrels.loc[
            qrels["relevance"] == filter_value, ["qid", "docno"]
        ].drop_duplicates()

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.result_frame(inp)
        if len(inp) == 0 or len(self._flagged) == 0:
            return inp
        out = inp.merge(self._flagged.assign(_ban=1), on=["qid", "docno"], how="left")
        return out[out["_ban"].isna()].drop(columns=["_ban"])


class _BRIGHT(Suite):
    """
    BRIGHT suite for evaluating retrieval that requires reasoning.
    """

    _datasets = datasets
    _measures = measures
    _query_field = "text"
    _metadata = {
        "official_measures": measures,
        "description": " BRIGHT is a suite datasets for evaluating retrieval that requires reasoning.",
    }

    def wrap_pipeline(
        self, pipeline: Transformer, context: DatasetContext
    ) -> Transformer:
        """Append a filter removing documents flagged as excluded in the qrels."""
        return pipeline >> DocumentFilter(context.dataset.get_qrels())


BRIGHT = _BRIGHT()

__all__ = ["BRIGHT", "DocumentFilter"]
