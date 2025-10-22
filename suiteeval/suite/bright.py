import builtins
from collections.abc import Sequence as runtime_Sequence
from typing import Optional, Tuple

import pyterrier as pt
import pandas as pd
from ir_measures import nDCG

from suiteeval.context import DatasetContext
from suiteeval.suite.base import Suite

datasets = [
    'bright/aops',
    'bright/biology',
    'bright/earth-science',
    'bright/economics',
    'bright/leetcode',
    'bright/pony',
    'bright/psychology',
    'bright/robotics',
    'bright/stackoverflow',
    'bright/sustainable-living',
    'bright/theoremqa-questions',
    'bright/theoremqa-theorems',
]

measures = [nDCG@10]


class DocumentFilter(pt.Transformer):
    def __init__(self, qrels: pd.DataFrame, filter_value: int = -100):
        super().__init__()
        self.pairs = qrels[qrels['relevance'] == filter_value][['qid', 'docno']].values.tolist()

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        mask = inp.apply(lambda row: (row['qid'], row['docno']) not in self.pairs, axis=1)
        return inp[mask]


class _BRIGHT(Suite):
    """
    BRIGHT suite for evaluating retrieval that requires reasoning.
    """

    _datasets = datasets
    _measures = measures
    _query_field = 'text'
    _metadata = {
        "official_measures": measures,
        "description": " BRIGHT is a suite datasets for evaluating retrieval that requires reasoning.",
    }

    def coerce_pipelines_sequential(
        self,
        context: DatasetContext,
        pipeline_generators: "runtime_Sequence|builtins.callable",
    ):
        """
        Wrap each streamed pipeline with a dataframe filter only for Quora,
        preserving (pipeline, name) pairs and not materialising the sequence.
        """
        for p, nm in super().coerce_pipelines_sequential(context, pipeline_generators):
            p = p >> DocumentFilter(context.dataset.get_qrels())
            yield p, nm

    def coerce_pipelines_grouped(
        self,
        context: DatasetContext,
        pipeline_generators: "runtime_Sequence|builtins.callable",
    ) -> Tuple[list[pt.Transformer], Optional[list[str]]]:
        """
        Materialise all pipelines (and names) via the superclass, then
        append a dataframe filter for qrels.
        """
        pipelines, names = super().coerce_pipelines_grouped(context, pipeline_generators)
        pipelines = [
            p >> DocumentFilter(context.dataset.get_qrels())
            for p in pipelines
        ]

        return pipelines, names


BRIGHT = _BRIGHT()

__all__ = ["BRIGHT"]
