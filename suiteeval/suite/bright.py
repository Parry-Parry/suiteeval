from typing import Any, Sequence, Optional, Union

from ir_measures import nDCG
from suiteeval.suite.base import Suite
import pandas as pd

from suiteeval.utility import geometric_mean

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

    # TODO: override __call__ to disregard documents for queries with -100 in the qrels


BRIGHT = _BRIGHT()

__all__ = ["BRIGHT"]
