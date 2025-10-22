# tests/test_bright_filter_minimal.py
import pandas as pd
import pyterrier as pt
import pytest

from suiteeval.suite.bright import BRIGHT
from suiteeval.suite.base import Suite


# ---- Lightweight stand-ins ----
class Identity(pt.Transformer):
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        return inp


class _DummyDataset:
    def __init__(self, qrels: pd.DataFrame):
        self._qrels = qrels
    def get_qrels(self) -> pd.DataFrame:
        return self._qrels


class _DummyContext:
    def __init__(self, qrels: pd.DataFrame):
        self.dataset = _DummyDataset(qrels)


@pytest.fixture
def qrels_flagged():
    # Two flagged pairs that must be removed by the wrapper filter.
    return pd.DataFrame(
        {
            "qid": ["q1", "q1", "q2", "q2", "q3"],
            "docno": ["d1", "dX", "d3", "dY", "dZ"],
            "relevance": [-100, 1, -100, 0, 0],
        }
    )


@pytest.fixture
def run_with_flagged_pairs():
    # Run contains both flagged and unflagged pairs
    return pd.DataFrame(
        {
            "qid":   ["q1", "q1", "q2", "q2", "q3"],
            "docno": ["d1", "dX", "d3", "dY", "dZ"],
            "score": [10.0, 9.0, 8.0, 7.0, 6.0],
            "rank":  [1, 2, 1, 2, 1],
        }
    )


def _assert_no_minus100_pairs(out: pd.DataFrame, qrels: pd.DataFrame):
    flagged = set(map(tuple, qrels.loc[qrels["relevance"] == -100, ["qid", "docno"]].itertuples(index=False, name=None)))
    assert not any((row.qid, row.docno) in flagged for row in out.itertuples(index=False))


def test_grouped_coercion_filters_minus100(monkeypatch, qrels_flagged, run_with_flagged_pairs):
    ctx = _DummyContext(qrels_flagged)

    def _fake_grouped(_self, _context, _pipeline_generators):
        return [Identity()], ["id"]
    monkeypatch.setattr(Suite, "coerce_pipelines_grouped", _fake_grouped, raising=True)

    pipelines, names = BRIGHT.coerce_pipelines_grouped(ctx, pipeline_generators=[Identity()])
    assert len(pipelines) == 1 and names == ["id"]

    out = pipelines[0](run_with_flagged_pairs)
    _assert_no_minus100_pairs(out, qrels_flagged)


def test_sequential_coercion_filters_minus100(monkeypatch, qrels_flagged, run_with_flagged_pairs):
    ctx = _DummyContext(qrels_flagged)

    def _fake_sequential(_self, _context, _pipeline_generators):
        yield Identity(), "id"
    monkeypatch.setattr(Suite, "coerce_pipelines_sequential", _fake_sequential, raising=True)

    seq = BRIGHT.coerce_pipelines_sequential(ctx, pipeline_generators=[Identity()])
    (pipeline, name), = list(seq)
    assert name == "id"

    out = pipeline(run_with_flagged_pairs)
    _assert_no_minus100_pairs(out, qrels_flagged)