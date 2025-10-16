import math
import pytest
import pandas as pd
from pyterrier import Transformer
from ir_measures import nDCG


from suiteeval.suite.beir import BEIR


# ---------- Helpers ----------

class TinyDataset:
    """Minimal stub with an _irds_id attribute to drive Quora / non-Quora logic."""
    def __init__(self, irds_id: str):
        self._irds_id = irds_id

class TinyContext:
    """Minimal stub matching the attributes used by BEIR.coerce_*."""
    def __init__(self, irds_id: str):
        self.dataset = TinyDataset(irds_id)

class EmitsRanking(Transformer):
    """A tiny transformer that fabricates a ranking from incoming topics."""
    def transform(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        # Expect at least qid in the input dataframe
        qids = topics_df["qid"].tolist()

        # Produce two rows per qid: one with docno == qid (filtered by Quora rule),
        # and one with a different docno (always retained).
        rows = []
        for i, q in enumerate(qids):
            rows.append({"qid": q, "docno": q,          "rank": 0, "score": 2.0})
            rows.append({"qid": q, "docno": f"d{i}",    "rank": 1, "score": 1.0})
        return pd.DataFrame(rows)


def run_pipeline(p, qids=("q1", "q2")) -> pd.DataFrame:
    """Execute a PT pipeline on a minimal topics dataframe."""
    topics = pd.DataFrame({"qid": list(qids)})
    return p.transform(topics)


# ---------- Tests for coerce_pipelines_sequential ----------

def test_sequential_appends_filter_for_quora():
    beir = BEIR
    ctx = TinyContext("beir/quora/test")

    # A generator returning a single Transformer
    def gen(_ctx):
        yield EmitsRanking(), "dummy"

    out = list(beir.coerce_pipelines_sequential(ctx, gen))
    assert len(out) == 1
    p, nm = out[0]
    assert nm == "dummy"

    # Run the resulting pipeline and verify rows with docno == qid are filtered
    df = run_pipeline(p)
    # The filter should have removed the self-matching rows
    assert not ((df["qid"] == df["docno"]).any())
    # Only the retained rows remain: one per qid
    assert len(df) == 2


def test_sequential_does_not_append_filter_for_non_quora():
    beir = BEIR
    ctx = TinyContext("beir/arguana")

    def gen(_ctx):
        yield EmitsRanking(), "dummy"

    out = list(beir.coerce_pipelines_sequential(ctx, gen))
    assert len(out) == 1
    p, nm = out[0]
    assert nm == "dummy"

    df = run_pipeline(p)
    # Non-Quora: the self-matching rows should remain
    assert ((df["qid"] == df["docno"]).any())
    assert len(df) == 4  # two rows per qid


# ---------- Tests for coerce_pipelines_grouped ----------

def test_grouped_appends_filter_for_quora():
    beir = BEIR
    ctx = TinyContext("beir/quora/test")

    def gen(_ctx):
        # Return a tuple (pipeline(s), name) as allowed by the base class
        return ([EmitsRanking(), EmitsRanking()], ["a", "b"])

    pipelines, names = beir.coerce_pipelines_grouped(ctx, gen)
    assert names == ["a", "b"]
    assert len(pipelines) == 2

    for p in pipelines:
        df = run_pipeline(p)
        # Quora: self-matching rows filtered
        assert not ((df["qid"] == df["docno"]).any())
        assert len(df) == 2


def test_grouped_no_filter_for_non_quora():
    beir = BEIR
    ctx = TinyContext("beir/fever/test")

    def gen(_ctx):
        return ([EmitsRanking(), EmitsRanking()], ["a", "b"])

    pipelines, names = beir.coerce_pipelines_grouped(ctx, gen)
    assert names == ["a", "b"]
    assert len(pipelines) == 2

    for p in pipelines:
        df = run_pipeline(p)
        # Non-Quora: no filtering
        assert ((df["qid"] == df["docno"]).any())
        assert len(df) == 4
