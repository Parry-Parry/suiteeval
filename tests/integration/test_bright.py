import pandas as pd
import pyterrier as pt
import pytest

from suiteeval.suite.bright import BRIGHT, DocumentFilter


# ---- Lightweight stand-ins ----
class Identity(pt.Transformer):
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        return inp


class _DummyDataset:
    def __init__(self, qrels: pd.DataFrame):
        self._qrels = qrels
        self._irds_id = "bright/dummy"

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
            "qid": ["q1", "q1", "q2", "q2", "q3"],
            "docno": ["d1", "dX", "d3", "dY", "dZ"],
            "score": [10.0, 9.0, 8.0, 7.0, 6.0],
            "rank": [1, 2, 1, 2, 1],
        }
    )


def _assert_no_minus100_pairs(out: pd.DataFrame, qrels: pd.DataFrame):
    flagged = set(
        map(
            tuple,
            qrels.loc[qrels["relevance"] == -100, ["qid", "docno"]].itertuples(
                index=False, name=None
            ),
        )
    )
    assert not any(
        (row.qid, row.docno) in flagged for row in out.itertuples(index=False)
    )


def _generator(*named_pipelines):
    def gen(_context):
        yield from named_pipelines

    return gen


def test_wrap_pipeline_filters_minus100(qrels_flagged, run_with_flagged_pairs):
    ctx = _DummyContext(qrels_flagged)

    pipeline = BRIGHT.wrap_pipeline(Identity(), ctx)

    out = pipeline(run_with_flagged_pairs)
    _assert_no_minus100_pairs(out, qrels_flagged)
    assert len(out) == 3


def test_grouped_coercion_filters_minus100(qrels_flagged, run_with_flagged_pairs):
    ctx = _DummyContext(qrels_flagged)

    pipelines, names = BRIGHT.coerce_pipelines_grouped(
        ctx, _generator((Identity(), "id"))
    )
    assert len(pipelines) == 1 and names == ["id"]

    out = pipelines[0](run_with_flagged_pairs)
    _assert_no_minus100_pairs(out, qrels_flagged)


def test_sequential_coercion_filters_minus100(qrels_flagged, run_with_flagged_pairs):
    ctx = _DummyContext(qrels_flagged)

    seq = BRIGHT.coerce_pipelines_sequential(ctx, _generator((Identity(), "id")))
    ((pipeline, name),) = list(seq)
    assert name == "id"

    out = pipeline(run_with_flagged_pairs)
    _assert_no_minus100_pairs(out, qrels_flagged)


def test_batches_apply_filter_in_both_modes(qrels_flagged, run_with_flagged_pairs):
    """Grouped and sequential batching both route through wrap_pipeline."""
    ctx = _DummyContext(qrels_flagged)
    gen = _generator((Identity(), "a"), (Identity(), "b"))

    grouped = list(BRIGHT.iter_pipeline_batches(ctx, gen, grouped=True))
    sequential = list(BRIGHT.iter_pipeline_batches(ctx, gen, grouped=False))

    assert len(grouped) == 1 and len(grouped[0]) == 2
    assert len(sequential) == 2 and all(len(b) == 1 for b in sequential)

    for batch in grouped + sequential:
        for pipeline, _ in batch:
            _assert_no_minus100_pairs(pipeline(run_with_flagged_pairs), qrels_flagged)


def test_document_filter_passes_through_empty_frame(qrels_flagged):
    empty = pd.DataFrame(columns=["qid", "docno", "score", "rank"])
    out = DocumentFilter(qrels_flagged).transform(empty)
    assert len(out) == 0


def test_document_filter_without_flagged_pairs_is_a_no_op(run_with_flagged_pairs):
    qrels = pd.DataFrame({"qid": ["q1"], "docno": ["d1"], "relevance": [1]})
    out = DocumentFilter(qrels).transform(run_with_flagged_pairs)
    assert len(out) == len(run_with_flagged_pairs)
