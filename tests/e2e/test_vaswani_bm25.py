"""End-to-end gate: the Vaswani suite must run through BM25.

This is the only test that exercises the whole stack together — dataset
download, PISA indexing, retrieval, evaluation and aggregation — through the
public :class:`~suiteeval.suite.base.Suite` entry point. Everything else in
``tests/`` either mocks PyTerrier or only inspects dataset declarations, so a
pipeline that is wired up wrongly can pass the rest of the suite.

Vaswani is used because it is small (~11k documents, 93 topics) and ships with
qrels, so a full run finishes in CI minutes without a GPU.

Marked ``e2e`` so the unit workflow can exclude it; it runs in its own
``integration`` workflow, which is the check required before merge.
"""

from __future__ import annotations

import pandas as pd
import pytest
from ir_measures import AP, nDCG

from suiteeval._optional import pyterrier_pisa_available
from suiteeval.context import DatasetContext
from suiteeval.suite.base import Suite

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not pyterrier_pisa_available(), reason="requires pyterrier_pisa"
    ),
]

MEASURES = [nDCG @ 10, AP]

#: Floors well below published BM25 scores on Vaswani. They catch a broken run
#: (empty rankings, mismatched identifiers, qrels joined on the wrong column)
#: without pinning the exact numbers a retriever version bump may shift.
MIN_SCORES = {"nDCG@10": 0.3, "AP": 0.2}


def bm25(context: DatasetContext):
    """Index the corpus with PISA and return a named BM25 retriever."""
    from pyterrier_pisa import PisaIndex

    index = PisaIndex(f"{context.path}/index.pisa")
    index.index(context.get_corpus_iter())
    return index.bm25(), "BM25"


@pytest.fixture(scope="module")
def suite() -> Suite:
    """A single-dataset suite, registered under its own name to avoid clashes."""
    return Suite.register(
        "e2e/vaswani",
        datasets=["vaswani"],
        metadata={
            "official_measures": MEASURES,
            "description": "Vaswani, used as the end-to-end smoke corpus.",
        },
    )


@pytest.fixture(scope="module")
def results(suite: Suite) -> pd.DataFrame:
    """Run BM25 over the suite once and share the table across assertions."""
    return suite(bm25)


def test_run_reports_the_dataset_and_an_overall_row(results: pd.DataFrame) -> None:
    assert not results.empty
    assert results["dataset"].tolist() == ["vaswani", "Overall"]
    assert results["name"].unique().tolist() == ["BM25"]


def test_run_reports_every_official_measure(results: pd.DataFrame) -> None:
    for measure in MEASURES:
        assert str(measure) in results.columns


def test_bm25_scores_are_plausible(results: pd.DataFrame) -> None:
    scores = results[results["dataset"] == "vaswani"].iloc[0]
    for measure, floor in MIN_SCORES.items():
        assert scores[measure] >= floor, f"{measure}={scores[measure]:.4f} < {floor}"


def test_overall_row_matches_the_single_dataset(results: pd.DataFrame) -> None:
    """With one dataset, the geometric mean must reproduce its scores."""
    dataset_row = results[results["dataset"] == "vaswani"].iloc[0]
    overall_row = results[results["dataset"] == "Overall"].iloc[0]
    for measure in MIN_SCORES:
        assert overall_row[measure] == pytest.approx(dataset_row[measure], abs=1e-6)
