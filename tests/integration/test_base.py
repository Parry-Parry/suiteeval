import os
import pytest
import pandas as pd
import suiteeval
from suiteeval import Suite
from suiteeval._optional import (
    pyterrier_available,
    pyterrier_dr_available,
    pyterrier_pisa_available,
    pyterrier_splade_available,
)

# Dummy index class for testing Temporary
class DummyIndex:
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        # create a marker file to verify directory creation
        open(os.path.join(path, "marker.txt"), "w").close()

def test_optional_dependency_flags_return_boolean():
    assert isinstance(pyterrier_available(), bool)
    assert isinstance(pyterrier_dr_available(), bool)
    assert isinstance(pyterrier_pisa_available(), bool)
    assert isinstance(pyterrier_splade_available(), bool)


def test_reranking_exports():
    from suiteeval import reranking

    expected = {"BM25", "SPLADE", "HgfBiEncoder"}
    assert set(reranking.__all__) == expected


def test_suite_singleton_behavior():
    first = Suite.register("foo_suite", ["dataset/a"])
    second = Suite.register("foo_suite", ["dataset/a"])
    assert first is second


def test_beir_empty_evaluation_returns_dataframe(monkeypatch):
    from suiteeval.suite import BEIR

    # Monkey-patch Experiment to avoid external calls
    import pyterrier as pt
    monkeypatch.setattr(pt, 'Experiment', lambda *args, **kwargs: pd.DataFrame())

    suite = BEIR()
    df = suite(pipelines=[], ranking_generators=[])
    assert isinstance(df, pd.DataFrame)
    assert df.empty
