import pytest
from suiteeval.reranking.bm25 import BM25
from suiteeval.reranking.splade import SPLADE
from suiteeval.reranking.biencoder import HgfBiEncoder
from suiteeval._optional import (
    pyterrier_pisa_available,
    pyterrier_dr_available,
)


class DummyContext:
    def get_corpus_iter(self):
        return []
    def text_loader(self, *args, **kwargs):
        return lambda x: x

@pytest.mark.skipif(not pyterrier_pisa_available(), reason="PISA not available")
def test_bm25_factory_returns_callable():
    f = BM25(lambda x: x)
    assert callable(f)
    gen = f(DummyContext())
    assert hasattr(gen, "__iter__")

@pytest.mark.skipif(not pyterrier_pisa_available(), reason="PISA not available")
def test_splade_factory_returns_callable():
    f = SPLADE(lambda x: x)
    assert callable(f)
    gen = f(DummyContext())
    assert hasattr(gen, "__iter__") or callable(gen)

@pytest.mark.skipif(not pyterrier_dr_available(), reason="pyterrier-dr not available")
def test_hgfbi_encoder_factory_returns_callable():
    f = HgfBiEncoder(lambda x: x)
    assert callable(f)
    gen = f(DummyContext())
    assert hasattr(gen, "__iter__")