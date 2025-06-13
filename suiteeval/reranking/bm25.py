from suiteeval.index import TemporaryPISAIndex
from suiteeval._optional import pyterrier_pisa_available


def BM25(ranking_pipeline):
    if not pyterrier_pisa_available():
        raise ImportError("pyterrier_pisa is required for BM25 pipeline.")

    def yeild_pipe(documents):
        with TemporaryPISAIndex(documents) as pisa_index:
            bm25 = pisa_index.bm25()
            yield bm25 >> ranking_pipeline
    return yeild_pipe


__all__ = ["BM25"]
