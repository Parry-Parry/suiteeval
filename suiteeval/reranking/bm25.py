from suiteeval.index import Temporary
from suiteeval._optional import pyterrier_pisa_available

if pyterrier_pisa_available():
    from pyterrier_pisa import PisaIndex


def BM25(ranking_pipeline):
    if not pyterrier_pisa_available():
        raise ImportError("pyterrier_pisa is required for BM25 pipeline.")

    def yeild_pipe(context):
        pisa_index = PisaIndex(context.path + "/index.pisa", stemmer="none")
        pisa_index.index(context.get_corpus_iter())
        bm25 = pisa_index.bm25()
        yield bm25 >> context.text_loader() >> ranking_pipeline

    return yeild_pipe


__all__ = ["BM25"]
