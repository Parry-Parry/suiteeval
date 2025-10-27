from suiteeval._optional import pyterrier_pisa_available

if pyterrier_pisa_available():
    from pyterrier_pisa import PisaIndex


def BM25(ranking_pipeline):
    """
    Constructs a BM25 retrieval pipeline generator using PyTerrier-PISA.

    Args:
        ranking_pipeline: A PyTerrier pipeline to apply after retrieval.
    """
    if not pyterrier_pisa_available():
        raise ImportError("pyterrier_pisa is required for BM25 pipeline.")

    def yeild_pipe(context):
        pisa_index = PisaIndex(context.path + "/index.pisa")
        pisa_index.index(context.get_corpus_iter())
        bm25 = pisa_index.bm25()
        yield bm25 >> context.text_loader() >> ranking_pipeline

    return yeild_pipe


__all__ = ["BM25"]
