from pyterrier_dr import HgfBiEncoder as HgfBiEncoder_base
from suiteeval._optional import pyterrier_dr_available
from suiteeval.index import TemporaryFlexIndex

def HgfBiEncoder(ranking_pipeline, checkpoint: str = "sentence-transformers/all-mpnet-base-v2"):
    if not pyterrier_dr_available():
        raise ImportError("pyterrier_dr is required for HgfBiEncoder pipeline.")

    hgf_be = HgfBiEncoder_base.from_pretrained(checkpoint)

    def yield_pipe(documents):
        with TemporaryFlexIndex(hgf_be, documents) as flex_index:
            yield flex_index.np_retriever() >> ranking_pipeline

    return yield_pipe


__all__ = ["HgfBiEncoder"]
