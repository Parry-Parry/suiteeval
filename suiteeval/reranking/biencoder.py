from pyterrier_dr import HgfBiEncoder as HgfBiEncoder_base, FlexIndex
from suiteeval._optional import pyterrier_dr_available
from suiteeval.index import Temporary


def HgfBiEncoder(
    ranking_pipeline, checkpoint: str = "sentence-transformers/all-mpnet-base-v2"
):
    if not pyterrier_dr_available():
        raise ImportError("pyterrier_dr is required for HgfBiEncoder pipeline.")

    hgf_be = HgfBiEncoder_base.from_pretrained(checkpoint)

    def yield_pipe(context):
        with Temporary(FlexIndex) as flex_index:
            pipe = hgf_be >> flex_index.indexer()
            pipe.index(context.docs_iter())
            yield hgf_be >> flex_index.np_retriever() >> context.text() >> ranking_pipeline

    return yield_pipe


__all__ = ["HgfBiEncoder"]
