from suiteeval.index import Temporary
from suiteeval._optional import pyterrier_pisa_available, pyterrier_splade_available

if pyterrier_splade_available():
    from pyt_splade import Splade

if pyterrier_pisa_available():
    from pyterrier_pisa import PisaIndex


def SPLADE(ranking_pipeline, checkpoint: str = "naver/splade-cocondenser-ensembledistil"):
    splade_model = Splade(model=checkpoint)

    def yield_pipe(context):
        with Temporary(PisaIndex, stemmer='none') as splade_index:
            index_pipeline = splade_model.doc_encoder() >> splade_index
            index_pipeline.index(context.docs_iter())
            return splade_model.query_encoder() >> splade_index.quantized() >> context.text() >> ranking_pipeline

    return yield_pipe


__all__ = ["SPLADE"]
