from suiteeval.index import Temporary
from suiteeval._optional import pyterrier_pisa_available, pyterrier_splade_available

if pyterrier_splade_available():
    from pyt_splade import Splade

if pyterrier_pisa_available():
    from pyterrier_pisa import PisaIndex


def SPLADE(
    ranking_pipeline, checkpoint: str = "naver/splade-cocondenser-ensembledistil"
):
    if not pyterrier_pisa_available():
        raise ImportError("pyterrier_pisa is required for SPLADE pipeline.")
    if not pyterrier_splade_available():
        raise ImportError("pyterrier_splade is required for SPLADE pipeline.")
    splade_model = Splade(model=checkpoint)

    def yield_pipe(context):
        splade_index = PisaIndex(context.path + '/index.splade.pisa', stemmer="none")
        index_pipeline = splade_model.doc_encoder() >> splade_index
        index_pipeline.index(context.get_corpus_iter())
        yield (
            splade_model.query_encoder()
            >> splade_index.quantized()
            >> context.text_loader()
            >> ranking_pipeline
        )

    return yield_pipe


__all__ = ["SPLADE"]
