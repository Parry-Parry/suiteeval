from suiteeval.index import TemporarySpladeIndex
from suiteeval._optional import pyterrier_pisa_available, pyterrier_splade_available

if pyterrier_splade_available():
    from pyt_splade import Splade


def SPLADE(ranking_pipeline, checkpoint: str = "naver/splade-cocondenser-ensembledistil"):
    if not pyterrier_pisa_available():
        raise ImportError("pyterrier_pisa is required for SPLADE pipeline.")
    if not pyterrier_splade_available():
        raise ImportError("pyt_splade is required for SPLADE pipeline.")

    splade_model = Splade(model=checkpoint)

    def yield_pipe(documents):
        with TemporarySpladeIndex(splade_model, documents) as splade_index:
            yield splade_index >> ranking_pipeline

    return yield_pipe


__all__ = ["SPLADE"]
