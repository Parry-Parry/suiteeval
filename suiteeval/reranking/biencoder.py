from suiteeval._optional import pyterrier_dr_available

if pyterrier_dr_available():
    from pyterrier_dr import HgfBiEncoder as HgfBiEncoder_base, FlexIndex


def HgfBiEncoder(
    ranking_pipeline, checkpoint: str = "sentence-transformers/all-mpnet-base-v2"
):
    """
    Loads a HgfBiEncoder bi-encoder model from the given checkpoint, and constructs a PyTerrier pipeline generator

    Args:
        ranking_pipeline: A PyTerrier pipeline to apply after retrieval.
        checkpoint: The checkpoint to load the bi-encoder model from.
    """
    if not pyterrier_dr_available():
        raise ImportError("pyterrier_dr is required for HgfBiEncoder pipeline.")

    hgf_be = HgfBiEncoder_base.from_pretrained(checkpoint)

    def yield_pipe(context):
        flex_index = FlexIndex(context.path + "/bienc.flex", stemmer="none")
        pipe = hgf_be >> flex_index.indexer()
        pipe.index(context.get_corpus_iter())
        yield (
            hgf_be
            >> flex_index.np_retriever()
            >> context.text_loader()
            >> ranking_pipeline
        )

    return yield_pipe


__all__ = ["HgfBiEncoder"]
