from ir_measures import nDCG

from suiteeval.suite.beir import _BEIR

datasets = [
    "nano-beir/arguana",
    "nano-beir/climate-fever",
    "nano-beir/dbpedia-entity",
    "nano-beir/fever",
    "nano-beir/fiqa",
    "nano-beir/hotpotqa",
    "nano-beir/msmarco",
    "nano-beir/nfcorpus",
    "nano-beir/nq",
    "nano-beir/quora",
    "nano-beir/scidocs",
    "nano-beir/scifact",
    "nano-beir/webis-touche2020",
]

measures = [nDCG @ 10]


class _NanoBEIR(_BEIR):
    """
    Nano BEIR suite for evaluating retrieval systems on various datasets.

    This suite includes a subset and subsampling of datasets from the BEIR benchmark,
    covering domains like question answering, fact verification, and more.
    It uses nDCG@10 as the primary measure for evaluation.

    Example:
        from suiteeval.suite import NanoBEIR
        results = NanoBEIR(pipeline)
    """

    _datasets = datasets
    _measures = measures
    _metadata = {
        "official_measures": measures,
        "description": (
            "Nano Beir is a smaller version (max 50 queries per benchmark) of the "
            "Beir suite of benchmarks to test zero-shot transfer."
        ),
    }


NanoBEIR = _NanoBEIR()

__all__ = ["NanoBEIR"]
