from suiteeval.suite.base import Suite
from ir_measures import nDCG

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

measures = [nDCG@10]

NanoBEIR = Suite.register(
    "nano-beir",
    datasets,
    metadata={
        "official_measures": measures,
        "description": "Nano Beir is a smaller version (max 50 queries per benchmark) of the Beir suite of benchmarks to test zero-shot transfer.",
    },
)

__all__ = ["NanoBEIR"]
