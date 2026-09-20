"""Regression tests for the declared BEIR / NanoBEIR dataset lists.

These guard against datasets silently going missing from a suite, and against
typos in the ``ir_datasets`` identifiers used to declare them.
"""

import ir_datasets
import pytest

from suiteeval.suite.beir import BEIR
from suiteeval.suite.nanobeir import NanoBEIR

BEIR_EXPECTED = {
    "arguana",
    "climate-fever",
    "cqadupstack",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
}

NANOBEIR_EXPECTED = BEIR_EXPECTED - {"cqadupstack", "trec-covid"}


def collection_names(datasets: list[str], prefix: str) -> set[str]:
    """Reduce dataset ids to their BEIR collection names.

    ``beir/cqadupstack/android`` and ``beir/scifact/test`` both collapse to the
    collection they belong to, so sub-collections and splits do not affect the
    comparison.
    """
    return {dataset.removeprefix(prefix).split("/")[0] for dataset in datasets}


def test_beir_declares_every_collection():
    assert collection_names(BEIR._datasets, "beir/") == BEIR_EXPECTED


def test_nanobeir_declares_every_collection():
    assert collection_names(NanoBEIR._datasets, "nano-beir/") == NANOBEIR_EXPECTED


@pytest.mark.parametrize("dataset", BEIR._datasets + NanoBEIR._datasets)
def test_declared_dataset_is_loadable(dataset: str):
    """Every declared id must resolve and expose docs, queries and qrels.

    ``ir_datasets.load`` only builds the dataset object, so this stays offline.
    """
    loaded = ir_datasets.load(dataset)
    assert loaded.has_docs()
    assert loaded.has_queries()
    assert loaded.has_qrels()
