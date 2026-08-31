"""Shared fixtures for the unit tests.

These mock out dataset downloads and PyTerrier experiments so the suite
orchestration logic can be exercised without touching the network or disk.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ir_measures import nDCG
from pyterrier import Transformer

from suiteeval.suite.base import Suite, SuiteMeta


class DummyTransformer(Transformer):
    """Minimal transformer producing one result row per topic."""

    def transform(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"qid": row["qid"], "docno": "d1", "rank": 0, "score": 1.0}
                for _, row in topics_df.iterrows()
            ]
        )


@pytest.fixture
def temp_dir():
    """Create and cleanup a temporary directory."""
    tmpdir = tempfile.mkdtemp(prefix="suiteeval_unit_test_")
    yield tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


@pytest.fixture
def cleanup_suite_registry():
    """Clean up suite registry after test to avoid pollution between tests."""
    yield
    for key in [k for k in SuiteMeta._classes if k.startswith("test_")]:
        del SuiteMeta._classes[key]
        SuiteMeta._instances.pop(key, None)


@pytest.fixture
def mock_text_loader():
    """Create a mock text loader (IRDSTextLoader-like object)."""
    mock_loader = MagicMock()
    mock_loader.transform.return_value = pd.DataFrame(
        {
            "qid": ["1"],
            "docno": ["d1"],
            "text": ["sample document text"],
        }
    )
    return mock_loader


@pytest.fixture
def mock_dataset(mock_text_loader):
    """Create a mock PyTerrier dataset with text_loader support."""
    mock_ds = MagicMock()
    mock_ds._irds_id = "vaswani"
    mock_ds.get_topics.return_value = pd.DataFrame(
        {"qid": ["1", "2"], "query": ["q1", "q2"]}
    )
    mock_ds.get_qrels.return_value = pd.DataFrame(
        {"qid": ["1", "1", "2"], "docno": ["d1", "d2", "d1"], "label": [1, 0, 1]}
    )
    mock_ds.text_loader.return_value = mock_text_loader
    mock_ds.get_corpus_iter.return_value = iter(
        [
            {"docno": "d1", "text": "document one"},
            {"docno": "d2", "text": "document two"},
        ]
    )
    return mock_ds


@pytest.fixture
def mock_pt_get_dataset(mock_dataset):
    """Mock pt.get_dataset to return our mock dataset."""
    with patch("pyterrier.get_dataset") as mock_get:
        mock_get.return_value = mock_dataset
        yield mock_get


@pytest.fixture
def mock_pt_experiment():
    """Mock pt.Experiment to track calls and return minimal DataFrame."""
    with patch("pyterrier.Experiment") as mock_exp:
        mock_exp.return_value = pd.DataFrame(
            {"name": ["test_system"], "nDCG@10": [0.5]}
        )
        yield mock_exp


@pytest.fixture
def mock_irds_docs_parent_id():
    """Mock ir_datasets.docs_parent_id so each dataset is its own corpus."""
    with patch("ir_datasets.docs_parent_id") as mock_parent:
        mock_parent.side_effect = lambda x: x
        yield mock_parent


@pytest.fixture
def vaswani_suite(cleanup_suite_registry):
    """Create a reusable vaswani test suite."""
    return Suite.register(
        "test_vaswani_suite",
        datasets=["vaswani"],
        metadata={"official_measures": [nDCG @ 10]},
    )
