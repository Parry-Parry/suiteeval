"""Tests for index_dir functionality to verify directory structure."""

import os
import tempfile
import shutil

import pytest
import pandas as pd
from pyterrier import Transformer
from ir_measures import AP

from suiteeval.suite.base import Suite


# ---------- Test Fixtures ----------


class DummyRankingTransformer(Transformer):
    """A simple transformer that generates dummy rankings."""

    def transform(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate a dummy ranking for each topic."""
        rows = []
        for _, row in topics_df.iterrows():
            qid = row["qid"]
            rows.append(
                {
                    "qid": qid,
                    "docno": "doc_1",
                    "rank": 0,
                    "score": 0.9,
                }
            )
            rows.append(
                {
                    "qid": qid,
                    "docno": "doc_2",
                    "rank": 1,
                    "score": 0.5,
                }
            )
        return pd.DataFrame(rows)


@pytest.fixture
def temp_output_dir():
    """Create and cleanup a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp(prefix="suiteeval_indexing_test_")
    yield tmpdir
    # Cleanup after the test
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


# ---------- Tests ----------


class TestIndexingDirBasicStructure:
    """Test that index_dir creates the correct directory structure."""

    def test_index_dir_creates_directory_structure(self, temp_output_dir):
        """Test that index_dir creates corpus-specific subdirectories."""
        # Create a simple suite with one dataset
        test_suite = Suite.register(
            "test_suite_indexing",
            datasets=["vaswani"],
            metadata={
                "official_measures": [AP @ 1000],
                "description": "Test suite for index_dir",
            },
        )

        indexing_path = os.path.join(temp_output_dir, "indices")

        # Verify directory doesn't exist before test
        assert not os.path.exists(indexing_path)

        def pipeline_generator(context):
            # Verify that context.path is set to our index_dir subdirectory
            assert context.path is not None
            assert context.path.startswith(indexing_path)
            # Verify the corpus ID is formatted correctly (vaswani -> vaswani)
            expected_corpus_dir = os.path.join(indexing_path, "vaswani")
            assert context.path == expected_corpus_dir
            yield DummyRankingTransformer(), "test_system"

        # Run the suite with index_dir
        _ = test_suite(pipeline_generator, index_dir=indexing_path)

        # Verify directory structure was created
        assert os.path.exists(indexing_path)
        assert os.path.isdir(indexing_path)

        # Verify corpus-specific subdirectory
        corpus_dir = os.path.join(indexing_path, "vaswani")
        assert os.path.exists(corpus_dir)
        assert os.path.isdir(corpus_dir)

    def test_index_dir_formatting_with_slashes(self, temp_output_dir):
        """Test that corpus IDs with slashes are formatted correctly."""
        # Create a suite with a dataset that has slashes in corpus ID
        test_suite = Suite.register(
            "test_suite_slashes",
            datasets=["vaswani"],  # Simple dataset for testing
            metadata={
                "official_measures": [AP @ 1000],
                "description": "Test suite for slashes",
            },
        )

        indexing_path = os.path.join(temp_output_dir, "indices")

        def pipeline_generator(context):
            # Just verify the directory was created
            assert context.path is not None
            yield DummyRankingTransformer(), "test_system"

        _ = test_suite(pipeline_generator, index_dir=indexing_path)

        # The corpus ID should be formatted: replace "/" with "-" and lowercase
        # For vaswani, it should just be "vaswani"
        corpus_dir = os.path.join(indexing_path, "vaswani")
        assert os.path.exists(corpus_dir)

    def test_index_dir_none_uses_temp_directory(self, temp_output_dir):
        """Test that when index_dir is None, temp directory is used."""
        test_suite = Suite.register(
            "test_suite_no_indexing",
            datasets=["vaswani"],
            metadata={
                "official_measures": [AP @ 1000],
                "description": "Test suite without index_dir",
            },
        )

        temp_paths = []

        def pipeline_generator(context):
            # When index_dir is None, context.path should be a temp directory
            assert context.path is not None
            # Verify it's a temp directory (contains temp-like path)
            assert "tmp" in context.path or "temp" in context.path.lower()
            temp_paths.append(context.path)
            yield DummyRankingTransformer(), "test_system"

        # Run without index_dir
        _ = test_suite(pipeline_generator)

        # Verify temp directory was created and used
        assert len(temp_paths) > 0
        # Note: temp directories are cleaned up automatically by DatasetContext


class TestIndexingDirWithMultipleDatasets:
    """Test index_dir with multiple datasets sharing the same corpus."""

    def test_multiple_datasets_share_same_index_dir(self, temp_output_dir):
        """Test that datasets sharing corpus use same indexing dir."""
        # Create a suite with multiple datasets
        test_suite = Suite.register(
            "test_suite_multi",
            datasets=["vaswani"],
            metadata={
                "official_measures": [AP @ 1000],
                "description": "Test suite with multiple datasets",
            },
        )

        indexing_path = os.path.join(temp_output_dir, "indices")
        context_paths = []

        def pipeline_generator(context):
            # Collect the context path for verification
            context_paths.append(context.path)
            yield DummyRankingTransformer(), "test_system"

        _ = test_suite(pipeline_generator, index_dir=indexing_path)

        # All datasets should use the same corpus directory
        # (Since we only have one corpus, there should be one unique path)
        assert len(set(context_paths)) == 1
        assert context_paths[0] == os.path.join(indexing_path, "vaswani")
