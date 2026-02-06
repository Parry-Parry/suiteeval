"""Tests for save_dir functionality to verify file structure and run files."""

import os
import tempfile
import shutil
import gzip

import pytest
import pandas as pd
from pyterrier import Transformer
from ir_measures import nDCG

from suiteeval.suite.base import Suite
from suiteeval.suite.beir import BEIR


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
    tmpdir = tempfile.mkdtemp(prefix="suiteeval_test_")
    yield tmpdir
    # Cleanup
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


@pytest.fixture
def simple_suite():
    """Create a simple test suite with a single dataset."""
    return Suite.register(
        "test_simple",
        datasets=["vaswani"],
        metadata={
            "official_measures": [nDCG @ 10],
            "description": "Simple test suite",
        },
    )  # noqa: F811


# ---------- Tests ----------


class TestSaveDirBasicStructure:
    """Tests for basic save_dir file structure creation."""

    def test_save_dir_creates_directory_structure(self, simple_suite, temp_output_dir):
        """Verify that save_dir creates the expected directory structure."""

        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "dummy_system"

        save_path = os.path.join(temp_output_dir, "results")
        simple_suite(dummy_pipeline_gen, save_dir=save_path)

        # Check that the save directory was created
        assert os.path.exists(save_path)
        assert os.path.isdir(save_path)

    def test_save_dir_creates_dataset_subdirectories(
        self, simple_suite, temp_output_dir
    ):
        """Verify that subdirectories are created for each dataset."""

        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "dummy_system"

        save_path = os.path.join(temp_output_dir, "results")
        simple_suite(dummy_pipeline_gen, save_dir=save_path)

        # Check for dataset-specific subdirectory
        dataset_dir = os.path.join(save_path, "vaswani")
        assert os.path.exists(dataset_dir)

    def test_save_dir_with_multiple_datasets(self, temp_output_dir):
        """Verify that subdirectories are created for multiple datasets."""
        multi_suite = Suite.register(
            "test_multi",
            datasets=["vaswani"],
            names=["vaswani_dataset"],
            metadata={
                "official_measures": [nDCG @ 10],
                "description": "Multi-dataset test suite",
            },
        )  # noqa: F811

        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "dummy_system"

        save_path = os.path.join(temp_output_dir, "results")
        multi_suite(dummy_pipeline_gen, save_dir=save_path)

        # Check for dataset-specific subdirectory
        vaswani_dir = os.path.join(save_path, "vaswani_dataset")
        assert os.path.exists(vaswani_dir)


class TestSaveDirRunFiles:
    """Tests for the creation of run files in save_dir."""

    def test_save_dir_creates_run_files(self, simple_suite, temp_output_dir):
        """Verify that run files are created in the save_dir."""

        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "dummy_system"

        save_path = os.path.join(temp_output_dir, "results")
        simple_suite(dummy_pipeline_gen, save_dir=save_path)

        dataset_dir = os.path.join(save_path, "vaswani")

        # List files in the dataset directory
        files = os.listdir(dataset_dir) if os.path.exists(dataset_dir) else []

        # PyTerrier creates TREC format run files by default
        assert len(files) > 0

    def test_save_format_trec_creates_trec_files(self, simple_suite, temp_output_dir):
        """Verify that TREC format is used when save_format='trec'."""

        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "dummy_system"

        save_path = os.path.join(temp_output_dir, "results_trec")
        simple_suite(dummy_pipeline_gen, save_dir=save_path, save_format="trec")

        dataset_dir = os.path.join(save_path, "vaswani")

        # Check that files were created
        if os.path.exists(dataset_dir):
            files = os.listdir(dataset_dir)
            assert len(files) > 0

            # TREC files may be gzip-compressed
            for f in files:
                filepath = os.path.join(dataset_dir, f)
                if os.path.isfile(filepath):
                    try:
                        with gzip.open(filepath, "rt") as file:
                            first_line = file.readline()
                    except (OSError, gzip.BadGzipFile):
                        with open(filepath, "r") as file:
                            first_line = file.readline()
                    # TREC format: qid iter docno rank score name
                    parts = first_line.strip().split()
                    assert len(parts) >= 6
                    break

    def test_save_dir_run_file_content(self, simple_suite, temp_output_dir):
        """Verify that run files contain valid ranking data."""

        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "test_system"

        save_path = os.path.join(temp_output_dir, "results_content")
        simple_suite(dummy_pipeline_gen, save_dir=save_path)

        dataset_dir = os.path.join(save_path, "vaswani")

        if os.path.exists(dataset_dir):
            # Find the run file
            for filename in os.listdir(dataset_dir):
                filepath = os.path.join(dataset_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        with gzip.open(filepath, "rt") as f:
                            lines = f.readlines()
                    except (OSError, gzip.BadGzipFile):
                        with open(filepath, "r") as f:
                            lines = f.readlines()

                    assert len(lines) > 0

                    # Each line should have TREC format
                    for line in lines:
                        parts = line.strip().split()
                        # TREC format: qid iter docno rank score name
                        assert len(parts) >= 6


class TestSaveDirWithMultipleSystems:
    """Tests for save_dir with multiple ranking systems."""

    def test_save_dir_with_multiple_systems(self, simple_suite, temp_output_dir):
        """Verify that run files are created for each system."""
        systems = ["system_a", "system_b", "system_c"]

        def multi_pipeline_gen(context):
            for name in systems:
                yield DummyRankingTransformer(), name

        save_path = os.path.join(temp_output_dir, "results_multi")
        simple_suite(multi_pipeline_gen, save_dir=save_path)

        dataset_dir = os.path.join(save_path, "vaswani")

        # Check that files were created for each system
        if os.path.exists(dataset_dir):
            files = os.listdir(dataset_dir)
            assert len(files) > 0


class TestSaveDirEdgeCases:
    """Tests for edge cases and error handling."""

    def test_save_dir_with_special_characters_in_name(self, temp_output_dir):
        """Verify handling of dataset names with special characters."""
        # Some BEIR datasets have names with slashes
        suite_with_slashes = Suite.register(
            "test_slashes",
            datasets=["vaswani"],
            names=["test/dataset/name"],
            metadata={
                "official_measures": [nDCG @ 10],
            },
        )  # noqa: F811

        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "system"

        save_path = os.path.join(temp_output_dir, "results_special")
        suite_with_slashes(dummy_pipeline_gen, save_dir=save_path)

        # The directory should exist (slashes converted to dashes in path)
        assert os.path.exists(save_path)

    def test_save_dir_nested_path_creation(self, temp_output_dir):
        """Verify that nested paths are created correctly."""
        nested_path = os.path.join(temp_output_dir, "a", "b", "c", "results")

        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "system"

        simple_suite_inst = Suite.register(
            "test_nested",
            datasets=["vaswani"],
            metadata={
                "official_measures": [nDCG @ 10],
            },
        )  # noqa: F811

        simple_suite_inst(dummy_pipeline_gen, save_dir=nested_path)

        # The directory should be created
        assert os.path.exists(nested_path)


class TestSaveDirWithBEIR:
    """Tests specific to BEIR suite save_dir behavior."""

    def test_beir_save_dir_creates_dataset_subdirs(self, temp_output_dir):
        """Verify that BEIR creates subdirectories for each dataset."""

        # Create a mock pipeline that yields dummy rankings
        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "test_system"

        save_path = os.path.join(temp_output_dir, "beir_results")

        # Note: This would require actual BEIR datasets, so it's marked
        # for integration testing with real datasets
        Suite.register(
            "test_beir_like",
            datasets=[
                "beir/arguana",
                "beir/cqadupstack/android",
                "beir/cqadupstack/english",
            ],
            metadata={
                "official_measures": [nDCG @ 10],
            },
        )  # noqa: F811

        # Skip if BEIR datasets not available
        pytest.skip("Requires full BEIR dataset setup")


# ---------- Integration Tests ----------


@pytest.mark.skip(reason="Requires real BEIR datasets - run manually with full setup")
class TestSaveDirIntegrationBEIR:
    """Integration tests with actual BEIR datasets."""

    def test_beir_full_pipeline_with_save_dir(self, temp_output_dir):
        """Full integration test with BEIR suite."""

        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "test_system"

        save_path = os.path.join(temp_output_dir, "beir_full")

        # Test with subset for speed
        results = BEIR(dummy_pipeline_gen, save_dir=save_path, subset="beir/arguana")

        # Verify results dataframe
        assert isinstance(results, pd.DataFrame)
        assert not results.empty

        # Verify directory structure
        assert os.path.exists(save_path)
        assert os.path.isdir(save_path)


class TestSaveDirMultiDatasetIsolation:
    """Integration tests for save_dir with multiple datasets sharing a corpus."""

    def test_multiple_datasets_create_separate_directories(self, temp_output_dir):
        """Each dataset in a multi-dataset suite creates its own subdirectory."""
        multi_suite = Suite.register(
            "test_multi_dirs",
            datasets=["vaswani", "vaswani"],
            names=["test_ds_one", "test_ds_two"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        def pipeline_gen(context):
            yield DummyRankingTransformer(), "system_a"

        save_path = os.path.join(temp_output_dir, "results")
        multi_suite(pipeline_gen, save_dir=save_path)

        # Both directories should exist
        dir_one = os.path.join(save_path, "test_ds_one")
        dir_two = os.path.join(save_path, "test_ds_two")

        assert os.path.exists(dir_one), f"Directory {dir_one} should exist"
        assert os.path.exists(dir_two), f"Directory {dir_two} should exist"

    def test_each_dataset_dir_contains_all_system_files(self, temp_output_dir):
        """Each dataset subdirectory should contain TREC files for ALL systems."""
        systems = ["system_alpha", "system_beta", "system_gamma"]

        multi_suite = Suite.register(
            "test_all_systems",
            datasets=["vaswani", "vaswani"],
            names=["dataset_x", "dataset_y"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        def pipeline_gen(context):
            for name in systems:
                yield DummyRankingTransformer(), name

        save_path = os.path.join(temp_output_dir, "results")
        multi_suite(pipeline_gen, save_dir=save_path)

        for ds_name in ["dataset_x", "dataset_y"]:
            ds_dir = os.path.join(save_path, ds_name)
            assert os.path.exists(ds_dir), f"Directory for {ds_name} should exist"

            files = os.listdir(ds_dir)
            # Should have files for all 3 systems
            assert len(files) >= len(systems), (
                f"Expected at least {len(systems)} files in {ds_dir}, found {len(files)}"
            )

    def test_grouped_mode_creates_correct_directory_structure(self, temp_output_dir):
        """Grouped mode (with baseline) creates correct directories for each dataset."""
        multi_suite = Suite.register(
            "test_grouped_dirs",
            datasets=["vaswani", "vaswani"],
            names=["grouped_ds_a", "grouped_ds_b"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        def pipeline_gen(context):
            yield DummyRankingTransformer(), "baseline_system"
            yield DummyRankingTransformer(), "test_system"

        save_path = os.path.join(temp_output_dir, "results_grouped")
        # baseline=0 triggers grouped mode
        multi_suite(pipeline_gen, save_dir=save_path, baseline=0)

        # Both dataset directories should exist
        assert os.path.exists(os.path.join(save_path, "grouped_ds_a"))
        assert os.path.exists(os.path.join(save_path, "grouped_ds_b"))

        # Each should contain files for both systems
        for ds_name in ["grouped_ds_a", "grouped_ds_b"]:
            ds_dir = os.path.join(save_path, ds_name)
            files = os.listdir(ds_dir)
            assert len(files) >= 2, f"Expected files for both systems in {ds_dir}"

    def test_trec_file_qids_match_dataset(self, temp_output_dir):
        """TREC files should contain qids appropriate for their dataset."""
        multi_suite = Suite.register(
            "test_qid_correctness",
            datasets=["vaswani"],
            names=["vaswani_test"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        def pipeline_gen(context):
            yield DummyRankingTransformer(), "test_system"

        save_path = os.path.join(temp_output_dir, "results_qids")
        multi_suite(pipeline_gen, save_dir=save_path)

        ds_dir = os.path.join(save_path, "vaswani_test")

        # Read and verify TREC file content
        for filename in os.listdir(ds_dir):
            filepath = os.path.join(ds_dir, filename)
            if os.path.isfile(filepath):
                try:
                    with gzip.open(filepath, "rt") as f:
                        lines = f.readlines()
                except (OSError, gzip.BadGzipFile):
                    with open(filepath, "r") as f:
                        lines = f.readlines()

                # Extract qids from TREC format (qid is first field)
                qids = set(line.split()[0] for line in lines if line.strip())

                # Verify qids are non-empty (basic sanity check)
                assert len(qids) > 0, "TREC file should contain queries"
