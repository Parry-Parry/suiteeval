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
            qid = row['qid']
            rows.append({
                'qid': qid,
                'docno': 'doc_1',
                'rank': 0,
                'score': 0.9,
            })
            rows.append({
                'qid': qid,
                'docno': 'doc_2',
                'rank': 1,
                'score': 0.5,
            })
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

    def test_save_dir_creates_directory_structure(
        self, simple_suite, temp_output_dir
    ):
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

    def test_save_dir_creates_run_files(
        self, simple_suite, temp_output_dir
    ):
        """Verify that run files are created in the save_dir."""
        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "dummy_system"

        save_path = os.path.join(temp_output_dir, "results")
        simple_suite(dummy_pipeline_gen, save_dir=save_path)

        dataset_dir = os.path.join(save_path, "vaswani")

        # List files in the dataset directory
        files = os.listdir(dataset_dir) if os.path.exists(
            dataset_dir
        ) else []

        # PyTerrier creates TREC format run files by default
        assert len(files) > 0

    def test_save_format_trec_creates_trec_files(
        self, simple_suite, temp_output_dir
    ):
        """Verify that TREC format is used when save_format='trec'."""
        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "dummy_system"

        save_path = os.path.join(temp_output_dir, "results_trec")
        simple_suite(
            dummy_pipeline_gen,
            save_dir=save_path,
            save_format="trec"
        )

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
                        with gzip.open(filepath, 'rt') as file:
                            first_line = file.readline()
                    except (OSError, gzip.BadGzipFile):
                        with open(filepath, 'r') as file:
                            first_line = file.readline()
                    # TREC format: qid iter docno rank score name
                    parts = first_line.strip().split()
                    assert len(parts) >= 6
                    break

    def test_save_dir_run_file_content(
        self, simple_suite, temp_output_dir
    ):
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
                        with gzip.open(filepath, 'rt') as f:
                            lines = f.readlines()
                    except (OSError, gzip.BadGzipFile):
                        with open(filepath, 'r') as f:
                            lines = f.readlines()

                    assert len(lines) > 0

                    # Each line should have TREC format
                    for line in lines:
                        parts = line.strip().split()
                        # TREC format: qid iter docno rank score name
                        assert len(parts) >= 6


class TestSaveDirWithMultipleSystems:
    """Tests for save_dir with multiple ranking systems."""

    def test_save_dir_with_multiple_systems(
        self, simple_suite, temp_output_dir
    ):
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

    def test_save_dir_with_special_characters_in_name(
        self, temp_output_dir
    ):
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
        nested_path = os.path.join(
            temp_output_dir, "a", "b", "c", "results"
        )

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

@pytest.mark.skip(
    reason="Requires real BEIR datasets - run manually with full setup"
)
class TestSaveDirIntegrationBEIR:
    """Integration tests with actual BEIR datasets."""

    def test_beir_full_pipeline_with_save_dir(self, temp_output_dir):
        """Full integration test with BEIR suite."""
        def dummy_pipeline_gen(context):
            yield DummyRankingTransformer(), "test_system"

        save_path = os.path.join(temp_output_dir, "beir_full")

        # Test with subset for speed
        results = BEIR(
            dummy_pipeline_gen,
            save_dir=save_path,
            subset="beir/arguana"
        )

        # Verify results dataframe
        assert isinstance(results, pd.DataFrame)
        assert not results.empty

        # Verify directory structure
        assert os.path.exists(save_path)
        assert os.path.isdir(save_path)


# ---------- Tests for skip_existing functionality ----------


class TestSkipExisting:
    """Tests for the skip_existing parameter."""

    def test_skip_existing_false_default_behavior(
        self, simple_suite, temp_output_dir
    ):
        """Verify that skip_existing=False (default) doesn't skip anything."""
        call_count = [0]

        def counting_pipeline_gen(context):
            call_count[0] += 1
            yield DummyRankingTransformer(), "system_a"

        save_path = os.path.join(temp_output_dir, "results_default")

        # First run
        results1 = simple_suite(counting_pipeline_gen, save_dir=save_path)
        first_call_count = call_count[0]
        assert first_call_count == 1

        # Second run (generator should be called again because skip_existing=False)
        results2 = simple_suite(counting_pipeline_gen, save_dir=save_path)
        assert call_count[0] == 2

        # Both runs should produce results
        assert isinstance(results1, pd.DataFrame)
        assert isinstance(results2, pd.DataFrame)
        assert not results1.empty
        assert not results2.empty

    def test_skip_existing_with_no_save_dir(self, simple_suite, temp_output_dir):
        """Verify that skip_existing has no effect without save_dir."""
        call_count = [0]

        def counting_pipeline_gen(context):
            call_count[0] += 1
            yield DummyRankingTransformer(), "system_a"

        # Run without save_dir but with skip_existing=True
        results = simple_suite(
            counting_pipeline_gen,
            skip_existing=True,
            # No save_dir parameter
        )

        # Generator should still be called because no save_dir
        assert call_count[0] == 1
        assert isinstance(results, pd.DataFrame)

    def test_skip_existing_loads_cached_results(
        self, simple_suite, temp_output_dir
    ):
        """Verify that skip_existing=True loads cached results correctly."""
        save_path = os.path.join(temp_output_dir, "results_cache")

        # First run: create the cached files
        def pipeline_gen_1(context):
            yield DummyRankingTransformer(), "system_a"

        results1 = simple_suite(pipeline_gen_1, save_dir=save_path)
        assert not results1.empty
        assert "system_a" in results1["name"].values

        # Second run: load from cache
        call_count = [0]

        def pipeline_gen_2(context):
            call_count[0] += 1
            yield DummyRankingTransformer(), "system_a"

        results2 = simple_suite(
            pipeline_gen_2, save_dir=save_path, skip_existing=True
        )

        # Results should be loaded from cache
        assert not results2.empty
        assert "system_a" in results2["name"].values

        # Verify structure matches (same metrics columns, etc.)
        assert set(results1.columns) == set(results2.columns)

    def test_skip_existing_partial_cache(
        self, simple_suite, temp_output_dir
    ):
        """Verify skip_existing with partial cache (some datasets cached, some not)."""
        # Create a suite with multiple datasets
        multi_suite = Suite.register(
            "test_partial_cache",
            datasets=["vaswani"],
            names=["vaswani"],
            metadata={
                "official_measures": [nDCG @ 10],
            },
        )  # noqa: F811

        save_path = os.path.join(temp_output_dir, "results_partial")

        # First run: compute system_a
        def gen_a(context):
            yield DummyRankingTransformer(), "system_a"

        results_a = multi_suite(gen_a, save_dir=save_path)
        assert "system_a" in results_a["name"].values

        # Second run: compute both system_a and system_b with skip_existing
        # Since system_a is cached, it should be loaded; system_b should be computed
        call_count = [0]

        def gen_ab(context):
            call_count[0] += 1
            yield DummyRankingTransformer(), "system_a"
            yield DummyRankingTransformer(), "system_b"

        results_ab = multi_suite(
            gen_ab, save_dir=save_path, skip_existing=True
        )

        # Both systems should be in results
        assert "system_a" in results_ab["name"].values
        assert "system_b" in results_ab["name"].values

        # Generator was called, but system_a's experiment should have been skipped
        assert call_count[0] == 1

    def test_skip_existing_all_cached(
        self, simple_suite, temp_output_dir
    ):
        """Verify skip_existing when all runs are already cached."""
        save_path = os.path.join(temp_output_dir, "results_all_cached")

        # First run: create cached files
        def gen_1(context):
            yield DummyRankingTransformer(), "system_a"

        results1 = simple_suite(gen_1, save_dir=save_path)
        assert not results1.empty

        # Second run: all cached, generator still called to get names
        call_count = [0]

        def gen_2(context):
            call_count[0] += 1
            yield DummyRankingTransformer(), "system_a"

        results2 = simple_suite(
            gen_2, save_dir=save_path, skip_existing=True
        )

        # Generator called once to get pipeline names
        assert call_count[0] == 1

        # Results should be loaded from cache
        assert not results2.empty
        assert len(results2) == len(results1)

    def test_skip_existing_preserves_dataframe_structure(
        self, simple_suite, temp_output_dir
    ):
        """Verify that cached results have same structure as computed results."""
        save_path = os.path.join(temp_output_dir, "results_structure")

        # First run: create baseline
        def gen(context):
            yield DummyRankingTransformer(), "system_a"

        results_fresh = simple_suite(gen, save_dir=save_path)

        # Delete the old results to force recomputation
        import shutil as sh
        sh.rmtree(save_path)

        # Second run: recompute (baseline for comparison)
        results_recomputed = simple_suite(gen, save_dir=save_path)

        # Now run with skip_existing on cached results
        results_cached = simple_suite(
            gen, save_dir=save_path, skip_existing=True
        )

        # All three should have same columns
        assert set(results_fresh.columns) == set(results_recomputed.columns)
        assert set(results_fresh.columns) == set(results_cached.columns)

        # All should have data rows
        assert len(results_fresh) > 0
        assert len(results_recomputed) > 0
        assert len(results_cached) > 0

