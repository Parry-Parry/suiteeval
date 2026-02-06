"""Unit tests for save_dir, index_dir, and DatasetContext functionality with mocking.

These tests verify that:
1. Run files are saved correctly in subdirectories when save_dir is provided
2. Index directories exist when index_dir is specified
3. Path formatting is correct (slashes -> dashes, lowercase)
4. DatasetContext.text_loader() correctly delegates to dataset.text_loader()

Uses mocking to avoid actual dataset downloads and large file creation.
"""

import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
from pyterrier import Transformer
from ir_measures import nDCG

from suiteeval.suite.base import Suite, SuiteMeta
from suiteeval.context import DatasetContext


# ---------- Shared Test Utilities ----------


class DummyTransformer(Transformer):
    """Minimal transformer for testing."""

    def transform(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in topics_df.iterrows():
            rows.append({"qid": row["qid"], "docno": "d1", "rank": 0, "score": 1.0})
        return pd.DataFrame(rows)


# ---------- Shared Fixtures ----------


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
    # Remove test suites from registry
    to_remove = [k for k in SuiteMeta._classes if k.startswith("test_")]
    for k in to_remove:
        del SuiteMeta._classes[k]
        if k in SuiteMeta._instances:
            del SuiteMeta._instances[k]


@pytest.fixture
def mock_text_loader():
    """Create a mock text loader (IRDSTextLoader-like object)."""
    mock_loader = MagicMock()
    mock_loader.transform.return_value = pd.DataFrame({
        "qid": ["1"],
        "docno": ["d1"],
        "text": ["sample document text"],
    })
    return mock_loader


@pytest.fixture
def mock_dataset(mock_text_loader):
    """Create a mock PyTerrier dataset with text_loader support."""
    mock_ds = MagicMock()
    mock_ds._irds_id = "vaswani"
    mock_ds.get_topics.return_value = pd.DataFrame({"qid": ["1", "2"], "query": ["q1", "q2"]})
    mock_ds.get_qrels.return_value = pd.DataFrame(
        {"qid": ["1", "1", "2"], "docno": ["d1", "d2", "d1"], "label": [1, 0, 1]}
    )
    mock_ds.text_loader.return_value = mock_text_loader
    mock_ds.get_corpus_iter.return_value = iter([
        {"docno": "d1", "text": "document one"},
        {"docno": "d2", "text": "document two"},
    ])
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
        mock_exp.return_value = pd.DataFrame({"name": ["test_system"], "nDCG@10": [0.5]})
        yield mock_exp


@pytest.fixture
def mock_irds_docs_parent_id():
    """Mock ir_datasets.docs_parent_id to return dataset ID as corpus ID."""
    with patch("ir_datasets.docs_parent_id") as mock_parent:
        mock_parent.side_effect = lambda x: x  # Return dataset ID as its own corpus
        yield mock_parent


@pytest.fixture
def vaswani_suite(cleanup_suite_registry):
    """Create a reusable vaswani test suite."""
    return Suite.register(
        "test_vaswani_suite",
        datasets=["vaswani"],
        metadata={"official_measures": [nDCG @ 10]},
    )


# ---------- save_dir Tests ----------


class TestSaveDirUnit:
    """Unit tests for save_dir functionality."""

    def test_save_dir_creates_dataset_subdirectory(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """Verify os.makedirs is called with correct dataset-specific path."""
        test_suite = Suite.register(
            "test_save_subdir",
            datasets=["vaswani"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        save_path = os.path.join(temp_dir, "results")

        def pipeline_gen(context):
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen, save_dir=save_path)

        # Verify dataset subdirectory was created
        expected_dir = os.path.join(save_path, "vaswani")
        assert os.path.exists(expected_dir), f"Expected directory {expected_dir} to exist"

    def test_save_dir_formats_slashes_to_dashes(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """Dataset names with slashes become dashes in path."""
        test_suite = Suite.register(
            "test_save_slashes",
            datasets=["vaswani"],
            names=["beir/arguana"],  # Name with slash
            metadata={"official_measures": [nDCG @ 10]},
        )

        save_path = os.path.join(temp_dir, "results")

        def pipeline_gen(context):
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen, save_dir=save_path)

        # Slash should be converted to dash
        expected_dir = os.path.join(save_path, "beir-arguana")
        assert os.path.exists(expected_dir), f"Expected directory {expected_dir} to exist"

    def test_save_dir_lowercases_dataset_name(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """Dataset names are lowercased in path."""
        test_suite = Suite.register(
            "test_save_lowercase",
            datasets=["vaswani"],
            names=["VASWANI"],  # Uppercase name
            metadata={"official_measures": [nDCG @ 10]},
        )

        save_path = os.path.join(temp_dir, "results")

        def pipeline_gen(context):
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen, save_dir=save_path)

        # Name should be lowercased
        expected_dir = os.path.join(save_path, "vaswani")
        assert os.path.exists(expected_dir), f"Expected directory {expected_dir} to exist"

    def test_save_dir_passed_to_experiment(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """Verify pt.Experiment receives correct save_dir kwarg."""
        test_suite = Suite.register(
            "test_save_to_exp",
            datasets=["vaswani"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        save_path = os.path.join(temp_dir, "results")

        def pipeline_gen(context):
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen, save_dir=save_path)

        # Check that pt.Experiment was called with save_dir
        mock_pt_experiment.assert_called_once()
        call_kwargs = mock_pt_experiment.call_args[1]
        expected_save_dir = os.path.join(save_path, "vaswani")
        assert call_kwargs.get("save_dir") == expected_save_dir

    def test_save_dir_multiple_datasets(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """Each dataset gets its own subdirectory."""
        test_suite = Suite.register(
            "test_save_multi",
            datasets=["vaswani", "vaswani"],  # Same underlying dataset
            names=["dataset_a", "dataset_b"],  # Different names
            metadata={"official_measures": [nDCG @ 10]},
        )

        save_path = os.path.join(temp_dir, "results")

        def pipeline_gen(context):
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen, save_dir=save_path)

        # Both subdirectories should exist
        assert os.path.exists(os.path.join(save_path, "dataset_a"))
        assert os.path.exists(os.path.join(save_path, "dataset_b"))

    def test_save_dir_none_not_passed_to_experiment(
        self,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """When save_dir is None, it's not passed to pt.Experiment."""
        test_suite = Suite.register(
            "test_save_none",
            datasets=["vaswani"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        def pipeline_gen(context):
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen)  # No save_dir

        # Check that pt.Experiment was called WITHOUT save_dir
        mock_pt_experiment.assert_called_once()
        call_kwargs = mock_pt_experiment.call_args[1]
        assert "save_dir" not in call_kwargs


class TestSaveDirKwargsIsolation:
    """Tests to verify save_dir kwargs are properly isolated between datasets."""

    def test_sequential_mode_multiple_datasets_get_correct_save_dir(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        cleanup_suite_registry,
    ):
        """Sequential mode: each dataset gets its own save_dir in pt.Experiment calls."""
        with patch("ir_datasets.docs_parent_id") as mock_parent:
            mock_parent.return_value = "shared_corpus"

            test_suite = Suite.register(
                "test_seq_multi_ds",
                datasets=["vaswani", "vaswani"],
                names=["dataset_a", "dataset_b"],
                metadata={"official_measures": [nDCG @ 10]},
            )

            save_path = os.path.join(temp_dir, "results")

            def pipeline_gen(context):
                yield DummyTransformer(), "system_1"
                yield DummyTransformer(), "system_2"

            test_suite(pipeline_gen, save_dir=save_path)  # No baseline = sequential

            # Verify pt.Experiment was called 4 times (2 pipelines x 2 datasets)
            assert mock_pt_experiment.call_count == 4

            # Verify each call got the correct save_dir
            # Sequential mode: for each pipeline, iterate through all datasets
            expected_calls = [
                os.path.join(save_path, "dataset_a"),  # pipeline1, ds_a
                os.path.join(save_path, "dataset_b"),  # pipeline1, ds_b
                os.path.join(save_path, "dataset_a"),  # pipeline2, ds_a
                os.path.join(save_path, "dataset_b"),  # pipeline2, ds_b
            ]

            for i, call in enumerate(mock_pt_experiment.call_args_list):
                actual_save_dir = call[1].get("save_dir")
                assert actual_save_dir == expected_calls[i], (
                    f"Call {i}: expected {expected_calls[i]}, got {actual_save_dir}"
                )

    def test_grouped_mode_multiple_datasets_get_correct_save_dir(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        cleanup_suite_registry,
    ):
        """Grouped mode: each dataset gets its own save_dir in pt.Experiment calls."""
        with patch("ir_datasets.docs_parent_id") as mock_parent:
            mock_parent.return_value = "shared_corpus"

            test_suite = Suite.register(
                "test_grp_multi_ds",
                datasets=["vaswani", "vaswani"],
                names=["dataset_a", "dataset_b"],
                metadata={"official_measures": [nDCG @ 10]},
            )

            save_path = os.path.join(temp_dir, "results")

            def pipeline_gen(context):
                yield DummyTransformer(), "system_1"
                yield DummyTransformer(), "system_2"

            test_suite(pipeline_gen, save_dir=save_path, baseline=0)  # baseline = grouped

            # Grouped mode: pt.Experiment called once per dataset, all pipelines together
            assert mock_pt_experiment.call_count == 2

            # Verify each call got the correct save_dir
            expected_dirs = [
                os.path.join(save_path, "dataset_a"),
                os.path.join(save_path, "dataset_b"),
            ]

            for i, call in enumerate(mock_pt_experiment.call_args_list):
                actual_save_dir = call[1].get("save_dir")
                assert actual_save_dir == expected_dirs[i], (
                    f"Call {i}: expected {expected_dirs[i]}, got {actual_save_dir}"
                )

    def test_three_datasets_all_get_unique_save_dirs(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        cleanup_suite_registry,
    ):
        """With 3 datasets sharing corpus, each gets unique save_dir."""
        with patch("ir_datasets.docs_parent_id") as mock_parent:
            mock_parent.return_value = "shared_corpus"

            test_suite = Suite.register(
                "test_three_ds",
                datasets=["vaswani", "vaswani", "vaswani"],
                names=["ds_alpha", "ds_beta", "ds_gamma"],
                metadata={"official_measures": [nDCG @ 10]},
            )

            save_path = os.path.join(temp_dir, "results")

            def pipeline_gen(context):
                yield DummyTransformer(), "system"

            test_suite(pipeline_gen, save_dir=save_path)

            assert mock_pt_experiment.call_count == 3

            save_dirs = [
                call[1].get("save_dir") for call in mock_pt_experiment.call_args_list
            ]
            expected = [
                os.path.join(save_path, "ds_alpha"),
                os.path.join(save_path, "ds_beta"),
                os.path.join(save_path, "ds_gamma"),
            ]
            assert save_dirs == expected


# ---------- index_dir Tests ----------


class TestIndexDirUnit:
    """Unit tests for index_dir functionality."""

    def test_index_dir_creates_corpus_subdirectory(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """Verify os.makedirs is called with corpus-specific path."""
        test_suite = Suite.register(
            "test_index_subdir",
            datasets=["vaswani"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        index_path = os.path.join(temp_dir, "indices")

        def pipeline_gen(context):
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen, index_dir=index_path)

        # Verify corpus subdirectory was created
        expected_dir = os.path.join(index_path, "vaswani")
        assert os.path.exists(expected_dir), f"Expected directory {expected_dir} to exist"

    def test_index_dir_formats_slashes_to_dashes(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        cleanup_suite_registry,
    ):
        """Corpus IDs with slashes become dashes in path."""
        # Mock docs_parent_id to return a corpus ID with slashes
        with patch("ir_datasets.docs_parent_id") as mock_parent:
            mock_parent.return_value = "beir/corpus/arguana"

            test_suite = Suite.register(
                "test_index_slashes",
                datasets=["vaswani"],
                metadata={"official_measures": [nDCG @ 10]},
            )

            index_path = os.path.join(temp_dir, "indices")

            def pipeline_gen(context):
                yield DummyTransformer(), "test_system"

            test_suite(pipeline_gen, index_dir=index_path)

            # Slash should be converted to dash
            expected_dir = os.path.join(index_path, "beir-corpus-arguana")
            assert os.path.exists(expected_dir), f"Expected directory {expected_dir} to exist"

    def test_index_dir_lowercases_corpus_id(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        cleanup_suite_registry,
    ):
        """Corpus IDs are lowercased in path."""
        with patch("ir_datasets.docs_parent_id") as mock_parent:
            mock_parent.return_value = "VASWANI"

            test_suite = Suite.register(
                "test_index_lowercase",
                datasets=["vaswani"],
                metadata={"official_measures": [nDCG @ 10]},
            )

            index_path = os.path.join(temp_dir, "indices")

            def pipeline_gen(context):
                yield DummyTransformer(), "test_system"

            test_suite(pipeline_gen, index_dir=index_path)

            # Corpus ID should be lowercased
            expected_dir = os.path.join(index_path, "vaswani")
            assert os.path.exists(expected_dir), f"Expected directory {expected_dir} to exist"

    def test_index_dir_passed_to_context(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """Verify DatasetContext receives correct path."""
        test_suite = Suite.register(
            "test_index_context",
            datasets=["vaswani"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        index_path = os.path.join(temp_dir, "indices")
        captured_context_path = []

        def pipeline_gen(context):
            captured_context_path.append(context.path)
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen, index_dir=index_path)

        # Context path should match expected index path
        expected_path = os.path.join(index_path, "vaswani")
        assert len(captured_context_path) == 1
        assert captured_context_path[0] == expected_path

    def test_index_dir_none_uses_temp(
        self,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """When index_dir is None, DatasetContext uses temp directory."""
        test_suite = Suite.register(
            "test_index_temp",
            datasets=["vaswani"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        captured_context_path = []

        def pipeline_gen(context):
            captured_context_path.append(context.path)
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen)  # No index_dir

        # Context path should be a temp directory
        assert len(captured_context_path) == 1
        # Temp directories typically contain "tmp" or "temp"
        assert "tmp" in captured_context_path[0].lower() or "temp" in captured_context_path[0].lower()

    def test_index_dir_shared_across_datasets_same_corpus(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        cleanup_suite_registry,
    ):
        """Datasets sharing same corpus use same index directory."""
        # Mock docs_parent_id to return same corpus for different datasets
        with patch("ir_datasets.docs_parent_id") as mock_parent:
            mock_parent.return_value = "shared_corpus"

            test_suite = Suite.register(
                "test_index_shared",
                datasets=["vaswani", "vaswani"],
                names=["dataset_a", "dataset_b"],
                metadata={"official_measures": [nDCG @ 10]},
            )

            index_path = os.path.join(temp_dir, "indices")
            captured_context_paths = []

            def pipeline_gen(context):
                captured_context_paths.append(context.path)
                yield DummyTransformer(), "test_system"

            test_suite(pipeline_gen, index_dir=index_path)

            # Both datasets should use the same corpus directory
            expected_path = os.path.join(index_path, "shared_corpus")
            # All captured paths should be the same
            assert all(p == expected_path for p in captured_context_paths)


# ---------- Combined Tests ----------


class TestSaveAndIndexDirCombined:
    """Tests for using both save_dir and index_dir together."""

    def test_both_save_and_index_dir(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """Verify both save_dir and index_dir work together correctly."""
        test_suite = Suite.register(
            "test_both_dirs",
            datasets=["vaswani"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        save_path = os.path.join(temp_dir, "results")
        index_path = os.path.join(temp_dir, "indices")
        captured_context_path = []

        def pipeline_gen(context):
            captured_context_path.append(context.path)
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen, save_dir=save_path, index_dir=index_path)

        # Verify save_dir subdirectory was created
        assert os.path.exists(os.path.join(save_path, "vaswani"))

        # Verify index_dir subdirectory was created
        assert os.path.exists(os.path.join(index_path, "vaswani"))

        # Verify context received correct index path
        assert captured_context_path[0] == os.path.join(index_path, "vaswani")

        # Verify pt.Experiment received save_dir
        call_kwargs = mock_pt_experiment.call_args[1]
        assert call_kwargs.get("save_dir") == os.path.join(save_path, "vaswani")

    def test_nested_paths_created(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        cleanup_suite_registry,
    ):
        """Verify deeply nested paths are created correctly."""
        test_suite = Suite.register(
            "test_nested_paths",
            datasets=["vaswani"],
            metadata={"official_measures": [nDCG @ 10]},
        )

        # Use deeply nested paths
        save_path = os.path.join(temp_dir, "a", "b", "c", "results")
        index_path = os.path.join(temp_dir, "x", "y", "z", "indices")

        def pipeline_gen(context):
            yield DummyTransformer(), "test_system"

        test_suite(pipeline_gen, save_dir=save_path, index_dir=index_path)

        # Verify nested directories were created
        assert os.path.exists(os.path.join(save_path, "vaswani"))
        assert os.path.exists(os.path.join(index_path, "vaswani"))


# ---------- DatasetContext Tests ----------


class TestDatasetContext:
    """Unit tests for DatasetContext functionality."""

    def test_context_stores_dataset(self, mock_dataset):
        """Verify DatasetContext stores the dataset reference."""
        context = DatasetContext(mock_dataset, path="/tmp/test")
        assert context.dataset is mock_dataset

    def test_context_uses_provided_path(self, mock_dataset):
        """Verify DatasetContext uses the provided path."""
        context = DatasetContext(mock_dataset, path="/custom/path")
        assert context.path == "/custom/path"

    def test_context_creates_temp_path_when_none(self, mock_dataset):
        """Verify DatasetContext creates temp directory when path is None."""
        context = DatasetContext(mock_dataset, path=None)
        # Should contain temp indicator and formatted dataset ID
        assert "tmp" in context.path.lower() or "temp" in context.path.lower()
        assert "vaswani" in context.path

    def test_context_formats_irds_id_in_temp_path(self, mock_dataset):
        """Verify slashes in irds_id are replaced in temp path suffix."""
        mock_dataset._irds_id = "beir/arguana"
        context = DatasetContext(mock_dataset, path=None)
        # Slashes should be replaced with dashes in the suffix
        assert "beir-arguana" in context.path


class TestTextLoader:
    """Unit tests for DatasetContext.text_loader() method."""

    def test_text_loader_delegates_to_dataset(self, mock_dataset, mock_text_loader):
        """Verify text_loader() calls dataset.text_loader()."""
        context = DatasetContext(mock_dataset, path="/tmp/test")

        result = context.text_loader()

        mock_dataset.text_loader.assert_called_once_with(fields="*")
        assert result is mock_text_loader

    def test_text_loader_passes_fields_parameter(self, mock_dataset, mock_text_loader):
        """Verify text_loader() passes fields parameter correctly."""
        context = DatasetContext(mock_dataset, path="/tmp/test")

        # Test with list of fields
        context.text_loader(fields=["title", "body"])
        mock_dataset.text_loader.assert_called_with(fields=["title", "body"])

        # Test with single field
        mock_dataset.text_loader.reset_mock()
        context.text_loader(fields="text")
        mock_dataset.text_loader.assert_called_with(fields="text")

        # Test with wildcard
        mock_dataset.text_loader.reset_mock()
        context.text_loader(fields="*")
        mock_dataset.text_loader.assert_called_with(fields="*")

    def test_text_loader_default_fields_is_wildcard(self, mock_dataset, mock_text_loader):
        """Verify text_loader() uses '*' as default for fields."""
        context = DatasetContext(mock_dataset, path="/tmp/test")

        context.text_loader()

        mock_dataset.text_loader.assert_called_once_with(fields="*")

    def test_text_loader_returns_transformer(self, mock_dataset, mock_text_loader):
        """Verify text_loader() returns a transformer-like object."""
        context = DatasetContext(mock_dataset, path="/tmp/test")

        loader = context.text_loader()

        # Should be callable with transform method (like PyTerrier Transformer)
        assert hasattr(loader, "transform")
        # Verify it can process a DataFrame
        input_df = pd.DataFrame({"qid": ["1"], "docno": ["d1"]})
        result = loader.transform(input_df)
        assert isinstance(result, pd.DataFrame)

    def test_text_loader_in_pipeline_context(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        mock_dataset,
        mock_text_loader,
        vaswani_suite,
    ):
        """Verify text_loader works correctly within a suite pipeline."""
        captured_loaders = []

        def pipeline_gen(context):
            # Capture the text_loader for verification
            loader = context.text_loader(fields=["text"])
            captured_loaders.append(loader)
            yield DummyTransformer(), "test_system"

        vaswani_suite(pipeline_gen, index_dir=os.path.join(temp_dir, "indices"))

        # Verify text_loader was called with correct fields
        assert len(captured_loaders) == 1
        mock_dataset.text_loader.assert_called_with(fields=["text"])


class TestGetCorpusIter:
    """Unit tests for DatasetContext.get_corpus_iter() method."""

    def test_get_corpus_iter_delegates_to_dataset(self, mock_dataset):
        """Verify get_corpus_iter() calls dataset.get_corpus_iter()."""
        context = DatasetContext(mock_dataset, path="/tmp/test")

        result = context.get_corpus_iter()

        mock_dataset.get_corpus_iter.assert_called_once_with()
        # Should return an iterator
        assert hasattr(result, "__iter__")

    def test_get_corpus_iter_passes_kwargs(self, mock_dataset):
        """Verify get_corpus_iter() passes keyword arguments."""
        context = DatasetContext(mock_dataset, path="/tmp/test")

        context.get_corpus_iter(verbose=True, fields=["docno", "text"])

        mock_dataset.get_corpus_iter.assert_called_once_with(
            verbose=True, fields=["docno", "text"]
        )

    def test_get_corpus_iter_returns_documents(self, mock_dataset):
        """Verify get_corpus_iter() returns document iterator."""
        context = DatasetContext(mock_dataset, path="/tmp/test")

        docs = list(context.get_corpus_iter())

        assert len(docs) == 2
        assert docs[0]["docno"] == "d1"
        assert docs[1]["docno"] == "d2"

    def test_get_corpus_iter_in_pipeline_context(
        self,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
        mock_dataset,
        vaswani_suite,
    ):
        """Verify get_corpus_iter works correctly within a suite pipeline."""
        captured_docs = []

        def pipeline_gen(context):
            # Capture corpus documents for verification
            for doc in context.get_corpus_iter():
                captured_docs.append(doc)
            yield DummyTransformer(), "test_system"

        vaswani_suite(pipeline_gen, index_dir=os.path.join(temp_dir, "indices"))

        # Verify corpus iteration worked
        assert len(captured_docs) == 2
        assert captured_docs[0]["docno"] == "d1"
