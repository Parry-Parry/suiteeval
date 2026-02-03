"""Unit tests for geometric_mean and compute_overall_mean functionality.

These tests verify:
1. geometric_mean function handles edge cases correctly
2. compute_overall_mean auto-detects all metric columns
3. No duplicate Overall rows when called multiple times
4. BEIR and NanoBEIR produce correct results without duplicates
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from ir_measures import nDCG

from suiteeval.utility import geometric_mean
from suiteeval.suite.base import Suite, SuiteMeta


# ---------- Fixtures ----------


@pytest.fixture
def cleanup_suite_registry():
    """Clean up suite registry after test."""
    yield
    to_remove = [k for k in SuiteMeta._classes if k.startswith("test_")]
    for k in to_remove:
        del SuiteMeta._classes[k]
        if k in SuiteMeta._instances:
            del SuiteMeta._instances[k]


@pytest.fixture
def mock_suite(cleanup_suite_registry):
    """Create a mock suite for testing compute_overall_mean."""
    suite = Suite.register(
        "test_compute_overall_mean_suite",
        datasets=["vaswani"],
        metadata={"official_measures": [nDCG @ 10]},
    )
    return suite


@pytest.fixture
def sample_results_single_metric():
    """Sample results with a single metric."""
    return pd.DataFrame({
        "dataset": ["ds1", "ds1", "ds2", "ds2"],
        "name": ["model_a", "model_b", "model_a", "model_b"],
        "nDCG@10": [0.5, 0.6, 0.7, 0.8],
    })


@pytest.fixture
def sample_results_multiple_metrics():
    """Sample results with multiple metrics."""
    return pd.DataFrame({
        "dataset": ["ds1", "ds1", "ds2", "ds2"],
        "name": ["model_a", "model_b", "model_a", "model_b"],
        "nDCG@10": [0.5, 0.6, 0.7, 0.8],
        "AP@10": [0.4, 0.5, 0.6, 0.7],
        "R@100": [0.8, 0.85, 0.9, 0.95],
    })


# ---------- Tests for geometric_mean ----------


class TestGeometricMean:
    """Unit tests for geometric_mean function."""

    def test_empty_sequence(self):
        """Empty sequence returns 0.0."""
        assert geometric_mean([]) == 0.0

    def test_single_element(self):
        """Single element returns itself."""
        assert geometric_mean([5.0]) == 5.0
        assert geometric_mean([0.1]) == pytest.approx(0.1)

    def test_multiple_identical_values(self):
        """Multiple identical values returns that value."""
        assert geometric_mean([3.0, 3.0, 3.0]) == pytest.approx(3.0)

    def test_normal_positive_values(self):
        """Normal positive values compute correct geometric mean."""
        # geometric_mean([2, 8]) = sqrt(2*8) = sqrt(16) = 4
        assert geometric_mean([2.0, 8.0]) == pytest.approx(4.0)
        # geometric_mean([1, 2, 4]) = (1*2*4)^(1/3) = 8^(1/3) = 2
        assert geometric_mean([1.0, 2.0, 4.0]) == pytest.approx(2.0)

    def test_values_near_zero(self):
        """Very small positive values are handled."""
        result = geometric_mean([1e-10, 1e-10])
        assert result == pytest.approx(1e-10)


# ---------- Tests for compute_overall_mean ----------


class TestComputeOverallMean:
    """Unit tests for compute_overall_mean method."""

    def test_single_model_single_dataset(self, mock_suite):
        """Single model, single dataset produces one Overall row."""
        results = pd.DataFrame({
            "dataset": ["ds1"],
            "name": ["model_a"],
            "nDCG@10": [0.5],
        })

        output = mock_suite.compute_overall_mean(results)

        overall_rows = output[output["dataset"] == "Overall"]
        assert len(overall_rows) == 1
        assert overall_rows.iloc[0]["name"] == "model_a"
        assert overall_rows.iloc[0]["nDCG@10"] == pytest.approx(0.5)

    def test_multiple_models_multiple_datasets(self, mock_suite, sample_results_single_metric):
        """Multiple models across multiple datasets aggregate correctly."""
        output = mock_suite.compute_overall_mean(sample_results_single_metric)

        overall_rows = output[output["dataset"] == "Overall"]
        assert len(overall_rows) == 2  # One per model

        # model_a: gmean(0.5, 0.7) = sqrt(0.35) ~ 0.5916
        model_a_overall = overall_rows[overall_rows["name"] == "model_a"]
        assert len(model_a_overall) == 1
        assert model_a_overall.iloc[0]["nDCG@10"] == pytest.approx(
            geometric_mean([0.5, 0.7])
        )

    def test_auto_detects_all_metric_columns(self, mock_suite, sample_results_multiple_metrics):
        """CRITICAL: All metric columns should get Overall values, not just defaults."""
        output = mock_suite.compute_overall_mean(sample_results_multiple_metrics)

        overall_rows = output[output["dataset"] == "Overall"]

        # Should have Overall values for ALL metrics
        for metric in ["nDCG@10", "AP@10", "R@100"]:
            assert metric in overall_rows.columns
            # Verify values are computed (not NaN)
            assert overall_rows[metric].notna().all()

    def test_no_duplicates_when_called_twice(self, mock_suite, sample_results_single_metric):
        """CRITICAL: Calling compute_overall_mean twice should not create duplicates."""
        # First call
        output1 = mock_suite.compute_overall_mean(sample_results_single_metric)
        overall_count_1 = len(output1[output1["dataset"] == "Overall"])

        # Second call on the same results
        output2 = mock_suite.compute_overall_mean(output1)
        overall_count_2 = len(output2[output2["dataset"] == "Overall"])

        # Should have same number of Overall rows
        assert overall_count_1 == overall_count_2
        assert overall_count_2 == 2  # One per model

    def test_handles_empty_dataframe(self, mock_suite):
        """Empty DataFrame returns empty DataFrame."""
        results = pd.DataFrame(columns=["dataset", "name", "nDCG@10"])
        output = mock_suite.compute_overall_mean(results)
        # Should not raise, should return DataFrame (possibly empty)
        assert isinstance(output, pd.DataFrame)

    def test_computes_correct_geometric_mean_for_multiple_metrics(
        self, mock_suite, sample_results_multiple_metrics
    ):
        """Verify geometric mean computation is correct for each metric."""
        output = mock_suite.compute_overall_mean(sample_results_multiple_metrics)

        overall_rows = output[output["dataset"] == "Overall"]
        model_a_overall = overall_rows[overall_rows["name"] == "model_a"].iloc[0]

        # model_a values: ds1=[0.5, 0.4, 0.8], ds2=[0.7, 0.6, 0.9]
        assert model_a_overall["nDCG@10"] == pytest.approx(geometric_mean([0.5, 0.7]))
        assert model_a_overall["AP@10"] == pytest.approx(geometric_mean([0.4, 0.6]))
        assert model_a_overall["R@100"] == pytest.approx(geometric_mean([0.8, 0.9]))
