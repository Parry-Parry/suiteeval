"""Unit tests for the helpers behind ``Suite.register``."""

from __future__ import annotations

import logging

import pytest
from ir_measures import nDCG

from suiteeval.suite.base import Suite, SuiteMeta
from suiteeval.suite.registration import (
    dataset_map,
    is_per_dataset,
    normalise_metadata,
)


class TestDatasetMap:
    def test_identifiers_are_their_own_display_names(self):
        assert dataset_map(["a", "b"]) == {"a": "a", "b": "b"}

    def test_names_are_paired_in_order(self):
        assert dataset_map(["ds/a", "ds/b"], ["a", "b"]) == {"a": "ds/a", "b": "ds/b"}

    def test_empty_names_falls_back_to_identifiers(self):
        assert dataset_map(["a"], []) == {"a": "a"}

    def test_mismatched_names_truncate_and_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            mapping = dataset_map(["ds/a", "ds/b"], ["only-one"])

        assert mapping == {"only-one": "ds/a"}
        assert "only the first 1" in caplog.text


class TestIsPerDataset:
    @pytest.mark.parametrize(
        "metadata, expected",
        [
            ({}, False),
            ({"description": "text", "official_measures": [nDCG @ 10]}, False),
            ({"a": {"official_measures": []}}, True),
            ({"a": {}, "b": {}}, True),
        ],
    )
    def test_shape_detection(self, metadata, expected):
        assert is_per_dataset(metadata) is expected

    def test_mixed_shape_is_per_dataset_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert is_per_dataset({"a": {}, "description": "text"}) is True

        assert "Ambiguous" in caplog.text


class TestNormaliseMetadata:
    def test_none_gives_an_empty_dict_per_dataset(self):
        assert normalise_metadata(None, ["a", "b"]) == {"a": {}, "b": {}}

    def test_flat_metadata_is_kept_flat(self):
        """Fanning it out hid keys like `description` under a dataset name."""
        flat = {"description": "text"}

        assert normalise_metadata(flat, ["a", "b"]) is flat

    def test_per_dataset_metadata_passes_through(self):
        per_dataset = {"a": {"official_measures": [nDCG @ 10]}}

        assert normalise_metadata(per_dataset, ["a"]) == per_dataset

    def test_list_metadata_is_aligned_with_names(self):
        assert normalise_metadata([{"x": 1}, {"x": 2}], ["a", "b"]) == {
            "a": {"x": 1},
            "b": {"x": 2},
        }

    def test_mismatched_list_is_rejected(self):
        with pytest.raises(ValueError, match="must match number of datasets"):
            normalise_metadata([{"x": 1}], ["a", "b"])

    def test_unsupported_type_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported metadata type"):
            normalise_metadata("nonsense", ["a"])  # type: ignore[arg-type]

    def test_entries_for_unknown_datasets_are_reported(self, caplog):
        with caplog.at_level(logging.WARNING):
            normalise_metadata({"typo": {"official_measures": []}}, ["a"])

        assert "does not declare" in caplog.text


class TestRegistry:
    def test_registering_twice_returns_the_same_instance(self, cleanup_suite_registry):
        first = Suite.register("test_twice", datasets=["vaswani"])
        second = Suite.register("test_twice", datasets=["vaswani"])

        assert first is second

    def test_registered_lists_the_names(self, cleanup_suite_registry):
        Suite.register("test_listed", datasets=["vaswani"])

        assert "test_listed" in SuiteMeta.registered()

    def test_forget_frees_the_name(self, cleanup_suite_registry):
        first = Suite.register("test_forgotten", datasets=["vaswani"])
        SuiteMeta.forget("test_forgotten")
        second = Suite.register("test_forgotten", datasets=["vaswani"])

        assert first is not second

    def test_forget_ignores_unknown_names(self):
        SuiteMeta.forget("test_never_registered")

    def test_same_named_classes_do_not_share_an_instance(self, mock_dataset):
        def make():
            class _Clashing(Suite):
                _datasets = ["vaswani"]
                _measures = [nDCG @ 10]

            return _Clashing()

        assert make() is not make()

    def test_arguments_to_an_existing_singleton_are_reported(
        self, cleanup_suite_registry, caplog
    ):
        suite = Suite.register("test_singleton_args", datasets=["vaswani"])

        with caplog.at_level(logging.WARNING):
            again = type(suite)("unexpected")

        assert again is suite
        assert "singleton" in caplog.text
