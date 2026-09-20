"""Unit tests for the dataset declaration normaliser."""

from __future__ import annotations

import pytest

from suiteeval.suite.datasets import (
    DatasetSpec,
    irds_id_of,
    is_dataset_like,
    normalise_datasets,
    validate_dataset,
)


class FakeDataset:
    """Minimal stand-in for a PyTerrier dataset."""

    def __init__(self, irds_id: str):
        self._irds_id = irds_id

    def get_topics(self, *args, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError

    def get_qrels(self, *args, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError


class TestNormaliseDatasets:
    def test_list_of_ids_uses_the_id_as_display_name(self):
        specs = normalise_datasets(["vaswani", "beir/nfcorpus/test"])

        assert [spec.name for spec in specs] == ["vaswani", "beir/nfcorpus/test"]
        assert [spec.irds_id for spec in specs] == ["vaswani", "beir/nfcorpus/test"]
        assert [spec.key for spec in specs] == ["vaswani", "beir/nfcorpus/test"]

    def test_mapping_keeps_declaration_order_and_display_names(self):
        specs = normalise_datasets({"b": "ds/b", "a": "ds/a"})

        assert [spec.name for spec in specs] == ["b", "a"]
        assert [spec.irds_id for spec in specs] == ["ds/b", "ds/a"]

    def test_dataset_objects_resolve_their_irds_id(self):
        dataset = FakeDataset("vaswani")

        (spec,) = normalise_datasets([dataset])

        assert spec.name == "vaswani"
        assert spec.irds_id == "vaswani"
        assert spec.ref is dataset

    def test_named_dataset_object_keeps_its_display_name(self):
        (spec,) = normalise_datasets({"mine": FakeDataset("vaswani")})

        assert spec.name == "mine"
        assert spec.irds_id == "vaswani"

    def test_as_item_round_trips_the_declaration(self):
        (spec,) = normalise_datasets({"mine": "vaswani"})

        assert spec.as_item() == ("mine", "vaswani")

    def test_specs_are_frozen(self):
        (spec,) = normalise_datasets(["vaswani"])

        with pytest.raises(Exception):
            spec.name = "other"  # type: ignore[misc]

    @pytest.mark.parametrize("empty", [[], {}, None])
    def test_empty_declaration_is_rejected(self, empty):
        with pytest.raises(AssertionError, match="at least one dataset"):
            normalise_datasets(empty)

    def test_unsupported_container_is_rejected(self):
        with pytest.raises(AssertionError, match="dict\\[name->id\\] or a list"):
            normalise_datasets(("vaswani",))

    def test_non_string_keys_are_rejected(self):
        with pytest.raises(AssertionError, match="keys must be strings"):
            normalise_datasets({1: "vaswani"})

    def test_invalid_reference_names_its_position(self):
        with pytest.raises(AssertionError, match=r"_datasets\[1\]"):
            normalise_datasets(["vaswani", 42])

    def test_invalid_reference_names_its_key(self):
        with pytest.raises(AssertionError, match=r"_datasets\['broken'\]"):
            normalise_datasets({"broken": 42})


class TestHelpers:
    def test_is_dataset_like_requires_every_attribute(self):
        assert is_dataset_like(FakeDataset("vaswani"))
        assert not is_dataset_like(object())

    def test_irds_id_of_accepts_both_forms(self):
        assert irds_id_of("vaswani") == "vaswani"
        assert irds_id_of(FakeDataset("vaswani")) == "vaswani"

    def test_validate_dataset_passes_both_forms(self):
        validate_dataset("vaswani", "0")
        validate_dataset(FakeDataset("vaswani"), "0")

    def test_dataset_spec_is_hashable(self):
        spec = DatasetSpec(key="a", ref="ds/a", name="a", irds_id="ds/a")

        assert {spec}
