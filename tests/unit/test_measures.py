"""Unit tests for measure parsing and discovery."""

from __future__ import annotations

import pytest
from ir_measures import AP, Measure, nDCG

from suiteeval.suite.measures import (
    discover_measures,
    measures_for_dataset,
    parse_measures,
)


class TestParseMeasures:
    def test_measure_objects_pass_through_in_order(self):
        assert parse_measures([nDCG @ 10, AP]) == [nDCG @ 10, AP]

    def test_ir_measures_strings_are_parsed(self):
        parsed = parse_measures(["nDCG@10", "P@5"])

        assert all(isinstance(measure, Measure) for measure in parsed)
        assert str(parsed[0]) == "nDCG@10"

    def test_strings_and_objects_mix(self):
        assert parse_measures(["nDCG@10", AP]) == [nDCG @ 10, AP]

    def test_empty_input_gives_empty_output(self):
        assert parse_measures([]) == []

    def test_unrecognised_string_is_rejected(self):
        with pytest.raises(ValueError, match="Unrecognised measure"):
            parse_measures(["not-a-measure"])

    def test_non_measure_object_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid measure type"):
            parse_measures([object()])


class TestDiscoverMeasures:
    """``dataset_ids=None`` skips the ir_datasets lookup, keeping these local."""

    def test_global_metadata_wins_first(self):
        discovered = discover_measures(
            ["ds"], None, {"official_measures": [nDCG @ 10]}, [AP]
        )

        assert discovered == [nDCG @ 10]

    def test_per_dataset_metadata_is_aggregated_in_declaration_order(self):
        metadata = {
            "b": {"official_measures": [AP]},
            "a": {"official_measures": [nDCG @ 10]},
        }

        discovered = discover_measures(["a", "b"], None, metadata, [])

        assert discovered == [nDCG @ 10, AP]

    def test_duplicates_are_dropped_keeping_the_first(self):
        metadata = {
            "official_measures": [nDCG @ 10],
            "a": {"official_measures": [nDCG @ 10, AP]},
        }

        discovered = discover_measures(["a"], None, metadata, [])

        assert discovered == [nDCG @ 10, AP]

    def test_strings_in_metadata_are_parsed(self):
        discovered = discover_measures(
            ["ds"], None, {"official_measures": ["nDCG@10"]}, []
        )

        assert discovered == [nDCG @ 10]

    def test_falls_back_to_the_default(self):
        assert discover_measures(["ds"], None, {}, [AP]) == [AP]

    def test_non_dict_metadata_is_ignored(self):
        assert discover_measures(["ds"], None, "nonsense", [AP]) == [AP]

    def test_non_dict_per_dataset_entry_is_ignored(self):
        assert discover_measures(["ds"], None, {"ds": "nonsense"}, [AP]) == [AP]

    def test_unknown_dataset_id_does_not_propagate(self, caplog):
        discovered = discover_measures(
            ["ds"], {"ds": "no-such-dataset-anywhere"}, {}, [AP]
        )

        assert discovered == [AP]


class TestMeasuresForDataset:
    def test_a_list_applies_to_every_dataset(self):
        assert measures_for_dataset([nDCG @ 10], "anything", [AP]) == [nDCG @ 10]

    def test_a_mapping_is_looked_up_by_name(self):
        assert measures_for_dataset({"a": [AP]}, "a", [nDCG @ 10]) == [AP]

    def test_a_missing_mapping_entry_falls_back(self):
        assert measures_for_dataset({"a": [AP]}, "b", [nDCG @ 10]) == [nDCG @ 10]
