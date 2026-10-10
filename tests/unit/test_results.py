"""Unit tests for result aggregation."""

from __future__ import annotations

import gzip
import os

import pandas as pd
import pytest

from suiteeval.suite.results import (
    append_overall,
    has_overall,
    mean_per_dataset,
    overall_value,
)
from suiteeval.suite.runfiles import read_run, replay_run


def results_frame(rows):
    return pd.DataFrame(rows)


class TestHasOverall:
    def test_a_frame_without_a_dataset_column_counts_as_done(self):
        assert has_overall(pd.DataFrame({"name": ["sys"]})) is True

    def test_detects_existing_overall_rows(self):
        frame = results_frame([{"dataset": "Overall", "name": "sys", "m": 1.0}])

        assert has_overall(frame) is True

    def test_a_plain_frame_is_not_done(self):
        frame = results_frame([{"dataset": "a", "name": "sys", "m": 1.0}])

        assert has_overall(frame) is False


class TestOverallValue:
    def test_geometric_mean_of_positive_values(self):
        assert overall_value(pd.Series([2.0, 8.0])) == pytest.approx(4.0)

    def test_non_numeric_entries_are_dropped(self):
        assert overall_value(pd.Series([2.0, "n/a", 8.0])) == pytest.approx(4.0)

    def test_missing_entries_are_dropped(self):
        assert overall_value(pd.Series([2.0, None, 8.0])) == pytest.approx(4.0)

    def test_a_zero_does_not_collapse_the_others(self):
        value = overall_value(pd.Series([0.0, 0.5]))

        assert value > 0.0
        assert value == pytest.approx((1e-12 * 0.5) ** 0.5, rel=1e-6)

    def test_only_the_offending_values_are_floored(self):
        """The rest are left exact, rather than every value being shifted."""
        floored = overall_value(pd.Series([0.0, 4.0, 9.0]))

        assert floored == pytest.approx((1e-12 * 4.0 * 9.0) ** (1 / 3))

    def test_positive_values_are_untouched_by_the_floor(self):
        assert overall_value(pd.Series([4.0, 9.0])) == pytest.approx(6.0)

    def test_negative_values_are_reported(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            overall_value(pd.Series([-1.0, 1.0]))

        assert "Negative metric values" in caplog.text


class TestMeanPerDataset:
    def test_repeated_runs_of_one_system_are_averaged(self):
        frame = results_frame(
            [
                {"dataset": "a", "name": "sys", "m": 0.2},
                {"dataset": "a", "name": "sys", "m": 0.4},
                {"dataset": "b", "name": "sys", "m": 0.9},
            ]
        )

        averaged = mean_per_dataset(frame, ["m"])

        assert len(averaged) == 2
        assert averaged.set_index("dataset").loc["a", "m"] == pytest.approx(0.3)


class TestAppendOverall:
    def test_one_row_per_system(self):
        frame = results_frame(
            [
                {"dataset": "a", "name": "x", "m": 0.4},
                {"dataset": "b", "name": "x", "m": 0.9},
                {"dataset": "a", "name": "y", "m": 0.1},
                {"dataset": "b", "name": "y", "m": 0.1},
            ]
        )

        out = append_overall(frame, ["m"])
        overall = out[out["dataset"] == "Overall"]

        assert len(overall) == 2
        assert set(overall["name"]) == {"x", "y"}
        assert overall.set_index("name").loc["y", "m"] == pytest.approx(0.1)

    def test_is_idempotent(self):
        frame = results_frame([{"dataset": "a", "name": "x", "m": 0.4}])

        once = append_overall(frame, ["m"])
        twice = append_overall(once, ["m"])

        assert len(once) == len(twice)

    def test_a_frame_with_no_measure_columns_is_untouched(self):
        frame = results_frame([{"dataset": "a", "name": "x"}])

        assert append_overall(frame, []).equals(frame)

    def test_the_original_rows_survive(self):
        frame = results_frame([{"dataset": "a", "name": "x", "m": 0.4}])

        out = append_overall(frame, ["m"])

        assert list(out[out["dataset"] == "a"]["m"]) == [0.4]


class TestRunFiles:
    def write(self, directory, text):
        path = os.path.join(directory, "run.res.gz")
        with gzip.open(path, "wt") as handle:
            handle.write(text)
        return path

    def test_read_run_keeps_result_columns_with_string_ids(self, temp_dir):
        path = self.write(temp_dir, "1 Q0 d1 0 2.0 sys\n1 Q0 d2 1 1.0 sys\n")

        run = read_run(path)

        assert list(run.columns) == ["qid", "docno", "score", "rank"]
        assert run["qid"].tolist() == ["1", "1"]
        assert run["score"].tolist() == [2.0, 1.0]

    def test_read_run_does_not_warn_about_copies(self, temp_dir, recwarn):
        path = self.write(temp_dir, "1 Q0 d1 0 2.0 sys\n")

        read_run(path)

        assert not [w for w in recwarn if "SettingWithCopy" in str(w.category)]

    def test_replay_run_yields_the_stored_ranking(self, temp_dir):
        path = self.write(temp_dir, "q1 Q0 d1 0 2.0 sys\nq1 Q0 d2 1 1.0 sys\n")

        replayed = replay_run(path)(pd.DataFrame({"qid": ["q1"], "query": ["a query"]}))

        assert replayed["docno"].tolist() == ["d1", "d2"]
