"""Aggregation of the results a suite collects.

A suite yields one frame per experiment and concatenates them; these functions
turn that table into the one a caller reads, in particular the ``Overall`` rows
that summarise a system across datasets.
"""

from __future__ import annotations

from logging import getLogger
from typing import Sequence

import numpy as np
import pandas as pd

from suiteeval.utility import geometric_mean

logger = getLogger(__name__)

#: Value of the ``dataset`` column on a summary row.
OVERALL = "Overall"

#: Added to every value of a metric before taking its geometric mean, when any
#: of them is zero or negative and would otherwise collapse or invalidate it.
EPSILON = 1e-12


def has_overall(results: pd.DataFrame) -> bool:
    """Whether ``results`` already carries its summary rows."""
    return "dataset" not in results.columns or OVERALL in results["dataset"].values


def mean_per_dataset(
    results: pd.DataFrame, measure_columns: Sequence[str]
) -> pd.DataFrame:
    """
    Average repeated runs of one system on one dataset.

    Args:
        results: A results table with ``dataset`` and ``name`` columns.
        measure_columns: The columns holding measure values.

    Returns:
        pandas.DataFrame: One row per ``(dataset, name)``.
    """
    return (
        results.groupby(["dataset", "name"], dropna=False)[list(measure_columns)]
        .mean()
        .reset_index()
    )


def overall_value(values: pd.Series) -> float:
    """
    The geometric mean of one metric across datasets.

    Non-numeric and missing entries are dropped. A zero would collapse the
    product and a negative would invalidate the root, so every value is shifted
    by :data:`EPSILON` when either appears.

    Args:
        values: The per-dataset values of one metric for one system.

    Returns:
        float: The geometric mean.
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna().values
    if np.any(numeric < 0):
        logger.warning(
            "Negative metric values in an Overall row; its geometric mean is "
            "not meaningful."
        )
    if np.any(numeric <= 0):
        numeric = numeric + EPSILON
    return geometric_mean(numeric)


def append_overall(
    results: pd.DataFrame, measure_columns: Sequence[str]
) -> pd.DataFrame:
    """
    Append one ``Overall`` row per system, summarising it across datasets.

    Repeated runs are averaged per dataset first, then combined across datasets
    with a geometric mean. Idempotent: a table that already has ``Overall`` rows
    is returned unchanged, as is one with no measure columns.

    Args:
        results: The concatenated results table.
        measure_columns: The columns holding measure values.

    Returns:
        pandas.DataFrame: ``results`` with the summary rows appended.
    """
    if has_overall(results) or not len(measure_columns):
        return results

    per_dataset = mean_per_dataset(results, measure_columns)
    rows = [
        {
            "dataset": OVERALL,
            "name": name,
            **{column: overall_value(group[column]) for column in measure_columns},
        }
        for name, group in per_dataset.groupby("name", dropna=False)
    ]
    return pd.concat([results, pd.DataFrame(rows)], ignore_index=True)


__all__ = [
    "EPSILON",
    "OVERALL",
    "append_overall",
    "has_overall",
    "mean_per_dataset",
    "overall_value",
]
