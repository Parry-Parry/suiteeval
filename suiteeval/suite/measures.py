"""Parsing and discovery of the measures a suite evaluates.

A suite may state its measures directly as ``_measures``, leave them in its
metadata under ``official_measures``, or leave them entirely to the
``ir_datasets`` documentation of its datasets. These functions are pure, so
what a suite will evaluate can be worked out without constructing one.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any, Sequence

import ir_datasets as irds
from ir_measures import Measure, parse_measure, parse_trec_measure

logger = getLogger(__name__)

#: Metadata key holding the measures a dataset or suite is scored on.
OFFICIAL_MEASURES = "official_measures"

MeasureLike = str | Measure


def parse_measures(measures: Sequence[MeasureLike]) -> list[Measure]:
    """
    Convert measure strings and :class:`ir_measures.Measure` objects to measures.

    Both the ``ir_measures`` syntax (``"nDCG@10"``) and the trec_eval syntax
    (``"ndcg_cut_10"``) are accepted.

    Args:
        measures: Measure strings and/or already-parsed measures.

    Returns:
        list[Measure]: The parsed measures, in the order given.

    Raises:
        ValueError: If a string cannot be parsed by either syntax, or if an
            entry is neither a string nor a measure.
    """
    parsed: list[Measure] = []
    for measure in measures:
        if isinstance(measure, Measure):
            parsed.append(measure)
        elif isinstance(measure, str):
            parsed.extend(parse_measure_string(measure))
        else:
            raise ValueError(f"Invalid measure type: {type(measure)}")
    return parsed


def parse_measure_string(measure: str) -> list[Measure]:
    """
    Parse one measure string, trying the ir_measures syntax then trec_eval.

    Args:
        measure: The measure string.

    Returns:
        list[Measure]: One measure for the ir_measures syntax, possibly several
            for a trec_eval name that expands to a family.

    Raises:
        ValueError: If neither syntax recognises the string.
    """
    for parser in (parse_measure, parse_trec_measure):
        try:
            result = parser(measure)
        except ValueError:
            continue
        return [result] if isinstance(result, Measure) else list(result)
    raise ValueError(f"Unrecognised measure string: {measure!r}")


def documented_measures(dataset_id: str) -> list[MeasureLike]:
    """
    The official measures ``ir_datasets`` documents for one dataset.

    Args:
        dataset_id: An IRDS dataset identifier.

    Returns:
        list[MeasureLike]: The documented measures, empty when the dataset is
            unknown locally or documents none.
    """
    documentation = getattr(irds.load(dataset_id), "documentation", lambda: None)()
    if not isinstance(documentation, dict):
        return []
    return list(documentation.get(OFFICIAL_MEASURES) or [])


def discover_measures(
    dataset_names: Sequence[str],
    dataset_ids: dict[str, str] | None,
    metadata: Any,
    default: Sequence[Measure],
) -> list[Measure]:
    """
    Aggregate the measures a suite should evaluate, in priority order.

    Sources are read in turn and deduplicated by string form, preserving the
    order in which they were first seen:

    1. ``metadata['official_measures']``.
    2. ``metadata[name]['official_measures']`` for each dataset.
    3. The ``ir_datasets`` documentation for each dataset.

    Args:
        dataset_names: Display names of the suite's datasets.
        dataset_ids: Display name → IRDS identifier, or ``None`` to skip the
            documentation lookup.
        metadata: The suite metadata, in any of the shapes it accepts.
        default: Measures to fall back on when nothing is discovered.

    Returns:
        list[Measure]: The discovered measures, or ``default`` if there are none.
    """
    discovered: list[Measure] = []
    seen: set[str] = set()

    def add(candidates: Sequence[MeasureLike] | None) -> None:
        for measure in parse_measures(candidates or []):
            if str(measure) not in seen:
                discovered.append(measure)
                seen.add(str(measure))

    if isinstance(metadata, dict):
        add(metadata.get(OFFICIAL_MEASURES))
        for name in dataset_names:
            per_dataset = metadata.get(name, {})
            if isinstance(per_dataset, dict):
                add(per_dataset.get(OFFICIAL_MEASURES))

    if isinstance(dataset_ids, dict):
        for name, dataset_id in dataset_ids.items():
            try:
                add(documented_measures(dataset_id))
            except Exception as error:
                logger.warning(
                    f"Failed to load measures from documentation for "
                    f"'{name}' ({dataset_id}): {error}"
                )

    if not discovered:
        logger.warning(f"No measures discovered; defaulting to {default}.")
        return list(default)
    return discovered


def measures_for_dataset(
    measures: list[Measure] | dict[str, list[Measure]],
    dataset: str,
    default: Sequence[Measure],
) -> list[Measure]:
    """
    Resolve the measures configured for one dataset.

    Args:
        measures: A suite-wide list, or a mapping of display name to list.
        dataset: The dataset display name.
        default: Measures to use when a mapping has no entry for the dataset.

    Returns:
        list[Measure]: The measures for that dataset.
    """
    if isinstance(measures, list):
        return measures
    return measures.get(dataset, default)


__all__ = [
    "OFFICIAL_MEASURES",
    "MeasureLike",
    "discover_measures",
    "documented_measures",
    "measures_for_dataset",
    "parse_measure_string",
    "parse_measures",
]
