"""Normalisation of a suite's dataset declaration.

A suite declares its datasets as ``_datasets``, which may be a list of IRDS
identifiers, a list of dataset-like objects, or a mapping of display name to
either. Every consumer used to re-derive the display name, the IRDS identifier
and the PyTerrier dataset from that raw declaration at its own call site. This
module does it once, up front, into a list of :class:`DatasetSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyterrier as pt

DATASET_ATTRIBUTES = ("_irds_id", "get_topics", "get_qrels")
"""Attributes an object must expose to stand in for an IRDS identifier."""


def is_dataset_like(value: Any) -> bool:
    """Whether ``value`` can be used in place of an IRDS identifier."""
    return all(hasattr(value, attribute) for attribute in DATASET_ATTRIBUTES)


def irds_id_of(ref: Any) -> str:
    """The IRDS identifier of a string identifier or a dataset-like object."""
    if isinstance(ref, str):
        return ref
    return ref._irds_id


def dataset_of(ref: Any) -> pt.datasets.Dataset:
    """The PyTerrier dataset for a string identifier or a dataset-like object."""
    if isinstance(ref, str):
        return pt.get_dataset(f"irds:{ref}")
    return ref


@dataclass(frozen=True)
class DatasetSpec:
    """
    One dataset of a suite, resolved from the raw declaration.

    Attributes:
        key: The declaration key, kept so hooks still see what was written:
            the display name for a mapping, the reference itself for a list.
        ref: The string identifier or dataset-like object as declared.
        name: The display name used in results tables and on disk.
        irds_id: The IRDS identifier, used for corpus grouping and metadata.
    """

    key: Any
    ref: Any
    name: str
    irds_id: str

    def dataset(self) -> pt.datasets.Dataset:
        """Resolve this spec to a PyTerrier dataset."""
        return dataset_of(self.ref)

    def as_item(self) -> tuple[Any, Any]:
        """The ``(key, ref)`` pair hooks receive as a group member."""
        return self.key, self.ref


def validate_dataset(ref: Any, where: str) -> None:
    """
    Raise unless ``ref`` is a string identifier or a dataset-like object.

    Args:
        ref: The declared dataset reference.
        where: Human-readable position, used in the error message.

    Raises:
        AssertionError: If the reference is neither form.
    """
    if isinstance(ref, str) or is_dataset_like(ref):
        return
    raise AssertionError(
        f"Suite _datasets[{where}] must be a string ID or a dataset "
        "object with _irds_id, get_topics, and get_qrels"
    )


def normalise_datasets(datasets: Any) -> list[DatasetSpec]:
    """
    Resolve a ``_datasets`` declaration into specs, validating as it goes.

    Args:
        datasets: A mapping of display name to reference, or a list of
            references. References are string IRDS identifiers or dataset-like
            objects.

    Returns:
        list[DatasetSpec]: One spec per declared dataset, in declaration order.

    Raises:
        AssertionError: If the declaration is empty, has an unsupported type,
            uses non-string mapping keys, or contains an invalid reference.
    """
    if not datasets:
        raise AssertionError(
            "Suite must have at least one dataset defined in _datasets"
        )

    if not isinstance(datasets, (dict, list)):
        raise AssertionError(
            "Suite _datasets must be a dict[name->id] or a list[dataset_id]"
        )

    items: list[tuple[Any, Any]]
    if isinstance(datasets, dict):
        for key in datasets:
            if not isinstance(key, str):
                raise AssertionError(
                    f"Suite _datasets keys must be strings, got {type(key)}"
                )
        items = list(datasets.items())
        positions = [repr(key) for key in datasets]
    else:
        items = [(ref, ref) for ref in datasets]
        positions = [str(index) for index in range(len(datasets))]

    specs: list[DatasetSpec] = []
    for (key, ref), where in zip(items, positions):
        validate_dataset(ref, where)
        irds_id = irds_id_of(ref)
        name = key if isinstance(key, str) else irds_id
        specs.append(DatasetSpec(key=key, ref=ref, name=name, irds_id=irds_id))
    return specs


__all__ = [
    "DATASET_ATTRIBUTES",
    "DatasetSpec",
    "dataset_of",
    "irds_id_of",
    "is_dataset_like",
    "normalise_datasets",
    "validate_dataset",
]
