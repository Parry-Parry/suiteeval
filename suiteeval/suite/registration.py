"""Helpers behind :meth:`~suiteeval.suite.base.SuiteMeta.register`.

Registering a suite means turning a loose declaration - some dataset
identifiers, maybe display names, maybe metadata in one of four shapes - into
the attributes a :class:`~suiteeval.suite.base.Suite` subclass expects. The
shape-guessing lives here so the metaclass is left with class creation alone.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any, Sequence

logger = getLogger(__name__)

MetadataInput = list[dict[str, Any]] | dict[str, Any] | None


def dataset_map(
    datasets: Sequence[str], names: Sequence[str] | None = None
) -> dict[str, str]:
    """
    Pair display names with dataset identifiers.

    A mismatched ``names`` truncates the suite rather than failing, which is
    long-standing behaviour; it is reported so it does not pass unnoticed.

    Args:
        datasets: IRDS dataset identifiers.
        names: Display names aligned with ``datasets``; the identifiers
            themselves are used when omitted or empty.

    Returns:
        dict[str, str]: Display name → dataset identifier, in declaration order.
    """
    display_names = names or datasets
    if len(display_names) != len(datasets):
        logger.warning(
            f"`names` has {len(display_names)} entries for {len(datasets)} "
            f"datasets; only the first {min(len(display_names), len(datasets))} "
            "will be registered."
        )
    return dict(zip(display_names, datasets))


def is_per_dataset(metadata: dict[str, Any]) -> bool:
    """
    Whether a metadata dict maps dataset names to their own metadata dicts.

    A dict whose values are all non-dicts is flat metadata that applies to
    every dataset. Anything else is read as a per-dataset mapping, matching
    long-standing behaviour; a mixed dict is ambiguous and is reported.
    """
    values = list(metadata.values())
    dicts = [value for value in values if isinstance(value, dict)]
    if dicts and len(dicts) != len(values):
        logger.warning(
            "Ambiguous `metadata`: some values are dicts and some are not. "
            "Reading it as a per-dataset mapping; the non-dict entries are ignored."
        )
    return bool(dicts)


def normalise_metadata(metadata: MetadataInput, names: Sequence[str]) -> dict[str, Any]:
    """
    Resolve any accepted metadata shape into a per-dataset mapping.

    Accepted shapes:

    * ``None`` → an empty dict per dataset.
    * ``list[dict]`` → entry *i* applies to ``names[i]``.
    * ``dict[str, dict]`` → already a per-dataset mapping.
    * ``dict[str, Any]`` with no dict values → flat metadata, applied to all.

    Args:
        metadata: The metadata as supplied to ``register``.
        names: Display names of the suite's datasets.

    Returns:
        dict[str, Any]: The per-dataset mapping.

    Raises:
        ValueError: If the shape is unsupported, or a list does not match the
            number of datasets.
    """
    if metadata is None:
        return {name: {} for name in names}

    if isinstance(metadata, list):
        if len(metadata) != len(names):
            raise ValueError("`metadata` list must match number of datasets")
        return dict(zip(names, metadata))

    if isinstance(metadata, dict):
        if not is_per_dataset(metadata):
            return {name: metadata for name in names}
        unknown = [key for key in metadata if key not in set(names)]
        if unknown:
            logger.warning(
                f"`metadata` has entries for datasets this suite does not "
                f"declare, which will be ignored: {sorted(unknown)}"
            )
        return metadata

    raise ValueError(f"Unsupported metadata type: {type(metadata)}")


__all__ = [
    "MetadataInput",
    "dataset_map",
    "is_per_dataset",
    "normalise_metadata",
]
