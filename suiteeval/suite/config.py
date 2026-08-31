"""Run configuration and the frame conventions shared across a suite run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import pandas as pd

#: Identifier columns coerced to strings before any join or merge.
ID_COLUMNS: Tuple[str, ...] = ("qid", "docno")

#: Columns of a results table that never hold measure values.
NON_METRIC_COLUMNS = frozenset(
    {"dataset", "name", "qid", "docno", "rank", "score", "query"}
)

#: Column order of a TREC run file.
RUN_FILE_COLUMNS = ["qid", "iter", "docno", "rank", "score", "name"]


def slugify(identifier: str) -> str:
    """Convert a dataset or corpus identifier into a filesystem-safe name."""
    return identifier.replace("/", "-").lower()


def ensure_string_ids(
    df: pd.DataFrame, cols: Sequence[str] = ID_COLUMNS
) -> pd.DataFrame:
    """Coerce identifier columns to strings for safe joins/merges."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def metric_columns(results: pd.DataFrame) -> list[str]:
    """Numeric columns of ``results`` that hold measure values."""
    return [
        col
        for col in results.columns
        if col not in NON_METRIC_COLUMNS and pd.api.types.is_numeric_dtype(results[col])
    ]


@dataclass(frozen=True)
class RunConfig:
    """
    Resolved settings for one :meth:`~suiteeval.suite.base.Suite.__call__` invocation.

    Produced by :meth:`~suiteeval.suite.base.Suite.resolve_config`, which separates
    suite-level options from the keyword arguments forwarded verbatim to
    :func:`pyterrier.Experiment`.

    Attributes:
        eval_metrics: Explicit metrics overriding the suite's configuration.
        subset: Restrict evaluation to a single dataset display name.
        compute_overall: Whether to append geometric-mean ``Overall`` rows.
        index_dir: Root directory for per-corpus indexes, if any.
        save_dir: Root directory for per-dataset run files, if any.
        save_mode: PyTerrier save mode; ``"overwrite"`` disables run-file reuse.
        perquery: Whether results are reported per query.
        grouped: Whether pipelines must be materialised together (significance tests).
        experiment_kwargs: Remaining kwargs passed to :func:`pyterrier.Experiment`.
    """

    eval_metrics: Optional[Sequence[Any]] = None
    subset: Optional[str] = None
    compute_overall: bool = True
    index_dir: Optional[str] = None
    save_dir: Optional[str] = None
    save_mode: str = "warn"
    perquery: bool = False
    grouped: bool = False
    experiment_kwargs: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ID_COLUMNS",
    "NON_METRIC_COLUMNS",
    "RUN_FILE_COLUMNS",
    "RunConfig",
    "ensure_string_ids",
    "metric_columns",
    "slugify",
]
