"""Reading back the TREC run files a previous evaluation wrote.

A suite decides *where* run files live - see the ``run_file_path``,
``save_dir_for`` and ``index_dir_for`` hooks on
:class:`~suiteeval.suite.base.Suite` - and this module decides how one is read
once located.
"""

from __future__ import annotations

import gzip

import pandas as pd
import pyterrier as pt
from pyterrier import Transformer

from suiteeval.suite.config import RUN_FILE_COLUMNS, ensure_string_ids

#: Extension of a run file written by a suite.
RUN_FILE_SUFFIX = ".res.gz"

#: Columns of a run file that a result frame needs.
RESULT_COLUMNS = ["qid", "docno", "score", "rank"]


def read_run(filepath: str) -> pd.DataFrame:
    """
    Read a gzipped TREC run file into a result frame.

    Args:
        filepath: Path to the ``.res.gz`` file.

    Returns:
        pandas.DataFrame: Columns ``qid``, ``docno``, ``score`` and ``rank``,
            with identifiers as strings.
    """
    with gzip.open(filepath, "rt") as handle:
        run = pd.read_csv(handle, sep=r"\s+", header=None, names=RUN_FILE_COLUMNS)
    return ensure_string_ids(run[RESULT_COLUMNS].copy())


def replay_run(filepath: str) -> Transformer:
    """
    Load a run file into a transformer that replays the stored ranking.

    Args:
        filepath: Path to the ``.res.gz`` file.

    Returns:
        Transformer: A transformer yielding the stored ranking.
    """
    return pt.Transformer.from_df(read_run(filepath))


__all__ = [
    "RESULT_COLUMNS",
    "RUN_FILE_SUFFIX",
    "read_run",
    "replay_run",
]
