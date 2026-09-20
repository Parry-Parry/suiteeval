"""Parity guards for the :class:`~suiteeval.suite.base.Suite` refactor.

Two jobs:

* Pin the public surface and the resolved configuration of every shipped suite,
  so a layout change cannot quietly drop a hook or alter what a suite evaluates.
* Characterise known defects with ``xfail``, so the commit that fixes one is the
  commit that flips its marker.
"""

from __future__ import annotations

import inspect

import pytest
from ir_measures import Measure, nDCG

from suiteeval.suite.base import Suite


DECLARATION_ATTRIBUTES = frozenset(
    {
        "_datasets",
        "_dataset_ids",
        "_metadata",
        "_measures",
        "_default_measures",
        "_query_field",
    }
)
"""Attributes a suite is declared with."""

DATASET_HOOKS = frozenset(
    {
        "_dataset_items",
        "_display_name",
        "_get_dataset_object",
        "_get_irds_id",
        "_validate_dataset",
        "datasets",
        "iter_corpus_groups",
        "select_members",
    }
)
"""Hooks resolving the declaration into datasets to evaluate."""

MEASURE_HOOKS = frozenset(
    {"coerce_measures", "get_measures", "measures_for", "parse_measures"}
)
"""Hooks choosing what a dataset is scored on."""

PIPELINE_HOOKS = frozenset(
    {
        "coerce_pipelines_grouped",
        "coerce_pipelines_sequential",
        "iter_pipeline_batches",
        "wrap_pipeline",
    }
)
"""Hooks turning generators into the pipelines an experiment sees."""

RUN_FILE_HOOKS = frozenset(
    {
        "has_cached_run",
        "index_dir_for",
        "load_cached_run",
        "prepare_save_dir",
        "run_file_path",
        "save_dir_for",
    }
)
"""Hooks deciding where runs and indexes live, and what may be replayed."""

EVALUATION_HOOKS = frozenset(
    {
        "annotate_results",
        "build_context",
        "evaluate_batch",
        "prepare_topics_qrels",
        "release_pipelines",
        "run_experiment",
    }
)
"""Hooks performing the evaluation itself."""

RESULT_HOOKS = frozenset(
    {"compute_overall_mean", "metric_columns", "postprocess_results"}
)
"""Hooks shaping the results table that is returned."""

ENTRY_POINT_HOOKS = frozenset({"resolve_config", "run"})
"""Hooks of the entry point itself."""

SUITE_SURFACE = (
    DECLARATION_ATTRIBUTES
    | DATASET_HOOKS
    | MEASURE_HOOKS
    | PIPELINE_HOOKS
    | RUN_FILE_HOOKS
    | EVALUATION_HOOKS
    | RESULT_HOOKS
    | ENTRY_POINT_HOOKS
)
"""
Every attribute a subclass may override or a caller may reach for. Adding to this set is
a feature; removing from it is a breaking change.
"""


def test_public_surface_is_unchanged():
    """No documented hook may disappear during a refactor."""
    present = {
        name for name in dir(Suite) if not name.startswith("__") and name != "_abc_impl"
    }

    assert SUITE_SURFACE - present == set(), "hooks removed from Suite"


def test_suite_is_still_callable():
    assert callable(Suite.__call__)


@pytest.mark.parametrize(
    "hook, parameters",
    [
        ("select_members", ["self", "members", "subset"]),
        ("wrap_pipeline", ["self", "pipeline", "context"]),
        ("build_context", ["self", "corpus_id", "corpus_ds", "config"]),
        ("prepare_topics_qrels", ["self", "dataset", "dataset_name"]),
        ("measures_for", ["self", "dataset_name", "eval_metrics"]),
        ("run_file_path", ["self", "save_dir", "dataset_name", "pipeline_name"]),
        (
            "has_cached_run",
            ["self", "save_dir", "dataset_name", "pipeline_name", "save_mode"],
        ),
        ("load_cached_run", ["self", "filepath"]),
        ("annotate_results", ["self", "results", "dataset_name", "corpus_id"]),
        ("postprocess_results", ["self", "results", "config"]),
        ("metric_columns", ["results"]),
        (
            "iter_pipeline_batches",
            ["self", "context", "pipeline_generators", "grouped"],
        ),
    ],
)
def test_hook_signatures_are_unchanged(hook, parameters):
    """Subclasses override these positionally; parameter names are the contract."""
    signature = inspect.signature(getattr(Suite, hook))

    assert list(signature.parameters) == parameters


def shipped_suites():
    """Import the shipped suites lazily so a collection error is attributable."""
    from suiteeval.suite import (
        BEIR,
        BRIGHT,
        Lotte,
        MSMARCODocument,
        MSMARCOPassage,
        NanoBEIR,
    )

    return {
        "BEIR": BEIR,
        "BRIGHT": BRIGHT,
        "Lotte": Lotte,
        "MSMARCODocument": MSMARCODocument,
        "MSMARCOPassage": MSMARCOPassage,
        "NanoBEIR": NanoBEIR,
    }


SHIPPED_EXPECTATIONS = {
    "BEIR": (25, "beir/arguana", "text"),
    "BRIGHT": (12, "bright/aops", "text"),
    "Lotte": (12, "lotte/lifestyle/test/forum", None),
    "MSMARCODocument": (3, "msmarco-document/trec-dl-2019/judged", None),
    "MSMARCOPassage": (3, "msmarco-passage/trec-dl-2019/judged", None),
    "NanoBEIR": (13, "nano-beir/arguana", "text"),
}
"""
``(dataset count, first dataset name, query field)`` per shipped suite.

Read from ``_datasets`` rather than the ``datasets`` property: not every
collection is registered with the installed ir_datasets, and this pins the
declaration, not the local provider.
"""

SHIPPED_MEASURES = {
    "BEIR": "[nDCG@10]",
    "BRIGHT": "[nDCG@10]",
    "NanoBEIR": "[nDCG@10]",
    "Lotte": "[Success@10, Success@5]",
    "MSMARCODocument": "[nDCG@10, RR, AP]",
    "MSMARCOPassage": "[nDCG@10, RR(rel=2), AP(rel=2)]",
}
"""
Measures each shipped suite resolves after construction. Registered suites pick up extra
measures from the ir_datasets documentation; that is current behaviour and is pinned
here deliberately.
"""


@pytest.mark.parametrize("suite_name", sorted(SHIPPED_EXPECTATIONS))
def test_shipped_suite_datasets_are_unchanged(suite_name):
    suite = shipped_suites()[suite_name]
    count, first, query_field = SHIPPED_EXPECTATIONS[suite_name]
    names = [name for name, _ in suite._dataset_items()]

    assert len(names) == count
    assert names[0] == first
    assert suite._query_field == query_field


@pytest.mark.parametrize("suite_name", sorted(SHIPPED_EXPECTATIONS))
def test_shipped_suite_measures_resolve_to_real_measures(suite_name):
    """Every shipped suite resolves a non-empty list of parsed measures."""
    suite = shipped_suites()[suite_name]
    measures = suite.get_measures(SHIPPED_EXPECTATIONS[suite_name][1])

    assert measures
    assert all(isinstance(measure, Measure) for measure in measures)


@pytest.mark.parametrize("suite_name", sorted(SHIPPED_MEASURES))
def test_shipped_suite_measure_values_are_unchanged(suite_name):
    suite = shipped_suites()[suite_name]
    resolved = suite.get_measures(SHIPPED_EXPECTATIONS[suite_name][1])

    assert str(resolved) == SHIPPED_MEASURES[suite_name]


def test_suites_are_singletons():
    assert shipped_suites()["BEIR"] is shipped_suites()["BEIR"]
    assert type(shipped_suites()["BEIR"])() is shipped_suites()["BEIR"]


@pytest.mark.xfail(
    reason="B1: parse_measure raises NameError, which parse_measures does not catch, "
    "so the parse_trec_measure fallback is unreachable",
    strict=True,
)
@pytest.mark.parametrize("measure_string", ["map", "ndcg_cut_10", "recip_rank"])
def test_trec_measure_strings_are_parsed(measure_string):
    assert Suite.parse_measures([measure_string])


@pytest.mark.xfail(
    reason="B2: Suite.register folds metadata into a per-dataset map, so the "
    "'description' key never reaches __doc__",
    strict=True,
)
@pytest.mark.parametrize("suite_name", ["Lotte", "MSMARCODocument", "MSMARCOPassage"])
def test_registered_suites_take_their_description_as_docstring(suite_name):
    assert shipped_suites()[suite_name].__doc__


def test_dataset_ids_are_populated_for_class_defined_suites():
    """B3, fixed: _dataset_ids is derived from the declaration, not assigned."""
    beir = shipped_suites()["BEIR"]

    assert beir._dataset_ids["beir/arguana"] == "beir/arguana"
    assert len(beir._dataset_ids) == 25


@pytest.mark.xfail(
    reason="B4: replayed runs are evaluated without config.experiment_kwargs, so "
    "they silently lose perquery, baseline and friends",
    strict=True,
)
def test_cached_replay_keeps_experiment_kwargs(
    cleanup_suite_registry,
    temp_dir,
    mock_pt_get_dataset,
    mock_pt_experiment,
    mock_irds_docs_parent_id,
):
    import gzip
    import os

    from tests.unit.conftest import DummyTransformer

    suite = Suite.register(
        "test_cached_kwargs",
        datasets=["vaswani"],
        metadata={"official_measures": [nDCG @ 10]},
    )
    path = suite.run_file_path(temp_dir, "vaswani", "sys")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("1 Q0 d1 0 1.0 sys\n")

    def generator(_context):
        yield DummyTransformer(), "sys"

    suite(generator, save_dir=temp_dir, perquery=True)

    assert mock_pt_experiment.call_args_list[0].kwargs.get("perquery") is True
