"""Unit tests for the overridable hooks that make up a suite evaluation.

Covers the seams documented on :class:`suiteeval.suite.base.Suite`:
configuration resolution, corpus grouping, pipeline batching and wrapping,
run-file caching, and result post-processing.
"""

import gzip
import os

import pandas as pd
import pytest
from ir_measures import AP, nDCG
from pyterrier import Transformer

from suiteeval.context import DatasetContext
from suiteeval.suite.base import RunConfig, Suite
from tests.unit.conftest import DummyTransformer


# ---------- Helpers ----------


class Tag(Transformer):
    """Transformer that records its own label, used to trace pipeline wrapping."""

    def __init__(self, label: str):
        super().__init__()
        self.label = label

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        return inp


def generator_of(*named_pipelines):
    """Build a pipeline generator yielding the given ``(pipeline, name)`` pairs."""

    def gen(_context):
        yield from named_pipelines

    return gen


def experiment_kwargs_for(mock_experiment, index: int) -> dict:
    """Keyword arguments of the ``index``-th recorded ``pt.Experiment`` call."""
    return mock_experiment.call_args_list[index].kwargs


@pytest.fixture
def two_corpus_suite(cleanup_suite_registry):
    """Two datasets that do not share a corpus, so each forms its own group."""
    return Suite.register(
        "test_two_corpora",
        datasets=["corpus-a/test", "corpus-b/test"],
        names=["ds_a", "ds_b"],
        metadata={"official_measures": [nDCG @ 10]},
    )


# ---------- Regression: save_dir must survive every corpus group ----------


class TestSaveDirAcrossCorpora:
    """``save_dir`` used to be consumed by the first corpus group only."""

    def test_sequential_mode_saves_runs_for_every_corpus(
        self,
        two_corpus_suite,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
    ):
        save_path = os.path.join(temp_dir, "runs")

        two_corpus_suite(generator_of((DummyTransformer(), "sys")), save_dir=save_path)

        assert mock_pt_experiment.call_count == 2
        saved = [
            experiment_kwargs_for(mock_pt_experiment, i).get("save_dir")
            for i in range(2)
        ]
        assert saved == [
            os.path.join(save_path, "ds_a"),
            os.path.join(save_path, "ds_b"),
        ]
        assert os.path.isdir(os.path.join(save_path, "ds_b"))

    def test_grouped_mode_saves_runs_for_every_corpus(
        self,
        two_corpus_suite,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
    ):
        save_path = os.path.join(temp_dir, "runs")

        two_corpus_suite(
            generator_of((DummyTransformer(), "sys")),
            save_dir=save_path,
            baseline=0,
        )

        assert mock_pt_experiment.call_count == 2
        saved = [
            experiment_kwargs_for(mock_pt_experiment, i).get("save_dir")
            for i in range(2)
        ]
        assert saved == [
            os.path.join(save_path, "ds_a"),
            os.path.join(save_path, "ds_b"),
        ]

    def test_index_dir_created_for_every_corpus(
        self,
        two_corpus_suite,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
    ):
        index_path = os.path.join(temp_dir, "indices")

        two_corpus_suite(
            generator_of((DummyTransformer(), "sys")), index_dir=index_path
        )

        assert os.path.isdir(os.path.join(index_path, "corpus-a-test"))
        assert os.path.isdir(os.path.join(index_path, "corpus-b-test"))


# ---------- resolve_config ----------


class TestResolveConfig:
    """``resolve_config`` splits suite settings from experiment kwargs."""

    def test_consumes_directory_arguments(self, vaswani_suite):
        config = vaswani_suite.resolve_config(
            save_dir="/runs", index_dir="/idx", verbose=True
        )

        assert config.save_dir == "/runs"
        assert config.index_dir == "/idx"
        assert config.experiment_kwargs == {"verbose": True}

    def test_forwards_but_also_reads_shared_arguments(self, vaswani_suite):
        config = vaswani_suite.resolve_config(save_mode="overwrite", perquery=True)

        assert config.save_mode == "overwrite"
        assert config.perquery is True
        # Both remain available to pt.Experiment.
        assert config.experiment_kwargs == {
            "save_mode": "overwrite",
            "perquery": True,
        }

    def test_baseline_selects_grouped_mode(self, vaswani_suite):
        assert vaswani_suite.resolve_config(baseline=0).grouped is True
        assert vaswani_suite.resolve_config().grouped is False

    def test_defaults(self, vaswani_suite):
        config = vaswani_suite.resolve_config()

        assert config == RunConfig(experiment_kwargs={})
        assert config.compute_overall is True


# ---------- Pipeline batching and coercion ----------


class TestPipelineBatching:
    """Grouped and sequential modes differ only in batch size."""

    def test_grouped_yields_one_batch(self, vaswani_suite, mock_dataset):
        context = DatasetContext(mock_dataset)
        gen = generator_of((Tag("a"), "a"), (Tag("b"), "b"))

        batches = list(vaswani_suite.iter_pipeline_batches(context, gen, grouped=True))

        assert len(batches) == 1
        assert [name for _, name in batches[0]] == ["a", "b"]

    def test_sequential_yields_one_batch_per_pipeline(
        self, vaswani_suite, mock_dataset
    ):
        context = DatasetContext(mock_dataset)
        gen = generator_of((Tag("a"), "a"), (Tag("b"), "b"))

        batches = list(vaswani_suite.iter_pipeline_batches(context, gen, grouped=False))

        assert [len(batch) for batch in batches] == [1, 1]
        assert [batch[0][1] for batch in batches] == ["a", "b"]

    def test_grouped_without_names_does_not_raise(self, vaswani_suite, mock_dataset):
        context = DatasetContext(mock_dataset)

        def gen(_context):
            yield Tag("a")
            yield Tag("b")

        (batch,) = list(vaswani_suite.iter_pipeline_batches(context, gen, grouped=True))

        assert [name for _, name in batch] == [None, None]

    def test_tuple_of_transformers_keeps_every_pipeline(
        self, vaswani_suite, mock_dataset
    ):
        """A 2-tuple of transformers is a pair of pipelines, not (pipeline, name)."""
        context = DatasetContext(mock_dataset)

        def gen(_context):
            return (Tag("a"), Tag("b"))

        out = list(vaswani_suite.coerce_pipelines_sequential(context, gen))

        assert [pipeline.label for pipeline, _ in out] == ["a", "b"]
        assert [name for _, name in out] == [None, None]

    def test_single_name_applies_to_every_pipeline(self, vaswani_suite, mock_dataset):
        context = DatasetContext(mock_dataset)

        def gen(_context):
            return ([Tag("a"), Tag("b")], "shared")

        out = list(vaswani_suite.coerce_pipelines_sequential(context, gen))

        assert [name for _, name in out] == ["shared", "shared"]

    def test_mismatched_name_count_raises(self, vaswani_suite, mock_dataset):
        context = DatasetContext(mock_dataset)

        def gen(_context):
            return ([Tag("a"), Tag("b")], ["only-one"])

        with pytest.raises(ValueError, match="does not match"):
            list(vaswani_suite.coerce_pipelines_sequential(context, gen))

    def test_invalid_item_raises(self, vaswani_suite, mock_dataset):
        context = DatasetContext(mock_dataset)

        def gen(_context):
            yield "not a transformer"

        with pytest.raises(ValueError, match="invalid item"):
            list(vaswani_suite.coerce_pipelines_sequential(context, gen))

    def test_leaked_stop_iteration_is_reported(self, vaswani_suite, mock_dataset):
        context = DatasetContext(mock_dataset)

        def gen(_context):
            yield next(iter([]))

        with pytest.raises(ValueError, match="StopIteration"):
            list(vaswani_suite.coerce_pipelines_sequential(context, gen))

    def test_empty_grouped_coercion_raises(self, vaswani_suite, mock_dataset):
        context = DatasetContext(mock_dataset)

        def gen(_context):
            return iter([])

        with pytest.raises(ValueError, match="No pipelines generated"):
            vaswani_suite.coerce_pipelines_grouped(context, gen)

    def test_non_callable_generator_raises(self, vaswani_suite, mock_dataset):
        context = DatasetContext(mock_dataset)

        with pytest.raises(TypeError, match="must be a callable"):
            list(vaswani_suite.coerce_pipelines_sequential(context, 42))


class TestWrapPipeline:
    """``wrap_pipeline`` is the one seam both batching modes share."""

    def test_applies_to_grouped_and_sequential(
        self, cleanup_suite_registry, mock_dataset
    ):
        class _WrappingSuite(Suite):
            _datasets = ["vaswani"]
            _measures = [nDCG @ 10]

            def wrap_pipeline(self, pipeline, context):
                return Tag(f"wrapped:{pipeline.label}")

        suite = _WrappingSuite()
        context = DatasetContext(mock_dataset)
        gen = generator_of((Tag("a"), "a"))

        sequential = list(suite.coerce_pipelines_sequential(context, gen))
        pipelines, _ = suite.coerce_pipelines_grouped(context, gen)

        assert sequential[0][0].label == "wrapped:a"
        assert pipelines[0].label == "wrapped:a"


# ---------- Dataset selection ----------


class TestSelectMembers:
    def test_returns_all_members_without_subset(self, vaswani_suite):
        members = [("a", "ds/a"), ("b", "ds/b")]

        assert vaswani_suite.select_members(members, None) == members

    def test_filters_to_the_named_subset(self, vaswani_suite):
        members = [("a", "ds/a"), ("b", "ds/b")]

        assert vaswani_suite.select_members(members, "b") == [("b", "ds/b")]

    def test_unknown_subset_selects_nothing(self, vaswani_suite):
        assert vaswani_suite.select_members([("a", "ds/a")], "zzz") == []

    def test_subset_skips_other_corpus_groups(
        self,
        two_corpus_suite,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
    ):
        two_corpus_suite(generator_of((DummyTransformer(), "sys")), subset="ds_b")

        assert mock_pt_experiment.call_count == 1


# ---------- Measures ----------


class TestMeasures:
    def test_explicit_metrics_win(self, vaswani_suite):
        assert vaswani_suite.measures_for("vaswani", [AP]) == [AP]

    def test_falls_back_to_suite_configuration(self, vaswani_suite):
        assert vaswani_suite.measures_for("vaswani") == [nDCG @ 10]

    def test_per_dataset_mapping_falls_back_to_defaults(self, cleanup_suite_registry):
        class _MappedSuite(Suite):
            _datasets = ["vaswani"]
            _measures = {"vaswani": [AP]}

        suite = _MappedSuite()

        assert suite.get_measures("vaswani") == [AP]
        assert suite.get_measures("unknown") == _MappedSuite._default_measures

    def test_parse_measures_accepts_strings_and_objects(self, vaswani_suite):
        parsed = vaswani_suite.parse_measures(["nDCG@10", AP])

        assert nDCG @ 10 in parsed
        assert AP in parsed

    def test_parse_measures_rejects_nonsense(self, vaswani_suite):
        with pytest.raises(ValueError, match="Unrecognised measure"):
            vaswani_suite.parse_measures(["not-a-measure"])

        with pytest.raises(ValueError, match="Invalid measure type"):
            vaswani_suite.parse_measures([object()])


# ---------- Run-file caching ----------


class TestRunFileCache:
    def test_paths_are_slugified(self, vaswani_suite):
        path = vaswani_suite.run_file_path("/runs", "beir/CQADupStack/Tex", "bm25")

        assert path == os.path.join("/runs", "beir-cqadupstack-tex", "bm25.res.gz")

    def test_overwrite_mode_ignores_existing_runs(self, vaswani_suite, temp_dir):
        path = vaswani_suite.run_file_path(temp_dir, "ds", "sys")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()

        assert vaswani_suite.has_cached_run(temp_dir, "ds", "sys", "warn") is True
        assert vaswani_suite.has_cached_run(temp_dir, "ds", "sys", "overwrite") is False

    def test_missing_run_is_not_cached(self, vaswani_suite, temp_dir):
        assert vaswani_suite.has_cached_run(temp_dir, "ds", "sys", "warn") is False

    def test_load_cached_run_replays_the_ranking(self, vaswani_suite, temp_dir):
        path = vaswani_suite.run_file_path(temp_dir, "ds", "sys")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wt") as f:
            f.write("q1 Q0 d1 0 2.0 sys\nq1 Q0 d2 1 1.0 sys\n")

        replayed = vaswani_suite.load_cached_run(path)(
            pd.DataFrame({"qid": ["q1"], "query": ["a query"]})
        )

        assert replayed["docno"].tolist() == ["d1", "d2"]
        assert replayed["score"].tolist() == [2.0, 1.0]

    def test_cached_runs_skip_inference(
        self,
        two_corpus_suite,
        temp_dir,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
    ):
        save_path = os.path.join(temp_dir, "runs")
        for ds_name in ("ds_a", "ds_b"):
            path = two_corpus_suite.run_file_path(save_path, ds_name, "sys")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with gzip.open(path, "wt") as f:
                f.write("1 Q0 d1 0 1.0 sys\n")

        two_corpus_suite(generator_of((DummyTransformer(), "sys")), save_dir=save_path)

        # Both datasets replayed from disk: no experiment writes a run file.
        assert mock_pt_experiment.call_count == 2
        assert all(
            "save_dir" not in experiment_kwargs_for(mock_pt_experiment, i)
            for i in range(2)
        )

    def test_custom_run_file_layout_is_honoured(
        self, cleanup_suite_registry, temp_dir, mock_pt_get_dataset, mock_pt_experiment
    ):
        class _FlatCacheSuite(Suite):
            _datasets = {"ds": "vaswani"}
            _dataset_ids = {"ds": "vaswani"}
            _measures = [nDCG @ 10]

            def run_file_path(self, save_dir, dataset_name, pipeline_name):
                return os.path.join(save_dir, f"{dataset_name}--{pipeline_name}.res.gz")

        suite = _FlatCacheSuite()
        path = suite.run_file_path(temp_dir, "ds", "sys")
        with gzip.open(path, "wt") as f:
            f.write("1 Q0 d1 0 1.0 sys\n")

        assert suite.has_cached_run(temp_dir, "ds", "sys", "warn") is True


# ---------- Results ----------


class TestResultHooks:
    def test_annotate_results_can_add_columns(
        self,
        cleanup_suite_registry,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
    ):
        class _AnnotatingSuite(Suite):
            _datasets = {"ds": "vaswani"}
            _dataset_ids = {"ds": "vaswani"}
            _measures = [nDCG @ 10]

            def annotate_results(self, results, dataset_name, corpus_id):
                results = super().annotate_results(results, dataset_name, corpus_id)
                results["corpus"] = corpus_id
                return results

        results = _AnnotatingSuite()(generator_of((DummyTransformer(), "sys")))

        assert set(results.loc[results["dataset"] == "ds", "corpus"]) == {"vaswani"}

    def test_postprocess_results_runs_before_the_overall_row(
        self,
        cleanup_suite_registry,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
    ):
        seen = {}

        class _PostprocessingSuite(Suite):
            _datasets = {"ds": "vaswani"}
            _dataset_ids = {"ds": "vaswani"}
            _measures = [nDCG @ 10]

            def postprocess_results(self, results, config):
                seen["datasets_before"] = set(results["dataset"])
                results = results.assign(extra=1)
                return super().postprocess_results(results, config)

        results = _PostprocessingSuite()(generator_of((DummyTransformer(), "sys")))

        assert seen["datasets_before"] == {"ds"}
        assert "extra" in results.columns
        assert "Overall" in set(results["dataset"])

    def test_compute_overall_can_be_disabled(
        self,
        vaswani_suite,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
    ):
        results = vaswani_suite(
            generator_of((DummyTransformer(), "sys")), compute_overall=False
        )

        assert "Overall" not in set(results["dataset"])

    def test_perquery_skips_the_overall_row(
        self,
        vaswani_suite,
        mock_pt_get_dataset,
        mock_pt_experiment,
        mock_irds_docs_parent_id,
    ):
        results = vaswani_suite(
            generator_of((DummyTransformer(), "sys")), perquery=True
        )

        assert "Overall" not in set(results["dataset"])

    def test_metric_columns_ignores_bookkeeping_columns(self, vaswani_suite):
        frame = pd.DataFrame(
            {
                "dataset": ["a"],
                "name": ["sys"],
                "nDCG@10": [0.5],
                "note": ["text"],
            }
        )

        assert vaswani_suite.metric_columns(frame) == ["nDCG@10"]
