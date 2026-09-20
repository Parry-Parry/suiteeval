from __future__ import annotations

import os
from abc import ABC, ABCMeta
from collections.abc import Iterator, Sequence
from logging import getLogger
from typing import Any

import ir_datasets as irds
import pandas as pd
import pyterrier as pt
from ir_measures import Measure, nDCG
from pyterrier import Transformer

from suiteeval.context import DatasetContext
from suiteeval.suite.config import RunConfig, ensure_string_ids, slugify
from suiteeval.suite.config import metric_columns as _metric_columns
from suiteeval.suite.datasets import (
    DatasetSpec,
    dataset_of,
    irds_id_of,
    normalise_datasets,
    validate_dataset,
)
from suiteeval.suite.measures import discover_measures, measures_for_dataset
from suiteeval.suite.measures import parse_measures as _parse_measures
from suiteeval.suite.pipelines import (
    NamedPipeline,
    PipelineGenerators,
    fill_names,
    iter_generator_output,
)
from suiteeval.suite.registration import (
    MetadataInput,
    dataset_map,
    normalise_metadata,
)
from suiteeval.suite.results import append_overall
from suiteeval.suite.runfiles import RUN_FILE_SUFFIX, replay_run

logger = getLogger(__name__)


class SuiteMeta(ABCMeta):
    """
    Metaclass for :class:`Suite`.

    Responsibilities:
    - Maintain a registry of suite classes by name.
    - Enforce a singleton instance per suite class (i.e., one instance per subclass).
    - Provide a :meth:`register` helper to dynamically create and register suites.
    """

    _classes: dict[str, type] = {}
    """Suite classes created by :meth:`register`, by the name they were given."""

    _instances: dict[type, "Suite"] = {}
    """
    One instance per suite class. Keyed by the class itself, so two suites that happen
    to share a name do not share an instance.
    """

    def __call__(cls, *args, **kwargs):
        if cls not in SuiteMeta._instances:
            SuiteMeta._instances[cls] = super().__call__(*args, **kwargs)
        elif args or kwargs:
            logger.warning(
                f"{cls.__name__} is a singleton and already exists; the "
                "arguments given to this call are ignored."
            )
        return SuiteMeta._instances[cls]

    @classmethod
    def register(
        mcs,
        suite_name: str,
        datasets: list[str],
        names: list[str] | None = None,
        metadata: MetadataInput = None,
        query_field: str | None = None,
    ) -> "Suite":
        """
        Create (or retrieve) a Suite singleton that wraps the given datasets.

        Args:
            suite_name: Name to assign to the dynamically created suite subclass.
            datasets: IRDS dataset identifiers (e.g., ``"msmarco-passage/trec-dl-2019"``).
            names: Optional display names corresponding one-to-one with ``datasets``.
                Defaults to ``datasets`` when omitted.
            metadata: Optional metadata, in any shape accepted by
                :func:`suiteeval.suite.registration.normalise_metadata`.
            query_field: Optional topic field name to use when fetching topics (e.g., ``"title"``).

        Returns:
            Suite: The singleton instance of the dynamically created suite class.

        Raises:
            ValueError: If ``metadata`` has an unsupported shape or length.
        """
        if suite_name in mcs._classes:
            return mcs._classes[suite_name]()

        name_to_id = dataset_map(datasets, names)
        new_cls = mcs(
            suite_name,
            (Suite,),
            {
                "_datasets": name_to_id,
                "_metadata": normalise_metadata(metadata, list(name_to_id)),
                "_query_field": query_field,
            },
        )

        mcs._classes[suite_name] = new_cls
        return new_cls()

    @classmethod
    def forget(mcs, suite_name: str) -> None:
        """
        Drop a registered suite, so the name can be registered again.

        Args:
            suite_name: The name the suite was registered under. Unknown names
                are ignored.
        """
        suite_cls = mcs._classes.pop(suite_name, None)
        if suite_cls is not None:
            mcs._instances.pop(suite_cls, None)

    @classmethod
    def registered(mcs) -> list[str]:
        """The names of every suite created through :meth:`register`."""
        return sorted(mcs._classes)


class TopicsQrelsCache:
    """
    Topics and qrels prepared once per dataset, for the life of a corpus group.

    Sequential mode evaluates one pipeline at a time against every dataset
    sharing a corpus, so without this the topics and qrels of each dataset
    would be re-read once per pipeline rather than once per dataset.

    The cached frames are handed to every batch, on the usual PyTerrier
    assumption that a transformer does not modify its input in place.
    """

    def __init__(self, suite: "Suite"):
        self._suite = suite
        self._prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def get(
        self, dataset_ref: Any, dataset_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        The topics and qrels for one dataset, preparing them on first use.

        Args:
            dataset_ref: String identifier or dataset-like object.
            dataset_name: Dataset display name.

        Returns:
            tuple[pandas.DataFrame, pandas.DataFrame]: ``(topics, qrels)``.
        """
        if dataset_name not in self._prepared:
            self._prepared[dataset_name] = self._suite.prepare_topics_qrels(
                dataset_of(dataset_ref), dataset_name
            )
        return self._prepared[dataset_name]


class Suite(ABC, metaclass=SuiteMeta):
    """
    Abstract base class for a set of related evaluations across one or more datasets.

    Subclasses (or classes created via :meth:`SuiteMeta.register`) must populate:

    Attributes:
        _datasets: Either a ``dict[str, str]`` mapping display name → IRDS dataset ID,
            or a ``list[str]`` of IRDS dataset IDs. Values may also be dataset-like
            objects exposing ``_irds_id``, ``get_topics`` and ``get_qrels``.
        _metadata: Optional per-dataset or global metadata.
        _measures: A list of :class:`ir_measures.Measure` or a mapping from dataset name
            to such a list. When not provided, defaults are derived from metadata or
            IRDS documentation; ultimately falling back to ``_default_measures``.
        _default_measures: Fallback measures when nothing else is configured.
        _query_field: Optional topic field name to use when fetching topics.

    Evaluation runs through a chain of small, overridable steps. In call order:

    ==============================  ==========================================
    Hook                            Responsibility
    ==============================  ==========================================
    :meth:`resolve_config`          Split kwargs into a :class:`RunConfig`
    :meth:`iter_corpus_groups`      Group datasets by shared corpus
    :meth:`select_members`          Pick the datasets to evaluate in a group
    :meth:`run_corpus_group`        Evaluate one corpus group start to finish
    :meth:`build_context`           Build the shared per-corpus context
    :meth:`iter_pipeline_batches`   Decide how pipelines are batched
    :meth:`wrap_pipeline`           Decorate each pipeline (filters, etc.)
    :meth:`run_batch`               Take one batch across the group's datasets
    :meth:`prepare_topics_qrels`    Fetch and normalise topics/qrels
    :meth:`measures_for`            Choose metrics for a dataset
    :meth:`has_cached_run` /
    :meth:`load_cached_run` /
    :meth:`run_file_path`           Reuse run files written by a previous call
    :meth:`run_experiment`          The :func:`pyterrier.Experiment` call itself
    :meth:`annotate_results`        Tag result rows with their dataset
    :meth:`release_pipelines` /
    :meth:`release_context`         Free memory between batches and groups
    :meth:`postprocess_results`     Aggregate the concatenated results
    ==============================  ==========================================

    Notes:
        Instances are singletons per subclass (enforced by :class:`SuiteMeta`).
    """

    _datasets: list[str] | dict[str, str] = {}
    _metadata: dict[str, Any] = {}
    _measures: list[Measure] | dict[str, list[Measure]] | None = None
    _default_measures: list[Measure] = [nDCG @ 10]
    _query_field: str | None = None

    def __init__(self):
        self._specs = normalise_datasets(self._datasets)
        self.coerce_measures(self._metadata)
        if "description" in self._metadata:
            self.__doc__ = self._metadata["description"]
        self.__post_init__()

    def __post_init__(self):
        """Validate the declaration. Override to add suite-specific checks."""
        normalise_datasets(self._datasets)
        if self._measures is None:
            raise AssertionError("Suite must have measures defined in _measures")

    _validate_dataset = staticmethod(validate_dataset)

    _get_irds_id = staticmethod(irds_id_of)
    _get_dataset_object = staticmethod(dataset_of)

    @property
    def _dataset_ids(self) -> dict[str, str]:
        """Display name → IRDS identifier, derived from the declaration."""
        return {spec.name: spec.irds_id for spec in self._specs}

    @classmethod
    def _display_name(cls, name_or_obj: Any) -> str:
        """Resolve a dataset key to the string used in results and paths."""
        if isinstance(name_or_obj, str):
            return name_or_obj
        return irds_id_of(name_or_obj)

    def _dataset_items(self) -> list[tuple[Any, Any]]:
        """Normalise ``_datasets`` to a list of ``(display_key, dataset_ref)``."""
        return [spec.as_item() for spec in self._specs]

    def iter_corpus_groups(
        self,
    ) -> Iterator[tuple[str, pt.datasets.Dataset, list[tuple[Any, Any]]]]:
        """
        Group the suite's datasets by the corpus they share.

        Membership is decided by :func:`ir_datasets.docs_parent_id`, so datasets
        built on the same document collection are indexed once. Groups are
        yielded in the order their first dataset was declared. Override to
        impose a different grouping.

        Yields:
            tuple[str, pyterrier.datasets.Dataset, list[tuple[Any, Any]]]:
                ``(corpus_id, corpus_dataset, [(display_key, dataset_ref), ...])``.
        """
        corpus_datasets: dict[str, pt.datasets.Dataset] = {}
        members: dict[str, list[tuple[Any, Any]]] = {}

        for spec in self._specs:
            corpus_id = self._corpus_id_for(spec)
            if corpus_id not in members:
                corpus_datasets[corpus_id] = dataset_of(
                    corpus_id if isinstance(corpus_id, str) else spec.irds_id
                )
                members[corpus_id] = []
            members[corpus_id].append(spec.as_item())

        for corpus_id, group in members.items():
            yield corpus_id, corpus_datasets[corpus_id], group

    @staticmethod
    def _corpus_id_for(spec: DatasetSpec) -> str:
        """The corpus a dataset belongs to, falling back to the dataset itself."""
        try:
            return irds.docs_parent_id(spec.irds_id) or spec.irds_id
        except Exception as error:
            logger.debug(
                f"No parent corpus for '{spec.irds_id}'; indexing it alone: {error}"
            )
            return spec.irds_id

    def select_members(
        self, members: Sequence[tuple[Any, Any]], subset: str | None
    ) -> list[tuple[Any, Any]]:
        """
        Choose which members of a corpus group to evaluate.

        Args:
            members: ``(display_key, dataset_ref)`` pairs sharing one corpus.
            subset: Optional display name to restrict evaluation to.

        Returns:
            list[tuple[Any, Any]]: The members to evaluate; empty skips the group.
        """
        if subset is None:
            return list(members)
        return [
            (name, ds) for name, ds in members if self._display_name(name) == subset
        ]

    @property
    def datasets(self) -> Iterator[tuple[str, pt.datasets.Dataset]]:
        """
        Iterate over declared datasets yielding display name and PyTerrier dataset.

        Each access resolves the datasets afresh, so this is an iterator rather
        than a list: take ``list(suite.datasets)`` to hold on to them.

        Yields:
            tuple[str, pyterrier.datasets.Dataset]: Pairs of (name, dataset object).
        """
        for spec in self._specs:
            yield spec.name, spec.dataset()

    parse_measures = staticmethod(_parse_measures)

    def coerce_measures(self, metadata: dict[str, Any]) -> None:
        """
        Populate ``self._measures`` when the suite does not state them directly.

        Sources are aggregated in priority order: global metadata, then
        per-dataset metadata, then the IRDS documentation of each dataset,
        falling back to ``_default_measures``. See
        :func:`suiteeval.suite.measures.discover_measures`.

        Args:
            metadata: The suite metadata dictionary as configured at construction time.
        """
        if self._measures is not None:
            return
        self._measures = discover_measures(
            [spec.name for spec in self._specs],
            self._dataset_ids,
            metadata,
            self._default_measures,
        )

    def get_measures(self, dataset: str) -> list[Measure]:
        """
        The measures configured for one dataset.

        ``_measures`` may be a suite-wide list or a per-dataset mapping; an
        unknown dataset falls back to ``_default_measures``.
        """
        return measures_for_dataset(self._measures, dataset, self._default_measures)

    def measures_for(
        self, dataset_name: str, eval_metrics: Sequence[Any] | None = None
    ) -> Sequence[Any]:
        """
        Decide which metrics to evaluate for one dataset.

        Metrics the caller supplied win; otherwise the suite's own
        configuration is used. Override to vary metrics per dataset.
        """
        if eval_metrics is not None:
            return eval_metrics
        return self.get_measures(dataset_name)

    def wrap_pipeline(
        self, pipeline: Transformer, context: DatasetContext
    ) -> Transformer:
        """
        Decorate a single pipeline before it is evaluated.

        This is the single seam for suite-specific pipeline modifications such as
        appending a result filter. It is applied by both sequential and grouped
        coercion, so overriding it alone covers every execution mode.

        Args:
            pipeline: The pipeline produced by a generator.
            context: The shared context for the corpus being evaluated.

        Returns:
            Transformer: The pipeline to evaluate. The default returns it unchanged.
        """
        return pipeline

    def coerce_pipelines_sequential(
        self,
        context: DatasetContext,
        pipeline_generators: PipelineGenerators,
    ) -> Iterator[NamedPipeline]:
        """
        Yield pipelines lazily, one at a time, without materializing the full set.

        Use this when you want to minimize memory/VRAM footprint and you do not require
        joint analysis across all systems at once (e.g., significance testing).

        Args:
            context: The shared :class:`DatasetContext` for the current corpus group.
            pipeline_generators: Callable or sequence of callables that produce either:
                * a single :class:`pyterrier.Transformer`,
                * a sequence of transformers,
                * a tuple ``(pipelines, name_or_names)`` where names may be a single label
                applied to all pipelines or a sequence aligned with ``pipelines``.

        Yields:
            tuple[Transformer, str | None]: The pipeline and an optional display name.

        Raises:
            ValueError: If a generator yields an invalid structure.
        """
        for pipeline, name in iter_generator_output(context, pipeline_generators):
            yield self.wrap_pipeline(pipeline, context), name

    def coerce_pipelines_grouped(
        self,
        context: DatasetContext,
        pipeline_generators: PipelineGenerators,
    ) -> tuple[list[Transformer], list[str] | None]:
        """
        Materialize all pipelines (and optional names) into lists.

        Use this when downstream evaluation requires access to the full set of systems
        simultaneously (e.g., significance tests).

        Args:
            context: The shared :class:`DatasetContext` for the current corpus group.
            pipeline_generators: Callable or sequence of callables following the same
                conventions as in :meth:`coerce_pipelines_sequential`.

        Returns:
            tuple[list[Transformer], list[str] | None]:
                A list of pipelines and, if provided, a list of corresponding names.
                If no names were supplied, returns ``None`` for the second element.

        Raises:
            ValueError: If the generators produce no pipelines or an invalid structure.
        """
        pipelines: list[Transformer] = []
        names: list[str | None] = []
        for pipeline, name in self.coerce_pipelines_sequential(
            context, pipeline_generators
        ):
            pipelines.append(pipeline)
            names.append(name)

        if not pipelines:
            raise ValueError(
                "No pipelines generated. Ensure your generators produce valid Transformers."
            )

        return pipelines, fill_names(names)

    def iter_pipeline_batches(
        self,
        context: DatasetContext,
        pipeline_generators: PipelineGenerators,
        grouped: bool,
    ) -> Iterator[list[NamedPipeline]]:
        """
        Yield the units of work that are evaluated together.

        Grouped mode yields a single batch holding every pipeline, so that
        :func:`pyterrier.Experiment` can run significance tests across systems.
        Sequential mode yields one single-pipeline batch at a time, so that each
        pipeline can be released before the next is built.

        Args:
            context: The shared :class:`DatasetContext` for the current corpus group.
            pipeline_generators: Callable or sequence of callables producing pipelines.
            grouped: Whether all pipelines must be materialized together.

        Yields:
            list[tuple[Transformer, str | None]]: One batch of named pipelines.
        """
        if not grouped:
            for named_pipeline in self.coerce_pipelines_sequential(
                context, pipeline_generators
            ):
                yield [named_pipeline]
            return

        pipelines, names = self.coerce_pipelines_grouped(context, pipeline_generators)
        yield list(zip(pipelines, names or [None] * len(pipelines)))

    def index_dir_for(self, index_dir: str, corpus_id: str) -> str:
        """Directory holding the shared index for one corpus."""
        return os.path.join(index_dir, slugify(corpus_id))

    def save_dir_for(self, save_dir: str, dataset_name: str) -> str:
        """Directory holding the run files for one dataset."""
        return os.path.join(save_dir, slugify(dataset_name))

    def run_file_path(
        self, save_dir: str, dataset_name: str, pipeline_name: str
    ) -> str:
        """Path of the run file for one pipeline on one dataset."""
        return os.path.join(
            self.save_dir_for(save_dir, dataset_name),
            f"{pipeline_name}{RUN_FILE_SUFFIX}",
        )

    def has_cached_run(
        self,
        save_dir: str,
        dataset_name: str,
        pipeline_name: str,
        save_mode: str,
    ) -> bool:
        """
        Whether a previously written run can be replayed instead of re-run.

        A PyTerrier ``save_mode`` of ``"overwrite"`` always re-runs. Override
        alongside :meth:`run_file_path` to change where runs are looked for.
        """
        if save_mode == "overwrite":
            return False
        return os.path.exists(self.run_file_path(save_dir, dataset_name, pipeline_name))

    def load_cached_run(self, filepath: str) -> Transformer:
        """
        Load the run file at ``filepath`` into a transformer that replays it.

        See :func:`suiteeval.suite.runfiles.replay_run`.
        """
        return replay_run(filepath)

    def prepare_save_dir(self, save_dir: str, dataset_name: str) -> str:
        """Create and return the run-file directory for one dataset."""
        path = self.save_dir_for(save_dir, dataset_name)
        os.makedirs(path, exist_ok=True)
        return path

    def build_context(
        self,
        corpus_id: str,
        corpus_ds: pt.datasets.Dataset,
        config: RunConfig,
    ) -> DatasetContext:
        """
        Build the context shared by every dataset in a corpus group.

        Indexing happens once per context, so overriding this is the way to
        inject a custom context type or a bespoke index location.

        Args:
            corpus_id: The shared corpus identifier.
            corpus_ds: The PyTerrier dataset for that corpus.
            config: The resolved run configuration.

        Returns:
            DatasetContext: The context handed to pipeline generators.
        """
        if config.index_dir is None:
            return DatasetContext(corpus_ds)
        path = self.index_dir_for(config.index_dir, corpus_id)
        os.makedirs(path, exist_ok=True)
        return DatasetContext(corpus_ds, path=path)

    def prepare_topics_qrels(
        self, dataset: pt.datasets.Dataset, dataset_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch ``(topics, qrels)`` for one dataset, with identifiers as strings.

        ``_query_field`` selects the topic field. Prepared once per dataset per
        corpus group, so an override may do real work here.
        """
        topics = ensure_string_ids(dataset.get_topics(self._query_field), ("qid",))
        qrels = ensure_string_ids(dataset.get_qrels(), ("qid", "docno"))
        return topics, qrels

    def run_experiment(
        self,
        pipelines: Sequence[Transformer],
        names: Sequence[str | None],
        topics: pd.DataFrame,
        qrels: pd.DataFrame,
        dataset_name: str,
        config: RunConfig,
        **experiment_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Evaluate pipelines against one dataset.

        The single point where :func:`pyterrier.Experiment` is called; override to
        swap the evaluation backend or inject extra arguments.

        Args:
            pipelines: Pipelines to evaluate together.
            names: Display names aligned with ``pipelines``; ``None`` entries are labelled positionally.
            topics: Topics frame.
            qrels: Qrels frame.
            dataset_name: Dataset display name, used to resolve metrics.
            config: The resolved run configuration.
            **experiment_kwargs: Extra arguments forwarded to the experiment.

        Returns:
            pandas.DataFrame: The raw experiment results.
        """
        return pt.Experiment(
            list(pipelines),
            eval_metrics=self.measures_for(dataset_name, config.eval_metrics),
            topics=topics,
            qrels=qrels,
            names=fill_names(names),
            **experiment_kwargs,
        )

    def evaluate_batch(
        self,
        batch: Sequence[NamedPipeline],
        topics: pd.DataFrame,
        qrels: pd.DataFrame,
        dataset_name: str,
        config: RunConfig,
    ) -> Iterator[pd.DataFrame]:
        """
        Evaluate one batch of pipelines against one dataset.

        Pipelines with a reusable run file are replayed from disk one by one; the
        remainder are evaluated together in a single experiment so that
        cross-system tests still see the full set.

        Args:
            batch: ``(pipeline, name)`` pairs to evaluate.
            topics: Topics frame.
            qrels: Qrels frame.
            dataset_name: Dataset display name.
            config: The resolved run configuration.

        Yields:
            pandas.DataFrame: One frame per experiment performed.
        """
        pending: list[NamedPipeline] = []

        for pipeline, name in batch:
            if not (
                config.save_dir
                and name
                and self.has_cached_run(
                    config.save_dir, dataset_name, name, config.save_mode
                )
            ):
                pending.append((pipeline, name))
                continue

            filepath = self.run_file_path(config.save_dir, dataset_name, name)
            logger.info(f"Loading '{name}' for {dataset_name} from {filepath}")
            yield self.run_experiment(
                [self.load_cached_run(filepath)],
                [name],
                topics,
                qrels,
                dataset_name,
                config,
            )

        if not pending:
            return

        kwargs = dict(config.experiment_kwargs)
        if config.save_dir is not None:
            kwargs["save_dir"] = self.prepare_save_dir(config.save_dir, dataset_name)

        yield self.run_experiment(
            [pipeline for pipeline, _ in pending],
            [name for _, name in pending],
            topics,
            qrels,
            dataset_name,
            config,
            **kwargs,
        )

    def annotate_results(
        self, results: pd.DataFrame, dataset_name: str, corpus_id: str
    ) -> pd.DataFrame:
        """
        Tag a result frame with the dataset it came from.

        The default sets ``dataset`` and ignores ``corpus_id``; override to
        record more about where a row came from. Modifies ``results`` in place.
        """
        results["dataset"] = dataset_name
        return results

    @staticmethod
    def release_pipelines() -> None:
        """
        Best-effort memory cleanup between pipeline batches.

        Collects garbage and, when torch is installed with a CUDA device,
        empties its cache. Failures are ignored: both are optional.
        """
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def release_context(self, context: DatasetContext) -> None:
        """
        Release a corpus context once its group has been evaluated.

        The default does nothing: the context is dropped immediately
        afterwards, and the index it points at is meant to outlive the run.
        Override to close a handle or delete a scratch index.
        """

    @staticmethod
    def metric_columns(results: pd.DataFrame) -> list[str]:
        """
        Numeric columns of ``results`` that hold measure values.

        Override alongside :meth:`postprocess_results` when a suite reports
        columns that should not be aggregated as metrics.
        """
        return _metric_columns(results)

    def compute_overall_mean(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Append overall (geometric mean) rows across datasets for each system name.

        This first aggregates per-dataset means over repeated runs, then computes the
        geometric mean across datasets for each metric and appends rows with
        ``dataset == "Overall"``.

        Args:
            results: DataFrame with at least ``["dataset", "name"]`` and metric columns.

        Returns:
            pandas.DataFrame: The input results with additional ``Overall`` rows appended.
        """
        return append_overall(results, self.metric_columns(results))

    def postprocess_results(
        self, results: pd.DataFrame, config: RunConfig
    ) -> pd.DataFrame:
        """
        Transform the concatenated results before they are returned.

        Override to aggregate sub-datasets or reshape the table, then call
        ``super().postprocess_results(...)`` to keep the ``Overall`` rows.

        Args:
            results: Concatenation of every annotated result frame.
            config: The resolved run configuration.

        Returns:
            pandas.DataFrame: The final results table.
        """
        if results.empty or config.perquery or not config.compute_overall:
            return results
        return self.compute_overall_mean(results)

    def resolve_config(
        self,
        eval_metrics: Sequence[Any] | None = None,
        subset: str | None = None,
        compute_overall: bool = True,
        **experiment_kwargs: Any,
    ) -> RunConfig:
        """
        Split raw call arguments into suite settings and experiment kwargs.

        ``index_dir`` and ``save_dir`` are consumed here because the suite rewrites
        them per corpus and per dataset respectively; everything else is forwarded
        to :func:`pyterrier.Experiment` untouched.

        Args:
            eval_metrics: Explicit metrics overriding the suite configuration.
            subset: Optional dataset display name to restrict evaluation to.
            compute_overall: Whether to append geometric-mean ``Overall`` rows.
            **experiment_kwargs: Remaining arguments, including ``index_dir`` and ``save_dir``.

        Returns:
            RunConfig: The resolved configuration for this call.
        """
        index_dir = experiment_kwargs.pop("index_dir", None)
        save_dir = experiment_kwargs.pop("save_dir", None)
        grouped = experiment_kwargs.get("baseline") is not None
        if grouped:
            logger.warning(
                "Significance tests require pipelines to be grouped; this uses more memory."
            )
        return RunConfig(
            eval_metrics=eval_metrics,
            subset=subset,
            compute_overall=compute_overall,
            index_dir=index_dir,
            save_dir=save_dir,
            save_mode=experiment_kwargs.get("save_mode", "warn"),
            perquery=experiment_kwargs.get("perquery", False),
            grouped=grouped,
            experiment_kwargs=experiment_kwargs,
        )

    def run(
        self,
        ranking_generators: PipelineGenerators,
        config: RunConfig,
    ) -> Iterator[pd.DataFrame]:
        """
        Evaluate every pipeline against every selected dataset.

        Walks corpus groups, building one shared context per corpus so that
        indexing happens once, and releases each pipeline batch once it has been
        evaluated against all datasets sharing that corpus.

        Args:
            ranking_generators: Callable or sequence of callables producing pipelines.
            config: The resolved run configuration.

        Yields:
            pandas.DataFrame: Annotated result frames, in evaluation order.
        """
        for corpus_id, corpus_ds, members in self.iter_corpus_groups():
            selected = self.select_members(members, config.subset)
            if not selected:
                continue
            yield from self.run_corpus_group(
                corpus_id, corpus_ds, selected, ranking_generators, config
            )

    def run_corpus_group(
        self,
        corpus_id: str,
        corpus_ds: pt.datasets.Dataset,
        members: Sequence[tuple[Any, Any]],
        ranking_generators: PipelineGenerators,
        config: RunConfig,
    ) -> Iterator[pd.DataFrame]:
        """
        Evaluate every pipeline against the datasets sharing one corpus.

        One context is built for the group, so indexing happens once, and each
        batch of pipelines is released as soon as every dataset in the group
        has been evaluated against it.

        Args:
            corpus_id: The shared corpus identifier.
            corpus_ds: The PyTerrier dataset for that corpus.
            members: ``(display_key, dataset_ref)`` pairs to evaluate.
            ranking_generators: Callable or sequence of callables producing pipelines.
            config: The resolved run configuration.

        Yields:
            pandas.DataFrame: Annotated result frames, in evaluation order.
        """
        context = self.build_context(corpus_id, corpus_ds, config)
        topics_qrels = TopicsQrelsCache(self)
        try:
            for batch in self.iter_pipeline_batches(
                context, ranking_generators, config.grouped
            ):
                try:
                    yield from self.run_batch(
                        batch, members, topics_qrels, corpus_id, config
                    )
                finally:
                    del batch
                    self.release_pipelines()
        finally:
            self.release_context(context)
            del context

    def run_batch(
        self,
        batch: Sequence[NamedPipeline],
        members: Sequence[tuple[Any, Any]],
        topics_qrels: "TopicsQrelsCache",
        corpus_id: str,
        config: RunConfig,
    ) -> Iterator[pd.DataFrame]:
        """
        Evaluate one batch of pipelines against every dataset in a corpus group.

        Args:
            batch: ``(pipeline, name)`` pairs to evaluate together.
            members: ``(display_key, dataset_ref)`` pairs to evaluate against.
            topics_qrels: Per-group cache of prepared topics and qrels.
            corpus_id: Identifier of the corpus the datasets belong to.
            config: The resolved run configuration.

        Yields:
            pandas.DataFrame: Annotated result frames, in evaluation order.
        """
        for key, dataset_ref in members:
            dataset_name = self._display_name(key)
            topics, qrels = topics_qrels.get(dataset_ref, dataset_name)
            for frame in self.evaluate_batch(
                batch, topics, qrels, dataset_name, config
            ):
                yield self.annotate_results(frame, dataset_name, corpus_id)

    def __call__(
        self,
        ranking_generators: PipelineGenerators,
        eval_metrics: Sequence[Any] | None = None,
        subset: str | None = None,
        compute_overall: bool = True,
        **experiment_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Run the experiment(s) for each dataset in the suite and return a results table.

        If a ``baseline`` is provided in ``experiment_kwargs``, all pipelines are
        materialized together (grouped mode) to enable tests that require joint access
        (e.g., significance). Otherwise, pipelines are streamed one-by-one to reduce
        memory usage (sequential mode).

        Args:
            ranking_generators: Callable or sequence of callables producing pipelines
                per :class:`DatasetContext` (same conventions as in
                :meth:`coerce_pipelines_sequential`).
            eval_metrics: Optional explicit metrics to evaluate; defaults to the suite's
                configuration for each dataset.
            subset: Optional dataset display name to restrict evaluation to a single member.
            compute_overall: Whether to append geometric-mean ``Overall`` rows.
            **experiment_kwargs: Additional keyword arguments forwarded to
                :func:`pyterrier.Experiment`. If ``save_dir`` is provided, it is
                suffixed per dataset. If ``index_dir`` is provided, it is
                suffixed per corpus for index storage.

        Returns:
            pandas.DataFrame: The concatenated experiment results. When ``perquery`` is
                not set, an additional ``Overall`` row is appended per system with
                geometric-mean aggregation across datasets.

        Notes:
            This method reuses a single index per corpus group and cleans up GPU memory
            between pipeline evaluations. Subclasses should prefer overriding the
            individual hooks documented on :class:`Suite` over this method.
        """
        config = self.resolve_config(
            eval_metrics=eval_metrics,
            subset=subset,
            compute_overall=compute_overall,
            **experiment_kwargs,
        )
        frames = list(self.run(ranking_generators, config))
        results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.postprocess_results(results, config)


__all__ = ["RunConfig", "Suite", "SuiteMeta", "TopicsQrelsCache"]
