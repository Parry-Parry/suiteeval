from __future__ import annotations

from abc import ABC, ABCMeta
from collections.abc import Iterator
import gzip
import os
from typing import Any, Generator, Optional, Sequence, Tuple, Union
from logging import getLogger

import numpy as np
import ir_datasets as irds
from ir_measures import nDCG, Measure, parse_measure, parse_trec_measure
import pandas as pd
import pyterrier as pt
from pyterrier import Transformer

from suiteeval.context import DatasetContext
from suiteeval.suite.config import (
    RUN_FILE_COLUMNS,
    RunConfig,
    ensure_string_ids,
    slugify,
)
from suiteeval.suite.config import metric_columns as _metric_columns
from suiteeval.suite.pipelines import (
    NamedPipeline,
    PipelineGenerators,
    fill_names,
    iter_generator_output,
)
from suiteeval.utility import geometric_mean

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
    _instances: dict[str, "Suite"] = {}

    def __call__(cls, *args, **kwargs):
        # singleton: only one instance per class
        if cls.__name__ not in SuiteMeta._instances:
            SuiteMeta._instances[cls.__name__] = super().__call__(*args, **kwargs)
        return SuiteMeta._instances[cls.__name__]

    @classmethod
    def register(
        mcs,
        suite_name: str,
        datasets: list[str],
        names: Optional[list[str]] = None,
        metadata: Optional[Union[list[dict[str, Any]], dict[str, Any]]] = None,
        query_field: Optional[str] = None,
    ) -> "Suite":
        """
        Create (or retrieve) a Suite singleton that wraps the given datasets.

        Args:
            suite_name: Name to assign to the dynamically created suite subclass.
            datasets: IRDS dataset identifiers (e.g., ``"msmarco-passage/trec-dl-2019"``).
            names: Optional display names corresponding one-to-one with ``datasets``.
                Defaults to ``datasets`` when omitted.
            metadata: Optional metadata. Accepted forms:
                * ``None`` → per-dataset empty dicts
                * ``list[dict]`` → each entry applies to the corresponding dataset in ``names``/``datasets``
                * ``dict[str, dict]`` → explicit mapping from dataset name/ID to metadata dict
                * ``dict[str, Any]`` where values are not dicts → treated as flat metadata applied to all
            query_field: Optional topic field name to use when fetching topics (e.g., ``"title"``).

        Returns:
            Suite: The singleton instance of the dynamically created suite class.

        Raises:
            ValueError: If ``metadata`` has an unsupported shape or length.
        """
        # if already registered, return existing instance
        if suite_name in mcs._classes:
            return mcs._classes[suite_name]()

        # build the dataset name → dataset_id mapping
        ds_names = names or datasets
        dataset_map = dict(zip(ds_names, datasets))

        # normalise metadata:
        #  • None            → empty per-dataset dicts
        #  • list[dict]      → metadata[i] applies to ds_names[i]
        #  • dict[str,dict]  → per-dataset mapping (keys are names or IDs)
        #  • dict[k,v] where v is NOT a dict → flat metadata for all
        if metadata is None:
            metadata_map = {name: {} for name in ds_names}
        elif isinstance(metadata, list):
            if len(metadata) != len(ds_names):
                raise ValueError("`metadata` list must match number of datasets")
            metadata_map = dict(zip(ds_names, metadata))
        elif isinstance(metadata, dict):
            if all(not isinstance(v, dict) for v in metadata.values()):
                metadata_map = {name: metadata for name in ds_names}
            else:
                metadata_map = metadata
        else:
            raise ValueError(f"Unsupported metadata type: {type(metadata)}")

        # dynamically create subclass with mappings
        attrs = {
            "_datasets": dataset_map,  # display-name -> dataset_id
            "_dataset_ids": dataset_map,  # alias used by other methods
            "_metadata": metadata_map,
            "_query_field": query_field,
        }
        new_cls = mcs(suite_name, (Suite,), attrs)

        # store class and return its singleton instance
        mcs._classes[suite_name] = new_cls
        return new_cls()


class Suite(ABC, metaclass=SuiteMeta):
    """
    Abstract base class for a set of related evaluations across one or more datasets.

    Subclasses (or classes created via :meth:`SuiteMeta.register`) must populate:

    Attributes:
        _datasets: Either a ``dict[str, str]`` mapping display name → IRDS dataset ID,
            or a ``list[str]`` of IRDS dataset IDs.
        _dataset_ids: Normalized mapping of display name → IRDS dataset ID
            (filled in by registration helpers).
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
    :meth:`build_context`           Build the shared per-corpus context
    :meth:`iter_pipeline_batches`   Decide how pipelines are batched
    :meth:`wrap_pipeline`           Decorate each pipeline (filters, etc.)
    :meth:`prepare_topics_qrels`    Fetch and normalise topics/qrels
    :meth:`measures_for`            Choose metrics for a dataset
    :meth:`has_cached_run` /
    :meth:`load_cached_run` /
    :meth:`run_file_path`           Reuse run files written by a previous call
    :meth:`run_experiment`          The :func:`pyterrier.Experiment` call itself
    :meth:`annotate_results`        Tag result rows with their dataset
    :meth:`release_pipelines`       Free memory between batches
    :meth:`postprocess_results`     Aggregate the concatenated results
    ==============================  ==========================================

    Notes:
        Instances are singletons per subclass (enforced by :class:`SuiteMeta`).
    """

    _datasets: Union[list[str], dict[str, str]] = {}
    _dataset_ids: dict[str, str] = {}
    _metadata: dict[str, Any] = {}
    _measures: Union[list[Measure], dict[str, list[Measure]], None] = None
    _default_measures: list[Measure] = [nDCG @ 10]
    _query_field: Optional[str] = None

    def __init__(self):
        self.coerce_measures(self._metadata)
        if "description" in self._metadata:
            self.__doc__ = self._metadata["description"]
        self.__post_init__()

    def __post_init__(self):
        assert self._datasets, (
            "Suite must have at least one dataset defined in _datasets"
        )

        if not isinstance(self._datasets, (dict, list)):
            raise AssertionError(
                "Suite _datasets must be a dict[name->id] or a list[dataset_id]"
            )

        if isinstance(self._datasets, dict):
            for name, ds in self._datasets.items():
                if not isinstance(name, str):
                    raise AssertionError(
                        f"Suite _datasets keys must be strings, got {type(name)}"
                    )
                self._validate_dataset(ds, repr(name))
        else:
            for i, ds in enumerate(self._datasets):
                self._validate_dataset(ds, str(i))

        assert self._measures is not None, (
            "Suite must have measures defined in _measures"
        )

    @staticmethod
    def _validate_dataset(ds: Any, where: str) -> None:
        """Raise unless ``ds`` is a string ID or a dataset-like object."""
        if isinstance(ds, str):
            return
        if all(hasattr(ds, attr) for attr in ("_irds_id", "get_topics", "get_qrels")):
            return
        raise AssertionError(
            f"Suite _datasets[{where}] must be a string ID or a dataset "
            "object with _irds_id, get_topics, and get_qrels"
        )

    # ------------------------------------------------------------------
    # Dataset resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _get_irds_id(ds_id_or_obj: Any) -> str:
        """
        Extract the IRDS ID from either a string ID or a dataset object.

        Args:
            ds_id_or_obj: Either a string IRDS ID or an object with `_irds_id` attribute.

        Returns:
            str: The IRDS ID.
        """
        if isinstance(ds_id_or_obj, str):
            return ds_id_or_obj
        return ds_id_or_obj._irds_id

    @classmethod
    def _display_name(cls, name_or_obj: Any) -> str:
        """Resolve a dataset key to the string used in results and paths."""
        if isinstance(name_or_obj, str):
            return name_or_obj
        return cls._get_irds_id(name_or_obj)

    @staticmethod
    def _get_dataset_object(ds_id_or_obj: Any) -> pt.datasets.Dataset:
        """
        Get a PyTerrier Dataset object from either a string ID or a dataset object.

        Args:
            ds_id_or_obj: Either a string IRDS ID or a dataset object.

        Returns:
            pt.datasets.Dataset: The dataset object.
        """
        if isinstance(ds_id_or_obj, str):
            return pt.get_dataset(f"irds:{ds_id_or_obj}")
        return ds_id_or_obj

    def _dataset_items(self) -> list[Tuple[Any, Any]]:
        """Normalise ``_datasets`` to a list of ``(display_key, dataset_ref)``."""
        if isinstance(self._datasets, dict):
            return list(self._datasets.items())
        return [(ds, ds) for ds in self._datasets]

    def iter_corpus_groups(
        self,
    ) -> Iterator[Tuple[str, pt.datasets.Dataset, list[Tuple[Any, Any]]]]:
        """
        Group the suite's datasets by the corpus they share.

        Membership is decided by :func:`ir_datasets.docs_parent_id`, so datasets
        built on the same document collection are indexed once. Override to
        impose a different grouping.

        Yields:
            tuple[str, pyterrier.datasets.Dataset, list[tuple[Any, Any]]]:
                ``(corpus_id, corpus_dataset, [(display_key, dataset_ref), ...])``.
        """
        groups: dict[str, dict] = {}
        for name, ds_id_or_obj in self._dataset_items():
            irds_id = self._get_irds_id(ds_id_or_obj)
            try:
                corpus_id = irds.docs_parent_id(irds_id) or irds_id
            except Exception:
                corpus_id = irds_id

            if corpus_id not in groups:
                corpus_ds = self._get_dataset_object(
                    corpus_id if isinstance(corpus_id, str) else irds_id
                )
                groups[corpus_id] = {"corpus_ds": corpus_ds, "members": []}

            groups[corpus_id]["members"].append((name, ds_id_or_obj))

        # deterministic iteration order (insertion order is fine here)
        for corpus_id, group in groups.items():
            yield corpus_id, group["corpus_ds"], group["members"]

    def select_members(
        self, members: Sequence[Tuple[Any, Any]], subset: Optional[str]
    ) -> list[Tuple[Any, Any]]:
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
    def datasets(self) -> Generator[Tuple[str, pt.datasets.Dataset], None, None]:
        """
        Iterate over declared datasets yielding display name and PyTerrier dataset.

        Yields:
            tuple[str, pyterrier.datasets.Dataset]: Pairs of (name, dataset object).

        Raises:
            ValueError: If ``_datasets`` has an invalid type.
        """
        if not isinstance(self._datasets, (list, dict)):
            raise ValueError(
                "Suite _datasets must be a list or dict mapping names to dataset IDs."
            )
        for name, ds_id_or_obj in self._dataset_items():
            yield self._display_name(name), self._get_dataset_object(ds_id_or_obj)

    # ------------------------------------------------------------------
    # Measures
    # ------------------------------------------------------------------

    @staticmethod
    def parse_measures(measures: Sequence[Union[str, Measure]]) -> list[Measure]:
        """
        Convert a list of measure strings or :class:`ir_measures.Measure` objects
        into a flat ``list[Measure]``.

        Args:
            measures: A sequence containing measure strings (e.g., ``"nDCG@10"``)
                and/or :class:`ir_measures.Measure` instances.

        Returns:
            list[Measure]: Parsed measure objects.

        Raises:
            ValueError: If a string entry cannot be parsed by either
                :func:`ir_measures.parse_measure` or :func:`ir_measures.parse_trec_measure`,
                or if an entry has an invalid type.
        """
        out: list[Measure] = []
        for m in measures:
            if isinstance(m, Measure):
                out.append(m)
                continue

            if not isinstance(m, str):
                raise ValueError(f"Invalid measure type: {type(m)}")

            candidates: list[Measure] = []
            for parser in (parse_measure, parse_trec_measure):
                try:
                    parsed = parser(m)
                except ValueError:
                    continue
                candidates.extend(
                    [parsed] if isinstance(parsed, Measure) else list(parsed)
                )
            if not candidates:
                raise ValueError(f"Unrecognised measure string: {m!r}")
            out.extend(candidates)

        return out

    def coerce_measures(self, metadata: dict[str, Any]) -> None:
        """
        Populate ``self._measures`` by aggregating available sources in priority order:

        1. Global ``metadata['official_measures']`` if present.
        2. Per-dataset ``metadata[name]['official_measures']`` if present.
        3. IRDS documentation ``official_measures`` for each dataset (when available).

        If no measures are discovered, falls back to ``_default_measures``.

        Args:
            metadata: The suite metadata dictionary as configured at construction time.

        Returns:
            None
        """
        if self._measures is not None:
            return

        measures: list[Measure] = []
        seen: set[str] = set()

        def _add_many(items: Optional[Sequence[Union[str, Measure]]]) -> None:
            for m in self.parse_measures(items or []):
                if str(m) not in seen:
                    measures.append(m)
                    seen.add(str(m))

        if isinstance(metadata, dict):
            # (1) global metadata, then (2) per-dataset metadata
            _add_many(metadata.get("official_measures"))
            for name in self._datasets:
                per_dataset = metadata.get(name, {})
                if isinstance(per_dataset, dict):
                    _add_many(per_dataset.get("official_measures"))

        # (3) ir_datasets documentation
        if isinstance(self._dataset_ids, dict):
            for name, ds_id in self._dataset_ids.items():
                try:
                    docs = getattr(irds.load(ds_id), "documentation", lambda: None)()
                    if isinstance(docs, dict):
                        _add_many(docs.get("official_measures"))
                except Exception as e:
                    logger.warning(
                        f"Failed to load measures from documentation for '{name}' ({ds_id}): {e}"
                    )

        if not measures:
            logger.warning(
                f"No measures discovered; defaulting to {self._default_measures}."
            )
            measures = list(self._default_measures)

        self._measures = measures

    def get_measures(self, dataset: str) -> list[Measure]:
        """
        Resolve the measures configured for a given dataset name.

        Args:
            dataset: Dataset display name as used in this suite.

        Returns:
            list[Measure]: The list configured for this dataset (or the suite-wide
                list if a single list is maintained). Falls back to
                ``_default_measures`` when the dataset is unknown.
        """
        if isinstance(self._measures, list):
            return self._measures
        return self._measures.get(dataset, self._default_measures)

    def measures_for(
        self, dataset_name: str, eval_metrics: Optional[Sequence[Any]] = None
    ) -> Sequence[Any]:
        """
        Decide which metrics to evaluate for one dataset.

        Args:
            dataset_name: Dataset display name.
            eval_metrics: Explicit metrics supplied by the caller, if any.

        Returns:
            Sequence[Any]: ``eval_metrics`` when given, else the suite's configuration.
        """
        if eval_metrics is not None:
            return eval_metrics
        return self.get_measures(dataset_name)

    # ------------------------------------------------------------------
    # Pipeline coercion
    # ------------------------------------------------------------------

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
            tuple[Transformer, Optional[str]]: The pipeline and an optional display name.

        Raises:
            ValueError: If a generator yields an invalid structure.
        """
        for pipeline, name in iter_generator_output(context, pipeline_generators):
            yield self.wrap_pipeline(pipeline, context), name

    def coerce_pipelines_grouped(
        self,
        context: DatasetContext,
        pipeline_generators: PipelineGenerators,
    ) -> Tuple[list[Transformer], Optional[list[str]]]:
        """
        Materialize all pipelines (and optional names) into lists.

        Use this when downstream evaluation requires access to the full set of systems
        simultaneously (e.g., significance tests).

        Args:
            context: The shared :class:`DatasetContext` for the current corpus group.
            pipeline_generators: Callable or sequence of callables following the same
                conventions as in :meth:`coerce_pipelines_sequential`.

        Returns:
            tuple[list[Transformer], Optional[list[str]]]:
                A list of pipelines and, if provided, a list of corresponding names.
                If no names were supplied, returns ``None`` for the second element.

        Raises:
            ValueError: If the generators produce no pipelines or an invalid structure.
        """
        pipelines: list[Transformer] = []
        names: list[Optional[str]] = []
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
            list[tuple[Transformer, Optional[str]]]: One batch of named pipelines.
        """
        if not grouped:
            for named_pipeline in self.coerce_pipelines_sequential(
                context, pipeline_generators
            ):
                yield [named_pipeline]
            return

        pipelines, names = self.coerce_pipelines_grouped(context, pipeline_generators)
        yield list(zip(pipelines, names or [None] * len(pipelines)))

    # ------------------------------------------------------------------
    # Run files
    # ------------------------------------------------------------------

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
            self.save_dir_for(save_dir, dataset_name), f"{pipeline_name}.res.gz"
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

        Args:
            save_dir: Root run-file directory.
            dataset_name: Dataset display name.
            pipeline_name: Pipeline display name.
            save_mode: PyTerrier save mode; ``"overwrite"`` always re-runs.

        Returns:
            bool: True when a reusable run file exists.
        """
        if save_mode == "overwrite":
            return False
        return os.path.exists(self.run_file_path(save_dir, dataset_name, pipeline_name))

    def load_cached_run(self, filepath: str) -> Transformer:
        """
        Load a gzipped TREC run file into a transformer that replays it.

        Args:
            filepath: Path returned by :meth:`run_file_path`.

        Returns:
            Transformer: A transformer yielding the stored ranking.
        """
        with gzip.open(filepath, "rt") as f:
            run = pd.read_csv(f, sep=r"\s+", header=None, names=RUN_FILE_COLUMNS)
        run = ensure_string_ids(run[["qid", "docno", "score", "rank"]])
        return pt.Transformer.from_df(run)

    def prepare_save_dir(self, save_dir: str, dataset_name: str) -> str:
        """Create and return the run-file directory for one dataset."""
        path = self.save_dir_for(save_dir, dataset_name)
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch topics and qrels for one dataset, with identifier columns as strings.

        Args:
            dataset: A :class:`pyterrier.datasets.Dataset` instance.
            dataset_name: Dataset display name.

        Returns:
            tuple[pandas.DataFrame, pandas.DataFrame]: ``(topics, qrels)``.
        """
        topics = ensure_string_ids(dataset.get_topics(self._query_field), ("qid",))
        qrels = ensure_string_ids(dataset.get_qrels(), ("qid", "docno"))
        return topics, qrels

    def run_experiment(
        self,
        pipelines: Sequence[Transformer],
        names: Sequence[Optional[str]],
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

        Args:
            results: Raw frame returned by :meth:`run_experiment`.
            dataset_name: Dataset display name.
            corpus_id: Identifier of the corpus the dataset belongs to.

        Returns:
            pandas.DataFrame: The annotated frame.
        """
        results["dataset"] = dataset_name
        return results

    @staticmethod
    def release_pipelines() -> None:
        """
        Best-effort memory cleanup between pipeline batches.

        Calls ``gc.collect()`` and, if ``torch.cuda.is_available()``, empties the CUDA cache.
        Silently ignores any exceptions (CUDA and torch are optional).
        """
        import gc

        gc.collect()
        try:
            import torch  # noqa: WPS433 — optional dependency

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

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
        # Idempotency check: skip if Overall rows already exist
        if "dataset" not in results.columns or "Overall" in results["dataset"].values:
            return results

        measure_cols = self.metric_columns(results)
        if not measure_cols:
            return results

        per_dataset = (
            results.groupby(["dataset", "name"], dropna=False)[measure_cols]
            .mean()
            .reset_index()
        )

        overall_rows = []
        for name, group in per_dataset.groupby("name", dropna=False):
            row = {"dataset": "Overall", "name": name}
            for col in measure_cols:
                values = pd.to_numeric(group[col], errors="coerce").dropna().values
                if np.any(values <= 0):
                    values = values + 1e-12
                row[col] = geometric_mean(values)
            overall_rows.append(row)

        return pd.concat([results, pd.DataFrame(overall_rows)], ignore_index=True)

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

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def resolve_config(
        self,
        eval_metrics: Optional[Sequence[Any]] = None,
        subset: Optional[str] = None,
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

            context = self.build_context(corpus_id, corpus_ds, config)
            try:
                batches = self.iter_pipeline_batches(
                    context, ranking_generators, config.grouped
                )
                for batch in batches:
                    try:
                        for key, dataset_ref in selected:
                            dataset_name = self._display_name(key)
                            dataset = self._get_dataset_object(dataset_ref)
                            topics, qrels = self.prepare_topics_qrels(
                                dataset, dataset_name
                            )
                            for frame in self.evaluate_batch(
                                batch, topics, qrels, dataset_name, config
                            ):
                                yield self.annotate_results(
                                    frame, dataset_name, corpus_id
                                )
                    finally:
                        del batch
                        self.release_pipelines()
            finally:
                del context

    def __call__(
        self,
        ranking_generators: PipelineGenerators,
        eval_metrics: Optional[Sequence[Any]] = None,
        subset: Optional[str] = None,
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


__all__ = ["Suite", "SuiteMeta", "RunConfig"]
