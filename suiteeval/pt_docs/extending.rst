Extending a Suite
=========================================

Every suite evaluation runs through the same fixed sequence of small,
overridable steps. Instead of reimplementing :meth:`~suiteeval.suite.base.Suite.__call__`,
override the one hook that covers what you want to change.

The pipeline
-----------------------------------------

.. code-block:: text

    __call__
      └─ resolve_config          split call arguments into a RunConfig
      └─ run
         └─ iter_corpus_groups   group datasets by shared corpus
            └─ select_members    pick datasets to evaluate in this group
            └─ build_context     build the shared DatasetContext (indexes once)
            └─ iter_pipeline_batches
               └─ coerce_pipelines_sequential / _grouped
                  └─ wrap_pipeline          decorate each pipeline
               └─ prepare_topics_qrels      fetch topics and qrels
               └─ evaluate_batch
                  ├─ has_cached_run / run_file_path / load_cached_run
                  └─ run_experiment         the pt.Experiment call
                     └─ measures_for        pick metrics for a dataset
               └─ annotate_results          tag rows with their dataset
               └─ release_pipelines         free memory between batches
      └─ postprocess_results     aggregate the concatenated table
         └─ compute_overall_mean

Common changes
-----------------------------------------

Decorate every pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~suiteeval.suite.base.Suite.wrap_pipeline` is applied by both the
sequential and grouped code paths, so overriding it alone covers every
execution mode.

.. code-block:: python

    class MySuite(Suite):
        _datasets = ["my-corpus/test"]

        def wrap_pipeline(self, pipeline, context):
            return pipeline >> MyResultFilter(context.dataset.get_qrels())

Reshape the results table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~suiteeval.suite.base.Suite.postprocess_results` sees the concatenated
results before the ``Overall`` rows are added. Call ``super()`` to keep them.
This is how :class:`BEIR` collapses the twelve CQADupStack sub-collections into
a single row.

.. code-block:: python

    def postprocess_results(self, results, config):
        results = merge_sub_collections(results)
        return super().postprocess_results(results, config)

Change where runs are cached
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run files are reused automatically when ``save_dir`` is given and ``save_mode``
is not ``"overwrite"``. Override
:meth:`~suiteeval.suite.base.Suite.run_file_path` for a different on-disk
layout, or :meth:`~suiteeval.suite.base.Suite.load_cached_run` for a different
file format.

.. code-block:: python

    def run_file_path(self, save_dir, dataset_name, pipeline_name):
        return os.path.join(save_dir, f"{dataset_name}--{pipeline_name}.res.gz")

Swap the evaluation backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~suiteeval.suite.base.Suite.run_experiment` is the single place
:func:`pyterrier.Experiment` is called.

Add columns to every row
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~suiteeval.suite.base.Suite.annotate_results` receives each raw result
frame along with the dataset and corpus it came from.

.. code-block:: python

    def annotate_results(self, results, dataset_name, corpus_id):
        results = super().annotate_results(results, dataset_name, corpus_id)
        results["corpus"] = corpus_id
        return results
