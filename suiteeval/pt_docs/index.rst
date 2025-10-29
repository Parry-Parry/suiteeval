SuiteEval
=========

``suiteeval`` is a lightweight framework for running reproducible IR evaluation
suites over multiple datasets.

.. rubric:: Getting Started

.. code-block:: console
    :caption: Install with pip

    $ pip install suiteeval

Basic usage:

You should define a function which produces `pyterrier` pipelines, if you do not want to lookup an index, the `DatasetContext` object provides tempoary paths and a corpus iterator for indexing.

The function can either return one or more pipelines or yield pipelines in the case that more complex memory management is required. Here is an example where we only keep one neural re-ranker in memory at a time while evaluating the BEIR suite.

You can choose to either return named systems (useful for larger evaluation) or just return the systems!

.. code-block:: python
    :caption: Running a suite

    from suiteeval import BEIR
    from pyterrier_pisa import PisaIndex
    from pyterrier_dr import ElectraScorer
    from pyterrier_t5 import MonoT5ReRanker

    def pipelines(context):
       index = PisaIndex(context.path + "/index.pisa")
       index.index(context.get_corpus_iter())
       bm25 = index.bm25()
       yield bm25 >> context.text_loader() >>  MonoT5ReRanker(), "BM25 >> monoT5"
       yield bm25 >> context.text_loader() >> ElectraScorer(), "BM25 >> monoELECTRA"

    results = BEIR(pipelines)

.. toctree::
   :maxdepth: 1
   :caption: Suites

   api/suites

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/suiteeval.suite
   api/suiteeval.api