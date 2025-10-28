SuiteEval
=========

``suiteeval`` is a lightweight framework for running reproducible IR evaluation
suites over multiple datasets.

.. rubric:: Getting Started

.. code-block:: console
    :caption: Install with pip

    $ pip install suiteeval

Basic usage:

.. code-block:: python
    :caption: Running a suite

    from suiteeval import BEIR
    from pyterrier_pisa import PisaIndex
    from pyterrier_dr import ElectraScorer
    from pyterrier_t5 import MonoT5ReRanker

    def systems(context):
       index = PisaIndex(context.path + "/index.pisa")
       index.index(context.get_corpus_iter())
       bm25 = index.bm25()
       yield bm25 >> context.text_loader() >>  MonoT5ReRanker(), "BM25 >> monoT5"
       yield bm25 >> context.text_loader() >> ElectraScorer(), "BM25 >> monoELECTRA"

    results = BEIR(systems)

.. toctree::
   :maxdepth: 1
   :caption: Suites

   api/suiteeval.suite.beir
   api/suiteeval.suite.bright
   api/suiteeval.suite.lotte
   api/suiteeval.suite.msmarco
   api/suiteeval.suite.nanobeir

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/suiteeval.suite
   api/suiteeval.context
   api/suiteeval.utility
   api/suiteeval.reranking
   api/suiteeval.reranking.bm25
   api/suiteeval.reranking.biencoder
   api/suiteeval.reranking.splade