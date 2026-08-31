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

    from suiteeval import NanoBEIR
    from pyterrier_pisa import PisaIndex
    from pyterrier_dr import ElectraScorer
    from pyterrier_t5 import MonoT5ReRanker

    def pipelines(context):
       index = PisaIndex(context.path + "/index.pisa")
       index.index(context.get_corpus_iter())
       bm25 = index.bm25(num_results=10)
       yield bm25 >> context.text_loader() >>  MonoT5ReRanker(), "BM25 >> monoT5"
       yield bm25 >> context.text_loader() >> ElectraScorer(), "BM25 >> monoELECTRA"

    results = BEIR(pipelines)

would produce a table as follows:

====  ===================  =========  ==========================
  ..  name                   nDCG@10  dataset
====  ===================  =========  ==========================
   0  BM25 >> monoT5        0.26704   nano-beir/arguana
   1  BM25 >> monoELECTRA   0.311608  nano-beir/arguana
   2  BM25 >> monoT5        0.35844   nano-beir/climate-fever
   3  BM25 >> monoELECTRA   0.369699  nano-beir/climate-fever
   4  BM25 >> monoT5        0.647339  nano-beir/dbpedia-entity
   5  BM25 >> monoELECTRA   0.647961  nano-beir/dbpedia-entity
   6  BM25 >> monoT5        0.895196  nano-beir/fever
   7  BM25 >> monoELECTRA   0.896831  nano-beir/fever
   8  BM25 >> monoT5        0.482808  nano-beir/fiqa
   9  BM25 >> monoELECTRA   0.478316  nano-beir/fiqa
  10  BM25 >> monoT5        0.881539  nano-beir/hotpotqa
  11  BM25 >> monoELECTRA   0.878735  nano-beir/hotpotqa
  12  BM25 >> monoT5        0.58611   nano-beir/msmarco
  13  BM25 >> monoELECTRA   0.576923  nano-beir/msmarco
  14  BM25 >> monoT5        0.33767   nano-beir/nfcorpus
  15  BM25 >> monoELECTRA   0.333013  nano-beir/nfcorpus
  16  BM25 >> monoT5        0.721034  nano-beir/nq
  17  BM25 >> monoELECTRA   0.696591  nano-beir/nq
  18  BM25 >> monoT5        0.900461  nano-beir/quora
  19  BM25 >> monoELECTRA   0.877153  nano-beir/quora
  20  BM25 >> monoT5        0.359567  nano-beir/scidocs
  21  BM25 >> monoELECTRA   0.351578  nano-beir/scidocs
  22  BM25 >> monoT5        0.761235  nano-beir/scifact
  23  BM25 >> monoELECTRA   0.741878  nano-beir/scifact
  24  BM25 >> monoT5        0.709486  nano-beir/webis-touche2020
  25  BM25 >> monoELECTRA   0.704123  nano-beir/webis-touche2020
  26  BM25 >> monoELECTRA   0.56563   Overall
  27  BM25 >> monoT5        0.564356  Overall
====  ===================  =========  ==========================


.. toctree::
   :maxdepth: 1
   :caption: Contents

   Suites <suites>
   Extending a Suite <extending>
   API Reference <api>
