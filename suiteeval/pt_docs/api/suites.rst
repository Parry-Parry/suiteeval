Availible Suites
===============

BEIR
====

BEIR is a heterogeneous benchmark containing diverse IR tasks.

Usage
-----
.. code-block:: python

   from suiteeval.suite import BEIR
   results = BEIR(pipelines)  # instance-style call

NanoBEIR
========

Compact BEIR subset for faster iteration.

Usage
-----
.. code-block:: python

   from suiteeval.suite import NanoBEIR
   results = NanoBEIR(pipelines)

LoTTE
=====

LoTTE (Long-Tail Topic-stratified Evaluation) is a set of test collections focused on out-of-domain evaluation. 

Usage
-----
.. code-block:: python

   from suiteeval.suite import Lotte
   results = Lotte(pipelines)

BRIGHT
======

BRIGHT comprises 12 diverse datasets, spanning biology, economics, robotics, math, code and more. The queries can be long StackExchange posts, math or code question. 

Usage
-----
.. code-block:: python

   from suiteeval.suite import BRIGHT
   results = BRIGHT(pipelines)

MS MARCO (Document & Passage)
=============================

MSMARCO is a large-scale dataset for training and evaluating information retrieval models. These suites contain TREC Deep Learning queries and relevance judgments for both document and passage retrieval tasks.

Usage
-----
.. code-block:: python

   from suiteeval.suite import MSMARCODocument, MSMARCOPassage
   doc_results = MSMARCODocument(pipelines)
   pas_results = MSMARCOPassage(pipelines)

