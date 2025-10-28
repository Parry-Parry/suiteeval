MS MARCO (Document & Passage)
=============================

MSMARCO is a large-scale dataset for training and evaluating information retrieval models. These suites contain TREC Deep Learning queries and relevance judgments for both document and passage retrieval tasks.

Usage
-----
.. code-block:: python

   from suiteeval.suite import MSMARCODocument, MSMARCOPassage
   doc_results = MSMARCODocument(pipelines)
   pas_results = MSMARCOPassage(pipelines)