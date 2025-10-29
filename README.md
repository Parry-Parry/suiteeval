# 🍬 SuiteEval

Tools for running IR Evaluation Suites with PyTerrier.

## Installation

```
pip install suiteeval
```

## Example Usage

You should define a function which produces `pyterrier` pipelines, if you do not want to lookup an index, the `DatasetContext` object provides tempoary paths and a corpus iterator for indexing.

The function can either return one or more pipelines or yield pipelines in the case that more complex memory management is required. Here is an example where we only keep one neural re-ranker in memory at a time while evaluating the BEIR suite.

You can choose to either return named systems (useful for larger evaluation) or just return the systems!

```python
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
```
