## 🍬 SuiteEval

Tools for running IR Evaluation Suites with PyTerrier.

### Example Usage

The `systems` function defines the retrieval systems to be evaluated. It takes a `context` object as an argument, which provides access to the corpus and other resources needed for indexing and retrieval. The function can either return one or more pipelines or yield pipelines in the case that more complex memory management is required. Here is an example where we only keep one neural re-ranker in memory at a time while evaluating the BEIR suite.

```python
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
```
