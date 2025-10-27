## 🍬 SuiteEval

Tools for running IR Evaluation Suites with PyTerrier.

### Example Usage

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
