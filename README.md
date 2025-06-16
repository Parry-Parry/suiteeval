## 🍬 SuiteEval

Tools for running IR Evaluation Suites

### Example Usage

```python
from suiteeval import BEIR
from pyterrier_pisa import PisaIndex
import pyterrier_dr
import pyterrier_t5

monot5 = pyterrier_t5.MonoT5ReRanker
monoelectra = pyterrier_dr.ElectraScorer

def yield_my_stages(context):
    index = PisaIndex(context.path + "/index.pisa")
    index.index(context.get_corpus_iter())
    bm25 = index.bm25()
    yield bm25 >> context.text_loader() >> monot5, "BM25 >> monoT5"
    yield bm25 >> context.text_loader() >> monoelectra, "BM25 >> monoELECTRA"

results = BEIR(yield_my_stages)
```
