## 🍬 SuiteEval

Tools for running IR Evaluation Suites

### Example Usage

```
from suiteeval import BEIR, Temporary
from pyterrier_pisa import PisaIndex
import pyterrier_dr
import pyterrier_t5

monot5 = pyterrier_t5.MonoT5ReRanker
monoelectra = pyterrier_dr.ElectraScorer

def yield_my_stages(context):
  with Temporary(PisaIndex) as index:
    index.index(context.docs_iter())
    bm25 = index.bm25()
    yield bm25 >> context.text() >> monot5
    yield bm25 >> context.text() >> monoelectra

results = BEIR(yield_my_stages)
```
