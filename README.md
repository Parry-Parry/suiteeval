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

def system(context):
    index = PisaIndex(context.path/'index.pisa')
    index.index(context.docs_iter())
    return index.bm25() >> context.text_loader() >> monot5
    # or return/yield multiple pipelines

results = BEIR(system)
```
