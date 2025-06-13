"""Top-level package for SuiteEval."""
__version__ = "0.1.0"

from suiteeval.suite import Suite, BEIR, Lotte, MSMARCODocument, MSMARCOPassage, NanoBEIR
from suiteeval.index import Temporary as Temporary

__all__ = [
    "Suite",
    "BEIR",
    "Lotte",
    "MSMARCODocument",
    "MSMARCOPassage",
    "NanoBEIR",
    "Temporary",
]


"""

class GeneratorContext:
    dataset: irds.Dataset

    def yeild_docs():



def whatever(documents):
    with Temporary(PisaIndex, kwarg=bla) as index:
        index.index(documents)
        return index.bm25()

crossenc = X

def mycrossencoder():
    with Temporary(PisaIndex, kwarg=bla) as index:
        index.index(documents)
        yield index.bm25() >> crossencoder
        yield index.bm25() >> crossencoder2



crossencoder 

def get_all_first_stages(context):
    pisa_index = PisaIndex(context.path + "/pisa")
    terrier_index = TerrierIndex(context.path + "/terrier")

    yield pisa.bm25() >> crossencoder
    yield terrier_index.bm25() >> crossencoder

from suiteeval import BEIR 

results = BEIR(get_all_first_stages)

"""