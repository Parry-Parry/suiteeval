from ir_measures import Success
from suiteeval.suite.base import Suite

datasets = [
    "lotte/lifestyle/test/forum",
    "lotte/lifestyle/test/search",
    "lotte/pooled/test/forum",
    "lotte/pooled/test/search",
    "lotte/recreation/test/forum",
    "lotte/recreation/test/search",
    "lotte/science/test/forum",
    "lotte/science/test/search",
    "lotte/technology/test/forum",
    "lotte/technology/test/search",
    "lotte/writing/test/forum",
    "lotte/writing/test/search",
]

measures = [Success @ 10]

Lotte = Suite.register(
    "lotte",
    datasets,
    metadata={
        "official_measures": measures,
        "description": "LoTTE (Long-Tail Topic-stratified Evaluation) is a set of test collections focused on out-of-domain evaluation. It consists of data from several StackExchanges, with relevance assumed by either by upvotes (at least 1) or being selected as the accepted answer by the question's author.",
    },
)

__all__ = ["Lotte"]
