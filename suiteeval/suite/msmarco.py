from suiteeval.suite.base import Suite
from ir_measures import nDCG

passage_datasets = [
    "msmarco-passage/trec-dl-2019/judged",
    "msmarco-passage/trec-dl-2020/judged",
    "msmarco-passage-v2/trec-dl-2022/judged",
    "msmarco-passage-v2/trec-dl-2023",
]
document_datasets = [
    "msmarco-document/trec-dl-2019/judged",
    "msmarco-document/trec-dl-2020/judged",
    "msmarco-document-v2/trec-dl-2022/judged",
    "msmarco-document-v2/trec-dl-2023",
]
measures = [nDCG @ 10]

MSMARCODocument = Suite.register(
    "msmarco/document",
    document_datasets,
    metadata={
        "official_measures": measures,
        "description": "MS MARCO Document datasets for evaluating retrieval systems on document-level tasks.",
    },
)

MSMARCOPassage = Suite.register(
    "msmarco/passage",
    passage_datasets,
    metadata={
        "official_measures": measures,
        "description": "MS MARCO Passage datasets for evaluating retrieval systems on passage-level tasks.",
    },
)

__all__ = ["MSMARCODocument", "MSMARCOPassage"]
