import pytest
import pandas as pd
import ir_datasets as irds
from suiteeval.suite.base import SuiteMeta, Suite, parse_measures
from ir_measures import nDCG, Measure


class DummyDataset:
    def __init__(self, docs, queries, qrels):
        self._docs = docs
        self._queries = queries
        self._qrels = qrels

    def documents_iter(self):
        for doc in self._docs:
            yield {'doc_id': doc['doc_id'], 'text': doc['text']}

    def queries_iter(self):
        for q in self._queries:
            yield q

    def qrels_iter(self):
        for r in self._qrels:
            yield r


@pytest.fixture(autouse=True)
def patch_ir_datasets(monkeypatch):
    """
    Monkey-patch `irds.load` so that 'dummy' returns our DummyDataset.
    """
    def load(ds_id):
        if ds_id == 'dummy':
            return DummyDataset(
                docs=[
                    {'doc_id': 'd1', 'text': 'text1'},
                    {'doc_id': 'd2', 'text': 'text2'}
                ],
                queries=[
                    {'query_id': 'q1', 'text': 'foo'},
                    {'query_id': 'q2', 'text': 'bar'}
                ],
                qrels=[
                    {'query_id': 'q1', 'document_id': 'd1', 'relevance': 1}
                ]
            )
        raise ValueError(f"Unknown dataset: {ds_id}")
    monkeypatch.setattr(irds, 'load', load)
    yield


class TestParseMeasures:
    def test_parse_valid_string(self):
        measures = parse_measures(['ndcg'])
        assert all(isinstance(m, Measure) for m in measures)
        assert any('ndcg' in m.name.lower() for m in measures)

    def test_parse_invalid_string_raises(self):
        with pytest.raises(ValueError):
            parse_measures(['this_does_not_exist'])

    def test_parse_instance_passthrough(self):
        m = nDCG @ 10
        measures = parse_measures([m])
        assert measures == [m]


class TestSuiteMetaRegistration:
    def test_register_returns_singleton(self):
        suite1 = SuiteMeta.register('TestSuite', ['dummy'])
        suite2 = SuiteMeta.register('TestSuite', ['dummy'])
        assert suite1 is suite2
        assert hasattr(suite1, '_datasets')
        assert 'dummy' in suite1._datasets.values()

    def test_register_with_names_and_metadata(self):
        metadata = [{'official_measures': ['ndcg@5']}]
        suite = SuiteMeta.register('CustomSuite', ['dummy'], names=['D'], metadata=metadata)
        # The dataset mapping should use the provided name
        assert 'D' in suite._datasets
        # coerce_measures should pick up official_measures from metadata
        suite2 = SuiteMeta._instances['CustomSuite']
        assert any(m.name.lower().startswith('ndcg') for m in suite2._measures)


class TestSuiteCoreFunctionality:
    @pytest.fixture
    def suite(self):
        # Create a fresh suite instance wrapping the 'dummy' dataset
        return SuiteMeta.register('TmpSuite', ['dummy'])

    def test_get_topics(self, suite):
        ds = irds.load('dummy')
        df = suite.get_topics(ds)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['qid', 'query']
        assert set(df['qid']) == {'q1', 'q2'}

    def test_get_qrels(self, suite):
        ds = irds.load('dummy')
        df = suite.get_qrels(ds)
        assert list(df.columns) == ['qid', 'docno', 'label']
        assert df.loc[0, 'qid'] == 'q1'
        assert df.loc[0, 'docno'] == 'd1'
        assert df.loc[0, 'label'] == 1

    def test_datasets_generator(self, suite):
        names = [name for name, _ in suite.datasets]
        assert names == ['dummy']

    def test_coerce_pipelines_passes_through_docs(self, suite):
        ds = irds.load('dummy')
        # A trivial generator: returns the iterator unchanged
        gens = [lambda doc_iter: doc_iter]
        pipelines = list(suite.coerce_pipelines(ds, gens))
        assert len(pipelines) == 1
        docs_from_pipeline = list(pipelines[0]())
        docs_direct = list(ds.documents_iter())
        assert docs_from_pipeline == docs_direct

    def test_get_measures_list(self, suite):
        # When _measures is a list, get_measures should return it regardless of dataset key
        suite._measures = [nDCG @ 10]
        ms = suite.get_measures('anything')
        assert ms == suite._measures