# File: tests/unit/test_suite_base.py
import pytest
import pandas as pd
import pyterrier as pt

from suiteeval.suite.base import Suite

VaswaniSuite = Suite.register(
    "vaswani",
    datasets=["vaswani"],
    metadata={
        "description": "Vaswani is a dataset for evaluating retrieval systems on a variety of topics.",
    },
)

@pytest.fixture(scope="module")
def suite():
    return VaswaniSuite

@pytest.fixture(scope="module")
def vaswani_dataset():
    # load the Vaswani dataset
    return pt.get_dataset("irds:vaswani")

def test_datasets_property_returns_dataset_instances(suite):
    datasets = list(suite.datasets)
    assert len(datasets) == 1
    name, ds = datasets[0]
    assert name == "vaswani"

class DummyTransformer(pt.Transformer):
    def transform(self, inp):
        # return an empty ranking DataFrame with standard columns
        qids = inp['qid'].unique()

        output = {
            'qid': [],
            'docno': [],
            'rank': [],
            'score': [],
        }
        for qid in qids:
            output['qid'].append(qid)
            output['docno'].append('dummy_doc')
            output['rank'].append(0)
            output['score'].append(1.0)
        return pd.DataFrame(output)

def test_call_runs_experiment_and_returns_dataframe(suite):
    ds_id, ds = list(suite.datasets)[0]
    dummy = DummyTransformer()
    def yield_pipe(context):
        yield dummy
    # Use the dummy transformer as a ranking generator
    results = suite(ranking_generators=yield_pipe)
    assert isinstance(results, pd.DataFrame)
    assert 'dataset' in results.columns
    # If not empty, all rows should have the same dataset name
    if not results.empty:
        assert results['dataset'].unique().tolist() == [ds_id]