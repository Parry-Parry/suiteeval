from abc import ABC
import pandas as pd
import pyterrier as pt
from functools import cached_property
import ir_datasets as irds
from typing import Dict, Generator


class Suite(ABC):
    """
    Abstract base class for a suite of evaluations.
    """
    _datasets: Dict[str, str] = {}

    @staticmethod
    def get_topics(dataset) -> pd.DataFrame:
        """
        Get the topics from a dataset.

        Args:
            dataset (ir_datasets.Dataset): The dataset to get topics from.
        """
        topics = pd.DataFrame(dataset.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})
        return topics

    @staticmethod
    def get_qrels(dataset) -> pd.DataFrame:
        """
        Get the qrels from a dataset.

        Args:
            dataset (ir_datasets.Dataset): The dataset to get qrels from.
        """
        qrels = pd.DataFrame(dataset.qrels_iter()).rename(columns={'query_id': 'qid', 'document_id': 'docno', 'relevance': 'label'})
        return qrels

    @cached_property
    def datasets(self) -> Generator[str, irds.Dataset]:
        """
        List of (name, ir_datasets.Dataset) pairs included in the suite.
        """
        for name, dataset in self._datasets.items():
            yield name, irds.load(dataset)

    def __call__(self,
                 pipelines,
                 eval_metrics,
                 names=None,
                 perquery=False,
                 dataframe=True,
                 batch_size=None,
                 filter_by_qrels=False,
                 filter_by_topics=True,
                 baseline=None,
                 test='t',
                 correction=None,
                 correction_alpha=0.05,
                 highlight=None,
                 round=None,
                 verbose=False,
                 save_dir=None,
                 save_mode='warn',
                 save_format='trec',
                 precompute_prefix=True,
                 **kwargs) -> pd.DataFrame:
        """
        Run the evaluation over all datasets in the suite.

        Args:
            pipelines (list): A list of PyTerrier transformer pipelines.
            names (list of str, optional): Names for each pipeline.
            **kwargs: All other keyword arguments accepted by pt.Experiment,
                      e.g. eval_metrics, perquery, dataframe, batch_size, etc.

        Returns:
            pd.DataFrame: Concatenated results from all datasets, with an
                          added 'dataset' column.
        """
        results = []
        for ds_name, ds in self.datasets:
            # run per‐dataset evaluation
            topics = self.get_topics(ds)
            qrels = self.get_qrels(ds)
            df = self.evaluate(
                pipelines=pipelines,
                eval_metrics=eval_metrics,
                topics=topics,
                qrels=qrels,
                names=names,
                perquery=perquery,
                dataframe=dataframe,
                batch_size=batch_size,
                filter_by_qrels=filter_by_qrels,
                filter_by_topics=filter_by_topics,
                baseline=baseline,
                test=test,
                correction=correction,
                correction_alpha=correction_alpha,
                highlight=highlight,
                round=round,
                verbose=verbose,
                save_dir=save_dir,
                save_mode=save_mode,
                save_format=save_format,
                precompute_prefix=precompute_prefix,
                **kwargs
            )
            # tag with dataset name
            df['dataset'] = ds_name
            results.append(df)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            # no datasets: empty dataframe
            return pd.DataFrame()

    def evaluate(
        self,
        pipelines,
        eval_metrics,
        topics,
        qrels,
        names=None,
        perquery=False,
        dataframe=True,
        batch_size=None,
        filter_by_qrels=False,
        filter_by_topics=True,
        baseline=None,
        test='t',
        correction=None,
        correction_alpha=0.05,
        highlight=None,
        round=None,
        verbose=False,
        save_dir=None,
        save_mode='warn',
        save_format='trec',
        precompute_prefix=False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Default evaluation using pt.Experiment on a single dataset.

        Args:
            pipelines (list): A list of PyTerrier transformer pipelines.
            dataset (ir_datasets.Dataset): The dataset to evaluate.
            eval_metrics (list of str): Measures to compute (e.g. ["map","ndcg"]).
            names (list of str, optional): Names for each pipeline.
            ... all other pt.Experiment kwargs ...

        Returns:
            pd.DataFrame: Evaluation results for this dataset.
        """

        return pt.Experiment(
            retr_systems=pipelines,
            topics=topics,
            qrels=qrels,
            eval_metrics=eval_metrics,
            names=names,
            perquery=perquery,
            dataframe=dataframe,
            batch_size=batch_size,
            filter_by_qrels=filter_by_qrels,
            filter_by_topics=filter_by_topics,
            baseline=baseline,
            test=test,
            correction=correction,
            correction_alpha=correction_alpha,
            highlight=highlight,
            round=round,
            verbose=verbose,
            save_dir=save_dir,
            save_mode=save_mode,
            save_format=save_format,
            precompute_prefix=precompute_prefix,
            **kwargs
        )
