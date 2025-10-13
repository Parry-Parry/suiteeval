import click
from pyterrier_colbert.indexing import ColBERTIndexer
from pyterrier_colbert.ranking import ColBERTFactory
from pyterrier_pisa import PisaIndex

from suiteeval.context import DatasetContext
from suiteeval import BEIR


@click.command()
@click.option("--save-path", type=str, required=True, help="Path to save the CSV results.")
@click.option("--checkpoint", type=str, default="bert-base-uncased", help="Checkpoint for ColBERT.")
@click.option("--batch-size", type=int, default=512, help="Batch size for processing.")
def main(
        save_path: str,
        checkpoint: str = "bert-base-uncased",
        batch_size: int = 512,
        ):
    def pipelines(context: DatasetContext):
        colbert_indexer = ColBERTIndexer(checkpoint, context.path + "/colbert_index", "colbert")
        pisa_index = PisaIndex(context.path + "/index.pisa", stemmer="none")

        colbert_indexer.index(context.get_corpus_iter())

        del colbert_indexer

        colbert = ColBERTFactory(checkpoint, context.path + "/colbert_index", "colbert")

        yield colbert.end_to_end(batch_size=batch_size), "ColBERT end-to-end"

        pisa_index.index(context.get_corpus_iter())
        bm25 = pisa_index.bm25()

        yield bm25 >> context.text_loader() >> colbert.text_scorer(batch_size=batch_size), "BM25 >> ColBERT"

    result = BEIR(pipelines)
    result.to_csv(save_path)


if __name__ == "__main__":
    main()
