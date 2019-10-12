import click
import shutil
import os
import spacy
import pandas as pd
from pathlib import Path

from spacy_pl.training.model import TrainParams, SpacyModel
from spacy.gold import GoldCorpus

@click.command()
@click.argument("train-data", type=str)
@click.argument("dev-data", type=str)
@click.argument("test-data", type=str)
@click.option(
    "-o", "--output-dir", type=str, required=True,
    help="Directory to which trained model will be saved"
)
@click.option(
    "-p", "--pipeline", type=str, required=True,
    help="Pipeline of tasks to train the model for, same format as for spacy.cli.train"
)
@click.option(
    "-v", "--vectors", type=str, default="models/blank/fasttext",
    help="Path to model from which vectors will be taken"
)
@click.option(
    "-r", "--refit", type=bool, default=True,
    help="Wipe the weights of specified model out"
)
@click.option(
    "--transfer_path", type=str,
    help="Path to the model to transfer from"
)
def run_train_test(train_data, dev_data, test_data, output_dir, pipeline, vectors, refit, transfer_path):
    if transfer_path:
        from spacy.pipeline import DependencyParser
        output_path = os.path.join(output_dir, "model-final")
        os.mkdir(output_dir)
        shutil.copytree(transfer_path, output_path)
        model = spacy.load(output_path)
        print(model.pipeline)
        parser = model.create_pipe("parser")
        model.add_pipe(parser, name="parser", last=True)
        corpus = GoldCorpus(Path(train_data), Path(dev_data), limit=0)
        model.begin_training(lambda: corpus.train_tuples)
        print(model.pipeline)
        model.to_disk(output_path)

    train_params = TrainParams(
        n_iter=5,
    )
    model_init_params = {
        "pipeline": pipeline,
        "vectors_path": vectors,
        "location": output_dir,
    }

    model = SpacyModel(**model_init_params)
    model.fit(train_path=train_data, dev_path=dev_data, train_params=train_params, refit=refit)

    scores = model.score(test_data)
    scores_df = pd.DataFrame(
        {"test_data": test_data, "model_location": model.location, **scores}, index=[0]
    )
    with pd.option_context("display.max_rows", 10, "display.max_columns", 20):
        print(scores_df)

    return scores_df


if __name__ == '__main__':
    run_train_test()
