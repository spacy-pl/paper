import click
import pandas as pd

from spacy_pl.training.model import TrainParams, SpacyModel


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
def run_train_test(train_data, dev_data, test_data, output_dir, pipeline, vectors):
    train_params = TrainParams(
        n_iter=5,
    )
    model_init_params = {
        "pipeline": pipeline,
        "vectors_path": vectors,
        "location": output_dir,
    }

    model = SpacyModel(**model_init_params)
    model.fit(train_path=train_data, dev_path=dev_data, train_params=train_params)

    scores = model.score(test_data)
    scores_df = pd.DataFrame(
        {"test_data": test_data, "model_location": model.location, **scores}, index=[0]
    )
    with pd.option_context("display.max_rows", 10, "display.max_columns", 20):
        print(scores_df)

    return scores_df


if __name__ == '__main__':
    run_train_test()
