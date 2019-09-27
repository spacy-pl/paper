import shutil
from tempfile import TemporaryDirectory
import json
from pathlib import Path
import typing as T
import os

import click
import numpy as np
import pandas as pd

from spacy_pl.training.model import SpacyModel, TrainParams

np.random.seed(42)


def split_data_kfold(input_file: str, output_dir: str, n_splits: int, train_frac: float):
    data = json.load(open(input_file))
    assert type(data == list)

    data_arr = np.array(data, dtype='object')
    fold_id_arr = np.random.choice(n_splits, len(data_arr), replace=True)

    print("Splitting data for k-fold cross validation:")
    for fold_idx in range(n_splits):
        test_ids = fold_id_arr == fold_idx
        train_and_dev_ids = fold_id_arr != fold_idx

        train_not_dev_ids = np.cumsum(train_and_dev_ids) < sum(train_and_dev_ids) * train_frac
        train_ids = np.logical_and(train_and_dev_ids, train_not_dev_ids)
        dev_ids = np.logical_and(train_and_dev_ids, ~train_not_dev_ids)

        fold_output_path = Path(output_dir) / f'fold-{fold_idx+1}'
        os.makedirs(fold_output_path, exist_ok=True)

        js_train = list(data_arr[train_ids])
        with open(fold_output_path / 'train.json', 'w') as f:
            json.dump(js_train, f)

        js_dev = list(data_arr[dev_ids])
        with open(fold_output_path / 'dev.json', 'w') as f:
            json.dump(js_dev, f)

        js_test = list(data_arr[test_ids])
        with open(fold_output_path / 'test.json', 'w') as f:
            json.dump(js_test, f)

        print(f"fold {fold_idx+1}, train_docs={len(js_train)}, dev_docs={len(js_dev)}, test_docs={len(js_test)}")

        yield fold_output_path


def kfold(
        input_file: str,
        output_dir: str,
        n_splits,
        train_frac,
        train_params: TrainParams,
        **model_init_params
) -> T.List[SpacyModel]:
    models = list()

    with TemporaryDirectory() as cv_dir:
        data_paths = list(split_data_kfold(input_file, cv_dir, n_splits, train_frac))

        for fold_idx, fold_dir in enumerate(data_paths):
            model_location = Path(output_dir) / f'fold-{fold_idx+1}'
            os.makedirs(model_location, exist_ok=True)
            print(f"Training model {model_location}...")

            model_init_params['location'] = model_location
            model = SpacyModel(**model_init_params)

            model.fit(
                train_path=os.path.join(fold_dir, 'train.json'),
                dev_path=os.path.join(fold_dir, 'dev.json'),
                train_params=train_params
            )

            # score dict is accessible via model.score_
            model.score(os.path.join(fold_dir, 'test.json'))
            shutil.rmtree(fold_dir)  # we keep only model location
            models.append(model)

    return models


@click.command()
@click.argument("input-file", type=str)
@click.argument("output-dir", type=str)
@click.option(
    "-p", "--pipeline", type=str, required=True,
    help="Pipeline of tasks to train the model for, same format as for spacy.cli.train"
)
@click.option(
    "-v", "--vectors", type=str, default="models/blank/fasttext",
    help="Path to model from which vectors will be taken"
)
@click.option("-n", "--n-splits", type=int, default=5)
@click.option(
    "-f", "--train-frac", type=float, default=0.75,
    help="Fraction of every fold that will be used for training (as opposed to validation)"
)
def run_kfold_cv(input_file, output_dir, pipeline, vectors, n_splits, train_frac):
    train_params = TrainParams(
        n_iter=5,
    )
    model_init_params = {
        "pipeline": pipeline,
        "vectors_path": vectors,
    }
    models = kfold(
        input_file,
        output_dir,
        n_splits,
        train_frac,
        train_params,
        **model_init_params
    )
    score_dfs = list()
    for idx, model in enumerate(models):
        score_row = pd.DataFrame(
            {"fold":idx+1, "model_location": model.location, **model.scores_}, index=[idx+1]
        )
        score_dfs.append(score_row)
    score_df = pd.concat(score_dfs, axis='index')
    with pd.option_context("display.max_rows", 10, "display.max_columns", 20):
        print(score_df)
    return score_df


if __name__ == "__main__":
    run_kfold_cv()
