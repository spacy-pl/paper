import json
from os import listdir
from os.path import isdir, join as join_path
from shutil import copytree, rmtree
from tempfile import TemporaryDirectory

import click

import train_test


@click.command()
@click.argument("train-data", type=str)
@click.argument("dev-data", type=str)
@click.option(
    "-o", "--output-dir", type=str, required=True,
    help="Directory to which trained model will be saved"
)
@click.option(
    "-v", "--vectors", type=str, default="models/blank/fasttext",
    help="Path to model from which vectors will be taken"
)
@click.option(
    "--transfer_path", type=str, required=True,
    help="Path to the model to transfer from"
)
def main(train_data, dev_data, output_dir, vectors, transfer_path):
    train_on_all_folds(train_data, dev_data, output_dir, vectors, transfer_path)


def train_on_all_folds(train_data, dev_data, output_dir, vectors, transfer_path):
    best_dir = TemporaryDirectory()
    best_mean = 0.0
    scores = []
    folds = [f for f in listdir(transfer_path) if isdir(join_path(transfer_path, f))]
    for fold_name in folds:
        tmpdir = TemporaryDirectory()
        fold_path = join_path(transfer_path, join_path(fold_name, "model-best"))
        print(f"Trans {fold_path}")
        model = train_test.run_train(
            train_data=train_data,
            dev_data=dev_data,
            output_dir=tmpdir.name,
            pipeline="parser",
            vectors=vectors,
            refit=False,
            transfer_path=fold_path,
        )

        with open(model.meta_path, "r") as f:
            meta = json.load(f)
        scores.append({"las": meta["accuracy"]["las"],
                       "uas": meta["accuracy"]["uas"]})
        print(f"Scores for model trained on {fold_name}: {scores[-1]}")

        fold_score = (scores[-1]["las"] + scores[-1]["uas"])/2
        if fold_score > best_mean:
            best_mean = fold_score
            best_dir = TemporaryDirectory()
            copytree(tmpdir.name, join_path(best_dir.name, "model-best"))

    rmtree(output_dir, ignore_errors=True)
    copytree(join_path(best_dir.name, "model-best"), output_dir)


if __name__ == "__main__":
    main()
