from pathlib import Path
import typing as T
import os
from time import time

import spacy
import morfeusz2

import click
from tqdm import tqdm
import pandas as pd
from conll_df import conll_df
from spacy.lang.pl import Polish


def load_data(path: Path, sample_size: T.Optional[int]=None):
    """
    Load conllu file to pandas DataFrame,
    optionally reducing number of samples to sample_size.
    """
    df = conll_df(str(path.resolve()), skip_meta=True)
    if sample_size is not None:
        df = df.iloc[:sample_size]
    df = df[['w', 'l', 'x']].reset_index(drop=True)
    df.columns = ['orth', 'lemma', 'UD_POS']
    return df


def run_benchmark(
        data_df: pd.DataFrame,
        spacy_model: spacy.lang.pl.Polish,
        spacy_lemmatizer: spacy.lemmatizer.Lemmatizer,
        benchmark_model: morfeusz2.Morfeusz
) -> T.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the benchmark on 3 models (spacy lemmatizer on true pos, spacy pos + lemmatizer, morfeusz lemmatizer)
    and returns predictions + summarized stats.
    :param data_df: Data loaded using load_data function (above)
    :param spacy_model: Polish model, containing at least tagger and lemmatizer
    :param spacy_lemmatizer: Raw polish lemmatizer, without any models
    :param benchmark_model: Morfeusz model, used as a benchmark
    :return: predictions_df (true + predicted lemma for all models), results_df (summary of results per model)
    """
    predictions_df = pd.DataFrame(
        columns=[
            'token',
            'true_pos',
            'true_lemma',
            'spacy_model_lemma',
            'spacy_lemmatizer_lemma',
            'benchmark_model_lemma',
        ]
    )
    predictions_df[['token', 'true_pos', 'true_lemma']] = data_df[['orth', 'UD_POS', 'lemma']].copy()
    predictions_df['is_trivial'] = predictions_df['true_pos'] == predictions_df['token']

    results_df = pd.DataFrame(columns=['all_lemma_accuracy', 'non_trivial_lemma_accuracy', 'runtime'])

    def build_result_df_row(predictions: pd.Series, runtime: float):
        prediction_accurate = predictions_df['true_lemma'] == predictions
        # noinspection PyUnresolvedReferences
        return {
            'all_lemma_accuracy': prediction_accurate.mean(),
            'non_trivial_lemma_accuracy': prediction_accurate[~predictions_df['is_trivial']].mean(),
            'runtime': runtime
        }

    # spacy predictions 1
    with tqdm(predictions_df[['token', 'true_pos']], desc="SpaCy (LEMMA)") as t:
        start = time()
        predictions_df['spacy_lemmatizer_lemma'] = list(map(
            lambda token, pos: spacy_lemmatizer(token, univ_pos=pos),
            t
        ))
        end = time()
        results_df.loc['spacy_lemmatizer'] = build_result_df_row(predictions_df['spacy_lemmatizer_lemma'], end-start)

    # spacy predictions 2
    with tqdm(predictions_df['token'], desc="SpaCy (POS + LEMMA)") as t:
        # TODO: Process entire text at once, align with dataframe later (to benchmark speed)
        predictions_df['spacy_model_lemma'] = list(map(
            lambda token: benchmark_model(token, disable=['parser', 'ner'])[0].lemma_,
            t
        ))
        end = time()
        results_df.loc['spacy_model'] = build_result_df_row(predictions_df['spacy_model_lemma'], end-start)

    # morfeusz predictions
    with tqdm(predictions_df['token'], desc="Morfeusz") as t:
        # TODO: Process entire text at once, align with dataframe later (to benchmark speed)
        predictions_df['benchmark_model_lemma'] = list(map(
            lambda token: list(map(
                lambda analysis_result: analysis_result[1],  # get the list of all possible lemmas
                benchmark_model.analyse(token)  #
            )),
            t
        ))
        end = time()
        # TODO: which lemma to select from the morfeusz output? (it returns a list)
        results_df.loc['benchmark_model'] = build_result_df_row(predictions_df['benchmark_model_lemma'], end - start)

    return predictions_df, results_df


@click.command(help="Benchmark lemmatization accuracy. Requires Morfeusz to be installed")
@click.argument(
    "data_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="./data/raw/UD_Polish-LFG-master/pl_lfg-ud-dev.conllu"
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="./results/lemmatizer",
    help="Path to output directory where predictions and result sumamry will be saved"
)
def benchmark_lemmatizer(data_path, output_dir):
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    spacy_model = spacy.load("pl_model")
    benchmark_model = morfeusz2.Morfeusz(generate=False)
    data_df = load_data(data_path)

    print("Running benchmark...")
    predictions_df, results_df = run_benchmark(data_df, spacy_model, benchmark_model)
    print("Benchmark complete")

    if not output_dir.exists():
        os.makedirs(str(output_dir.resolve()))

    predictions_df.to_csv(str(output_dir / "predictions.csv"))
    results_df.to_csv(str(output_dir / "results.csv"))
    print(results_df)
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    benchmark_lemmatizer()
