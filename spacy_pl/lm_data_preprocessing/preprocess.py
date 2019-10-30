from pathlib import Path
import os
import json

import click
from tqdm import tqdm


@click.command(help="Preprocess language modeling dataset")
@click.argument(
    "input-path", type=str, default="./data/raw/language_model_training_data.txt"
)
@click.argument(
    'output-path', type=str, default="./data/processed/lm/language_model_training_data.jsonl"
)
def preprocess(input_path, output_path):
    with open(Path(input_path), encoding='utf-8') as input_file:
        os.makedirs(os.path.dirname(Path(output_path)), exist_ok=True)
        with open(Path(output_path), mode='a', encoding='utf-8') as output_file:
            for line in tqdm(input_file):
                sentence = {'text': line.strip()}
                output_file.write(json.dumps(sentence, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    preprocess()
