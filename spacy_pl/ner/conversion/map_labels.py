from spacy_pl.ner.conversion.data_types import *
import click
import json
import os

MAPS_PATH = "./data/config/ner/conversion/label_maps"
MAP_FORMAT = ".json"


def get_available_label_maps():
    files = []
    for file in os.listdir(MAPS_PATH):
        if file.endswith(MAP_FORMAT):
            files.append(file[:-len(MAP_FORMAT)])

    return files


def load_map(map_name):
    path = os.path.join(MAPS_PATH, map_name + MAP_FORMAT)
    with open(path, 'r') as f:
        map = json.load(f)
    return map


def map_labels(tokens, map):
    for tok in tokens:
        tok.attribs = [{map[k]: v} for attrib in tok.attribs for k, v in attrib.items()]

    return tokens


@click.command()
@click.argument("input", type=click.File('r'))
@click.argument("output", type=click.File('w+'))
@click.option('--label_map',
              type=click.Choice(get_available_label_maps(), case_sensitive=False),
              required=True)
def main(input, output, label_map):
    json_corpus = json.load(input)

    corpus = Corpus.from_json(json_corpus)

    ner_label_map = load_map(label_map)

    for doc in corpus.documents:
        for paragraph in doc.paragraphs:
            for sent in paragraph.sentences:
                sent.tokens = map_labels(sent.tokens, ner_label_map)

    json.dump(corpus.to_json(), output)


if __name__ == "__main__":
    main()
