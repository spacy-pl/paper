from spacy_pl.ner.conversion.data_types import *
import click
import json

from spacy_pl.ner.conversion.ner_label_map import ner_label_map


def map_labels(tokens, map):
    for tok in tokens:
        tok.attribs = [{map[k]: v} for attrib in tok.attribs for k, v in attrib.items()]

    return tokens


@click.command()
@click.argument("input_path", type=str)
@click.argument("output_path", type=str)
def main(input_path, output_path):
    with open(input_path, 'r') as f:
        json_corpus = json.load(f)

    corpus = Corpus.from_json(json_corpus)

    for doc in corpus.documents:
        for paragraph in doc.paragraphs:
            for sent in paragraph.sentences:
                sent.tokens = map_labels(sent.tokens, ner_label_map)

    with open(output_path, 'w+') as f:
        json.dump(corpus.to_json(), f)


if __name__ == "__main__":
    main()
