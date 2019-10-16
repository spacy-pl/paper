from spacy_pl.ner.conversion.data_types import *
import click
import json


def convert_tokens_to_spacy(tokens):
    return [TokenSpacy(tok.orth, tok.get_NE(), tok.id) for tok in tokens]


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
                sent.tokens = convert_tokens_to_spacy(sent.tokens)

    with open(output_path, 'w+') as f:
        json.dump(corpus.to_json(), f)


if __name__ == "__main__":
    main()
