from spacy_pl.ner.conversion.data_types import *
import random
import click
import json


def convert_to_biluo(tokens):
    out = []
    in_ne = False
    for i, token in enumerate(tokens[:-1]):
        if in_ne:
            if token.is_NE():
                if tokens[i + 1].is_NE() and token.get_NE() == tokens[i + 1].get_NE():
                    # inner NE
                    out += [Token(token.orth, [{"I-" + token.get_NE(): '1'}], token.id)]
                else:
                    # last NE
                    out += [Token(token.orth, [{"L-" + token.get_NE(): '1'}], token.id)]
                    in_ne = False
            else:
                # we shouldn't ever get here
                assert False

        else:
            if token.is_NE():
                # new NE
                if tokens[i + 1].is_NE() and token.get_NE() == tokens[i + 1].get_NE():
                    # beginning NE
                    out += [Token(token.orth, [{"B-" + token.get_NE(): '1'}], token.id)]
                    in_ne = True
                else:
                    # unit NE
                    out += [Token(token.orth, [{"U-" + token.get_NE(): '1'}], token.id)]
                    in_ne = False
            else:
                # outside of NE
                out += [Token(token.orth, [{"O": '1'}], token.id)]

    # process last token
    token = tokens[-1]
    if in_ne:
        out += [Token(token.orth, [{"L-" + token.get_NE(): '1'}], token.id)]
    else:
        if token.is_NE():
            out += [Token(token.orth, [{"U-" + token.get_NE(): '1'}], token.id)]
        else:
            out += [Token(token.orth, [{"O": '1'}], token.id)]

    return out


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
                sent.tokens = convert_to_biluo(sent.tokens)

    with open(output_path, 'w+') as f:
        json.dump(corpus.to_json(), f)


if __name__ == "__main__":
    main()
