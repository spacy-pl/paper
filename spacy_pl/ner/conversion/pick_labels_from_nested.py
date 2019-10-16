from spacy_pl.ner.conversion.data_types import *
import random
import click
import json


def still_in_sequence(v1, v2):
    return any(v1e == v2e != "0" for v1e, v2e in zip(v1, v2))


def get_last_label(v):
    for i, e in enumerate(v):
        if e != "0":
            return i
    return None


def get_label_set(v):
    res = set()
    for i, e in enumerate(v):
        if e != "0":
            res.add(i)

    return res


def get_any_label(v):
    if v == emptyset():
        return None
    return random.sample(v, 1)[0]


def emptyset():
    return set()


def get_longest_sequences(tokens):
    res = []
    b = 0
    e = 0
    attribs = [k for d in tokens[0].attribs for k in d]
    last_set = None
    label_set = emptyset()
    while e != len(tokens) - 1:
        current_token = tokens[e]

        if last_set == None or label_set == emptyset():
            last_set = [v for d in current_token.attribs for k, v in d.items()]
            label_set = get_label_set(last_set)
            b = e
        else:
            new_set = [v for d in current_token.attribs for k, v in d.items()]
            label_set = label_set.intersection(get_label_set(new_set))
            if not still_in_sequence(last_set, new_set):
                label_id = get_any_label(label_set)
                if (label_id != None):
                    label = attribs[label_id]
                    res.append((b, e, label))
                b = e
                label_set = emptyset()

            last_set = new_set
        e += 1

    return res


def pick_tags(tokens):
    longest_sequences = get_longest_sequences(tokens)
    for b, e, label in longest_sequences:
        seq = tokens[b:e]
        for tok in seq:
            tok.attribs = [{label: '1'}]
        tokens[b:e] = seq
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
                sent.tokens = pick_tags(sent.tokens)

    with open(output_path, 'w+') as f:
        json.dump(corpus.to_json(), f)


if __name__ == "__main__":
    main()
