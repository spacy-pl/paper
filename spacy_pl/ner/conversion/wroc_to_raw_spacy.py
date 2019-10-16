import xml.etree.ElementTree as ET
from ner_label_map import ner_label_map
import json
import os
import click

path_prefix = './'
corpus_path = 'data/raw/kpwr-1.1/'


class setCounter:
    def __init__(self):
        self.contents = {}

    def count(self, k, times=1):
        if k in self.contents:
            self.contents[k] += times
        else:
            self.contents[k] = times

    def merge(self, other):
        for k in other.contents:
            self.count(k, other.contents[k])


def get_subdirs(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


class Token:
    def __init__(self, orth, attribs, id):
        self.orth = orth
        self.attribs = attribs
        self.id = id

    def is_NE(self):
        return self.get_NE() is not None and self.get_NE() != "O"

    def get_NE(self):
        for attrib in self.attribs:
            for k in attrib:
                if attrib[k] != "0":
                    return k

        return None

    def __str__(self):
        return (self.orth + ":" + str(self.attribs))


class Sentence:
    def __init__(self, tokens=None):
        self.tokens = tokens if tokens is not None else []

    def add(self, token):
        self.tokens.append(token)

    def to_json(self):
        return {'tokens': [{
            'orth': t.orth,
            'id': t.id,
            'ner': t.get_NE()}
            for t in self.tokens
        ], 'brackets': []
        }


class Paragraph:
    def __init__(self, sentences=None):
        self.sentences = sentences if sentences is not None else []

    def add(self, sentence):
        self.sentences.append(sentence)

    def to_json(self):
        return {'sentences': [sentence.to_json() for sentence in self.sentences]}


class Document:
    def __init__(self, id, paragraphs=None):
        self.id = id
        self.paragraphs = paragraphs if paragraphs is not None else []

    def add(self, paragraph):
        self.paragraphs.append(paragraph)

    def to_json(self):
        return {'id': self.id,
                'paragraphs': [p.to_json() for p in self.paragraphs]}


class Corpus:
    def __init__(self, documents=None):
        self.documents = documents if documents is not None else []

    def add(self, document):
        self.documents.append(document)

    def to_json(self):
        return [doc.to_json() for doc in self.documents]


def process_token(tok, id):
    attribs = []
    orth = tok.find("orth").text
    for ann in tok.iter("ann"):
        if ann.attrib['chan'].endswith("nam"):  # and ann.text != "0":
            attribs += [{ann.attrib['chan']: ann.text}]

    return Token(orth, attribs, id)


def get_common_tag(t1, t2):
    set1 = set(t1.attribs)
    set2 = set(t2.attribs)
    common = list(set1 & set2)
    return common[0] if len(common) > 0 else None


def get_all_labels(tokens):
    labels = set()
    for tok in tokens:
        for attr in tok.attribs:
            labels.add(attr)

    return labels


def get_all_labels_with_cardinalities(tokens):
    labels = setCounter()
    for tok in tokens:
        for attr in tok.attribs:
            labels.count(attr)

    return labels


def map_labels(tokens, map):
    for tok in tokens:
        tok.attribs = [{map[k]: v} for attrib in tok.attribs for k, v in attrib.items()]

    return tokens


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


import random


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


# emptyset = set()
def pick_tags(tokens):
    longest_sequences = get_longest_sequences(tokens)
    for b, e, label in longest_sequences:
        seq = tokens[b:e]
        for tok in seq:
            tok.attribs = [{label: '1'}]
        tokens[b:e] = seq
    return tokens


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
                assert (False)

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


def get_file_paths(index_path):
    with open(index_path) as index_file:
        files = []
        line = index_file.readline()
        while line:
            line = line.replace('\n', '')
            files.append(line)
            line = index_file.readline()

        return files


def get_file_iterator(file):
    file = os.path.join(path_prefix, corpus_path, file)
    assert (not file.endswith("rel.xml") and not file.endswith(".ini"))
    tree = ET.parse(file)
    root = tree.getroot()
    return root.iter("sentence")


def get_sentence_iterator(sent):
    return sent.iter("tok")


def extract_corpus():
    corpus = Corpus()
    doc_idx = 0
    file_paths = get_file_paths(os.path.join(path_prefix, corpus_path, 'index_names.txt'))
    for file in file_paths:
        paragraph = Paragraph()
        document = Document(doc_idx)
        token_idx_in_doc = 0
        for sent in get_file_iterator(file):
            sentence = Sentence()
            for tok in get_sentence_iterator(sent):
                token = process_token(tok, token_idx_in_doc)
                token_idx_in_doc += 1
                sentence.add(token)

            paragraph.add(sentence)

        document.add(paragraph)
        corpus.add(document)
        doc_idx += 1

    return corpus


@click.command()
@click.option("-m", "--use-label-map", type=bool, default=False)
@click.argument("output_path", type=str)
def main(
        use_label_map,
        output_path,
):
    corpus = extract_corpus()
    for doc in corpus.documents:
        for paragraph in doc.paragraphs:
            for sent in paragraph.sentences:
                sent.tokens = pick_tags(sent.tokens)

    if use_label_map:
        [map_labels(sent.tokens, ner_label_map) for doc in corpus.documents
         for paragraph in doc.paragraphs for sent in paragraph.sentences]

    for doc in corpus.documents:
        for paragraph in doc.paragraphs:
            for sent in paragraph.sentences:
                sent.tokens = convert_to_biluo(sent.tokens)
    with open(os.path.expanduser(output_path), 'w+') as f:
        json.dump(corpus.to_json(), f)


if __name__ == "__main__":
    main()


