import xml.etree.ElementTree as ET
from spacy_pl.ner.conversion.ner_label_map import ner_label_map
from spacy_pl.ner.conversion.data_types import *
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
    #
    # if use_label_map:
    #     [map_labels(sent.tokens, ner_label_map) for doc in corpus.documents
    #      for paragraph in doc.paragraphs for sent in paragraph.sentences]
    #

    with open(os.path.expanduser(output_path), 'w+') as f:
        json.dump(corpus.to_json(), f)


if __name__ == "__main__":
    main()
