#!/bin/bash

export INPUT_DIR=data/raw/UD_Polish-LFG-master
export OUTPUT_DIR=data/processed/trees

mkdir $OUTPUT_DIR

python -m spacy convert $INPUT_DIR/pl_lfg-ud-train.conllu $OUTPUT_DIR -t json -l pl
python -m spacy convert $INPUT_DIR/pl_lfg-ud-dev.conllu $OUTPUT_DIR -t json -l pl
python -m spacy convert $INPUT_DIR/pl_lfg-ud-test.conllu $OUTPUT_DIR -t json -l pl
