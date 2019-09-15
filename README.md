# spaCy PL - paper

Polishing, benchmarking & other preparation for publishing our work


## Workflow

### Repository structure

Key folders & their intended use:
```
.
├── data  # everything that is not a code or notebook
│   ├── processed  # outputs of pipeline steps - intermediate or final
│   └── raw  # data downloaded from sources
├── models
│   ├── dep  # dependency parser models
│   ├── mixed  # models performing more than 1 task
│   ├── ner  # named entity recognizers
│   └── pos  # part of speech taggers
├── notebooks  # Jupyter notebooks performing analysis
└── src  # all python source code (scripts and modules)
```

Let's keep the repo as clean as possible - in case you feel a new folder is necessary,
discuss it with everyone first.


### DVC & running experiments

All of the `.dvc` files should be stored in repository root folder.
Also, all scripts should be executed from the root folder - that way,
all commands line options related to file paths can be specified as
defaults and we'll keep commands short.

Pipelines should reflect the flow of data from raw downloaded files (`data/raw` folder)
through preprocessing (one or more items created in `processed` folder), finally creating
one or more models in `models` folder.

Ideally, we shouldn't have to specify too much dependencies or outputs for any `dvc` command -
to do this, let's keep grouping outputs of a given script within one folder. For example,
a script that has 3 output files should create a directory and specify it in dvc - not
the individual files. Names should reflect what the script does clearly - so that
people working on other pipelines can easily understand what it does.

Python files running executing long or complicated processing logic should have a
docstring at the beginning to explain what they do.


### Naming

`.dvc` files should always be named by `<verb>_<noun(s) separated by "_">` - an action
and its output(s).
For example: `create_pos_nkjp_word2vec.dvc` or `train_pos_nkjp_word2vec.dvc` or `benchmark_pos_nkjp_word2vec.dvc` or `package_pos_dep_ner_v1.0.1.dvc` - in case of models.
For scripts: `unpack_nkjp`, `merge_nkjp_pos` or `convert_spacy_pos`.

If the step executes a single python file, the python file should be named
the same as a script.


## DVC cheat sheet

Whenever you struggle with doing something with dvc, describe a problem and
its solution (step-by-step) here. Some example problems worth describing:
- how to change dependencies of an already computed step
- how to update .dvc file after renaming dependency or output
- how to commit and push your new steps
- how to re-run last step but pull all the previous ones from remote

This way we'll create a nice knowledge base and speed up our work in the future.
