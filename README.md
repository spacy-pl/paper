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
├── scripts  # useful bash scripts
└── spacy_pl  # all python source code (scripts and modules)
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


### What do I need make sure dvc works correctly?
1. `pip install -r requirements.txt` to install dvc (and other dependencies)
2. Find the google cloud key (let's name it `gc-key.json`) and place it in `path/to/a/folder/of/your/choice/`
3. In a shell from which you wish to use dvc, run `export GOOGLE_APPLICATION_CREDENTIALS=path/to/a/folder/of/your/choice/gc-key.json`
   (on Windows, in CMD, run: `set GOOGLE_APPLICATION_CREDENTIALS=path/to/a/folder/of/your/choice/gc-key.json`) 
   
Extra TIP: If you're using pycharm, make sure to [mark .dvc folder as excluded](https://stackoverflow.com/a/6535511) - otherwise it will keep indexing your dvc files (including cache).


### What should I do when creating a new branch for my task?
1. `git fetch --all`
2. `git checkout master` - necessary for steps 3 & 4
3. `git pull --rebase`
4. `dvc pull` - get the latest updated data from master, this will take some time
5. `git checkout -b your-branch-#10` where 10 is the number of github issue related to the branch
6. `dvc commit` - if this command changes anything in your repo, it means you messed something up


### How do I add new source of data?
I'm just adding the `data/raw/cc.pl.300.vec.gz` - what should I do to make it easy for others to work with?

1. Place the file in `data/raw/cc.pl.300.vec.gz`
2. Run `dvc add data/raw/cc.pl.300.vec.gz`
3. Move the file created by dvc to the right location: `mv data/raw/cc.pl.300.vec.gz.dvc ./add_fasttext_vectors.dvc`
4. Open the moved dvc file, make sure the path to the added data is correct (should be `data/raw/cc.pl.300.vec.gz`)
5. `git add data/raw/cc.pl.300.vec.gz.dvc data/raw/.gitignore`
6. Make sure the data file itself (`cc.pl.300.vec.gz`) is ignored in git, ie. doesn't show up in `git status` output
7. Check if it works: `dvc repro add_fasttext_vectors.dvc` - should print something like "stage didn't change, using cache"
8. `git commit`
9. `dvc push -j 1` - push your changes as early as possible to prevent problems later, `-j 1` option tells dvc to use 1 thread, 
   which may be slower but provides a progressbar so at least you know what is going on 
10. `git push` - same as for dvc, if you know remote contains your work you don't have to worry about breaking something locally :)


### How to run experiment that I just wrote?
I have just created a `spacy_pl/tagset` module with 2 python files, that I want to use to generate tagset and conversion map
from nltk to spacy format (for the selected NKJP POS tags). To do this:

1. I use click options in the python script to specify all paths to dependencies and outputs
2. However, I also specify the default values for them (to write shorter commands in shell later)
3. I run the script normally to make sure it works, list of key things to check:
    - **all output directories are created if they don't exist already**
    - all paths are relative to repository root folder (including python imports)
    - this step can also be performed with dvc, but since we don't care about result at first, why bother?
4. Now I can run it with dvc:
   `dvc run -d data/raw/NKJP_1.2_nltk -d spacy_pl/tagset -o data/processed/tagset -f generate_tagset_and_conversion_map.dvc python spacy_pl/tagset/generate_tagset_and_conversion_map.py`,
   note that:
    - I ran this from root folder of repository
    - even though I was running just one script, I specified entire module `spacy_pl/tagset` as dependency, as it contains related code
    - I output more than one file, so instead of listing them all, I group them by putting inside one folder
    - I specified the name of dvc file using conventions described above
5. After the script ends successfully, we can add and commit our changes:
    - `git add spacy_pl/tagset generate_tagset_and_conversion_map.dvc data/processed/.gitignore`
    - make sure the data file itself (`data/processed/tagset`) is ignored in git, ie. doesn't show up in `git status` output
    - check if it works: `dvc repro generate_tagset_and_conversion_map.dvc` - should print something like "stage didn't change, using cache"
    - `git commit`
    - `dvc push -j 1` push your changes as early as possible to prevent problems later, `-j 1` option tells dvc to use 1 thread 
    - `git push`

### How to open a pull request and ensure all my changes will be available for other people?
If you followed the guidelines for adding files to dvc and running experiments, everything will work.

Some useful commands in case you messed something up (documentation on dvc.org is ok for these):
- `dvc commit`
- `dvc pull`
- `dvc lock`


### How can I get the latest, trained version of the model?
Assuming your're on the right branch (ie. the models' dvc file exists on it). 
For example, for pulling cross-validation of pos-only tagger using fasttext vectors:

1. `dvc pull`
2. `dvc repro cv-pos-nkjp-justpos-fasttext.dvc`


### How can I re-train a model?
Assuming your're on the right branch (ie. the models' dvc file exists on it). 
For example, ro re-run the cross-validation of pos-only tagger using fasttext vectors:

1. `dvc pull`
2. View the dependency tree of pos tagger: `dvc pipeline show --ascii cv-pos-nkjp-justpos-fasttext.dvc`
3. For each immediate dependency, run `dvc repro dependency-name.dvc`
4. Make your changes to the code
5. To track the training results, follow steps from *How to run experiment that I just wrote?* described above
