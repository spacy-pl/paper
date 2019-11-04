import json
import os
from tempfile import TemporaryDirectory

import numpy as np

import train_test


scores = []
for i in range(1, 11):
    tmpdir = TemporaryDirectory()
    transfer_path = f"models/pos/cv_nkjp_justpos_fasttext/fold-{i}/model-best"
    print(f"Trans {transfer_path}")
    train_test.run_train_test(
        train_data="data/processed/trees/pl_lfg-ud-train.json",
        dev_data="data/processed/trees/pl_lfg-ud-dev.json",
        test_data="data/processed/trees/pl_lfg-ud-test.json",
        output_dir=tmpdir.name,
        pipeline="parser",
        vectors="models/blank/fasttext",
        refit=False,
        transfer_path=transfer_path,
    )
    with open(os.path.join(tmpdir.name, "model-best/meta.json"), "r") as f:
        meta = json.load(f)
    scores.append({"las": meta["accuracy"]["las"], "uas": meta["accuracy"]["uas"]})
    print(f"Scores in {i}-th fold: {scores[-1]}")

with open("data/processed/training/ten_splits_exp_results.json", "w") as f:
    json.dump(scores, f)

las = np.asarray([s["las"] for s in scores])
uas = np.asarray([s["uas"] for s in scores])
print(f"LAS var: {las.var()}, mean: {las.mean()}")
print(f"UAS var: {uas.var()}, mean: {uas.mean()}")
