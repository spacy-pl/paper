Replace 8888 with the related github issue number to enable automation:
closes: #8888

Don't change the text below, it will create checklists after you submit the form :)


**Checklist before requesting the review:**

- [ ] I checked that all experiments are organized in a dvc pipeline
- [ ] I listed all data & source code depencencies in every dvc file
- [ ] I listed and all outputs in every dvc file
- [ ] I ran `dvc commit` to check if dvc files reflect the state of my data
- [ ] I successfully ran `dvc push` before opening this pull request


**Checklist for reviewer:**

- [ ] Filenames and function names follow conventions set in `README.md`
- [ ] Scripts don't have hardcoded settings (global variables at the top of file and default `click.option` values are ok)
- [ ] After `git checkout branch-name` and `dvc pull`, `dvc repro related-pipeline-name.dvc` loads everything from cache
