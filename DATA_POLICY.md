# Data and Artifact Policy

This repository is intended to contain source code, documentation, configuration examples, and a few small illustrative images only.

Do not commit:

- raw KMA HSR archives or extracted `.bin.gz` files
- generated datasets, checkpoints, Excel reports, or bulk visualizations
- local absolute paths, API keys, W&B credentials, access tokens, or private experiment metadata
- notebooks with executed outputs that reveal local paths, logs, machine names, or private file names

Recommended local layout:

```text
data/
  archives/        # original .tar.gz files, ignored by git
  raw/             # extracted .bin.gz files, ignored by git
  dBZ_png/         # converted PNG files, ignored by git
  dataset_.../     # train/valid/test split, ignored by git
checkpoints/       # model weights, ignored by git
outputs/           # evaluation reports and generated visualizations, ignored by git
```

The README links to the public data source. Keep large data and trained weights in an external data/model registry such as Zenodo, Hugging Face, institutional storage, or a GitHub Release when licensing allows it.

If a credential was ever committed, revoke and rotate it at the provider first. Git history cleanup reduces exposure in the repository, but it does not guarantee that external caches, forks, clones, or provider-side logs have forgotten the value.
