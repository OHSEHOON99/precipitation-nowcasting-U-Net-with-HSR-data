# Data and Artifact Scope

This repository contains source code, documentation, configuration examples, and a compact illustrative visualization.

## External Artifacts

The following artifacts are distributed or generated outside the repository:

- raw KMA HSR archives and extracted `.bin.gz` files
- generated datasets and train/validation/test splits
- trained checkpoints and model weight files
- Excel reports, bulk predictions, and visualization batches
- private run metadata such as local paths, W&B credentials, and access tokens
- executed notebook outputs that expose local logs, machine names, or private file names

Recommended local layout:

```text
data/
  archives/
  raw/
  dBZ_png/
  dataset_.../
checkpoints/
outputs/
```

The README links to the public Zenodo record for the dataset and pretrained models. Additional releases can use an external data/model registry such as Zenodo, Hugging Face, institutional storage, or GitHub Releases when licensing allows it.
