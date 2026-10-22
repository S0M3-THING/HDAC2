# HDAC2 — Cancer Stage Classification

An experimental Python project exploring pancreatic cancer stage classification from HDAC gene measurements and patient metadata. The scripts train TensorFlow/Keras neural networks using gene-expression summaries, race, gender, and age, with `ajcc_pathologic_stage` as the target label.

## Project layout

| File | Purpose |
| --- | --- |
| [`final.py`](final.py) | Class-based HDAC2 pipeline with preprocessing, training, evaluation, cross-validation, and an example single-sample prediction. |
| [`HDAC2model.py`](HDAC2model.py) | Standalone HDAC2 training and evaluation script. |
| [`HDAC3model.py`](HDAC3model.py) | Standalone HDAC3 variant; requires `HDAC3.csv`, which is not included. |
| [`HDAC23model.py`](HDAC23model.py) | Experimental combined HDAC2/HDAC3 variant. |
| [`HDAC2.csv`](HDAC2.csv) | Input dataset used by the HDAC2 scripts. |
| [`requirements.txt`](requirements.txt) | Pinned environment package list; see the dependency caveats below. |

## Setup

Clone the repository and create an isolated environment with Conda:

```bash
git clone https://github.com/S0M3-THING/HDAC2.git
cd HDAC2
conda create --name hdac2 python=3.10
conda activate hdac2
```

Python 3.10 is a starting point for environment setup, not a tested compatibility guarantee. The existing dependency list is broad and is not a verified reproducible environment:

- The scripts import `tensorflow` and `imblearn`, but `requirements.txt` does not explicitly list TensorFlow or `imbalanced-learn`. The `tf` entry is not the TensorFlow package name.
- The scripts use `tensorflow.keras.optimizers.legacy.Adam`, so the TensorFlow/Keras version must support that API.
- The list includes packages unrelated to these scripts; installing the full list can require additional downloads and dependency resolution.

The existing dependency installation command is:

```bash
python -m pip install -r requirements.txt
```

Resolve the missing TensorFlow and `imbalanced-learn` dependencies in a compatible environment before running training. A fully validated installation recipe is not yet provided.

## Input data

Run scripts from the repository root so their relative CSV paths resolve correctly. `final.py` reads `HDAC2.csv` and uses these columns:

| Column | Role |
| --- | --- |
| `mean.x` | Gene-expression summary; copied to `mean2` during preprocessing. |
| `race` | Categorical input, label-encoded during training. |
| `gender` | Categorical input, label-encoded during training. |
| `age_at_index` | Numeric age input. |
| `ajcc_pathologic_stage` | Target stage label. |

`final.py` drops rows with missing values before selecting features. Its SMOTE and stratified splits require enough examples in each target class. Single-sample predictions must use race and gender categories seen during training.

## Run the HDAC2 pipeline

After resolving the dependencies:

```bash
python final.py
```

The pipeline trains a model, prints classification metrics, displays training-history, confusion-matrix, and ROC plots, and runs five-fold cross-validation. The final example prediction uses the model trained during that run. Plot windows may need to be closed for execution to continue. The script does not save a trained model to disk.

## Current limitations

- The code is experimental; reported metrics should not be treated as validated clinical performance.
- In the main pipeline, SMOTE runs before the train/test split. This can leak information into evaluation and should be addressed before drawing conclusions from the scores.
- `HDAC23model.py` references `X.mean()` while assigning `X`, so that variant needs a code fix before use; it also requires the absent `HDAC3.csv`.
- There is no automated test suite or CI workflow in this repository. Full training depends on a working scientific Python environment and suitable data.
