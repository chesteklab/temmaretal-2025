# TCN Analysis code for Temmar et al 2025

## 1. Introduction
The following code was used to produce all the figures used in 'Investigating the benefits of artificial neural networks over linear approaches to BMI decoding' by Temmar et al., published in the Journal of Neural Engineering in 2025 (DOI: 10.1088/1741-2552/ade568, [click here](https://doi.org/10.1088/1741-2552/ade568)). The code writes `.pdf` figures. Minor aesthetic changes were then applied in Adobe Illustrator.

## 2. Setup
### 1. Environment
Create the conda environment from `environment.yaml` and activate it:

```
conda env create -f environment.yaml
conda activate temmaretal-2025
```

That file matches the `temmaretal-2025` environment used for this analysis: Python 3.10, PyTorch 2.4 (CPU), NumPy, SciPy, Matplotlib, pandas, seaborn, h5py, and PyYAML.

### 2. Downloading the data

The data can be downloaded through Deep Blue Data, U-M's data depository, [here](https://deepblue.lib.umich.edu/data/concern/data_sets/z316q2934). Since the dataset is larger than 5GB, it needs to be downloaded using Globus. You will have to make a Globus account to download it. Once downloaded, extract raw.zip and move the raw folder wherever you'd like.

### 3. Where to put the data
After the data is downloaded, set the paths in `config.yaml`. These four entries are the ones to change:

| Key | What it is |
| --- | --- |
| `raw_data_dir` | Zstruct `.npy` files for each run |
| `data_dir` | Preprocessed arrays written by the analyses |
| `model_dir` | Trained decoders |
| `results_dir` | Figures, tables, and saved metrics |

At startup, `generateFiguresAndAnalyses.py` creates `data_dir`, `model_dir`, and `results_dir` if they do not exist, and relevant subfolders within them (each subfolder relates to one of the analysis blocks in the top-level generate script).

`architectures_path` and `training_params_path` point at `network_architectures.yaml` and `training_params.yaml` in this repository. Leave those as they are unless you move those files.

After downloading, `raw_data_dir` should look like this. The folder names `Monkey_N` and `Monkey_W` are the names the scripts pass as `mk_name`, and each `.npy` filename is built from that folder name, the date, and the run:

```
raw_data_dir/
  Monkey_N/
    2021-02-16/
      Run-003/
        Z_Monkey_N_2021-02-16_Run-003.npy
  Monkey_W/
    2020-12-08/
      Run-003/
        Z_Monkey_W_2020-12-08_Run-003.npy
```

## 3. Running the code
From the repository root, with the conda environment active:

```
python generateFiguresAndAnalyses.py
```

Each block in that script is one figure. Set `run_section` to `False` to skip a block. The monkey name in a block (`Monkey_N` or `Monkey_W`) must match the folder under `raw_data_dir`.

Inside a block:

- `preprocess=True` reads the `.npy` Zstruct for each run and writes preprocessed data under `data_dir`. Set it to `False` to load those saved files instead. Must be True for first time running.
- `train_rr`, `train_ds`, `train_tcn`, and `train_rnn` fit the decoders and write them under `model_dir`. Set a flag to `False` to load a decoder that is already saved. Online sections only have `preprocess`.
- The context block uses one `flag` for preprocessing and for training the ridge-regression, TCN, and RNN models. `firstpart` must be `True` the first time, so the block runs the analysis instead of loading `contextResults.pkl`.

Notes:
- For the `variance offline` figure, you need to run the `fits offline` block first to get the preprocessing done.
- For the `variance online` figure, you need to run the `variance online` part of the script in python, and then run the `nested_anova.R` script in R to get the stats.
