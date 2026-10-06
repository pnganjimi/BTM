# BTM 

This repository organises the eICU / MIMIC-III experiments, from the 'Geometric Characterisation and Structured Trajectory Surrogates for Clinical Dataset Condensation' paper into one pipeline:

1. learn **teacher optimisation trajectories**;
2. learn **quadratic Bézier mode connections** between each teacher's initial
   and final checkpoint;
3. optimise **synthetic data with BTM**;
4. repeatedly train/evaluate fresh models on the learned synthetic set.

## Repository layout

```text
btm_repo/
├── btm/
│   ├── __init__.py
│   ├── loaders.py          # all eICU + MIMIC loading/pre-processing
│   ├── models.py           # DNN, RNN/LSTM/TCN models 
│   └── utils.py            # utilities
├── notebooks/
│   ├── 01_learn_teacher_trajectories.ipynb
│   ├── 02_learn_mode_connections.ipynb
│   ├── 03_btm_eicu.ipynb
│   ├── 04_btm_mimic3_ihm.ipynb
│   └── 05_btm_mimic3_ph.ipynb
├── data/
│   └── README.md
├── trajectories/           # generated teacher trajectories + mode connections
├── stored/                 # generated synthetic datasets
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Dataset names

- `eicu`
- `mimic3_ihm`
- `mimic3_ph`


## Running the pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 1. Teacher trajectories

Open `notebooks/01_learn_teacher_trajectories.ipynb` and edit the configuration
cell:

```python
DATASET = "eicu"          # eicu | mimic3_ihm | mimic3_ph
NET_TYPE = None           # dataset default
OPTIM = "sgd"             # sgd | gsam | adam
NUM_EXPERTS = 50
```

The notebook writes:

```text
trajectories/<dataset>/<net_type>_<optim>_trajectories.pt
```

### 2. Mode connections

Open `notebooks/02_learn_mode_connections.ipynb`, select the same dataset/model/
optimiser, and run it after the teacher notebook.

It writes:

```text
trajectories/<dataset>/<net_type>_<optim>_modes.pt
```

The notebook also contains a one-expert diagnostic comparing the original
teacher path, a linear connection and the learned quadratic Bézier connection.

### 3. BTM

Use the dataset-specific notebook:

- `03_btm_eicu.ipynb`
- `04_btm_mimic3_ihm.ipynb`
- `05_btm_mimic3_ph.ipynb`


Synthetic datasets are stored under:

```text
stored/BTM/<dataset>/<net_type>_<optim>/
```