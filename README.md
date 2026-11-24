# RLPS

Official PyTorch implementation of **Resource-Efficient Hyperspectral Image Classification with Reinforcement-Learned Band Selection and Point-Set Modeling**.

RLPS is a two-stage framework for hyperspectral image classification under constrained computation and memory:

1. **DRLBS** learns a fixed subset of original spectral bands with a candidate-constrained masked dueling Double DQN, hybrid teacher guidance, and a point-set-aware marginal reward.
2. **PSGCN** (Point-Set Graph-Regularized Classification Network) represents each reduced local neighborhood as spectral-positional points. It uses shared point-wise extraction, two lightweight context blocks, and a category graph that regularizes training only.

## Repository Structure

```text
configs/selected_bands.json   Reported 20-band subsets from Supplementary Table D3
rlps/data.py                  Dataset loading, splits, normalization, and patch construction
rlps/drlbs/                   DRLBS agent, reward, teacher, and training procedure
rlps/psgcn/                   PSGCN architecture and training procedure
scripts/train_drlbs.py        Run band selection only
scripts/train_psgcn.py        Train and evaluate the classifier
scripts/run_rlps.py           Run both stages sequentially
tests/test_components.py      Lightweight component and loss checks
```

## Installation

Python 3.9 or newer is recommended. Create an environment, install a PyTorch build suitable for your platform, and install this repository:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The code runs on CPU, but CUDA is strongly recommended for PSGCN training.

## Datasets

Download the four public datasets from their original providers. Dataset files are not redistributed in this repository. Arrange them as follows:

```text
data/
├── Indian/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── PaviaU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── Houston/
│   ├── Houston.mat
│   └── Houston_gt.mat
└── WHU_LK/
    ├── WHU_Hi_LongKou.mat
    └── WHU_Hi_LongKou_gt.mat
```

Expected MATLAB variable names are defined in `rlps/config.py`. The loader also accepts a path that points directly to one dataset directory.

## Reproduce PSGCN Results

The reported DRLBS subsets are provided in `configs/selected_bands.json`. The classifier script uses these subsets by default, which avoids repeating the stochastic band-selection stage for every classifier run.

```bash
python scripts/train_psgcn.py --dataset Indian --data-root ./data
python scripts/train_psgcn.py --dataset PaviaU --data-root ./data
python scripts/train_psgcn.py --dataset Houston --data-root ./data
python scripts/train_psgcn.py --dataset WHU_LK --data-root ./data
```

The default protocol follows the paper: 20 bands, a `7 x 7` neighborhood, 150 epochs, Adam with an initial learning rate of `0.001`, and ten independent class-stratified splits. Training rates are 5% for Indian Pines, 3% for Pavia University, 3% for Houston 2013, and 0.5% for WHU-Hi-LongKou.

Results, per-run checkpoints, and timing information are written under `outputs/psgcn/<dataset>/`. Add `--predict-map` to export full-scene classification maps as MATLAB files.

## Run DRLBS

DRLBS is trained once for each dataset. All label-dependent teacher and reward quantities are estimated only from the labeled training partition.

```bash
python scripts/train_drlbs.py --dataset Indian --data-root ./data
```

The command writes a checkpoint, training curve, run metadata, and `selected_bands.json` under `outputs/drlbs/<dataset>/`. Exported band numbers are **1-based** and refer to the original input-band order. Internally, Python arrays use 0-based indices.

To train PSGCN with a newly generated subset:

```bash
python scripts/train_psgcn.py \
  --dataset Indian \
  --data-root ./data \
  --bands-file outputs/drlbs/Indian/selected_bands.json
```

The complete two-stage pipeline can be launched with:

```bash
python scripts/run_rlps.py --dataset Indian --data-root ./data
```

Band selection is stochastic and may vary slightly across hardware, PyTorch, and scikit-learn versions. Use `configs/selected_bands.json` when reproducing the downstream results reported in the paper.

## Implementation Notes

- The DRLBS candidate pool retains the top `ceil(0.9L)` bands using quality, point-set, and teacher scores with weights `(0.42, 0.33, 0.25)`.
- Teacher priors fuse filter, embedding, and wrapper guidance with weights `(0.30, 0.35, 0.35)`. Guidance probability decays from `0.65` to `0.05`.
- The masked dueling DQN uses Double-Q targets, Smooth L1 loss, discount factor `0.99`, replay capacity `50,000`, and mini-batches of 100 transitions.
- PSGCN uses five shared `1 x 1` layers with dimensions `24, 32, 48, 72, 104` and two context blocks.
- Category prototypes and graph affinities are computed from the current training split. The supervised and graph losses are mean-squared errors in probability space with a fixed `1:1` weight.
- The category graph is omitted during inference.

Run the lightweight checks with:

```bash
python -m unittest discover -s tests
```

## Citation

```bibtex
@article{bai2026rlps,
  title   = {Resource-Efficient Hyperspectral Image Classification with Reinforcement-Learned Band Selection and Point-Set Modeling},
  author  = {Bai, Jing and Xu, Ziyi and Xia, Beihao and Gu, Mansu and Deng, Wen and Xiao, Zhu and Jiao, Licheng},
  journal = {Science China Information Sciences},
  year    = {2026},
  note    = {Manuscript under review}
}
```
