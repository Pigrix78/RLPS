# RLPS

Official PyTorch implementation of **Resource-Efficient Hyperspectral Image Classification with Reinforcement-Learned Band Selection and Point-Set Modeling**.

RLPS combines reinforcement-learned band selection (DRLBS) with point-set classification (PSGCN). Please refer to the paper and supplementary material for the method and experimental settings.

## Installation

The experiments use an NVIDIA GPU. Use Python 3.9 or newer and install a CUDA-enabled PyTorch build compatible with your GPU, then install the repository:

```bash
git clone https://github.com/Pigrix78/RLPS.git
cd RLPS
python -m pip install -e .
```

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

Dataset file names and MATLAB variable names are defined in `rlps/config.py`.

## Training and Evaluation

Train and evaluate PSGCN using the fixed 20-band subsets from Supplementary Table D3:

```bash
python scripts/train_psgcn.py --dataset Indian --data-root ./data --device cuda
```

Set `--dataset` to `Indian`, `PaviaU`, `Houston`, or `WHU_LK`. Defaults are 150 epochs and ten runs. Outputs are saved to `outputs/psgcn/<dataset>/`.

To run band selection followed by classifier training and evaluation:

```bash
python scripts/run_rlps.py --dataset Indian --data-root ./data --device cuda
```

The stages can also be run separately:

```bash
python scripts/train_drlbs.py --dataset Indian --data-root ./data --device cuda
python scripts/train_psgcn.py --dataset Indian --data-root ./data --device cuda --bands-file outputs/drlbs/Indian/selected_bands.json
```

Band JSON files use 1-based indices. The paper-reported subsets are listed in `configs/selected_bands.json`.

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
