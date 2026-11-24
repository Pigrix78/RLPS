from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import scipy.io as sio
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

from .config import DATASETS


@dataclass
class HSIData:
    name: str
    cube: np.ndarray
    labels: np.ndarray
    classes: int
    labeled_indices: np.ndarray

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.cube.shape


def _dataset_directory(data_root: Path, dataset: str, data_file: str) -> Path:
    candidates = (data_root / dataset, data_root)
    for candidate in candidates:
        if (candidate / data_file).is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {data_file}. Searched: {searched}")


def load_hsi(data_root: str, dataset: str) -> HSIData:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    directory = _dataset_directory(Path(data_root), dataset, spec.data_file)
    cube = np.asarray(sio.loadmat(directory / spec.data_file)[spec.data_key])
    labels = np.asarray(sio.loadmat(directory / spec.label_file)[spec.label_key])
    if cube.ndim != 3 or labels.shape != cube.shape[:2]:
        raise ValueError(f"Unexpected shapes for {dataset}: cube={cube.shape}, labels={labels.shape}")
    labeled_indices = np.flatnonzero(labels.reshape(-1) > 0)
    return HSIData(dataset, cube, labels.astype(np.int64), int(labels.max()), labeled_indices)


def minmax_cube(cube: np.ndarray) -> np.ndarray:
    height, width, bands = cube.shape
    flat = MinMaxScaler().fit_transform(cube.reshape(-1, bands))
    return flat.reshape(height, width, bands).astype(np.float32)


def stratified_split(labels_zero: np.ndarray, train_rate: float, classes: int, seed: int):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for class_id in range(classes):
        positions = np.flatnonzero(labels_zero == class_id)
        if positions.size < 2:
            raise ValueError(f"Class {class_id + 1} needs at least two labeled samples.")
        rng.shuffle(positions)
        count = max(1, int(positions.size * train_rate)) if train_rate < 1 else int(train_rate)
        count = min(count, positions.size - 1)
        train.extend(positions[:count])
        test.extend(positions[count:])
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def prepare_point_cube(cube: np.ndarray, selected_zero: Iterable[int]):
    selected_zero = np.asarray(list(selected_zero), dtype=np.int64)
    height, width, bands = cube.shape
    if selected_zero.size == 0 or selected_zero.min() < 0 or selected_zero.max() >= bands:
        raise ValueError("Selected band indices are outside the input cube.")
    spectra = cube.reshape(-1, bands)[:, selected_zero]
    spectra = StandardScaler().fit_transform(spectra).astype(np.float32)

    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    coordinates = np.stack((rows, cols), axis=-1).reshape(-1, 2)
    coordinates = StandardScaler().fit_transform(coordinates).astype(np.float32)
    points = np.concatenate((spectra, coordinates), axis=1)
    return points.reshape(height, width, -1), spectra


class PatchPointDataset(Dataset):
    def __init__(self, point_cube: np.ndarray, labels: np.ndarray, scene_indices: np.ndarray, window: int = 7):
        if window % 2 != 1:
            raise ValueError("The patch window must be odd.")
        self.window = int(window)
        self.width = int(point_cube.shape[1])
        margin = window // 2
        self.points = np.pad(point_cube, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
        self.labels = labels.reshape(-1).astype(np.int64) - 1
        self.indices = np.asarray(scene_indices, dtype=np.int64)

    def __len__(self):
        return self.indices.size

    def __getitem__(self, item):
        flat_index = int(self.indices[item])
        row, col = divmod(flat_index, self.width)
        patch = self.points[row : row + self.window, col : col + self.window]
        patch = patch.reshape(self.window * self.window, -1)
        return torch.from_numpy(patch.astype(np.float32, copy=False)), int(self.labels[flat_index])
