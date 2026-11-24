import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_distances


def normalize(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    span = float(values.max() - values.min())
    return np.zeros_like(values) if span < 1.0e-12 else (values - values.min()) / span


@dataclass(frozen=True)
class RewardWeights:
    quality: float = 0.35
    diversity: float = 0.30
    pointset: float = 0.35
    teacher: float = 0.15
    cost: float = 0.02
    entropy_mix: float = 0.70
    snr_mix: float = 0.30


class PointSetReward:
    def __init__(
        self,
        cube: np.ndarray,
        train_scene_indices: np.ndarray,
        train_labels: np.ndarray,
        classes: int,
        window: int = 7,
        sample_limit: int = 8000,
        seed: int = 40,
        weights: RewardWeights = RewardWeights(),
    ):
        self.cube = cube.astype(np.float32, copy=False)
        self.height, self.width, self.bands = cube.shape
        self.train_scene_indices = np.asarray(train_scene_indices, dtype=np.int64)
        self.train_labels = np.asarray(train_labels, dtype=np.int64)
        self.classes = classes
        self.window = window
        self.sample_limit = sample_limit
        self.seed = seed
        self.weights = weights
        self.train_spectra = cube.reshape(-1, self.bands)[self.train_scene_indices]

        self.entropy_score = normalize(entropy(self.train_spectra.astype(np.float64) + 1.0e-12, axis=0))
        self.snr_score = self._spatial_snr()
        self.quality_score = normalize(
            weights.entropy_mix * self.entropy_score + weights.snr_mix * self.snr_score
        )
        self.pointset_score = self._pointset_score()
        self.diversity_distance = self._prototype_distances()
        self.teacher_score = np.zeros(self.bands, dtype=np.float64)

    def _spatial_snr(self):
        score = np.zeros(self.bands, dtype=np.float64)
        for band in range(self.bands):
            image = self.cube[:, :, band].astype(np.float64, copy=False)
            smooth = gaussian_filter(image, sigma=1.0)
            score[band] = np.var(smooth) / (np.var(image - smooth) + 1.0e-12)
        return normalize(np.log1p(score))

    def _sample_positions(self):
        count = self.train_scene_indices.size
        if self.sample_limit <= 0 or count <= self.sample_limit:
            return np.arange(count, dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        per_class = max(1, math.ceil(self.sample_limit / self.classes))
        chosen = []
        for class_id in range(self.classes):
            positions = np.flatnonzero(self.train_labels == class_id)
            chosen.extend(rng.choice(positions, min(per_class, positions.size), replace=False).tolist())
        return np.asarray(chosen[: self.sample_limit], dtype=np.int64)

    @staticmethod
    def _fisher(values, labels):
        values = np.asarray(values, dtype=np.float64)
        overall = values.mean(axis=0)
        between = np.zeros(values.shape[1], dtype=np.float64)
        within = np.zeros(values.shape[1], dtype=np.float64)
        for class_id in np.unique(labels):
            group = values[labels == class_id]
            mean = group.mean(axis=0)
            between += group.shape[0] * (mean - overall) ** 2
            within += ((group - mean) ** 2).sum(axis=0)
        return between / (within + 1.0e-12)

    def _pointset_score(self):
        positions = self._sample_positions()
        scene_indices = self.train_scene_indices[positions]
        labels = self.train_labels[positions]
        center = self.cube.reshape(-1, self.bands)[scene_indices]
        margin = self.window // 2
        padded = np.pad(self.cube, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
        local_mean = np.zeros_like(center)
        local_std = np.zeros_like(center)
        for index, flat in enumerate(scene_indices):
            row, col = divmod(int(flat), self.width)
            patch = padded[row : row + self.window, col : col + self.window]
            local_mean[index] = patch.mean(axis=(0, 1))
            local_std[index] = patch.std(axis=(0, 1))
        score = (self._fisher(center, labels) + self._fisher(local_mean, labels) + self._fisher(local_std, labels)) / 3.0
        return normalize(score)

    def _prototype_distances(self):
        prototypes = np.zeros((self.classes, self.bands), dtype=np.float64)
        for class_id in range(self.classes):
            prototypes[class_id] = self.train_spectra[self.train_labels == class_id].mean(axis=0)
        distance = cosine_distances(prototypes.T)
        distance = np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(distance, 0.0)
        return distance

    def candidate_mask(self, ratio: float = 0.9):
        count = max(1, min(self.bands, math.ceil(ratio * self.bands)))
        score = normalize(0.42 * self.quality_score + 0.33 * self.pointset_score + 0.25 * self.teacher_score)
        mask = np.zeros(self.bands, dtype=bool)
        mask[np.argsort(score)[-count:]] = True
        return mask

    def diversity(self, selected):
        if len(selected) < 2:
            return 0.0
        matrix = self.diversity_distance[np.ix_(selected, selected)]
        return float(matrix[np.triu_indices(len(selected), k=1)].mean())

    def utility(self, selected, target_size: int):
        if not selected:
            return 0.0
        progress = min(1.0, len(selected) / max(target_size, 1))
        alpha = self.weights.quality * (1.0 - 0.35 * progress)
        gamma = self.weights.pointset * (0.65 + 0.35 * progress)
        delta = self.weights.teacher * max(0.25, 1.0 - 0.55 * progress)
        value = (
            alpha * float(self.quality_score[selected].mean())
            + self.weights.diversity * self.diversity(selected)
            + gamma * float(self.pointset_score[selected].mean())
            + delta * float(self.teacher_score[selected].mean())
            - self.weights.cost * progress
        )
        return float(value)

    def marginal_reward(self, selected, action: int, target_size: int):
        return self.utility(selected + [action], target_size) - self.utility(selected, target_size)
