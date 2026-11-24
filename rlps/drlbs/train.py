import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from ..config import DATASETS
from ..data import load_hsi, minmax_cube, stratified_split
from .agent import DuelingDoubleDQN
from .reward import PointSetReward, normalize
from .teacher import HybridTeacher


@dataclass
class DRLBSConfig:
    n_select: int = 20
    episodes: int = 1000
    seed: int = 40
    point_window: int = 7
    sample_limit: int = 8000
    candidate_ratio: float = 0.9
    learning_rate: float = 1.0e-4
    gamma: float = 0.99
    memory_size: int = 50_000
    batch_size: int = 100
    target_update: int = 100
    epsilon_initial: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    teacher_initial: float = 0.65
    teacher_min: float = 0.05
    teacher_decay_episodes: int = 450
    teacher_switch_span: int = 120
    blend_q: float = 0.40
    blend_reward: float = 0.35
    blend_teacher: float = 0.25
    local_refine_passes: int = 1


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_available(values, available):
    output = np.zeros_like(np.asarray(values, dtype=np.float64))
    output[available] = normalize(np.asarray(values)[available])
    return output


def _fused_selection(agent, reward, teacher, config):
    state = np.zeros(reward.bands, dtype=np.float32)
    selected = []
    for _ in range(config.n_select):
        available = np.flatnonzero((state < 0.5) & agent.candidate_mask)
        q_score = _normalize_available(agent.q_values(state), available)
        teacher_score = _normalize_available(teacher.hybrid_score, available)
        marginal = np.zeros(reward.bands, dtype=np.float64)
        for action in available:
            marginal[action] = reward.marginal_reward(selected, int(action), config.n_select)
        reward_score = _normalize_available(marginal, available)
        fused = config.blend_q * q_score + config.blend_reward * reward_score + config.blend_teacher * teacher_score
        action = int(available[np.argmax(fused[available])])
        selected.append(action)
        state[action] = 1.0
    return selected


def _local_refinement(selected, reward, candidate_mask, passes, target_size):
    selected = list(selected)
    current = reward.utility(selected, target_size)
    candidates = np.flatnonzero(candidate_mask)
    for _ in range(passes):
        improved = False
        for position, old_band in enumerate(list(selected)):
            best_subset, best_value = None, current
            for new_band in candidates:
                if int(new_band) in selected:
                    continue
                trial = list(selected)
                trial[position] = int(new_band)
                value = reward.utility(trial, target_size)
                if value > best_value + 1.0e-12:
                    best_subset, best_value = trial, value
            if best_subset is not None:
                selected, current = best_subset, best_value
                improved = True
        if not improved:
            break
    return selected


def _write_outputs(output_dir, dataset, selected, config, train_count, elapsed, losses, agent):
    output_dir.mkdir(parents=True, exist_ok=True)
    bands_one = [int(index) + 1 for index in selected]
    payload = {
        "dataset": dataset,
        "index_base": 1,
        "selected_bands": bands_one,
        "note": "Indices refer to the input cube's original band order.",
    }
    (output_dir / "selected_bands.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "epsilon", "teacher_probability", "reward", "loss"])
        writer.writeheader()
        writer.writerows(losses)
    metadata = {
        "dataset": dataset,
        "training_samples": train_count,
        "selected_bands_one_based": bands_one,
        "elapsed_seconds": elapsed,
        "config": asdict(config),
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(agent.policy.state_dict(), output_dir / "drlbs.pth")


def train_drlbs(dataset: str, data_root: str, output_dir: str, config=None, device=None, quiet=False):
    config = config or DRLBSConfig()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASETS)}")
    _set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hsi = load_hsi(data_root, dataset)
    cube = minmax_cube(hsi.cube)
    labels_zero = hsi.labels.reshape(-1)[hsi.labeled_indices] - 1
    train_positions, _ = stratified_split(labels_zero, DATASETS[dataset].train_rate, hsi.classes, config.seed)
    train_scene_indices = hsi.labeled_indices[train_positions]
    train_labels = labels_zero[train_positions]

    reward = PointSetReward(
        cube,
        train_scene_indices,
        train_labels,
        hsi.classes,
        window=config.point_window,
        sample_limit=config.sample_limit,
        seed=config.seed,
    )
    train_spectra = cube.reshape(-1, hsi.shape[2])[train_scene_indices]
    teacher = HybridTeacher(train_spectra, train_labels, reward, seed=config.seed)
    reward.teacher_score = teacher.hybrid_score
    candidate_mask = reward.candidate_mask(config.candidate_ratio)
    if config.n_select > int(candidate_mask.sum()):
        raise ValueError("The requested band count exceeds the candidate pool.")

    agent = DuelingDoubleDQN(
        hsi.shape[2],
        candidate_mask,
        config.learning_rate,
        config.gamma,
        config.memory_size,
        config.batch_size,
        config.target_update,
        device,
    )
    epsilon = config.epsilon_initial
    curve = []
    start = time.perf_counter()
    progress = trange(config.episodes, desc=f"DRLBS {dataset}", disable=quiet)
    for episode in progress:
        state = np.zeros(hsi.shape[2], dtype=np.float32)
        selected = []
        episode_reward = 0.0
        decay = min(1.0, episode / max(1, config.teacher_decay_episodes))
        teacher_probability = max(config.teacher_min, config.teacher_initial * (1.0 - decay))
        for step in range(config.n_select):
            if np.random.random() < teacher_probability:
                action = teacher.suggest(state, candidate_mask, episode, config.teacher_switch_span)
            elif np.random.random() < epsilon:
                action = agent.random_action(state)
            else:
                action = agent.greedy_action(state)
            reward_value = reward.marginal_reward(selected, action, config.n_select)
            next_state = state.copy()
            next_state[action] = 1.0
            agent.store(state, action, reward_value, next_state, step == config.n_select - 1)
            state = next_state
            selected.append(action)
            episode_reward += reward_value
        loss = agent.learn()
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        curve.append(
            {
                "episode": episode + 1,
                "epsilon": epsilon,
                "teacher_probability": teacher_probability,
                "reward": episode_reward,
                "loss": "" if loss is None else loss,
            }
        )
        progress.set_postfix(epsilon=f"{epsilon:.3f}", reward=f"{episode_reward:.4f}")

    selected = _fused_selection(agent, reward, teacher, config)
    selected = _local_refinement(
        selected,
        reward,
        candidate_mask,
        config.local_refine_passes,
        config.n_select,
    )
    output_path = Path(output_dir)
    _write_outputs(
        output_path,
        dataset,
        selected,
        config,
        train_positions.size,
        time.perf_counter() - start,
        curve,
        agent,
    )
    return [index + 1 for index in selected]
