import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from ..config import DATASETS
from ..data import PatchPointDataset, load_hsi, prepare_point_cube, stratified_split
from ..metrics import classification_metrics
from .model import PSGCN


@dataclass
class PSGCNConfig:
    epochs: int = 150
    runs: int = 10
    seed: int = 40
    window: int = 7
    learning_rate: float = 0.001
    train_batch_size: int = 256
    test_batch_size: int = 256
    lr_step: int = 50
    lr_gamma: float = 0.1


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _category_adjacency(spectra, labels, classes, device):
    prototypes = np.zeros((classes, spectra.shape[1]), dtype=np.float32)
    for class_id in range(classes):
        prototypes[class_id] = spectra[labels == class_id].mean(axis=0)
    nodes = torch.as_tensor(prototypes, dtype=torch.float32, device=device)
    return torch.exp(-torch.cdist(nodes, nodes, p=2))


def _evaluate(model, loader, classes, device):
    targets, predictions = [], []
    model.eval()
    with torch.no_grad():
        for points, target in loader:
            probabilities, _ = model(points.to(device))
            targets.extend(target.numpy().tolist())
            predictions.extend(probabilities.argmax(dim=1).cpu().numpy().tolist())
    return classification_metrics(targets, predictions, classes)


def _predict_scene(model, point_cube, window, batch_size, device):
    height, width = point_cube.shape[:2]
    indices = np.arange(height * width, dtype=np.int64)
    dummy_labels = np.ones((height, width), dtype=np.int64)
    loader = DataLoader(PatchPointDataset(point_cube, dummy_labels, indices, window), batch_size=batch_size)
    prediction = []
    model.eval()
    with torch.no_grad():
        for points, _ in loader:
            probabilities, _ = model(points.to(device))
            prediction.extend((probabilities.argmax(dim=1) + 1).cpu().numpy().tolist())
    return np.asarray(prediction, dtype=np.int16).reshape(height, width)


def _write_summary(output_dir, dataset, selected_bands, config, run_metrics, parameters):
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = ("oa", "aa", "kappa")
    summary = {
        key: {
            "mean": float(np.mean([run[key] for run in run_metrics])),
            "std": float(np.std([run[key] for run in run_metrics])),
        }
        for key in keys
    }
    payload = {
        "dataset": dataset,
        "selected_bands_one_based": list(selected_bands),
        "train_rate": DATASETS[dataset].train_rate,
        "trainable_parameters": parameters,
        "config": asdict(config),
        "summary": summary,
        "runs": run_metrics,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "seed", "oa", "aa", "kappa", "train_seconds", "test_seconds"))
        for index, run in enumerate(run_metrics, start=1):
            writer.writerow((index, run["seed"], run["oa"], run["aa"], run["kappa"], run["train_seconds"], run["test_seconds"]))
    return summary


def train_psgcn(
    dataset: str,
    data_root: str,
    output_dir: str,
    selected_bands_one,
    config=None,
    device=None,
    predict_map=False,
    quiet=False,
):
    config = config or PSGCNConfig()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASETS)}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    selected_bands_one = [int(index) for index in selected_bands_one]
    selected_zero = np.asarray(selected_bands_one, dtype=np.int64) - 1
    hsi = load_hsi(data_root, dataset)
    point_cube, standardized_spectra = prepare_point_cube(hsi.cube, selected_zero)
    labeled_labels = hsi.labels.reshape(-1)[hsi.labeled_indices] - 1
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runs = []
    parameters = None

    for run in range(config.runs):
        seed = config.seed + run
        _set_seed(seed)
        train_positions, test_positions = stratified_split(
            labeled_labels,
            DATASETS[dataset].train_rate,
            hsi.classes,
            seed,
        )
        train_indices = hsi.labeled_indices[train_positions]
        test_indices = hsi.labeled_indices[test_positions]
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            PatchPointDataset(point_cube, hsi.labels, train_indices, config.window),
            batch_size=config.train_batch_size,
            shuffle=True,
            generator=generator,
        )
        test_loader = DataLoader(
            PatchPointDataset(point_cube, hsi.labels, test_indices, config.window),
            batch_size=config.test_batch_size,
            shuffle=False,
        )
        train_labels = hsi.labels.reshape(-1)[train_indices] - 1
        adjacency = _category_adjacency(
            standardized_spectra[train_indices],
            train_labels,
            hsi.classes,
            device,
        )
        model = PSGCN(len(selected_bands_one) + 2, hsi.classes).to(device)
        if parameters is None:
            parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)
        best_loss = float("inf")
        checkpoint = output_path / f"run_{run + 1:02d}.pth"
        train_start = time.perf_counter()
        epochs = trange(config.epochs, desc=f"PSGCN {dataset} run {run + 1}/{config.runs}", disable=quiet)
        for _ in epochs:
            model.train()
            loss_sum = 0.0
            sample_count = 0
            for points, target in train_loader:
                points = points.to(device)
                target = target.to(device)
                probabilities, graph_distribution = model(points, adjacency)
                hard_target = F.one_hot(target, hsi.classes).float()
                graph_target = graph_distribution[target]
                loss = F.mse_loss(probabilities, hard_target) + F.mse_loss(probabilities, graph_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * target.size(0)
                sample_count += target.size(0)
            scheduler.step()
            epoch_loss = loss_sum / max(1, sample_count)
            epochs.set_postfix(loss=f"{epoch_loss:.5f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint)
        train_seconds = time.perf_counter() - train_start

        model.load_state_dict(torch.load(checkpoint, map_location=device))
        test_start = time.perf_counter()
        metrics = _evaluate(model, test_loader, hsi.classes, device)
        test_seconds = time.perf_counter() - test_start
        metrics.update(
            {
                "seed": seed,
                "train_samples": int(train_indices.size),
                "test_samples": int(test_indices.size),
                "train_seconds": train_seconds,
                "test_seconds": test_seconds,
            }
        )
        runs.append(metrics)
        if predict_map:
            prediction = _predict_scene(model, point_cube, config.window, config.test_batch_size, device)
            sio.savemat(output_path / f"classification_map_run_{run + 1:02d}.mat", {"classification_map": prediction})

    summary = _write_summary(output_path, dataset, selected_bands_one, config, runs, parameters)
    return summary
