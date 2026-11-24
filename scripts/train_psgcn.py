import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlps.config import DATASETS, PAPER_BANDS
from rlps.psgcn import PSGCNConfig, train_psgcn


def read_bands(path, dataset):
    if path is None:
        return PAPER_BANDS[dataset]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "selected_bands" in payload:
        if int(payload.get("index_base", 1)) != 1:
            raise ValueError("Band files supplied to this script must use 1-based indices.")
        return payload["selected_bands"]
    if dataset in payload:
        if int(payload.get("index_base", 1)) != 1:
            raise ValueError("Band files supplied to this script must use 1-based indices.")
        return payload[dataset]
    raise KeyError(f"No selected bands for {dataset} in {path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PSGCN.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--bands-file", default=None, help="A 1-based DRLBS JSON export; defaults to Table D3 bands.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--predict-map", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    output = args.output or str(Path("outputs") / "psgcn" / args.dataset)
    config = PSGCNConfig(runs=args.runs, epochs=args.epochs, seed=args.seed)
    summary = train_psgcn(
        args.dataset,
        args.data_root,
        output,
        read_bands(args.bands_file, args.dataset),
        config,
        args.device,
        args.predict_map,
        args.quiet,
    )
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {Path(output).resolve()}")


if __name__ == "__main__":
    main()
