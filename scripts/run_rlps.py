import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlps.config import DATASETS
from rlps.drlbs import DRLBSConfig, train_drlbs
from rlps.psgcn import PSGCNConfig, train_psgcn


def main():
    parser = argparse.ArgumentParser(description="Run the two-stage RLPS pipeline.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    drlbs_output = root / "drlbs" / args.dataset
    psgcn_output = root / "psgcn" / args.dataset
    bands = train_drlbs(
        args.dataset,
        args.data_root,
        str(drlbs_output),
        DRLBSConfig(episodes=args.episodes, seed=args.seed),
        args.device,
        args.quiet,
    )
    summary = train_psgcn(
        args.dataset,
        args.data_root,
        str(psgcn_output),
        bands,
        PSGCNConfig(epochs=args.epochs, runs=args.runs, seed=args.seed),
        args.device,
        False,
        args.quiet,
    )
    print(json.dumps({"selected_bands_one_based": bands, "psgcn": summary}, indent=2))


if __name__ == "__main__":
    main()
