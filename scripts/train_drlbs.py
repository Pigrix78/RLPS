import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlps.config import DATASETS
from rlps.drlbs import DRLBSConfig, train_drlbs


def main():
    parser = argparse.ArgumentParser(description="Train DRLBS and export a fixed band subset.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    output = args.output or str(Path("outputs") / "drlbs" / args.dataset)
    config = DRLBSConfig(episodes=args.episodes, seed=args.seed)
    bands = train_drlbs(args.dataset, args.data_root, output, config, args.device, args.quiet)
    print(f"Selected bands (1-based): {bands}")
    print(f"Outputs: {Path(output).resolve()}")


if __name__ == "__main__":
    main()
