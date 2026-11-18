import argparse
import json
from pathlib import Path
from typing import Literal, Protocol, cast

from time_series import choose_device, evaluate_checkpoint


class _EvalArgs(Protocol):
    data: list[str]
    checkpoint: Path
    split: Literal["validation", "test"]
    batch_size: int
    num_workers: int
    device: Literal["auto", "cpu", "cuda"]


def _parse_args() -> _EvalArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved SST forecast checkpoint on its frozen "
            "validation or test period."
        )
    )
    _ = parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Annual reef_sst_*.nc files, glob patterns, or directories",
    )
    _ = parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/gbr_sst_lstm.pt"),
    )
    _ = parser.add_argument("--split", choices=("validation", "test"), default="test")
    _ = parser.add_argument("--batch-size", type=int, default=1024)
    _ = parser.add_argument("--num-workers", type=int, default=0)
    _ = parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return cast(_EvalArgs, cast(object, parser.parse_args()))


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    metrics = evaluate_checkpoint(
        data_inputs=args.data,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=choose_device(args.device),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
