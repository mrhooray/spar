from pathlib import Path
import argparse

from datasets import load_dataset


DATASET = "nvidia/SOL-ExecBench"
REVISION = "63699402f003496acc3af4eb534a5304a8ac1ea9"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("level")
    parser.add_argument("problem")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = load_dataset(
        DATASET,
        name=args.level,
        split="train",
        revision=REVISION,
    )
    reference = next(row["reference"] for row in rows if row["name"] == args.problem)
    args.output.write_text(reference, encoding="utf-8")


if __name__ == "__main__":
    main()
