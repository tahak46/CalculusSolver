import argparse
import os
from pathlib import Path
from typing import Iterable, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interleave JSONL sources and split into train/val/test."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source file or directory paths to interleave.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.9, 0.05, 0.05],
        help="Ratios for train/val/test splits.",
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for train/val/test jsonl files."
    )
    return parser.parse_args()


def list_jsonl_paths(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() == ".jsonl":
        return [source]
    if source.is_dir():
        return sorted(
            [
                p
                for p in source.iterdir()
                if p.is_file() and p.suffix.lower() == ".jsonl"
            ]
        )
    raise ValueError(f"Invalid source path: {source}")


def source_line_iterator(paths: List[Path]) -> Iterable[str]:
    for file_path in paths:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if line:
                    yield line


def count_lines(paths: List[Path]) -> int:
    total = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    total += 1
    return total


def interleave_sources(source_paths: List[List[Path]]):
    iterators = [iter(source_line_iterator(paths)) for paths in source_paths]
    active = [True] * len(iterators)

    while any(active):
        for index, it in enumerate(iterators):
            if not active[index]:
                continue
            try:
                yield next(it)
            except StopIteration:
                active[index] = False


def normalize_ratios(ratios: List[float]) -> List[float]:
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Ratios must sum to a positive value.")
    return [r / total for r in ratios]


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_lists = []
    for source in args.sources:
        source_path = Path(source)
        source_paths = list_jsonl_paths(source_path)
        if not source_paths:
            raise ValueError(f"No JSONL files found in source: {source_path}")
        source_lists.append(source_paths)

    total_lines = sum(count_lines(paths) for paths in source_lists)
    if total_lines == 0:
        raise ValueError("No lines found in the provided sources.")

    ratios = normalize_ratios(args.ratios)
    boundaries = [0]
    cumulative = 0
    for ratio in ratios[:-1]:
        cumulative += ratio
        boundaries.append(int(total_lines * cumulative))
    boundaries.append(total_lines)

    paths = {
        "train": out_dir / "train.jsonl",
        "val": out_dir / "val.jsonl",
        "test": out_dir / "test.jsonl",
    }

    writers = {
        "train": paths["train"].open("w", encoding="utf-8"),
        "val": paths["val"].open("w", encoding="utf-8"),
        "test": paths["test"].open("w", encoding="utf-8"),
    }

    try:
        index = 0
        for line in interleave_sources(source_lists):
            if index < boundaries[1]:
                writers["train"].write(line + "\n")
            elif index < boundaries[2]:
                writers["val"].write(line + "\n")
            else:
                writers["test"].write(line + "\n")
            index += 1

        print(f"Total lines processed: {index}")
        print(f"Train: {boundaries[1]} lines")
        print(f"Val: {boundaries[2] - boundaries[1]} lines")
        print(f"Test: {boundaries[3] - boundaries[2]} lines")
    finally:
        for writer in writers.values():
            writer.close()


if __name__ == "__main__":
    main()
