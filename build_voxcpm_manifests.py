from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build VoxCPM JSONL manifests from a prepared single-speaker dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/cosrigel_north_female"),
        help="Directory produced by prepare_north_female_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("finetune/manifests"),
        help="Where train.jsonl, val.jsonl, and stats.json will be written.",
    )
    parser.add_argument(
        "--train-filelist",
        type=Path,
        default=Path("train.txt"),
        help="Relative path inside dataset-dir for the train split filelist.",
    )
    parser.add_argument(
        "--val-filelist",
        type=Path,
        default=Path("val.txt"),
        help="Relative path inside dataset-dir for the val split filelist.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Drop clips shorter than this number of seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Drop clips longer than this number of seconds.",
    )
    parser.add_argument(
        "--ref-audio-ratio",
        type=float,
        default=0.4,
        help="Fraction of rows that should include ref_audio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic ref_audio pairing.",
    )
    return parser.parse_args()


def load_metadata(dataset_dir: Path) -> dict[str, dict]:
    metadata_path = dataset_dir / "metadata.csv"
    with metadata_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_name: dict[str, dict] = {}
    for row in rows:
        by_name[row["file_name"]] = {
            "file_name": row["file_name"],
            "text": row["text"],
            "duration_seconds": float(row["duration_seconds"]),
            "audio_path": (dataset_dir / "audio" / row["file_name"]).resolve(),
            "speaker": "default",
        }
    return by_name


def load_split(filelist_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with filelist_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            audio_rel, text = line.split("|", 1)
            rows.append((Path(audio_rel).name, text))
    return rows


def choose_ref_audio(
    records: list[dict],
    current: dict,
    rng: random.Random,
) -> Path | None:
    candidates = [
        record["audio_path"]
        for record in records
        if record["speaker"] == current["speaker"] and record["file_name"] != current["file_name"]
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def build_manifest_rows(
    split_rows: list[tuple[str, str]],
    metadata: dict[str, dict],
    min_duration: float,
    max_duration: float,
    ref_audio_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    resolved_rows: list[dict] = []
    dropped_too_short = 0
    dropped_too_long = 0
    missing = 0

    for file_name, _ in split_rows:
        record = metadata.get(file_name)
        if record is None:
            missing += 1
            continue
        duration = record["duration_seconds"]
        if duration < min_duration:
            dropped_too_short += 1
            continue
        if duration > max_duration:
            dropped_too_long += 1
            continue
        resolved_rows.append(record)

    manifest_rows: list[dict] = []
    for record in resolved_rows:
        payload = {
            "audio": str(record["audio_path"]),
            "text": record["text"],
            "duration": record["duration_seconds"],
            "dataset_id": 0,
        }
        if rng.random() < ref_audio_ratio:
            ref_audio = choose_ref_audio(resolved_rows, record, rng)
            if ref_audio is not None:
                payload["ref_audio"] = str(ref_audio)
        manifest_rows.append(payload)

    stats = {
        "raw_rows": len(split_rows),
        "kept_rows": len(manifest_rows),
        "dropped_too_short": dropped_too_short,
        "dropped_too_long": dropped_too_long,
        "missing_rows": missing,
        "with_ref_audio": sum(1 for row in manifest_rows if "ref_audio" in row),
    }
    return manifest_rows, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    rng = random.Random(args.seed)

    metadata = load_metadata(dataset_dir)
    train_split = load_split(dataset_dir / args.train_filelist)
    val_split = load_split(dataset_dir / args.val_filelist)

    train_rows, train_stats = build_manifest_rows(
        split_rows=train_split,
        metadata=metadata,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        ref_audio_ratio=args.ref_audio_ratio,
        rng=rng,
    )
    val_rows, val_stats = build_manifest_rows(
        split_rows=val_split,
        metadata=metadata,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        ref_audio_ratio=args.ref_audio_ratio,
        rng=rng,
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    stats = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "ref_audio_ratio": args.ref_audio_ratio,
        "seed": args.seed,
        "train": train_stats,
        "val": val_stats,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
