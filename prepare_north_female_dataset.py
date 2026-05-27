from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path

DATASET_ID = "cosrigel/vn_tts_medium_clean"
ROWS_API = "https://datasets-server.huggingface.co/rows"
REQUEST_RETRIES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the northern-female subset from cosrigel/vn_tts_medium_clean."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/cosrigel_north_female"),
        help="Directory where audio and manifests will be written.",
    )
    parser.add_argument(
        "--source",
        default="1",
        help='Speaker source id. "1" is female north according to the dataset card.',
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Rows to fetch per page from the dataset server.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=20,
        help="Every Nth sample goes to validation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for quick tests. 0 means download all matching rows.",
    )
    return parser.parse_args()


def fetch_rows(offset: int, length: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    url = f"{ROWS_API}?{query}"
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
            return [row["row"] for row in payload["rows"]]
        except Exception as exc:
            if attempt == REQUEST_RETRIES:
                raise RuntimeError(f"Failed to fetch rows at offset={offset}") from exc
            time.sleep(attempt)
    return []


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".tmp")
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with temp_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            temp_path.replace(destination)
            return
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            if attempt == REQUEST_RETRIES:
                raise RuntimeError(f"Failed to download audio to {destination}") from exc
            time.sleep(attempt)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def write_filelist(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"audio/{record['file_name']}|{record['text']}\n")


def select_reference(records: list[dict]) -> dict | None:
    if not records:
        return None
    return min(records, key=lambda item: abs(item["duration_seconds"] - 8.0))


def create_reference_symlink(output_dir: Path, record: dict | None) -> None:
    if record is None:
        return
    link_path = output_dir / "reference.wav"
    target = output_dir / "audio" / record["file_name"]
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    try:
        os.symlink(Path("audio") / record["file_name"], link_path)
    except OSError:
        shutil.copyfile(target, link_path)
    (output_dir / "reference.txt").write_text(record["text"], encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    downloaded = 0
    offset = 0
    page = 0

    while True:
        rows = fetch_rows(offset=offset, length=args.page_size)
        if not rows:
            break

        for row in rows:
            if row["source"] != args.source:
                continue

            downloaded += 1
            file_name = f"{downloaded:05d}.wav"
            destination = audio_dir / file_name
            audio_url = row["audio"][0]["src"]
            text = " ".join(row["text"].split())

            if not destination.exists():
                download_file(audio_url, destination)

            duration_seconds = wav_duration_seconds(destination)
            records.append(
                {
                    "file_name": file_name,
                    "text": text,
                    "source": row["source"],
                    "duration_seconds": duration_seconds,
                    "remote_url": audio_url,
                }
            )

            print(
                f"[{downloaded}] saved {file_name} "
                f"({duration_seconds:.2f}s)"
            , flush=True)

            if args.max_samples and downloaded >= args.max_samples:
                break

        page += 1
        offset += len(rows)
        print(f"Processed page {page}, offset now {offset}", flush=True)

        if args.max_samples and downloaded >= args.max_samples:
            break

        if len(rows) < args.page_size:
            break

        time.sleep(0.1)

    if not records:
        raise SystemExit(f"No rows matched source={args.source!r}")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "text",
                "source",
                "duration_seconds",
                "remote_url",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    val_records = records[:: args.val_every]
    val_names = {record["file_name"] for record in val_records}
    train_records = [record for record in records if record["file_name"] not in val_names]

    write_filelist(output_dir / "all.txt", records)
    write_filelist(output_dir / "train.txt", train_records)
    write_filelist(output_dir / "val.txt", val_records)

    reference_record = select_reference(records)
    create_reference_symlink(output_dir, reference_record)

    total_seconds = sum(record["duration_seconds"] for record in records)
    stats = {
        "dataset": DATASET_ID,
        "source": args.source,
        "samples": len(records),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "total_seconds": total_seconds,
        "total_hours": total_seconds / 3600.0,
        "reference_file": reference_record["file_name"] if reference_record else None,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
