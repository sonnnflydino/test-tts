from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import wave
from pathlib import Path

DATASET_ID = "cosrigel/vn_tts_medium_clean"


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


def save_audio(audio_data: dict, destination: Path) -> None:
    """Save audio từ HuggingFace datasets format ra WAV file."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Trường hợp 1: raw bytes (thường với streaming)
    if isinstance(audio_data, dict) and audio_data.get("bytes"):
        destination.write_bytes(audio_data["bytes"])
        return

    # Trường hợp 2: decoded numpy array
    if isinstance(audio_data, dict) and "array" in audio_data:
        try:
            import soundfile as sf  # type: ignore
            sf.write(str(destination), audio_data["array"], audio_data["sampling_rate"])
            return
        except ImportError:
            pass
        try:
            import scipy.io.wavfile as wavfile  # type: ignore
            import numpy as np
            arr = audio_data["array"]
            sr = audio_data["sampling_rate"]
            # soundfile not available — use scipy, convert float→int16
            if arr.dtype.kind == "f":
                arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
            wavfile.write(str(destination), sr, arr)
            return
        except ImportError:
            pass

    raise RuntimeError(
        f"Cannot save audio to {destination}: "
        "install soundfile (`pip install soundfile`) or scipy."
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Import datasets lazily để script vẫn chạy được khi không có thư viện
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Thư viện `datasets` chưa được cài. Chạy: pip install datasets"
        ) from exc

    print(f"Loading dataset {DATASET_ID} (streaming)...")
    ds = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    records: list[dict] = []
    downloaded = 0

    for row in ds:
        row_source = str(row.get("source", "")).strip()
        if row_source != args.source:
            continue

        downloaded += 1
        file_name = f"{downloaded:05d}.wav"
        destination = audio_dir / file_name

        if not destination.exists():
            audio_data = row.get("audio")
            if audio_data is None:
                print(f"[WARN] Row {downloaded} has no audio field, skipping.")
                downloaded -= 1
                continue
            save_audio(audio_data, destination)

        duration_seconds = wav_duration_seconds(destination)
        text = " ".join(row.get("text", "").split())

        records.append(
            {
                "file_name": file_name,
                "text": text,
                "source": row_source,
                "duration_seconds": duration_seconds,
                "remote_url": "",
            }
        )

        print(
            f"[{downloaded}] saved {file_name} ({duration_seconds:.2f}s) | {text[:60]}",
            flush=True,
        )

        if args.max_samples and downloaded >= args.max_samples:
            break

    if not records:
        raise SystemExit(f"No rows matched source={args.source!r}")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "text", "source", "duration_seconds", "remote_url"],
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
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
