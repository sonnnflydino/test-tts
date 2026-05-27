from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
from voxcpm import VoxCPM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local VoxCPM2 TTS test.")
    parser.add_argument("--model", default="openbmb/VoxCPM2")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--output", type=Path, default=Path("output_voxcpm2_hq.wav"))
    parser.add_argument(
        "--lora-weights",
        type=Path,
        help="Path to a LoRA checkpoint file or directory produced by fine-tuning.",
    )
    parser.add_argument(
        "--text",
        default="Xin chào bạn. This is a clearer English and Vietnamese speech test for VoxCPM2.",
    )
    parser.add_argument(
        "--voice-control",
        default="Warm, clear, natural female voice",
        help="Style hint used when no reference audio is provided.",
    )
    parser.add_argument("--cfg-value", type=float, default=1.6)
    parser.add_argument("--inference-timesteps", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--normalize-text", action="store_true", default=True)
    parser.add_argument("--no-normalize-text", dest="normalize_text", action="store_false")
    parser.add_argument("--reference-wav", type=Path)
    parser.add_argument("--prompt-wav", type=Path)
    parser.add_argument("--prompt-text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on CPU...")
    model = VoxCPM.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir),
        device="cpu",
        optimize=False,
        load_denoiser=False,
        lora_weights_path=str(args.lora_weights) if args.lora_weights else None,
    )

    print("Synthesizing text:")
    print(args.text)
    if args.lora_weights:
        print(f"Using LoRA weights from: {args.lora_weights}")

    generation_kwargs = {
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
        "max_len": args.max_len,
        "normalize": args.normalize_text,
    }

    if args.reference_wav:
        generation_kwargs["reference_wav_path"] = str(args.reference_wav)
        text = args.text
    else:
        text = f"({args.voice_control}){args.text}"

    if args.prompt_wav:
        generation_kwargs["prompt_wav_path"] = str(args.prompt_wav)
        if not args.prompt_text:
            raise SystemExit("--prompt-text is required when --prompt-wav is set")
        generation_kwargs["prompt_text"] = args.prompt_text

    wav = model.generate(text=text, **generation_kwargs)
    sf.write(args.output, wav, model.tts_model.sample_rate)

    duration_seconds = len(wav) / model.tts_model.sample_rate
    print(f"Saved {args.output} ({duration_seconds:.2f}s)")


if __name__ == "__main__":
    main()
