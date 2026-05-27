# Lightning AI Deployment

This folder is the shortest path to run VoxCPM2 LoRA fine-tuning on a free Lightning AI GPU Studio.

## Why Lightning

As of 2026-05-27:

- Lightning AI documents `~80 free GPU hours` per month on the free tier after verification.
- The pricing page lists single-GPU options including `L4 24 GB` and `A100 40 GB`.
- VoxCPM2's official fine-tuning guide estimates `~20 GB VRAM` for LoRA fine-tuning.

That makes `L4` or `A100` the realistic free-tier choices for this project.

## 1. Create and start a Studio

Option A: use the web UI at `studio.lightning.ai`.

Option B: use the local CLI after `pip install lightning-sdk -U` and `lightning login`:

```bash
TEAMSPACE="your-user/your-teamspace" \
STUDIO_NAME="voxcpm-north-female" \
MACHINE="L4" \
bash lightning/create_studio_cli.sh
```

Notes:

- Free Studios restart every 4 hours.
- Storage and installed packages persist across restarts.

## 2. Clone this repo in the Studio

Inside the Studio terminal:

```bash
git clone <your-repo-url>
cd test-tts
```

Or upload the folder directly if you are not using Git.

## 3. Bootstrap the environment

```bash
bash lightning/bootstrap_studio.sh
```

This creates `.venv-lightning/` and installs the training dependencies.

## 4. Install the on-stop action

This is optional but recommended for free-tier restarts:

```bash
bash lightning/install_on_stop_action.sh
```

It installs `$HOME/.lightning_studio/on_stop.sh`, which sends `SIGTERM` to the official VoxCPM training script. That script already saves a checkpoint on `SIGTERM`/`SIGINT`, so restarts can resume from `latest/`.

## 5. Train LoRA

```bash
bash lightning/train_lora.sh
```

This will:

- regenerate `finetune/manifests/train.jsonl` and `val.jsonl`
- resolve the local VoxCPM2 snapshot
- download the official OpenBMB training script
- start LoRA fine-tuning

Checkpoints land in:

```text
finetune/checkpoints/north_female_lora/
```

The current config is:

- `finetune/configs/voxcpm2_north_female_lora.yaml`

To resume after a restart or manual stop, run the same command again:

```bash
bash lightning/train_lora.sh
```

The upstream trainer resumes automatically from `latest/` when present.

## 6. Smoke-test the latest LoRA

```bash
bash lightning/smoke_test_latest_lora.sh
```

This loads:

```text
finetune/checkpoints/north_female_lora/latest
```

and writes:

```text
output_north_female_lora.wav
```

## 7. Full fine-tuning

Only use this if LoRA is not enough and you have enough VRAM:

```bash
bash lightning/train_full.sh
```

Config:

- `finetune/configs/voxcpm2_north_female_full.yaml`

## References

- Lightning AI Studio docs: https://lightning.ai/docs/overview/ai-studio
- Lightning CLI docs: https://lightning.ai/docs/overview/ai-studio/cli
- Lightning quickstart: https://lightning.ai/docs/overview/quick-start
- VoxCPM fine-tuning guide: https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html
