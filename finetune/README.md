# VoxCPM Fine-Tune Workflow

This project is set up for the `datasets/cosrigel_north_female` single-speaker dataset.

## 1. Build training manifests

```bash
python3 build_voxcpm_manifests.py
```

This writes:

- `finetune/manifests/train.jsonl`
- `finetune/manifests/val.jsonl`
- `finetune/manifests/stats.json`

Each line follows the official VoxCPM manifest format:

```json
{"audio": "/abs/path.wav", "text": "Transcript", "duration": 12.3, "dataset_id": 0, "ref_audio": "/abs/path_ref.wav"}
```

Notes:

- Clips shorter than `3.0s` or longer than `30.0s` are dropped by default.
- `ref_audio` is added to roughly 40% of samples, using another clip from the same speaker.

## 2. Run LoRA fine-tuning

```bash
.venv/bin/python run_voxcpm_finetune.py \
  --config finetune/configs/voxcpm2_north_female_lora.yaml
```

The runner will:

- download/resolve the base `openbmb/VoxCPM2` snapshot into `.cache/huggingface`
- download the official `train_voxcpm_finetune.py` from `OpenBMB/VoxCPM`
- resolve all local paths into `finetune/.resolved/active_config.yaml`
- launch the upstream training script

Outputs:

- checkpoints: `finetune/checkpoints/north_female_lora/latest/`
- tensorboard logs: `finetune/tensorboard/north_female_lora/`

The LoRA checkpoint directory should contain:

- `lora_weights.safetensors` or `lora_weights.ckpt`
- `lora_config.json`
- optimizer/scheduler/training state files

## 3. Use the fine-tuned voice later

Inference is already wired in `main.py`:

```bash
.venv/bin/python main.py \
  --lora-weights finetune/checkpoints/north_female_lora/latest \
  --text "Xin chào, đây là giọng đã fine-tune." \
  --output output_north_female_lora.wav
```

## 4. Full fine-tuning

If LoRA is not enough, use:

```bash
.venv/bin/python run_voxcpm_finetune.py \
  --config finetune/configs/voxcpm2_north_female_full.yaml
```

The resulting checkpoint is a complete model directory, and later inference would load that directory directly as `--model`.

## Practical note

This machine is currently CPU-only. Fine-tuning VoxCPM2 is realistically a GPU task. The official docs estimate roughly:

- LoRA on VoxCPM2: around `20 GB VRAM`
- Full fine-tuning on VoxCPM2: around `40 GB VRAM`
