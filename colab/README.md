# Colab Workflow

This setup is meant for Google Colab, with persistence through Google Drive.

## Reality check

VoxCPM's official guide estimates:

- `~20 GB VRAM` for VoxCPM2 LoRA fine-tuning
- `~40 GB VRAM` for full fine-tuning

That means:

- Colab free often fails this preflight because it commonly gives smaller GPUs.
- Colab Pro / PayGo with a larger GPU is much more realistic.

The notebook and scripts in this folder explicitly check VRAM before training starts.

## Files

- `colab/VoxCPM2_North_Female_Finetune.ipynb`: Colab notebook
- `colab/bootstrap_colab.sh`: install dependencies into the Colab runtime
- `colab/preflight_check.py`: GPU/VRAM gate
- `colab/train_lora_colab.sh`: prepare dataset + manifests + launch LoRA fine-tuning
- `colab/smoke_test_latest_lora_colab.sh`: run inference with the latest adapter
- `make_colab_bundle.sh`: export this repo as a zip for manual upload into Colab

## Fastest path

1. On your local machine, create a bundle:

```bash
bash make_colab_bundle.sh
```

2. Open the notebook:

- `colab/VoxCPM2_North_Female_Finetune.ipynb`

3. In Colab:

- mount Google Drive
- change runtime type to `GPU` before running `bootstrap_colab.sh`
- upload `test-tts-colab-bundle.zip` or clone your Git repo
- run `bash colab/bootstrap_colab.sh`
- run the preflight cell
- run `bash colab/train_lora_colab.sh`

## Resume

The trainer writes into:

- `finetune/checkpoints/north_female_lora/latest`

Because the notebook works out of Google Drive, you can reconnect later and run the same train command again to resume.
