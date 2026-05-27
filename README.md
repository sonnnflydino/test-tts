# VoxCPM2 TTS - Nữ Miền Bắc

Repo này chuẩn bị sẵn pipeline để:

- tải dataset giọng nữ miền Bắc
- build manifest train/val cho `VoxCPM2`
- fine-tune bằng `LoRA`
- chạy inference lại với adapter đã train

## Yêu cầu

- Windows 10/11
- Python 3.12 hoặc 3.11
- `git`
- GPU NVIDIA có CUDA nếu muốn train local

Nếu không có GPU đủ mạnh trên máy Windows, nên dùng Colab / Lightning / Modal để train, còn máy Windows dùng để chuẩn bị dữ liệu và chạy inference.

## 1. Clone repo

```powershell
git clone https://github.com/<your-username>/<your-repo>.git
cd test-tts
```

## 2. Tạo môi trường

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## 3. Cài package

```powershell
pip install -r requirements-colab.txt
```

File này đủ cho luồng chuẩn bị data và inference. Nếu bạn train local theo script chính thức của VoxCPM, cài thêm:

```powershell
pip install -r requirements-lightning.txt
```

## 4. Tải dataset giọng nữ miền Bắc

Dataset được chọn là `cosrigel/vn_tts_medium_clean`, trong đó:

- `source=1` là `Nữ - Miền Bắc`

Chạy lệnh sau để tải lại audio và tạo manifest cục bộ:

```powershell
python prepare_north_female_dataset.py
```

Sau lệnh này bạn sẽ có:

- `datasets\cosrigel_north_female\audio\*.wav`
- `datasets\cosrigel_north_female\metadata.csv`
- `datasets\cosrigel_north_female\train.txt`
- `datasets\cosrigel_north_female\val.txt`
- `datasets\cosrigel_north_female\reference.wav`

Lưu ý: các file `.wav` không được commit lên GitHub để tránh repo phình quá lớn. Khi clone về, bạn cần chạy lại bước này để tải audio về máy.

## 5. Build manifest train/val

```powershell
python build_voxcpm_manifests.py
```

Lệnh này sẽ sinh:

- `finetune\manifests\train.jsonl`
- `finetune\manifests\val.jsonl`
- `finetune\manifests\stats.json`

## 6. Fine-tune LoRA

### 6.1. Chạy local trên Windows

Nếu máy bạn có GPU đủ mạnh, chạy:

```powershell
python run_voxcpm_finetune.py --config finetune\configs\voxcpm2_north_female_lora.yaml
```

Nếu muốn test CPU-only cho bước resolve config, thêm:

```powershell
python run_voxcpm_finetune.py --config finetune\configs\voxcpm2_north_female_lora.yaml --allow-cpu --dry-run
```

### 6.2. Output sau train

Checkpoint LoRA sẽ nằm tại:

```text
finetune\checkpoints\north_female_lora\latest
```

Khi train xong, thư mục này là thứ bạn dùng lại cho các lần chạy sau.

## 7. Chạy inference bằng adapter đã fine-tune

```powershell
python main.py `
  --lora-weights finetune\checkpoints\north_female_lora\latest `
  --text "Xin chào, đây là giọng nữ miền Bắc đã fine-tune." `
  --output output_north_female_lora.wav
```

Nếu muốn dùng `reference.wav` thay vì `LoRA`, chạy:

```powershell
python main.py `
  --reference-wav datasets\cosrigel_north_female\reference.wav `
  --text "Xin chào, đây là giọng test clone." `
  --output output_reference_test.wav
```

## 8. Flow đầy đủ

### Cách nhanh nhất

Chạy một lệnh PowerShell để làm tuần tự toàn bộ flow:

```powershell
.\run_all_windows.ps1
```

Script này sẽ:

- tạo `.venv` nếu chưa có
- cài dependency
- tải lại dataset audio
- build manifest
- train LoRA nếu máy có CUDA
- chạy inference bằng adapter vừa train

Nếu máy Windows có GPU NVIDIA nhưng `torch` vẫn chưa nhận CUDA, bạn có thể cài lại PyTorch từ CUDA wheel index rồi chạy script với tham số `-TorchIndexUrl`.

Nếu muốn chỉ chạy từng bước thủ công, dùng các lệnh dưới đây:

```powershell
python prepare_north_female_dataset.py
python build_voxcpm_manifests.py
python run_voxcpm_finetune.py --config finetune\configs\voxcpm2_north_female_lora.yaml
python main.py --lora-weights finetune\checkpoints\north_female_lora\latest --text "Xin chào"
```

## 9. Vì sao thư mục `datasets` bị mất `.wav`?

Tôi đã xóa các file audio tạm và thêm `.gitignore` để không đẩy các file nặng lên GitHub.

Lý do:

- `.wav` của dataset là artifact lớn, không nên nằm trong repo source
- GitHub không phù hợp để lưu hàng trăm file audio train
- dữ liệu có thể tải lại bằng `python prepare_north_female_dataset.py`

Nói ngắn gọn: không mất dữ liệu nguồn, chỉ là repo được dọn sạch để push lên GitHub.

## 10. Script phụ

- `colab/README.md`: hướng dẫn chạy trên Colab
- `lightning/README.md`: hướng dẫn chạy trên Lightning AI
- `finetune/README.md`: giải thích workflow fine-tune chi tiết hơn
