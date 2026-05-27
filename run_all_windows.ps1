param(
    [string]$PythonExe = "py",
    [string]$PythonVersion = "-3.12",
    [string]$VenvDir = ".venv",
    [string]$ConfigPath = "finetune\configs\voxcpm2_north_female_lora.yaml",
    [string]$DefaultLoraWeights = "finetune\checkpoints\north_female_lora\latest",
    [string]$Text = "Xin chao, day la giong nu mien Bac da fine-tune.",
    [string]$Output = "output_north_female_lora.wav",
    [string]$ReferenceWav = "",
    [string]$PromptWav = "",
    [string]$PromptText = "",
    [string]$TorchIndexUrl = "",
    [string]$LoraWeights = "",
    [switch]$SkipDataset,
    [switch]$SkipTrain,
    [switch]$SkipInference,
    [switch]$AllowCpuTrain,
    [switch]$AllowLowVramTrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Arguments -join ' ')"
    }
}

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "[INFO] $Message"
}

Write-Step "Repo root: $ScriptDir"

if (-not (Test-Path $VenvDir)) {
    Write-Step "Creating virtual environment at $VenvDir"
    & $PythonExe $PythonVersion -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment."
    }
}

$Python = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python executable not found at $Python"
}

Write-Step "Upgrading packaging tools"
Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

if (-not [string]::IsNullOrWhiteSpace($TorchIndexUrl)) {
    Write-Step "Installing PyTorch wheels from custom index"
    Invoke-Python -Arguments @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchaudio",
        "torchvision",
        "--index-url",
        $TorchIndexUrl
    )
}

Write-Step "Installing project dependencies"
Invoke-Python -Arguments @("-m", "pip", "install", "-r", "requirements-colab.txt")

if (-not $SkipDataset) {
    Write-Step "Downloading dataset and building local audio cache"
    Invoke-Python -Arguments @("prepare_north_female_dataset.py")

    Write-Step "Building VoxCPM manifests"
    Invoke-Python -Arguments @("build_voxcpm_manifests.py")
}
else {
    Write-Step "Skipping dataset preparation"
}

if (-not $SkipTrain) {
    Write-Step "Checking CUDA availability"
    $cudaStatus = & $Python -c "import torch; print('cuda_available=' + str(torch.cuda.is_available())); print('cuda_device_count=' + str(torch.cuda.device_count()))"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to probe CUDA status."
    }
    Write-Host $cudaStatus

    if ($cudaStatus -notmatch "cuda_available=True" -and -not $AllowCpuTrain) {
        throw "CUDA is not available. Re-run on an NVIDIA GPU machine, or pass -AllowCpuTrain if you only want to test the pipeline."
    }

    Write-Step "Launching fine-tune"
    $trainArgs = @("run_voxcpm_finetune.py", "--config", $ConfigPath)
    if ($AllowCpuTrain) {
        $trainArgs += "--allow-cpu"
    }
    if ($AllowLowVramTrain) {
        $trainArgs += "--allow-low-vram"
    }
    Invoke-Python -Arguments $trainArgs
}
else {
    Write-Step "Skipping train"
}

if (-not $SkipInference) {
    $ResolvedLoraWeights = $LoraWeights
    if ([string]::IsNullOrWhiteSpace($ResolvedLoraWeights)) {
        $ResolvedLoraWeights = $DefaultLoraWeights
    }

    if (-not (Test-Path $ResolvedLoraWeights)) {
        Write-Warning "LoRA weights not found at $ResolvedLoraWeights. Inference will be skipped."
    }
    else {
        Write-Step "Running inference with fine-tuned weights"
        $inferArgs = @(
            "main.py",
            "--lora-weights",
            $ResolvedLoraWeights,
            "--text",
            $Text,
            "--output",
            $Output
        )

        if (-not [string]::IsNullOrWhiteSpace($ReferenceWav)) {
            $inferArgs += @("--reference-wav", $ReferenceWav)
        }

        if (-not [string]::IsNullOrWhiteSpace($PromptWav)) {
            if ([string]::IsNullOrWhiteSpace($PromptText)) {
                throw "--PromptText is required when --PromptWav is provided."
            }
            $inferArgs += @("--prompt-wav", $PromptWav, "--prompt-text", $PromptText)
        }

        Invoke-Python -Arguments $inferArgs
    }
}
else {
    Write-Step "Skipping inference"
}

Write-Step "Done"
