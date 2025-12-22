# Activate Conda environment with MKL BLAS configuration
# Usage: . .\activate_mmm.ps1

# Set BLAS flags for PyTensor
$env:PYTENSOR_FLAGS='blas__ldflags=-LC:/Users/jorda/miniconda3/envs/mmm-env/Library/lib -lmkl_rt'

# Initialize Conda for PowerShell and activate
$condaPath = "C:\Users\jorda\miniconda3\Scripts\conda.exe"
(& $condaPath "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate mmm-env

Write-Host ""
Write-Host "Environment activated: mmm-env" -ForegroundColor Green
Write-Host "BLAS: Intel MKL configured" -ForegroundColor Green
