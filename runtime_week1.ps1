# Week 1: 4-bit llama.cpp runtime vs the transformers bf16 setup, accuracy and energy on MBPP.
#   Start-Process pwsh -ArgumentList "-NoExit","-File","runtime_week1.ps1"       (keep the lid open)
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"; $d = "runs\runtime"; $log = "$d\week1.log"
$env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"
New-Item -ItemType Directory -Force $d | Out-Null
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'MM-dd HH:mm')] START $name" | Tee-Object -FilePath $log -Append
    & $cmd *>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "[$(Get-Date -Format 'MM-dd HH:mm')] FAILED $name (exit $LASTEXITCODE); rerun to resume" | Tee-Object -FilePath $log -Append; exit 1 }
    "[$(Get-Date -Format 'MM-dd HH:mm')] DONE $name" | Tee-Object -FilePath $log -Append
}
if (-not (docker info --format '{{.ServerVersion}}' 2>$null)) { "Docker Desktop is not running: start it first." | Tee-Object -FilePath $log -Append; exit 1 }
Step "llama.cpp 4-bit, all 257 tasks" { & $py -u runtime_bench.py --engine llamacpp --k 8 --out "$d\llamacpp_q4.jsonl" }
Step "transformers bf16, first 100 tasks (energy)" { & $py -u runtime_bench.py --engine hf --k 8 --limit 100 --out "$d\hf_bf16.jsonl" }
"[$(Get-Date -Format 'MM-dd HH:mm')] WEEK 1 COMPLETE" | Tee-Object -FilePath $log -Append
