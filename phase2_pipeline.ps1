# Phase 2, round 1: baseline -> collect verified solutions -> fine-tune -> re-test.
# Each step resumes from its output, so rerunning this script continues where it stopped.
#   Start-Process pwsh -ArgumentList "-NoExit","-File","phase2_pipeline.ps1"
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"; $d = "runs\phase2"; $log = "$d\pipeline.log"
$env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"      # model and data are cached: never depend on the network mid-run
New-Item -ItemType Directory -Force $d | Out-Null
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'MM-dd HH:mm')] START $name" | Tee-Object -FilePath $log -Append
    & $cmd *>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "[$(Get-Date -Format 'MM-dd HH:mm')] FAILED $name (exit $LASTEXITCODE); rerun this script to resume" | Tee-Object -FilePath $log -Append; exit 1 }
    "[$(Get-Date -Format 'MM-dd HH:mm')] DONE $name" | Tee-Object -FilePath $log -Append
}
if (-not (docker info --format '{{.ServerVersion}}' 2>$null)) { "Docker Desktop is not running: start it first (generated code runs in its sandbox)." | Tee-Object -FilePath $log -Append; exit 1 }

Step "baseline: best of 8 on the test set" { & $py -u phase2.py bestof --k 8 --greedy --out "$d\bestof_base.jsonl" }
Step "collect: verified solutions to 464 training problems" { & $py -u phase2.py collect --k 8 --out "$d\collect_r1.jsonl" }
if (-not (Test-Path "$d\adapter_r1\adapter_config.json")) {
    Step "train: LoRA on round-1 verified solutions" { & $py -u phase2.py train --data "$d\collect_r1.jsonl" --out "$d\adapter_r1" }
}
Step "re-test: fine-tuned model on the test set" { & $py -u phase2.py bestof --k 8 --greedy --adapter "$d\adapter_r1" --out "$d\bestof_r1.jsonl" }
"[$(Get-Date -Format 'MM-dd HH:mm')] PIPELINE COMPLETE" | Tee-Object -FilePath $log -Append
