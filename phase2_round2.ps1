# Phase 2, round 2: fix round 1's data mix.
#   1. Re-attack the 139 training problems round 1 never solved, with 32 samples instead of 8.
#   2. Fine-tune (from the base model) only on the FRONTIER: problems solved at most half the time,
#      one verified solution each, so easy problems cannot dominate the data again.
#   3. Re-test on the same 257 test tasks.
# Resumable: rerun this script to continue.   Start-Process pwsh -ArgumentList "-NoExit","-File","phase2_round2.ps1"
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"; $d = "runs\phase2"; $log = "$d\round2.log"
$env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'MM-dd HH:mm')] START $name" | Tee-Object -FilePath $log -Append
    & $cmd *>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "[$(Get-Date -Format 'MM-dd HH:mm')] FAILED $name (exit $LASTEXITCODE); rerun this script to resume" | Tee-Object -FilePath $log -Append; exit 1 }
    "[$(Get-Date -Format 'MM-dd HH:mm')] DONE $name" | Tee-Object -FilePath $log -Append
}
if (-not (docker info --format '{{.ServerVersion}}' 2>$null)) { "Docker Desktop is not running: start it first." | Tee-Object -FilePath $log -Append; exit 1 }

Step "re-attack: 32 samples on round-1 unsolved problems" { & $py -u phase2.py collect --k 32 --only-unsolved-from "$d\collect_r1.jsonl" --max-new 384 --out "$d\collect_r2.jsonl" }
if (-not (Test-Path "$d\adapter_r2\adapter_config.json")) {
    Step "train: frontier only, 1 solution per problem" { & $py -u phase2.py train --data "$d\collect_r1.jsonl" "$d\collect_r2.jsonl" --max-pass-rate 0.5 --per-problem 1 --epochs 3 --out "$d\adapter_r2" }
}
Step "re-test: round-2 model on the test set" { & $py -u phase2.py bestof --k 8 --greedy --adapter "$d\adapter_r2" --out "$d\bestof_r2.jsonl" }
"[$(Get-Date -Format 'MM-dd HH:mm')] ROUND 2 COMPLETE" | Tee-Object -FilePath $log -Append
