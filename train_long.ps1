# Long Sudoku training on a laptop GPU. Runs one training process at a time and:
#   - waits while the laptop is on battery (training checkpoints and pauses when unplugged)
#   - resumes from the last checkpoint after a GPU driver crash (at most 8 times)
#   - stops cleanly when you create a STOP file:   New-Item runs\STOP
#
#   .\train_long.ps1                              # 180 min of training into runs\long.pt
#   .\train_long.ps1 -Minutes 240 -Out runs\x.pt -Extra "--D","192","--H","192"
param([double]$Minutes = 180, [string]$Out = "runs\long.pt", [string[]]$Extra = @())
Set-Location $PSScriptRoot
Add-Type -AssemblyName System.Windows.Forms
$log = [IO.Path]::ChangeExtension($Out, ".log")
$stop = Join-Path (Split-Path $Out) "STOP"
Remove-Item $stop -ErrorAction SilentlyContinue
$PID | Out-File ([IO.Path]::ChangeExtension($Out, ".supervisor.pid"))
function Say($m) { "[supervisor $(Get-Date -Format HH:mm:ss)] $m" | Tee-Object -FilePath $log -Append }
function OnAC { [System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus -eq "Online" }

$crashes = 0; $waiting = $false
Say "start: $Minutes min budget -> $Out"
while ($true) {
    if (Test-Path $stop) { Say "STOP file found, exiting"; break }
    if (-not (OnAC)) {
        if (-not $waiting) { Say "on battery: waiting for AC power"; $waiting = $true }
        Start-Sleep 30; continue
    }
    if ($waiting) { Say "AC power back, resuming"; $waiting = $false }
    & .venv\Scripts\python.exe -u settle_graph.py train --minutes $Minutes --out $Out --resume @Extra *>> $log
    $code = $LASTEXITCODE
    if ($code -eq 0) { Say "training finished"; break }
    if ($code -eq 3) { continue }                                  # paused for battery; loop waits for AC
    $crashes++
    if ($crashes -gt 8) { Say "crashed $crashes times, giving up"; break }
    Say "training exited with code $code (crash $crashes), resuming in 30 s"; Start-Sleep 30
}
Remove-Item ([IO.Path]::ChangeExtension($Out, ".supervisor.pid")) -ErrorAction SilentlyContinue
