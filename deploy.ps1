# Rebuild the site and deploy it to Vercel (https://settle-three-ochre.vercel.app).
#
#   .\deploy.ps1                                   # redeploy current site (e.g. after editing site/*.html)
#   .\deploy.ps1 -Ckpt settle_sudoku.pt            # ship a newly trained Sudoku checkpoint
#   .\deploy.ps1 -Ckpt runs\x.pt -Preview          # deploy to a preview URL instead of production
#
# Stops before deploying if the browser engines no longer match the Python models.
param([string]$Ckpt = "", [switch]$Preview)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run($label, [scriptblock]$cmd) {
    Write-Host "== $label" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$label failed (exit $LASTEXITCODE). Nothing was deployed." }
}

if ($Ckpt) {
    Run "export $Ckpt" { .venv\Scripts\python settle_graph.py export --ckpt $Ckpt }
    Run "PyTorch reference outputs" { .venv\Scripts\python tests\make_sudoku_fixture.py $Ckpt }
}
Run "Sudoku engine parity (JS vs PyTorch)" { node tests\sudoku_parity.test.js }
Run "build maze page" { python build_web.py }
Run "Maze engine parity (JS vs NumPy)" { node tests\parity.test.js }
if (Test-Path arena\settle_nopath.json) {
    Run "Maze 'no path' model reference outputs" { python tests\make_parity_fixture.py arena/settle_nopath.json parity_fixture_nopath.json }
    Run "Maze 'no path' model parity (JS vs NumPy)" { node tests\parity.test.js arena/settle_nopath.json parity_fixture_nopath.json }
}

Push-Location site
try {
    if ($Preview) { Run "deploy (preview)" { vercel deploy --yes } }
    else { Run "deploy (production)" { vercel deploy --prod --yes } }
} finally { Pop-Location }
Write-Host "Live: https://settle-three-ochre.vercel.app" -ForegroundColor Green
