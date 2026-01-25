param(
  [string]$Game = "template",
  [int]$MaxCycles = 5,
  [string]$TrainArgs = ""
)

$env:SDL_VIDEODRIVER = "dummy"
$env:SDL_AUDIODRIVER = "dummy"

New-Item -ItemType Directory -Force videos, models, logs | Out-Null

$cmd = "python train.py --game $Game --max-cycles $MaxCycles $TrainArgs"
Write-Host "Running: $cmd"
Invoke-Expression $cmd
