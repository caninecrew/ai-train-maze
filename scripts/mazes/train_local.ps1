param(
  [string]$Game = "template",
  [int]$MaxCycles = 5,
  [string]$TrainArgs = "",
  [string]$MazeId = "",
  [int]$MazeMaxSteps = 3600
)

$env:SDL_VIDEODRIVER = "dummy"
$env:SDL_AUDIODRIVER = "dummy"
if ($MazeId) {
  $env:MAZE_ID = $MazeId
}
if ($MazeMaxSteps -gt 0) {
  $env:MAZE_MAX_STEPS = $MazeMaxSteps
}

New-Item -ItemType Directory -Force videos, models, logs | Out-Null

$cmd = "python train.py --game $Game --max-cycles $MaxCycles $TrainArgs"
Write-Host "Running: $cmd"
Invoke-Expression $cmd
