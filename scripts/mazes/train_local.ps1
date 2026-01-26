param(
  [string]$Game = "template",
  [int]$MaxCycles = 5,
  [string]$TrainArgs = "",
  [string]$MazeId = "",
  [int]$MazeMaxSteps = 3600,
  [int]$MazeCellSize = 8
)

$env:SDL_VIDEODRIVER = "dummy"
$env:SDL_AUDIODRIVER = "dummy"
if ($MazeId) {
  $env:MAZE_ID = $MazeId
}
if ($MazeMaxSteps -gt 0) {
  $env:MAZE_MAX_STEPS = $MazeMaxSteps
}
if ($MazeCellSize -gt 0) {
  $env:MAZE_CELL_SIZE = $MazeCellSize
}

New-Item -ItemType Directory -Force videos, models, logs | Out-Null

$cmd = "python train.py --game $Game --max-cycles $MaxCycles $TrainArgs"
if (Test-Path "logs\\metrics.csv") {
  $best = python scripts/mazes/find_best_checkpoint.py 2>$null
  if ($LASTEXITCODE -eq 0 -and $best) {
    $cmd = "python train.py --game $Game --max-cycles $MaxCycles --resume-from $best $TrainArgs"
  }
}
Write-Host "Running: $cmd"
Invoke-Expression $cmd
