param(
    [string]$IsaacSimRoot,
    [switch]$Headless,
    [switch]$ExecuteActionRequest,
    [switch]$ExecuteNonOraclePlan,
    [ValidateSet("minimal", "benchmark")]
    [string]$SceneProfile = "minimal"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$sceneFilename = if ($SceneProfile -eq "benchmark") {
    "open_container_benchmark.usda"
} else {
    "open_container_minimal.usda"
}
$scenePath = Join-Path $projectRoot "assets\scenes\$sceneFilename"

$candidateRoots = @(
    $IsaacSimRoot,
    $env:ISAAC_SIM_ROOT,
    "C:\isaacsim",
    "C:\IsaacSim",
    "D:\isaacsim",
    "D:\IsaacSim",
    "C:\Users\wunee\AppData\Local\ov\pkg\isaac_sim-*"
) | Where-Object { $_ }

$launcher = $null
$pythonLauncher = $null
foreach ($root in $candidateRoots) {
    $resolvedRoots = Resolve-Path -Path $root -ErrorAction SilentlyContinue
    foreach ($resolvedRoot in $resolvedRoots) {
        foreach ($name in @("isaac-sim.bat", "isaac-sim.selector.bat")) {
            $candidate = Join-Path $resolvedRoot.Path $name
            if (Test-Path -LiteralPath $candidate) {
                $launcher = $candidate
                break
            }
        }
        $pythonCandidate = Join-Path $resolvedRoot.Path "python.bat"
        if (Test-Path -LiteralPath $pythonCandidate) {
            $pythonLauncher = $pythonCandidate
        }
        if ($launcher) { break }
    }
    if ($launcher) { break }
}

if (-not $launcher -or -not $pythonLauncher) {
    throw "Isaac Sim launcher not found. Install Isaac Sim or pass -IsaacSimRoot <installation-directory>."
}

if ($Headless) {
    Start-Process -FilePath $launcher -ArgumentList @($scenePath, "--no-window") -WindowStyle Hidden
    Write-Output "Started Isaac Sim headless with scene: $scenePath"
    exit 0
}

$openScript = Join-Path $projectRoot "scripts\open_minimal_scene.py"
$scriptArguments = @($openScript, "--scene-profile", $SceneProfile)
if ($ExecuteActionRequest) {
    $scriptArguments += "--execute-action-request"
}
if ($ExecuteNonOraclePlan) {
    $scriptArguments += "--execute-non-oracle-plan"
}
Start-Process -FilePath $pythonLauncher -ArgumentList $scriptArguments -WindowStyle Normal
Write-Output "Started Isaac Sim Python runtime with scene loader: $openScript"
