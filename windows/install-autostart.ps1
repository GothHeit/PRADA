<#
.SYNOPSIS
Start the PRADA home node in WSL every time you log in to Windows.

.DESCRIPTION
Registers a Task Scheduler task that, at your logon, runs scripts/start-node.sh inside WSL
with no console window. The node (and Ollama, if it isn't already running) keeps running
until you log out; from Windows it answers on http://127.0.0.1:<Port>/health.

Run it from an elevated PowerShell (Windows requires that to register a logon task); the
node itself still runs as you, without admin rights. Running it again updates the task.

.PARAMETER Distro
WSL distribution that holds the repo. Defaults to the one in the script's \\wsl$ path, or
WSL's default distribution.

.PARAMETER RepoPath
Linux path of the PRADA repo, e.g. /home/kafka/projetos/PRADA. Defaults to the repo this
script is in, when it is run from a \\wsl$ or \\wsl.localhost path.

.EXAMPLE
.\install-autostart.ps1 -Distro Ubuntu -RepoPath /home/kafka/projetos/PRADA
#>
param(
    [string]$Distro,
    [string]$RepoPath,
    [int]$Port = 7723,
    [string]$TaskName = "PRADA node",
    [switch]$NoStart
)
$ErrorActionPreference = "Stop"

$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this from an elevated PowerShell (Run as administrator)."
}

# \\wsl$\Ubuntu\home\kafka\projetos\PRADA\windows -> Ubuntu, /home/kafka/projetos/PRADA
if ($PSScriptRoot -match '^\\\\wsl(?:\$|\.localhost)\\([^\\]+)\\(.+)\\windows$') {
    if (-not $Distro) { $Distro = $Matches[1] }
    if (-not $RepoPath) { $RepoPath = "/" + ($Matches[2] -replace '\\', '/') }
}
if (-not $RepoPath) {
    throw "Pass -RepoPath (the repo's Linux path, e.g. /home/you/projetos/PRADA)."
}

function Quote([string]$arg) {
    if ($arg -match '[\s"]') { return '"' + ($arg -replace '"', '\"') + '"' }
    return $arg
}

$wslArgs = @()
if ($Distro) { $wslArgs += "-d", $Distro }
$wslArgs += "--exec", "/bin/sh", "$RepoPath/scripts/start-node.sh", "--port", "$Port"
$line = "--headless wsl.exe " + (($wslArgs | ForEach-Object { Quote $_ }) -join " ")

# conhost --headless gives wsl.exe a console that is never shown, so no window at logon.
$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\conhost.exe" -Argument $line
$user = "$env:USERDOMAIN\$env:USERNAME"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force `
    -Description "Starts the PRADA home node in WSL at logon." | Out-Null
Write-Host "Registered task '$TaskName': conhost.exe $line"

if (-not $NoStart) {
    Start-ScheduledTask -TaskName $TaskName
    & "$PSScriptRoot\check-node.ps1" -Port $Port -Wait 60 -TaskName $TaskName
    exit $LASTEXITCODE
}
