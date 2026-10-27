<#
.SYNOPSIS
Stop starting the PRADA home node at logon, and stop the running node.

.PARAMETER Distro
WSL distribution the node runs in. Defaults to the one in the script's \\wsl$ path, or
WSL's default distribution.

.PARAMETER KeepRunning
Remove the task but leave a running node alone.
#>
param(
    [string]$Distro,
    [string]$TaskName = "PRADA node",
    [switch]$KeepRunning
)
$ErrorActionPreference = "Stop"

$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this from an elevated PowerShell (Run as administrator)."
}

if (-not $Distro -and $PSScriptRoot -match '^\\\\wsl(?:\$|\.localhost)\\([^\\]+)\\') {
    $Distro = $Matches[1]
}

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    if (-not $KeepRunning) { Stop-ScheduledTask -TaskName $TaskName }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed task '$TaskName'."
} else {
    Write-Host "Task '$TaskName' is not installed."
}

if (-not $KeepRunning) {
    # Stopping the task ends wsl.exe, which may leave the node running inside WSL.
    $wslArgs = @()
    if ($Distro) { $wslArgs += "-d", $Distro }
    & wsl.exe @wslArgs --exec pkill -f -- "-m prada node"
    if ($LASTEXITCODE -eq 0) { Write-Host "Stopped the PRADA node." }
}
exit 0
