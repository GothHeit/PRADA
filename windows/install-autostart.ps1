<#
.SYNOPSIS
Start the PRADA home node in WSL when Windows boots, before anyone logs in, and check on it
after every wake from sleep.

.DESCRIPTION
Registers a Task Scheduler task that runs scripts/start-node.sh inside WSL:
- at boot (30 s after Windows starts), with no logon and no autologin;
- 15 s after each wake from sleep or hibernation. If the node is still running this does
  nothing; if it died while the PC slept, this starts it again.
The node (and Ollama, if it isn't already running) keeps running until shutdown; from Windows
it answers on http://127.0.0.1:<Port>/health.

The task runs as you, without admin rights, whether or not you are logged in. For that Windows
stores your account password with the task: the script asks for it once (for a Microsoft
account, that account's password, not the PIN). With -S4U nothing is stored, but WSL may
refuse to start under that kind of logon; try it only if you'd rather not store a password.

Run it from an elevated PowerShell (Windows requires that to register a boot task). Running it
again updates the task, and replaces the older start-at-logon version.

.PARAMETER Distro
WSL distribution that holds the repo. Defaults to the one in the script's \\wsl$ path, or
WSL's default distribution.

.PARAMETER RepoPath
Linux path of the PRADA repo, e.g. /home/kafka/projetos/PRADA. Defaults to the repo this
script is in, when it is run from a \\wsl$ or \\wsl.localhost path.

.PARAMETER S4U
Register the task without storing your password (Windows "S4U" logon).

.EXAMPLE
.\install-autostart.ps1 -Distro Ubuntu -RepoPath /home/kafka/projetos/PRADA
#>
param(
    [string]$Distro,
    [string]$RepoPath,
    [int]$Port = 7723,
    [string]$TaskName = "PRADA node",
    [switch]$S4U,
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
$line = ($wslArgs | ForEach-Object { Quote $_ }) -join " "

# A task that runs without a logged-in user never shows a window, so wsl.exe runs directly.
$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\wsl.exe" -Argument $line

$boot = New-ScheduledTaskTrigger -AtStartup
$boot.Delay = "PT30S"
# Wake from sleep: System event 1 from Power-Troubleshooter ("returned from a low power state").
$eventTrigger = Get-CimClass -ClassName MSFT_TaskEventTrigger `
    -Namespace Root/Microsoft/Windows/TaskScheduler
$resume = New-CimInstance -CimClass $eventTrigger -ClientOnly
$resume.Enabled = $true
$resume.Delay = "PT15S"
$resume.Subscription = '<QueryList><Query Id="0" Path="System"><Select Path="System">' +
    "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]" +
    '</Select></Query></QueryList>'

# IgnoreNew: a wake while the node is up leaves it alone. No idle conditions, so sleep and
# idle never stop it.
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

$user = "$env:USERDOMAIN\$env:USERNAME"
$task = @{
    TaskName    = $TaskName
    Action      = $action
    Trigger     = @($boot, $resume)
    Settings    = $settings
    Description = "Starts the PRADA home node in WSL at boot and after wake, without logon."
    Force       = $true
}
if ($S4U) {
    $task.Principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
} else {
    $cred = Get-Credential -UserName $user `
        -Message "Your Windows password, so the PRADA node can start before you log in."
    if (-not $cred) { throw "No password given; nothing was changed." }
    $task.User = $cred.UserName
    $task.Password = $cred.GetNetworkCredential().Password
    $task.RunLevel = "Limited"
}
Register-ScheduledTask @task | Out-Null
Write-Host "Registered task '$TaskName' (at boot and after wake, as $user): wsl.exe $line"

if (-not $NoStart) {
    Start-ScheduledTask -TaskName $TaskName
    & "$PSScriptRoot\check-node.ps1" -Port $Port -Wait 60 -TaskName $TaskName
    exit $LASTEXITCODE
}
