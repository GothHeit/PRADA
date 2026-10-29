<#
.SYNOPSIS
Let the PC sleep when idle and wake when a Wake-on-LAN magic packet arrives.

.DESCRIPTION
Changes only Windows settings, and prints each one:
- the active power plan sleeps after -SleepMinutes idle on AC power, and never hibernates on
  its own (most network cards can't wake the PC from hibernation);
- each wired network card may wake the PC, but only for a magic packet (not for ordinary
  traffic, which would keep waking it).
It never touches the BIOS/UEFI: see docs/setup.md, "Sleep and Wake-on-LAN", for the settings to
check there. At the end it prints the MAC address the waker needs.

Run it from an elevated PowerShell. -DryRun only shows what it would change.

.PARAMETER SleepMinutes
Idle minutes on AC power before sleep. 0 means never sleep.

.PARAMETER Adapter
Name of the network card to set up (as in Get-NetAdapter). Defaults to every wired one.
#>
param(
    [int]$SleepMinutes = 30,
    [string]$Adapter,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"

$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $DryRun -and -not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this from an elevated PowerShell (Run as administrator)."
}

function Step([string]$what, [scriptblock]$do) {
    if ($DryRun) { Write-Host "would: $what"; return }
    Write-Host $what
    & $do
}

# S0 "Modern Standby" PCs sleep differently and usually only wake for a magic packet if the
# card supports it in that mode; S3 is the classic sleep this was written for.
Write-Host "Sleep states (powercfg /a):"
powercfg /a | Select-Object -First 8 | ForEach-Object { Write-Host "  $_" }

# Current AC timeouts, so the change can be undone (powercfg /change ... <minutes>).
function AcMinutes([string]$setting) {
    $out = powercfg /query SCHEME_CURRENT SUB_SLEEP $setting | Out-String
    if ($out -match 'Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)') {
        return [int]([Convert]::ToInt64($Matches[1], 16) / 60)
    }
    return "?"
}
Write-Host ("Before: sleep after $(AcMinutes STANDBYIDLE) min, hibernate after " +
    "$(AcMinutes HIBERNATEIDLE) min on AC (0 = never).")

Step "Power plan: sleep after $SleepMinutes min idle on AC, no hibernation timer" {
    powercfg /change standby-timeout-ac $SleepMinutes
    powercfg /change hibernate-timeout-ac 0
}

if ($Adapter) {
    $nics = @(Get-NetAdapter -Name $Adapter)
} else {
    $nics = @(Get-NetAdapter -Physical | Where-Object {
        $_.MediaType -eq "802.3" -and $_.InterfaceDescription -notmatch "Wi-?Fi|Wireless"
    })
}
if (-not $nics) {
    Write-Warning "No wired network card found; Wake-on-LAN needs Ethernet. Pass -Adapter."
    exit 1
}

foreach ($nic in $nics) {
    Step "$($nic.Name) ($($nic.InterfaceDescription)): wake on magic packet only" {
        Set-NetAdapterPowerManagement -Name $nic.Name `
            -WakeOnMagicPacket Enabled -WakeOnPattern Disabled -ErrorAction Continue
        # The "Allow this device to wake the computer" box in Device Manager.
        powercfg /deviceenablewake "$($nic.InterfaceDescription)" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning ("Couldn't let '$($nic.InterfaceDescription)' wake the PC; tick " +
                "'Allow this device to wake the computer' in Device Manager.")
        }
    }
    $pm = Get-NetAdapterPowerManagement -Name $nic.Name -ErrorAction SilentlyContinue
    if ($pm) {
        Write-Host "  now: magic packet $($pm.WakeOnMagicPacket), pattern $($pm.WakeOnPattern)"
    }
    $ip = (Get-NetIPAddress -InterfaceIndex $nic.ifIndex -AddressFamily IPv4 `
        -ErrorAction SilentlyContinue | Select-Object -First 1).IPAddress
    Write-Host "  MAC $($nic.MacAddress -replace '-', ':')  IP $ip  status $($nic.Status)"
}

Write-Host ""
Write-Host "Devices allowed to wake the PC (powercfg /devicequery wake_armed):"
powercfg /devicequery wake_armed | ForEach-Object { Write-Host "  $_" }
Write-Host "Put the MAC above in the waker's [wake] mac. Test: sleep the PC, then from another"
Write-Host "machine on the LAN run 'prada wake' (or any Wake-on-LAN tool)."
exit 0
