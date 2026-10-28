<#
.SYNOPSIS
Check that the PRADA home node answers on 127.0.0.1 (exit 0 if it does, 1 if not).

.PARAMETER Wait
Keep trying for this many seconds (the node may still be starting).
#>
param(
    [int]$Port = 7723,
    [int]$Wait = 0,
    [string]$TaskName = "PRADA node"
)

$url = "http://127.0.0.1:$Port/health"
$deadline = (Get-Date).AddSeconds($Wait)
while ($true) {
    try {
        $health = Invoke-RestMethod -Uri $url -TimeoutSec 2
        Write-Host "PRADA node up at $url (modules: $($health.modules -join ', '))"
        exit 0
    } catch {
        if ((Get-Date) -ge $deadline) { break }
        Start-Sleep -Seconds 1
    }
}

Write-Host "No PRADA node answering at $url."
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    $info = $task | Get-ScheduledTaskInfo
    Write-Host "Task '$TaskName' is $($task.State) (last result $($info.LastTaskResult))."
    Write-Host "Node log in WSL: ~/.local/share/prada/node.log"
    # 2147943726 = 0x8007052E, logon failure.
    if ($info.LastTaskResult -eq 2147943726) {
        Write-Host "Windows rejected the stored password: run install-autostart.ps1 again."
    } elseif ($task.State -eq "Ready") {
        Write-Host "Start it with: Start-ScheduledTask -TaskName '$TaskName'"
    }
} else {
    Write-Host "Task '$TaskName' is not installed (see windows\install-autostart.ps1)."
}
exit 1
