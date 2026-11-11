# PRADA's bridge to Windows' media controls (SMTC), started by prada.music.bridge.
#
# One request per line on stdin, one JSON reply per line on stdout:
#   {"id": 1, "cmd": "status"}                  -> {"id": 1, "ok": true, "current": APP, "players": [...]}
#   {"id": 2, "cmd": "control", "app": APP, "verb": "pause|play|next|previous"}
#                                               -> {"id": 2, "ok": true|false}
# A failed request answers {"id": N, "ok": false, "error": "..."}. The first line written is
# {"ready": true} (or {"fatal": "..."} before exiting). Closing stdin ends the script.
# Read-only except for the four player commands; it never touches files or the network.

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false

function Say($obj) {
    [Console]::Out.WriteLine(($obj | ConvertTo-Json -Compress -Depth 4))
    [Console]::Out.Flush()
}

try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
    $null = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager, Windows.Media.Control, ContentType = WindowsRuntime]
    $null = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionMediaProperties, Windows.Media.Control, ContentType = WindowsRuntime]

    function Await($op, [Type]$type, [int]$ms = 4000) {
        $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
        if (-not $task.Wait($ms)) { throw 'Windows did not answer in time' }
        $task.Result
    }

    $manager = Await ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]::RequestAsync()) `
        ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager])
} catch {
    Say @{ fatal = "$_" }
    exit 1
}

function Player($s) {
    # A slow player (a browser tab) is skipped rather than holding up the whole answer.
    $props = Await ($s.TryGetMediaPropertiesAsync()) `
        ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionMediaProperties]) 1500
    $state = "$($s.GetPlaybackInfo().PlaybackStatus)".ToLower()
    $t = $s.GetTimelineProperties()
    $position = $t.Position.TotalSeconds
    if ($state -eq 'playing' -and $t.LastUpdatedTime.Year -gt 2000) {
        # Players update the position now and then; add the time since.
        $position += ([DateTimeOffset]::Now - $t.LastUpdatedTime).TotalSeconds
    }
    $duration = ($t.EndTime - $t.StartTime).TotalSeconds
    [ordered]@{
        app      = $s.SourceAppUserModelId
        title    = "$($props.Title)"
        artist   = "$($props.Artist)"
        album    = "$($props.AlbumTitle)"
        state    = $state
        position = [math]::Round([math]::Max(0, $position), 1)
        duration = [math]::Round([math]::Max(0, $duration), 1)
    }
}

function Status {
    $players = New-Object System.Collections.ArrayList
    foreach ($s in $manager.GetSessions()) {
        try { $null = $players.Add((Player $s)) } catch { }
    }
    $current = $manager.GetCurrentSession()
    [ordered]@{
        ok      = $true
        current = if ($current) { $current.SourceAppUserModelId } else { $null }
        players = $players.ToArray()
    }
}

function Control($app, $verb) {
    $session = $null
    foreach ($s in $manager.GetSessions()) {
        if ($s.SourceAppUserModelId -ne $app) { continue }
        # An app can leave a stale session behind; prefer the one that is playing.
        if ($null -eq $session -or "$($s.GetPlaybackInfo().PlaybackStatus)" -eq 'Playing') { $session = $s }
    }
    if ($null -eq $session) { return @{ ok = $false; error = 'that player is not open' } }
    $op = switch ($verb) {
        'pause' { $session.TryPauseAsync() }
        'play' { $session.TryPlayAsync() }
        'next' { $session.TrySkipNextAsync() }
        'previous' { $session.TrySkipPreviousAsync() }
        default { $null }
    }
    if ($null -eq $op) { return @{ ok = $false; error = "unknown command $verb" } }
    @{ ok = [bool](Await $op ([bool])) }
}

Say @{ ready = $true }
$stdin = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), (New-Object System.Text.UTF8Encoding $false))
while ($null -ne ($line = $stdin.ReadLine())) {
    if (-not $line.Trim()) { continue }
    $id = $null
    try {
        $req = $line | ConvertFrom-Json
        $id = $req.id
        $reply = switch ($req.cmd) {
            'status' { Status }
            'control' { Control $req.app $req.verb }
            default { @{ ok = $false; error = "unknown request $($req.cmd)" } }
        }
    } catch {
        $reply = @{ ok = $false; error = "$_" }
    }
    $reply['id'] = $id
    Say $reply
}
