# Helper for "Start HobbyLM.bat": is a HobbyLM server from THIS folder already running?
# Identity check: a server listening on 127.0.0.1, ports 8080-8099, whose /props reports exactly this folder's model
# file path. Anything else on those ports (other programs, other llama.cpp servers, other HobbyLM copies) is ignored.
# Read-only: only HTTP GET requests to ports that are already listening; nothing is stopped or changed.
# No state file is used, so a shutdown or crash cannot leave stale state behind.
# If a llama.cpp server on these ports is still loading its model (HTTP 503 "Loading model"), wait up to
# WaitLoadingSec for it to finish, so a quick second double-click does not start a second copy.
# Exit code: the port number if this folder's server was found, otherwise 0.
param([Parameter(Mandatory = $true)][string]$Model, [int]$First = 8080, [int]$Last = 8099, [int]$WaitLoadingSec = 20)
$want = [System.IO.Path]::GetFullPath($Model)

function Get-Props([int]$port) {
    $req = [System.Net.HttpWebRequest]::Create("http://127.0.0.1:$port/props")
    $req.Proxy = $null; $req.Timeout = 1500; $req.ReadWriteTimeout = 1500
    try { $resp = $req.GetResponse() }
    catch [System.Net.WebException] { $resp = $_.Exception.Response; if (-not $resp) { return $null } }
    catch { return $null }
    try {
        $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
        $body = $reader.ReadToEnd(); $code = [int]$resp.StatusCode; $resp.Close()
        return @{ code = $code; json = ($body | ConvertFrom-Json) }
    } catch { return $null }
}

$deadline = (Get-Date).AddSeconds($WaitLoadingSec)
do {
    $loading = $false
    $ports = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -ge $First -and $_.LocalPort -le $Last -and $_.LocalAddress -in @('127.0.0.1', '0.0.0.0', '::', '::1') } |
        Select-Object -ExpandProperty LocalPort -Unique | Sort-Object)
    foreach ($port in $ports) {
        $r = Get-Props $port
        if ($null -eq $r) { continue }
        if ($r.code -eq 200 -and $r.json.model_path) {
            try { $got = [System.IO.Path]::GetFullPath([string]$r.json.model_path) } catch { continue }
            if ([string]::Equals($got, $want, [System.StringComparison]::OrdinalIgnoreCase)) { exit $port }
        } elseif ($r.code -eq 503 -and $r.json.error.message -eq 'Loading model') {
            $loading = $true
        }
    }
    if ($loading) { Start-Sleep -Milliseconds 500 }
} while ($loading -and (Get-Date) -lt $deadline)
exit 0
