# Helper for "Start HobbyLM.bat": waits until the local HobbyLM server answers /health, then opens the chat page in
# the default browser. Talks to 127.0.0.1 only (no proxy). Changes nothing on the system. Gives up quietly after the
# timeout (the launcher window still shows the address to open by hand).
param([Parameter(Mandatory = $true)][int]$Port, [int]$TimeoutSec = 180)
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    try {
        $req = [System.Net.HttpWebRequest]::Create("http://127.0.0.1:$Port/health")
        $req.Proxy = $null
        $req.Timeout = 2000
        $resp = $req.GetResponse()
        $code = [int]$resp.StatusCode
        $resp.Close()
        if ($code -eq 200) {
            Start-Process "http://127.0.0.1:$Port/"
            exit 0
        }
    } catch {
        # server still loading (503) or not up yet
    }
    Start-Sleep -Milliseconds 500
}
exit 1
