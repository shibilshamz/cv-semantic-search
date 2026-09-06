# Search the CV index from Windows, in one command.
#
#   .\cvsearch.ps1 "backend engineer comfortable with payments APIs, in the Gulf"
#   .\cvsearch.ps1 -Brief "ICU nurse, DHA licensed" -K 3 -NoExplain
#   .\cvsearch.ps1 -Ui          # opens the interactive docs page in a browser
#
# The API listens on the VPS's docker bridge and is not reachable from the
# internet -- by design. This opens a temporary SSH tunnel, runs the query
# through it, and closes it again.
#
# Auth: SSH keys if you have them, otherwise you are prompted for the password.
# Set $env:VPS_PASSWORD to avoid the prompt, but prefer keys.

param(
  [Parameter(Position = 0)][string]$Brief,
  [int]$K = 5,
  [switch]$NoExplain,     # skip the Claude ranking call: instant, free, no prose
  [switch]$Ui,            # open the interactive API docs instead of querying
  [string]$VpsHost = "72.61.233.142",
  [int]$Port = 5680
)

$ErrorActionPreference = "Stop"
$plink = "C:\Program Files\PuTTY\plink.exe"
$hostkey = "SHA256:R4mbXepGSsAARCsvcaIpSosiVCLeyLReh+3dFewnQek"

if (-not $Ui -and [string]::IsNullOrWhiteSpace($Brief)) {
  Write-Host 'Usage: .\cvsearch.ps1 "describe the role in plain English"' -ForegroundColor Yellow
  Write-Host '       .\cvsearch.ps1 -Ui       # browser interface'
  exit 1
}

$args = @("-batch", "-N", "-hostkey", $hostkey, "-L", "${Port}:172.17.0.1:${Port}")
if ($env:VPS_PASSWORD) { $args += @("-pw", $env:VPS_PASSWORD) }
$args += "root@$VpsHost"

Write-Host "opening tunnel to $VpsHost..." -ForegroundColor DarkGray
$tunnel = Start-Process -FilePath $plink -ArgumentList $args -PassThru -WindowStyle Hidden

try {
  # Wait for the tunnel rather than guessing at a sleep duration.
  $ready = $false
  foreach ($i in 1..20) {
    Start-Sleep -Milliseconds 700
    try {
      $h = Invoke-RestMethod "http://localhost:$Port/health" -TimeoutSec 3
      $ready = $true
      break
    } catch { }
  }
  if (-not $ready) { throw "tunnel did not come up; is the VPS reachable and cv-search running?" }

  Write-Host "index: $($h.chunks) chunks" -ForegroundColor DarkGray

  if ($Ui) {
    Write-Host "opening http://localhost:$Port/docs -- close this window to end the tunnel" -ForegroundColor Cyan
    Start-Process "http://localhost:$Port/docs"
    Write-Host "Press Enter to close the tunnel..." -ForegroundColor DarkGray
    [void][System.Console]::ReadLine()
    return
  }

  $body = @{ text = $Brief; k = $K; explain = (-not $NoExplain) } | ConvertTo-Json
  $r = Invoke-RestMethod "http://localhost:$Port/search" -Method Post `
        -ContentType "application/json" -Body $body -TimeoutSec 180

  Write-Host ""
  Write-Host "BRIEF: $Brief" -ForegroundColor White
  if ($r.weak) {
    Write-Host "No strong match in the index (top score $($r.top_score))." -ForegroundColor Yellow
  }
  Write-Host ""
  $rank = 1
  foreach ($m in $r.matches) {
    "{0}. {1,-28} {2,7:N3}  [{3}]" -f $rank, $m.name, $m.score, $m.section | Write-Host
    "   {0}" -f $m.candidate_id | Write-Host -ForegroundColor DarkGray
    $rank++
  }
  if ($r.answer) {
    Write-Host ""
    Write-Host ("-" * 70) -ForegroundColor DarkGray
    Write-Host $r.answer
  }
}
finally {
  if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force }
  Write-Host ""
  Write-Host "tunnel closed" -ForegroundColor DarkGray
}
