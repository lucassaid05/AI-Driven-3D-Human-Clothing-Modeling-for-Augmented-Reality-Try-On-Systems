# Opens inbound TCP for local dev so phones on the same LAN can reach Vite and the Python API.
# Run PowerShell as Administrator, then:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#   .\tools\windows-firewall-dev.ps1

$ErrorActionPreference = 'Stop'
$ports = @(5173, 4173, 8000)
foreach ($port in $ports) {
  $name = "pix2surf try-on dev TCP $port"
  $show = netsh advfirewall firewall show rule name="$name" 2>$null
  if ($LASTEXITCODE -ne 0) {
    netsh advfirewall firewall add rule name="$name" dir=in action=allow protocol=TCP localport=$port | Out-Null
    Write-Host "Added firewall rule: $name"
  } else {
    Write-Host "Rule already exists: $name"
  }
}
Write-Host "Done. Restart npm dev / server-local.py if they were already running."
