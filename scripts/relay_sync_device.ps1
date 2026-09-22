# relay_sync_device.ps1 — 把 relay 相关文件同步到设备，并**逐文件 sha256 自检**
#
# 为什么需要它（都实际犯过）：
#   ① 漏传 keep-head 绑定 ⇒ 设备端 `AttributeError: 'KeepHeadUpstream' object has no
#      attribute 'forward_hidden_to_token'`（tail 服务每次连接都失败）；
#   ② 漏传服务脚本 ⇒ 设备端 `unrecognized arguments: --heartbeat-interval`；
#     更隐蔽的是"隧道端口在监听、远端却是旧服务/无服务"，现象与"模型算错"难以区分。
# 因此本脚本在同步后**回读校验**（sha256 逐文件比对 + 关键符号抽查），而不是"传完就算完"。
#
# 用法：
#   pwsh -File scripts/relay_sync_device.ps1 -Device y700
#   pwsh -File scripts/relay_sync_device.ps1 -Device surface
#   pwsh -File scripts/relay_sync_device.ps1 -Device y700 -SkipShim      # 只同步脚本
#
# 退出码：0 = 全部一致；1 = 有文件不一致或缺符号（会打印差异清单）
param(
    [Parameter(Mandatory = $true)][ValidateSet('y700', 'surface')][string]$Device,
    [switch]$SkipShim,
    [string[]]$ExtraFiles = @()
)

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

if ($Device -eq 'y700') {
    $ssh = 'y700'
    $remoteRoot = '/data/data/com.termux/files/home/qlh-keephead'
    $shimLocal = 'build/termux-arm64/bin/libqlh_keep_head.so'
    $shimRemote = "$remoteRoot/bin/libqlh_keep_head.so"
    $sep = '/'
} else {
    $ssh = 'surface@100.100.52.106'
    $remoteRoot = 'C:/Users/surface/qlh-keephead'
    $shimLocal = 'build/keephead/build-cpu/bin/qlh_keep_head.dll'
    $shimRemote = "$remoteRoot/bin/qlh_keep_head.dll"
    $sep = '/'
}

# (本地路径, 远端相对路径) —— 布局与两端现有目录一致
$files = @(
    @('src/llama_keep_head.py', 'llama_keep_head.py'),
    @('src/llama_keep_head_worker.py', 'llama_keep_head_worker.py'),
    @('src/relay_transport.py', 'src/relay_transport.py'),
    @('scripts/relay_mid_service.py', 'relay_mid_service.py')
)
$shimName = Split-Path -Leaf $shimLocal
if (-not $SkipShim) { $files += , @($shimLocal, "bin/$shimName") }
foreach ($extra in $ExtraFiles) { $files += , @($extra, 'models/' + (Split-Path -Leaf $extra)) }

function Get-LocalHash([string]$path) {
    return (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLower()
}

function Get-RemoteHash([string]$remotePath) {
    # 设备侧探测走**工具脚本**（见 `Initialize-RemoteProbe`），避免多层引号转义地狱：
    # 早先试过 `sha256sum` / `certutil` / `python -c "..."` 三种内联写法，都在 Windows 目标上
    # 因引号被 PowerShell→ssh→cmd 逐层吃掉而失败（表现为"所有文件都缺"，把排查方向带偏）。
    $out = & ssh -o BatchMode=yes $ssh "python $probeTool hash $remotePath" 2>$null
    $hash = ($out | Select-Object -First 1)
    if ($null -eq $hash) { return '' }
    return ($hash -replace '\s', '').ToLower()
}

function Initialize-RemoteProbe {
    $toolLocal = 'build/_relay_remote_probe.py'
    if (-not (Test-Path $toolLocal)) {
        New-Item -ItemType Directory -Force -Path 'build' | Out-Null
        $body = @'
import hashlib
import sys

mode, path = sys.argv[1], sys.argv[2]
if mode == "hash":
    with open(path, "rb") as handle:
        print(hashlib.sha256(handle.read()).hexdigest())
elif mode == "count":
    with open(path, encoding="utf-8", errors="replace") as handle:
        print(handle.read().count(sys.argv[3]))
'@
        Set-Content -Path $toolLocal -Value $body -Encoding ASCII
    }
    & scp -q $toolLocal "${ssh}:$remoteRoot/_relay_remote_probe.py" 2>$null
    return "$remoteRoot/_relay_remote_probe.py"
}

$probeTool = Initialize-RemoteProbe

Write-Host "=== 同步到 $Device（$remoteRoot）==="
foreach ($item in $files) {
    $local, $rel = $item[0], $item[1]
    if (-not (Test-Path $local)) { Write-Host "  [skip] 本地缺 $local"; continue }
    & scp -q $local "${ssh}:$remoteRoot/$rel" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "  [FAIL] 传输失败：$rel" }
}

Write-Host "`n=== 自检（sha256 逐文件比对）==="
$mismatch = @()
foreach ($item in $files) {
    $local, $rel = $item[0], $item[1]
    if (-not (Test-Path $local)) { continue }
    $lh = Get-LocalHash $local
    $rh = Get-RemoteHash "$remoteRoot/$rel"
    if ($lh -eq $rh) {
        Write-Host ("  [ok]   {0,-34} {1}" -f $rel, $lh.Substring(0, 12))
    } else {
        Write-Host ("  [DIFF] {0,-34} local={1} remote={2}" -f $rel, $lh.Substring(0, 12),
                    ($(if ($rh) { $rh.Substring(0, [Math]::Min(12, $rh.Length)) } else { '(缺)' })))
        $mismatch += $rel
    }
}

# 关键符号/参数抽查（sha256 一致时通常已够，但符号表能在"文件同名不同内容"时给出更直接的提示）
Write-Host "`n=== 关键能力抽查 ==="
$checks = @(
    @('llama_keep_head.py', 'forward_hidden_to_token', '末段入口（tail 用）'),
    @('relay_mid_service.py', 'heartbeat-interval', '健康检查参数'),
    @('relay_mid_service.py', 'role head', '上游段角色'),
    @('src/relay_transport.py', 'TOKENS', '上游段帧类型')
)
foreach ($check in $checks) {
    $rel, $needle, $why = $check[0], $check[1], $check[2]
    # 同样走探测工具（Windows 目标没有 grep，内联引号也不可靠）
    $count = & ssh -o BatchMode=yes $ssh "python $probeTool count $remoteRoot/$rel $needle" 2>$null
    $value = ($count | Select-Object -First 1)
    if ($value -and [int]$value -gt 0) { Write-Host "  [ok]   $why" }
    else { Write-Host "  [FAIL] $why（$rel 缺 '$needle'）"; $mismatch += "${rel}:${needle}" }
}

if ($mismatch.Count -gt 0) {
    Write-Host "`n[verdict] 同步**不完整**：$($mismatch -join ', ')"
    exit 1
}
Write-Host "`n[verdict] 同步一致（文件与关键能力均已核对）"
exit 0
