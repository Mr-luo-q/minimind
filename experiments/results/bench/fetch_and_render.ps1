<#
.SYNOPSIS
  从 AutoDL 实例拉取评测结果，合并成最终结果集，并在本地渲染表格与分析。

.DESCRIPTION
  最终结果集口径：
    IFEval <- 修正版判分器的重跑（长度 count_units + 结尾容错）
    GSM8K  <- 首次跑批（两处修正都只影响 IFEval，GSM8K 判分逻辑未变）

  产物落在 -OutDir：
    raw/            合并后的数据集（gen_table.py 直接读它）+ manifest.json
    index.html      网页版结果表（含每题明细）
    RESULTS.md      对比表
    ANALYSIS.md     约束类型分析
    table.csv       表格数据
    batches/        各次跑批的原始拉取（保留可追溯性）

.EXAMPLE
  .\fetch_and_render.ps1 -Port 39365
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][int]$Port,
  [string]$HostName = "connect.bjb1.seetacloud.com",
  [string]$User = "root",
  [string]$RemoteBench = "/root/autodl-tmp/qwen_grpo/bench",
  [string]$IfevalBatch = "results_v3",
  [string]$Gsm8kBatch = "results",
  [string]$OutDir = ""
)

# $PSScriptRoot 在"嵌套 powershell -File 调用"下会解析成意外路径，改用脚本自身路径推导
if (-not $OutDir) {
  $OutDir = if ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }
}

$ErrorActionPreference = "Stop"
$Target = "$User@${HostName}"
$Opts = @("-o","BatchMode=yes","-o","StrictHostKeyChecking=no","-o","ConnectTimeout=20")
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Batches = Join-Path $OutDir "batches"
$Raw = Join-Path $OutDir "raw"

function Show-Filtered($lines) {
  $lines | Where-Object { $_ -notmatch "known_hosts|CategoryInfo|FullyQualifiedErrorId|At line:|^\s*\+|RemoteException" } | Write-Host
}
# 可选文件缺失不应打死脚本（顶部是 ErrorActionPreference=Stop）
function Copy-Optional($remote, $local) {
  $ErrorActionPreference = "Continue"
  Show-Filtered (& scp @Opts -P $Port $remote $local 2>&1)
  $ErrorActionPreference = "Stop"
}

Write-Host "== 1/5 连通性 ==" -ForegroundColor Cyan
Show-Filtered (& ssh @Opts -p $Port $Target "echo OK; ls -d $RemoteBench/$IfevalBatch/*/ 2>/dev/null | wc -l; ls $RemoteBench/$Gsm8kBatch/gsm8k_results.csv 2>/dev/null" 2>&1)
if ($LASTEXITCODE -ne 0) { throw "SSH 连接失败（端口 $Port 是否正确？实例是否开机？）" }

$IfDir = Join-Path $Batches "$IfevalBatch-ifeval"
$GsDir = Join-Path $Batches "$Gsm8kBatch-gsm8k"
New-Item -ItemType Directory -Force -Path $IfDir, $GsDir, $Raw | Out-Null

Write-Host "== 2/5 拉取 IFEval 重跑结果（$IfevalBatch，每模型一个子目录）==" -ForegroundColor Cyan
foreach ($tag in @("qwen3-4b_base", "qwen3-4b_sft", "qwen3-4b_grpo")) {
  $dst = Join-Path $IfDir $tag
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  Copy-Optional "${Target}:$RemoteBench/$IfevalBatch/$tag/ifeval_results.csv" (Join-Path $dst "ifeval_results.csv")
  Copy-Optional "${Target}:$RemoteBench/$IfevalBatch/$tag/ifeval_results_$tag.jsonl" (Join-Path $dst "ifeval_results_$tag.jsonl")
}
Copy-Optional "${Target}:$RemoteBench/$IfevalBatch.log" (Join-Path $IfDir "rerun.log")
# 带批次名的日志副本，避免以后误以为不同版本的日志是同一个文件
Copy-Optional "${Target}:$RemoteBench/$IfevalBatch.log" (Join-Path $Raw "rerun_$IfevalBatch.log")

Write-Host "== 3/5 拉取 GSM8K 结果（$Gsm8kBatch）==" -ForegroundColor Cyan
Copy-Optional "${Target}:$RemoteBench/$Gsm8kBatch/gsm8k_results.csv" $GsDir
Copy-Optional "${Target}:$RemoteBench/$Gsm8kBatch/gsm8k_results_*.jsonl" $GsDir
Copy-Optional "${Target}:$RemoteBench/$Gsm8kBatch/gsm8k_*.log" $GsDir

Write-Host "== 4/5 合并成最终结果集 ==" -ForegroundColor Cyan
& $py (Join-Path $OutDir "merge_results.py") --ifeval-v2 $IfDir --gsm8k-v1 $GsDir --out $Raw
if ($LASTEXITCODE -ne 0) { throw "merge_results.py 失败（可能缺某个模型的 IFEval 结果）" }

Write-Host "== 5/5 渲染结果表 + 约束分析 ==" -ForegroundColor Cyan
& $py (Join-Path $OutDir "gen_table.py") --dir $Raw
& $py (Join-Path $OutDir "analyze_constraints.py") --dir $Raw

foreach ($f in @("RESULTS.md", "index.html", "table.csv", "ANALYSIS.md")) {
  $src = Join-Path $Raw $f
  if (Test-Path $src) { Copy-Item $src (Join-Path $OutDir $f) -Force }
}
Write-Host "`n完成 -> $(Join-Path $OutDir 'index.html')" -ForegroundColor Green
