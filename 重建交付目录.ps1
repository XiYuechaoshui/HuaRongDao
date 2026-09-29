# 华容道 交付目录构建脚本 (把 dist\华容道 精简成可直接分发的 安装\)
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File .\重建交付目录.ps1
#
# 为什么要有这个脚本:
#   打包脚本第 5 步会把整个 训练数据\ 复制到 dist\华容道\, 而主程序跑一次分析就会往
#   里面写布局级存档 (graph_<hash>.pkl, 单个可达 100 MB)。所以 dist\华容道 会自己"长胖",
#   而交付包只需要三件全局模型 (gnn_global.pt / tda_ae_global.pt / ml_global.pkl)。
#   以前这一步是手工做的, 每次重新打包都要重来一遍, 容易漏也容易把 200 MB 存档发出去。
#
# 本脚本做四件事:
#   1. 用 dist\华容道 的 exe 与 _internal 覆盖交付目录 (即"重新发布")
#   2. 训练数据\ 只保留全局模型, 布局级存档一律不带; 布局缓存\ 建为空目录
#   3. 从 交付模板\ 取 启动.bat 与 使用说明.txt (它们只在这边维护, 见模板目录说明)
#   4. 校验: 三个全局模型齐全、不含布局存档、启动脚本仍是纯 ASCII
#
# 注意: 交付目录里**不要**放 自检报告.json —— 它是排查用的中间产物, 不是交付物。

param(
    [string]$交付目录 = '',      # 默认: 本工程上级目录的 安装\
    [switch]$保留旧目录          # 出问题时便于对比, 默认直接覆盖
)

$ErrorActionPreference = 'Stop'
$projDir = $PSScriptRoot
$distDir = Join-Path $projDir 'dist\华容道'
$模板Dir = Join-Path $projDir '交付模板'

if (-not $交付目录) {
    $交付目录 = Join-Path (Split-Path $projDir -Parent) '安装'
}

Write-Output "=== 交付目录构建 ==="
Write-Output "  源: $distDir"
Write-Output "  目标: $交付目录"

# ---------- 0. 前置校验 ----------
if (-not (Test-Path (Join-Path $distDir '华容道.exe'))) {
    Write-Error "dist\华容道\华容道.exe 不存在 —— 请先跑 打包exe.ps1"
    exit 1
}
if (-not (Test-Path (Join-Path $distDir '_internal'))) {
    Write-Error "dist\华容道\_internal 不存在 —— 这份 dist 不完整, 请重新打包"
    exit 1
}
if (-not (Test-Path $模板Dir)) {
    Write-Error "缺少 交付模板\ 目录 (启动.bat / 使用说明.txt 应当从那里取)"
    exit 1
}
foreach ($t in @('启动.bat', '使用说明.txt')) {
    if (-not (Test-Path (Join-Path $模板Dir $t))) {
        Write-Error "交付模板\$t 不存在"
        exit 1
    }
}
if (@(Get-Process 华容道 -ErrorAction SilentlyContinue).Count -gt 0) {
    Write-Error "有 华容道.exe 正在运行, 会锁住 _internal —— 请先关掉它"
    exit 1
}

# ---------- 1. 覆盖 exe 与 _internal ----------
Write-Output ""
Write-Output "=== 1. 更新程序本体 ==="
# exe 必须整体替换 (运行中的文件无法覆盖, 上面已检查)
Copy-Item (Join-Path $distDir '华容道.exe') (Join-Path $交付目录 '华容道.exe') -Force
Write-Output ("  华容道.exe  {0:N2} MB" -f ((Get-Item (Join-Path $交付目录 '华容道.exe')).Length / 1MB))

$旧internal = Join-Path $交付目录 '_internal'
if (Test-Path $旧internal) {
    Remove-Item $旧internal -Recurse -Force
    Write-Output "  已移除旧的 _internal"
}
Copy-Item (Join-Path $distDir '_internal') $旧internal -Recurse -Force
$内 = Get-ChildItem $旧internal -Recurse -File
Write-Output ("  _internal   {0:N0} MB / {1} 文件" -f (($内 | Measure-Object Length -Sum).Sum / 1MB), $内.Count)

Copy-Item (Join-Path $distDir 'app.ico') (Join-Path $交付目录 'app.ico') -Force
Write-Output "  app.ico"

# ---------- 2. 训练数据: 只留全局模型 ----------
Write-Output ""
Write-Output "=== 2. 训练数据 (只保留全局模型) ==="
$模型源 = Join-Path $distDir '训练数据\模型'
$模型目标 = Join-Path $交付目录 '训练数据\模型'
$布局缓存 = Join-Path $交付目录 '训练数据\布局缓存'

if (-not (Test-Path $模型源)) {
    Write-Error "源里没有 训练数据\模型 —— dist 不完整"
    exit 1
}

$全局模型 = @('gnn_global.pt', 'tda_ae_global.pt', 'ml_global.pkl')
foreach ($d in @($模型目标, $布局缓存)) {
    if (Test-Path $d) { Remove-Item $d -Recurse -Force }
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}
$缺失 = @()
foreach ($m in $全局模型) {
    $src = Join-Path $模型源 $m
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $模型目标 $m) -Force
        Write-Output ("  + {0,-20} {1,9:N0} 字节" -f $m, (Get-Item $src).Length)
    } else {
        $缺失 += $m
    }
}
if ($缺失.Count -gt 0) {
    Write-Error ("全局模型缺失: {0} —— 缺了 AI 分析会退化成每次从零训练" -f ($缺失 -join ', '))
    exit 1
}
# 布局缓存保持空目录 (程序运行后自动填)
$sum = Join-Path $distDir '训练数据\summary.json'
if (Test-Path $sum) {
    Copy-Item $sum (Join-Path $交付目录 '训练数据\summary.json') -Force
    Write-Output "  + summary.json (训练汇总, 供「模型」页显示)"
}
# 明确报告剔除了多少布局级存档
$存档 = @(Get-ChildItem $模型源 -File | Where-Object { $_.Name -match '^(graph_|gnn_|tda_ae_|ml_|embeddings_)' -and $_.Name -notin $全局模型 })
if ($存档.Count -gt 0) {
    $省 = ($存档 | Measure-Object Length -Sum).Sum
    Write-Output ("  已剔除布局级存档 {0} 个 / {1:N1} MB" -f $存档.Count, ($省 / 1MB))
} else {
    Write-Output "  源里没有布局级存档 (干净)"
}

# ---------- 3. 启动器与说明 ----------
Write-Output ""
Write-Output "=== 3. 启动器与说明 (取自 交付模板\) ==="
foreach ($t in @('启动.bat', '使用说明.txt')) {
    Copy-Item (Join-Path $模板Dir $t) (Join-Path $交付目录 $t) -Force
    Write-Output ("  {0}" -f $t)
}

# ---------- 4. 校验 ----------
Write-Output ""
Write-Output "=== 4. 交付校验 ==="
$通过 = $true

# 4a. 启动脚本必须仍然是 纯 ASCII + 无 BOM (否则 cmd 会读成乱码, 见工程文档)
$b = [System.IO.File]::ReadAllBytes((Join-Path $交付目录 '启动.bat'))
$nonAscii = 0
foreach ($x in $b) { if ($x -gt 0x7F) { $nonAscii++ } }
$hasBom = ($b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF)
if ($nonAscii -gt 0 -or $hasBom) {
    Write-Output ("  ✗ 启动.bat 不再是纯 ASCII 无 BOM (非ASCII {0} 字节, BOM {1})" -f $nonAscii, $hasBom)
    $通过 = $false
} else {
    Write-Output "  ✓ 启动.bat 纯 ASCII 无 BOM"
}

# 4b. 不含布局级存档
$残留 = @(Get-ChildItem $模型目标 -File | Where-Object { $_.Name -match '^(graph_|embeddings_|gnn_[0-9a-f]{8}|tda_ae_[0-9a-f]{8}|ml_[0-9a-f]{8})' })
if ($残留.Count -gt 0) {
    Write-Output ("  ✗ 仍含布局级存档 {0} 个" -f $残留.Count)
    $通过 = $false
} else {
    Write-Output "  ✓ 不含布局级存档"
}

# 4c. 三个全局模型齐全
$有 = @(Get-ChildItem $模型目标 -File | ForEach-Object { $_.Name })
$缺 = @($全局模型 | Where-Object { $_ -notin $有 })
if ($缺.Count -gt 0) {
    Write-Output ("  ✗ 缺全局模型: {0}" -f ($缺 -join ', '))
    $通过 = $false
} else {
    Write-Output "  ✓ 全局模型三件齐全"
}

# 4d. 不含自检报告等排查产物
$不该有 = @(Get-ChildItem $交付目录 -Force | Where-Object {
    $_.Name -match '自检|\.log$|tmp|torchinductor' })
if ($不该有.Count -gt 0) {
    Write-Output ("  ✗ 含排查产物: {0}" -f (($不该有 | ForEach-Object { $_.Name }) -join ', '))
    $通过 = $false
} else {
    Write-Output "  ✓ 无自检报告/日志等排查产物"
}

# 4e. 可搬移: 文本文件里不应有硬编码的本机绝对路径
$硬编码 = @()
Get-ChildItem $交付目录 -Recurse -File -Include '*.txt', '*.md', '*.json' -ErrorAction SilentlyContinue |
    ForEach-Object {
        $t = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
        if ($t -match '[A-Z]:\\XIYue|XIYueの奇妙冒险') { $硬编码 += $_.Name }
    }
if ($硬编码.Count -gt 0) {
    Write-Output ("  ⚠ 以下文件提到本机路径 (确认只是示例文字即可): {0}" -f (($硬编码 | Select-Object -Unique) -join ', '))
} else {
    Write-Output "  ✓ 文本文件无本机硬编码路径"
}

# ---------- 汇总 ----------
Write-Output ""
$exeMB = (Get-Item (Join-Path $交付目录 '华容道.exe')).Length / 1MB
$内MB = (Get-ChildItem (Join-Path $交付目录 '_internal') -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
$数据MB = (Get-ChildItem (Join-Path $交付目录 '训练数据') -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1MB
$总MB = (Get-ChildItem $交付目录 -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
$总数 = @(Get-ChildItem $交付目录 -Recurse -File).Count
Write-Output "=== 交付目录 ==="
Write-Output ("  {0}" -f $交付目录)
Write-Output ("  华容道.exe   {0,8:N2} MB" -f $exeMB)
Write-Output ("  _internal\   {0,8:N0} MB" -f $内MB)
Write-Output ("  训练数据\    {0,8:N2} MB" -f $数据MB)
Write-Output ("  ──────────────────────")
Write-Output ("  合计         {0,8:N1} MB / {1} 文件" -f $总MB, $总数)
Write-Output ""
if ($通过) {
    Write-Output "验证结论: ✓ 交付目录可用 (整目录拷走即可运行)"
    Write-Output "  建议再跑一次包内自检作为最终证据:"
    Write-Output ("    `$env:HRD_SELFTEST='1'; & '{0}'; Get-Content '{1}' -Encoding UTF8" -f `
        (Join-Path $交付目录 '华容道.exe'), (Join-Path $交付目录 '自检报告.json'))
} else {
    Write-Output "验证结论: ✗ 有问题 —— 这份交付不要发出去"
    exit 1
}
