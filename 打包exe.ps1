# 华容道 exe 打包脚本
# 使用方法: 在 PowerShell 中运行 & d:\华容道\华容道_打包\打包exe.ps1
#
# 设计要点 (2026-09 修订):
#   1. 源码同步是"非破坏性"的: 只复制打包目录**缺失**的文件, 已有文件一律不动。
#      历史问题: 原脚本无条件从 d:\华容道\华容道 覆盖 16 个文件, 而 d:\华容道\华容道 里是旧版本,
#      一跑打包就会把 d:\华容道\华容道_打包 里更新过的源码全部回退 (实测有 7 个文件不同)。
#      确实想用源码目录的版本覆盖时, 加 -PullSource (或 -Force 全量覆盖)。
#   2. 同步白名单按 *.py 全量枚举, 不再手工列举 (原白名单漏了
#      导入器.py / 代码生成器.py / 布局编辑器.py)。
#   3. 训练数据不再由 spec 打进 _internal (那份永不生效, 因为 布局哈希.py 用的是
#      exe 同级目录), 只在本脚本第 5 步复制一份到 exe 同级。
#   4. VC++ 运行时不再删除, 改为在启动测试中校验; 缺了会给出明确提示。
#   5. **验证有两道关口, 任一不过就以 exit 1 结束** (2026-09-17 加):
#      第 6 步 启动测试: 断言主窗口标题 == 华容道状态空间分析器。
#        不能只看"进程还活着" —— PyInstaller 窗口版遇导入期异常会弹模态对话框吊住进程,
#        实测就是这样把一份一打开就崩的包判成了"启动成功"。
#      第 7 步 包内自检: 设 HRD_SELFTEST=1 让 exe 自己跑 自检.py, 验包内 torch / 训练数据 /
#        模型载入 / GUI 真的能用 (冻结后的 import 与 DLL 路径源码模式复现不出来)。
#      详细事故记录见同目录 打包说明.md。
#   6. 别在沙箱/受限令牌里跑第 6/7 步: matplotlib 首跑要写 %TEMP% 字体缓存, 被拒会误报成包缺陷。
#   7. 第 1 步的同步是**单向**的: 打包目录是发布源, 默认只补缺、不覆盖任何已有文件。
#      历史事故: 旧逻辑按时间戳判断, 给 华容道\主窗口.py 打补丁后它时间戳变新, 于是打包目录里
#      AI 时代的 主窗口.py 被那边 RL 时代的旧版覆盖 → exe 启动 ImportError: RLSolver,
#      而且没有 git/备份可恢复。要用源码目录的版本请显式加 -PullSource。

param(
    [switch]$Force,          # 强制用 d:\华容道\华容道 的源码覆盖打包目录 (危险: 那个目录是旧版本!)
    [switch]$PullSource,     # 源文件比打包目录旧时也覆盖 (默认只在打包目录**缺失**该文件时才复制)
    [switch]$SkipStartTest   # 跳过启动测试
)

$ErrorActionPreference = "Stop"
$projDir = "d:\华容道\华容道_打包"
$srcDir = "d:\华容道\华容道"
$venvPython = "d:\华容道\华容道\.venv\Scripts\python.exe"
$specFile = Join-Path $projDir "打包exe.spec"   # 自带 spec, 不再依赖源码目录
$internalPath = Join-Path $projDir "dist\华容道\_internal"

# 这些文件留在源码目录作排障用, 不随工程分发 (注意: 第 1 步是"只补缺",
# 所以源码目录里有、打包目录里没有的文件**每次打包都会被复制回来** ——
# 想让它不在包里出现, 就必须列在这里, 而不是把文件删掉了事)。
$syncExclude = @('诊断torch.py')
Write-Output "=== 1. 同步源码到打包目录 (默认只补缺, 不覆盖) ==="
# 2026-09-17 事故 (必须记住): 旧逻辑按**时间戳**判断新旧, 认为"本地更新就保留"。
#   但给 华容道\主窗口.py 打补丁会让它时间戳变新 —— 于是打包目录里那份 AI 时代的
#   主窗口.py 被 华容道 里那份 RL 时代的覆盖, 重打包后 exe 启动即
#   ImportError: cannot import name 'RLSolver'。时间戳根本不能代表"哪份更完整"。
#   现在改为: 默认**只复制打包目录缺失的文件**, 已有文件一律不动;
#   确实想用 华容道 的版本覆盖, 显式加 -PullSource (或 -Force 全量覆盖)。
if (-not (Test-Path $srcDir)) {
    Write-Output "  源码目录 $srcDir 不存在, 跳过同步"
} else {
    $copied = 0
    $kept = 0
    $diff = @()
    foreach ($f in Get-ChildItem -Path $srcDir -Filter *.py -File) {
        if ($syncExclude -contains $f.Name) { continue }
        # 跳过历史一次性调试脚本 (_diag*.py / _fix*.py / _test*.py 等),
        # 它们对打包无用, 复制过来只会污染工程目录
        if ($f.Name.StartsWith('_')) { continue }
        $dst = Join-Path $projDir $f.Name
        if (Test-Path $dst) {
            $kept++
            if ((Get-FileHash $dst -Algorithm MD5).Hash -ne (Get-FileHash $f.FullName -Algorithm MD5).Hash) {
                $diff += $f.Name
            }
            if ($Force -or $PullSource) {
                Copy-Item -Path $f.FullName -Destination $dst -Force
                Write-Output ("  已按参数覆盖: {0}" -f $f.Name)
            } else {
                continue
            }
        } else {
            Copy-Item -Path $f.FullName -Destination $dst -Force
            $copied++
            Write-Output ("  补入打包目录缺失文件: {0}" -f $f.Name)
        }
    }
    Write-Output ("  同步完成: 仅补缺 {0} 个, 保留打包目录现有 {1} 个 (默认不回退)" -f $copied, $kept)
    if ($diff.Count -gt 0) {
        Write-Output ("  ⚠ 两侧内容不同的文件 {0} 个: {1}" -f $diff.Count, ($diff -join ', '))
        Write-Output "    打包目录是发布源, 已按原样保留; 要用源码目录的版本请加 -PullSource"
    }
}
# 入口脚本按打包目录自身为准, 单独确认存在
foreach ($need in @('主程序.py', '主窗口.py', 'AI模型.py')) {
    if (-not (Test-Path (Join-Path $projDir $need))) {
        Write-Error "打包目录缺少必需文件: $need"
        exit 1
    }
}

Write-Output ""
Write-Output "=== 2. 清理旧打包产物 ==="
$distPath = Join-Path $projDir "dist"
$buildPath = Join-Path $projDir "build"
$pycachePath = Join-Path $projDir "__pycache__"
if (Test-Path $distPath) { Remove-Item -Path $distPath -Recurse -Force }
if (Test-Path $buildPath) { Remove-Item -Path $buildPath -Recurse -Force }
if (Test-Path $pycachePath) { Remove-Item -Path $pycachePath -Recurse -Force }
# 自检.py 会在工程目录建 _自检临时_* 做存盘往返; 受限环境下它自己删不掉, 这里兜底清掉
$自检残留 = Get-ChildItem -Path $projDir -Directory -Filter "_自检临时_*" -ErrorAction SilentlyContinue
if ($自检残留) {
    foreach ($d in $自检残留) { Remove-Item -Path $d.FullName -Recurse -Force -ErrorAction SilentlyContinue }
    Write-Output ("  清理自检临时目录 {0} 个" -f @($自检残留).Count)
}
Start-Sleep -Seconds 1
Write-Output "  清理完成"

Write-Output ""
Write-Output "=== 3. 运行 PyInstaller 打包 ==="
Set-Location $projDir
$ErrorActionPreference = "Continue"
# 先收集完整输出再决定打印多少: 原来只 Select-Object -Last 3,
# 会把 PyInstaller 的警告(例如"排除的模块不存在"、缺 hiddenimport)连同关键报错一起吞掉,
# 事后只能靠翻 build\warn-*.txt 找线索。
$pyiOut = & $venvPython -m PyInstaller $specFile --distpath $distPath --workpath $buildPath 2>&1
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) {
    $pyiOut | Select-Object -Last 25
    Write-Error "打包失败: PyInstaller 退出码 $LASTEXITCODE"
    exit 1
}
# 正常路径下把警告行单独挑出来 (它们最容易指向"包能打出来但一跑就崩")
$pyiWarn = @($pyiOut | Select-String -Pattern "WARNING|ERROR" -SimpleMatch)
if ($pyiWarn.Count -gt 0) {
    Write-Output ("  ⚠ PyInstaller 有 {0} 条 WARNING/ERROR (末 8 条):" -f $pyiWarn.Count)
    $pyiWarn | Select-Object -Last 8 | ForEach-Object { Write-Output ("    " + $_.Line) }
    Write-Output "    完整清单见 build\warn-打包exe.txt"
} else {
    Write-Output "  PyInstaller 无 WARNING/ERROR"
}
$exePath = Join-Path $projDir "dist\华容道\华容道.exe"
if (-not (Test-Path $exePath)) {
    Write-Error "打包失败: exe 未生成"
    exit 1
}
$exeMB = [math]::Round((Get-Item $exePath).Length / 1MB, 2)
Write-Output "  打包成功 (exe $exeMB MB)"

Write-Output ""
Write-Output "=== 4. 统一 VC++ 运行时 (app-local) ==="
# 依据 (2026-09 实测):
#   torch_cpu.dll 的静态导入表含 MSVCP140.dll / VCRUNTIME140.dll / VCRUNTIME140_1.dll,
#   未装 VC++ 2015-2022 x64 可再发行组件的机器上缺了它们就会 WinError 1114 闪退
#   (console=False 时用户看不到任何提示), 所以必须随包分发。
#
#   但"让 PyInstaller 自己收集"不可靠: 实测它从
#     C:\WINDOWS\System32\DriverStore\FileRepository\hwaudioservice.inf_*\msvcp140.dll
#   抓到了 14.26.28720.3 的老副本, 而 MSVCP140_ATOMIC_WAIT.dll 却是 14.51.36247.0 ——
#   同一进程混用两个版本的 VC 运行时存在静态初始化崩溃风险 (本机之所以能跑, 只是因为
#   rthook 先从 System32 预加载了 14.51, 把这个不一致掩盖了)。
#   因此这里统一从 System32 覆盖一份, 确保几个文件同版本。
$vcFiles = @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
             'MSVCP140_ATOMIC_WAIT.dll')
$vcInstalled = @()
$vcVersions = @()
foreach ($n in $vcFiles) {
    $sys = Join-Path $env:SystemRoot "System32\$n"
    if (Test-Path $sys) {
        Copy-Item $sys (Join-Path $internalPath $n) -Force
        $vcInstalled += $n
        $vcVersions += (Get-Item $sys).VersionInfo.FileVersion
    }
}
if ($vcInstalled.Count -gt 0) {
    $uniq = @($vcVersions | Select-Object -Unique)
    Write-Output ("  已从 System32 写入: {0}" -f ($vcInstalled -join ', '))
    Write-Output ("  版本: {0}" -f ($uniq -join ' / '))
    if ($uniq.Count -gt 1) {
        Write-Output "  ⚠ System32 里这些 DLL 版本不一致, 建议先更新 VC++ 可再发行组件"
    } else {
        Write-Output "  ✓ 包内 VC++ 运行时版本一致, 目标机无需另装"
    }
} else {
    Write-Output "  ⚠ System32 里找不到 VC++ 运行时 —— 未装可再发行组件的机器会启动失败"
    Write-Output "    解决: 安装 VC++ 2015-2022 x64 可再发行组件后重新打包"
}
# 删掉子目录里的重复副本, 避免同一进程从不同路径加载到不同版本
$vcPattern = '^(vcruntime140|msvcp140|concrt140)'
$dupVc = Get-ChildItem -Path $internalPath -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match $vcPattern -and $_.DirectoryName -ne $internalPath }
foreach ($d in $dupVc) {
    $rel = $d.FullName.Replace("$internalPath\", "")
    $ver = $d.VersionInfo.FileVersion
    Remove-Item $d.FullName -Force
    Write-Output "  已移除子目录重复副本: $rel (版本 $ver)"
}

Write-Output ""
Write-Output "=== 5. 复制训练数据和图标到 exe 同级目录 ==="
# 注意: 布局哈希.py 在 frozen 模式下用 dirname(sys.executable) 作为基准目录,
# 所以训练数据必须放在 exe 同级 (spec 里不再打进 _internal, 避免两份且无效的那份)。
$srcTrainData = Join-Path $projDir "训练数据"
$dstTrainData = Join-Path $projDir "dist\华容道\训练数据"
if (Test-Path $srcTrainData) {
    Copy-Item -Path $srcTrainData -Destination $dstTrainData -Recurse -Force
    Write-Output "  已复制 训练数据"
} else {
    Write-Output "  ⚠ 源目录没有 训练数据, 跳过"
}
$icoSrc = Join-Path $projDir "app.ico"
$icoDst = Join-Path $projDir "dist\华容道\app.ico"
if (Test-Path $icoSrc) {
    Copy-Item -Path $icoSrc -Destination $icoDst -Force
    Write-Output "  已复制 app.ico"
}
$staleInternalTrain = Join-Path $internalPath "训练数据"
if (Test-Path $staleInternalTrain) {
    Write-Output "  ⚠ _internal\训练数据 仍存在 (spec 里应已移除该 datas 项)"
}

Write-Output ""
Write-Output "=== 6. 启动测试 (断言主窗口标题) ==="
# 判据修订 (2026-09-17): 原来只看"12 秒后进程还在"是不够的 ——
# PyInstaller 窗口版在导入期抛异常时会弹一个 "Unhandled exception in script" 模态对话框,
# 进程**一直不退出**, 于是坏包被误判成"启动成功" (实测: excludes 去掉 unittest 那次)。
# 现在必须等到主窗口标题 == 期望标题 才算通过; 窗口没起来就把 stderr 回溯打出来。
$期望标题 = '华容道状态空间分析器'
$启动通过 = $false
$自检_ok = $true   # 第 7 步若有问题会置 false; 只有两个标志都为真才算这份包可用
if ($SkipStartTest) {
    Write-Output "  已按参数跳过"
} else {
    $exeDir = Join-Path $projDir "dist\华容道"
    $torchLog = Join-Path $exeDir "torch_load_error.log"
    $dllLog = Join-Path $exeDir "dll_preload_errors.log"
    $errLog = Join-Path $env:TEMP "华容道_stderr.txt"
    if (Test-Path $torchLog) { Remove-Item $torchLog -Force }
    if (Test-Path $dllLog) { Remove-Item $dllLog -Force }
    if (Test-Path $errLog) { Remove-Item $errLog -Force }

    # 重定向 stderr: PyInstaller 窗口版把 Python 异常回溯写这里, 是唯一的现场
    $proc = Start-Process -FilePath $exePath -PassThru -RedirectStandardError $errLog
    $标题 = ''
    $标题出现秒 = $null
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt 40) {
        Start-Sleep -Milliseconds 400
        $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if (-not $p) { break }
        if ($p.MainWindowHandle -ne 0 -and $p.MainWindowTitle) {
            $标题 = $p.MainWindowTitle
            if ($标题 -eq $期望标题) { $标题出现秒 = [math]::Round($sw.Elapsed.TotalSeconds, 1); break }
        }
    }
    $sw.Stop()
    $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue

    if ($标题 -eq $期望标题 -and $p) {
        $mem = [math]::Round($p.WorkingSet64 / 1MB, 2)
        Write-Output "  ✓ 主窗口已出现: '$标题' (${标题出现秒}s, 内存 $mem MB)"
        $启动通过 = $true
    } else {
        Write-Output "  ✗ 主窗口没有出现 (期望标题: '$期望标题')"
        if ($标题) { Write-Output "    实际窗口标题: '$标题'" }
        if (-not $p) { Write-Output "    进程已退出, 退出码 $($proc.ExitCode)" }
    }

    if ($p) { Stop-Process -Id $proc.Id -Force; Start-Sleep -Seconds 2 }
    if (Test-Path $torchLog) {
        Write-Output "  ✗ PyTorch 加载失败:"; Get-Content $torchLog; $启动通过 = $false
    }
    if (Test-Path $dllLog) {
        Write-Output "  ✗ DLL 预加载有错误:"; Get-Content $dllLog; $启动通过 = $false
    }
    if ($启动通过) { Write-Output "  启动测试通过 (窗口标题正确, 且无 torch/DLL 错误日志)" }
    elseif ((Test-Path $errLog) -and ((Get-Item $errLog).Length -gt 0)) {
        Write-Output "  ---- stderr 回溯 (末 40 行) ----"
        Get-Content $errLog | Select-Object -Last 40
    }
}

Write-Output ""
# 启动都失败就没必要再跑包内自检了: 它只会以同一种方式再崩一次, 白白多等十几秒
if (-not $SkipStartTest -and -not $启动通过) {
    Write-Output "=== 7. 包内自检 ==="
    Write-Output "  已跳过 (第 6 步启动就没通过, 先把上面的回溯修掉)"
    Write-Output ""
    Write-Output "=== 打包失败 ==="
    Write-Output ("验证结论: ✗ 启动测试未通过 (期望主窗口标题 '{0}') —— 这份包不要发出去" -f $期望标题)
    Write-Output "exe 位置: $exePath"
    exit 1
}

Write-Output "=== 7. 包内自检 (由 exe 自己跑 自检.py) ==="
# 为什么必须由 exe 自己跑: 冻结后的 import 解析、_internal 的 DLL 搜索路径、
# 布局哈希.py 的 frozen 分支 (_BASE_DIR = exe 同级) 在源码模式下都测不出来。
# 自检覆盖 8 项: 冻结布局 / 17 个工程模块导入 / unittest·pyparsing 回归 / torch 计算 /
#   AI 训练流水线(GNN+sklearn+TDA+存盘往返) / 训练数据发现 / 模型文件真实载入 /
#   GUI 主窗口构造与标题。报告落盘, 因为窗口版 exe 的 stdout 是 None。
$自检报告 = Join-Path $projDir "dist\华容道\自检报告.json"
if ($SkipStartTest) {
    Write-Output "  已按参数跳过"
} else {
    if (Test-Path $自检报告) { Remove-Item $自检报告 -Force }
    $selfLog = Join-Path $env:TEMP "华容道_自检_stderr.txt"
    if (Test-Path $selfLog) { Remove-Item $selfLog -Force }
    $env:HRD_SELFTEST = '1'
    # 用 .NET 的 Process 而不是 Start-Process -PassThru:
    # 实测 Start-Process 一旦带 -RedirectStandardError, 返回对象的 .ExitCode 就恒为 $null
    # (连 .Refresh() 都刷不出来), 于是 `$sp.ExitCode -ne 0` 恒为真, 自检会被误判成失败。
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exePath
    $psi.UseShellExecute = $false
    $psi.RedirectStandardError = $true
    $sp = [System.Diagnostics.Process]::Start($psi)
    $已退出 = $sp.WaitForExit(300000)
    $退出码 = if ($已退出) { $sp.ExitCode } else { $null }
    if (-not $已退出) { try { $sp.Kill() } catch {} }
    Remove-Item Env:\HRD_SELFTEST -ErrorAction SilentlyContinue
    if (-not $已退出) {
        Write-Output "  ✗ 自检超时 (>300s)"; $自检_ok = $false
    } else {
        Write-Output "  自检退出码: $退出码  (0=全过, 2=有失败项)"
        # 报告落盘后再读: exe 是窗口版, stdout/stderr 拿不到内容, stderr 这里同时落一份便于排错
        try { $sp.StandardError.ReadToEnd() | Set-Content -Path $selfLog -Encoding UTF8 } catch {}
        if ($退出码 -ne 0) { $自检_ok = $false }
    }

    if (Test-Path $自检报告) {
        $rep = Get-Content $自检报告 -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($c in $rep.'检查') {
            $mark = if ($c.'通过') { '✓' } else { '✗' }
            Write-Output ("  {0} {1} ({2}s): {3}" -f $mark, $c.'名称', $c.'耗时秒', $c.'详情')
            if (-not $c.'通过' -and $c.'回溯') {
                Write-Output "      ---- 回溯 ----"
                ($c.'回溯' -split "`n") | Select-Object -Last 12 | ForEach-Object { Write-Output "      $_" }
            }
        }
        Write-Output ("  自检汇总: 通过 {0} 项, 失败 {1} 项" -f $rep.'通过数', $rep.'失败数')
        if ($rep.'失败数' -gt 0) { Write-Output ("  ✗ 失败项: " + ($rep.'失败项' -join ', ')); $自检_ok = $false }
    } else {
        Write-Output "  ✗ 未生成自检报告: $自检报告"
        if (Test-Path $selfLog) { Get-Content $selfLog | Select-Object -Last 40 }
        $自检_ok = $false
    }
}

Write-Output ""
Write-Output "=== 打包完成 ==="
Write-Output "exe 位置: $exePath"
if (-not $SkipStartTest) {
    if ($启动通过 -and $自检_ok) {
        Write-Output "验证结论: ✓ 启动测试与包内自检全部通过"
        # 自检报告是排查用的中间产物, 内容已在上面全部打印, 清掉避免被当成交付物一起发出去
        if (Test-Path $自检报告) { Remove-Item $自检报告 -Force; Write-Output "  已清理自检报告 (内容见上方输出)" }
    } else {
        Write-Output ("验证结论: ✗ 存在问题 (启动测试 {0} / 包内自检 {1}) —— 这份包不要发出去" -f `
            $(if ($启动通过) { '通过' } else { '失败' }), $(if ($自检_ok) { '通过' } else { '失败' }))
        exit 1
    }
}
