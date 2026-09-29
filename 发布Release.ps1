# GitHub Release 创建/上传脚本
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File .\发布Release.ps1
#   powershell -ExecutionPolicy Bypass -File .\发布Release.ps1 -ZipPath <路径>   # 顺带上传资产
#
# 为什么需要它:
#   本工程在国内网络下 git push 经常 "Connection was reset", 传几百 MB 的资产更是容易中断。
#   这个脚本把"建 Release"和"传资产"分开, 并默认**不覆盖已存在的 Release**,
#   所以可以反复运行: 第一次建好 Release, 之后只补传资产, 中断了再跑一次即可。
#
# 凭据来源: Windows 凭据管理器里的 git:https://github.com (git push 时存进去的那个)。
#   本脚本不打印、不落盘任何令牌。

param(
    [string]$Owner   = 'XiYuechaoshui',
    [string]$Repo    = 'HuaRongDao',
    [string]$Tag     = 'v1.0.0',
    [string]$Name    = '华容道状态空间分析器 v1.0.0',
    [string]$ZipPath = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# ---------- 1. 从 Windows 凭据管理器读令牌 ----------
if (-not ('CredMan' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class CredMan {
    [DllImport("advapi32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    static extern bool CredRead(string target, int type, int flags, out IntPtr credential);
    [DllImport("advapi32.dll")]
    static extern void CredFree(IntPtr cred);
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    struct CREDENTIAL {
        public int Flags; public int Type; public string TargetName; public string Comment;
        public long LastWritten; public int CredentialBlobSize; public IntPtr CredentialBlob;
        public int Persist; public int AttributeCount; public IntPtr Attributes;
        public string TargetAlias; public string UserName;
    }
    public static string Get(string target) {
        IntPtr p;
        if (!CredRead(target, 1, 0, out p)) return null;
        try {
            CREDENTIAL c = (CREDENTIAL)Marshal.PtrToStructure(p, typeof(CREDENTIAL));
            if (c.CredentialBlob == IntPtr.Zero || c.CredentialBlobSize == 0) return null;
            return Marshal.PtrToStringUni(c.CredentialBlob, c.CredentialBlobSize / 2);
        } finally { CredFree(p); }
    }
}
'@
}

Write-Output "=== 1. 读取凭据 ==="
$targets = @('git:https://github.com', "git:https://$Owner@github.com")
$token = $null
foreach ($t in $targets) {
    $v = [CredMan]::Get($t)
    if ($v) { $token = $v; Write-Output "  从 '$t' 取到凭据 ($($v.Length) 字符)"; break }
}
if (-not $token) { Write-Error "凭据管理器里没找到 GitHub 凭据, 请先成功 git push 一次"; exit 1 }
if ($token.Length -lt 20) { Write-Error "凭据看起来不是令牌 (长度 $($token.Length)), 可能是旧密码"; exit 1 }

$headers = @{
    Authorization = "Bearer $token"
    Accept        = 'application/vnd.github+json'
    'User-Agent'  = 'HuaRongDao-Release-Script'
    'X-GitHub-Api-Version' = '2022-11-28'
}
$api = "https://api.github.com/repos/$Owner/$Repo"

# ---------- 2. 验证令牌与仓库 ----------
Write-Output ""
Write-Output "=== 2. 验证令牌 ==="
try {
    $me = Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers -TimeoutSec 30
    Write-Output "  登录身份: $($me.login)"
} catch {
    Write-Error "令牌无效或网络不通: $($_.Exception.Message)"
    exit 1
}

if ($DryRun) { Write-Output ""; Write-Output "(-DryRun 模式, 到此为止)"; exit 0 }

# ---------- 3. 建 Release (已存在则复用) ----------
Write-Output ""
Write-Output "=== 3. Release ($Tag) ==="
$release = $null
try {
    $release = Invoke-RestMethod -Uri "$api/releases/tags/$Tag" -Headers $headers -TimeoutSec 30
    Write-Output "  已存在, 复用: $($release.html_url)"
} catch {
    $body = @{
        tag_name         = $Tag
        target_commitish = 'main'
        name             = $Name
        draft            = $false
        prerelease       = $false
        body             = @"
可以**直接下载运行**的 Windows 版本, 不需要装 Python / 联网 / VC++ 运行库。

## 下载哪个

``华容道-v1.0.0-win64.zip`` (约 250~350 MB)

解压后双击 ``华容道.exe`` 即可。**首次启动约 10~20 秒** (要解包并预热 torch)。

> ⚠️ 必须**整个文件夹一起**保留 (``_internal\`` 是 Python 运行时 + torch, 约 535 MB)。
> 单独把 exe 拷出来是跑不起来的。

## 这个程序做什么

把华容道这类滑块谜题的**状态图**当作数学对象研究:

| 分析 | 内容 |
|---|---|
| 状态空间生成 | 从任意布局 BFS 出完整状态图 (5×4/4 块 = 4.6 万节点; 5 块 = 15.4 万), 支持对称性模式开关 |
| 代数拓扑 | Betti 数 β₀/β₁、Euler 示性数 χ、围长、圈空间 |
| 谱图分析 | 拉普拉斯/邻接谱、谱半径、代数连通度 λ₂、有效电阻 |
| 持久同调 | 自实现 Vietoris–Rips 列约简、H₀/H₁ 条码、持久图像 |
| 社区检测 | Louvain / greedy modularity / GNN 嵌入聚类 |
| 图机器学习 | 两层 GraphSAGE、TDA 自编码器、sklearn 图级特征 |
| 可视化 | 3D 状态空间 (按 BFS 深度着色)、谱分布、持久条码、AI 综合增强图 |

界面快捷键: **F1** 3D 状态图 / **F2** 谱分布 / **F3** 持久条形码 / **F4** AI增强图

## 目标机要求

- Windows x64
- 内存约 330 MB
- 需要能写 ``%TEMP%`` / ``%LOCALAPPDATA%`` (matplotlib 首次运行要写字体缓存)

## 说明

- 自带三个全局预训练模型 (gnn / tda_ae / ml), 打开即可用 AI 分析, 不用先自己训
- 不含布局级分析缓存 —— 那是对特定棋盘算过的结果, 程序会在你分析时自动生成
- 完整源码与打包流程见仓库根目录 ``README.md`` / ``打包说明.md``

MIT License
"@
    }
    $json = $body | ConvertTo-Json -Depth 5
    $release = Invoke-RestMethod -Uri "$api/releases" -Method Post -Headers $headers `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($json)) -ContentType 'application/json; charset=utf-8' -TimeoutSec 60
    Write-Output "  ✓ 已创建: $($release.html_url)"
}

# ---------- 4. 上传资产 ----------
if (-not $ZipPath) {
    Write-Output ""
    Write-Output "=== 4. 资产上传 ==="
    Write-Output "  未指定 -ZipPath, 跳过。要上传请加:  -ZipPath 'D:\...\华容道-v1.0.0-win64.zip'"
    exit 0
}
if (-not (Test-Path $ZipPath)) { Write-Error "找不到文件: $ZipPath"; exit 1 }

$zip = Get-Item $ZipPath
Write-Output ""
Write-Output "=== 4. 上传资产 ==="
Write-Output ("  文件: {0}  ({1:N1} MB)" -f $zip.Name, ($zip.Length / 1MB))

# 已上传过就跳过 (可反复运行)
$existing = @($release.assets | Where-Object { $_.name -eq $zip.Name })
if ($existing.Count -gt 0) {
    Write-Output ("  已存在同名资产 ({0:N1} MB), 跳过上传" -f ($existing[0].size / 1MB))
    Write-Output "  若要重传, 请先在网页上删掉这个资产"
    exit 0
}

$uploadUrl = "https://uploads.github.com/repos/$Owner/$Repo/releases/$($release.id)/assets?name=$([uri]::EscapeDataString($zip.Name))"
Write-Output "  开始上传 (中途失败请重新运行本脚本, 会跳过已完成的步骤)..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $asset = Invoke-RestMethod -Uri $uploadUrl -Method Post -Headers $headers `
        -InFile $zip.FullName -ContentType 'application/zip' -TimeoutSec 7200
    $sw.Stop()
    Write-Output ("  ✓ 上传完成: {0:N1} MB, 耗时 {1:N0} 秒" -f ($asset.size / 1MB), $sw.Elapsed.TotalSeconds)
    Write-Output "  下载地址: $($asset.browser_download_url)"
} catch {
    $sw.Stop()
    Write-Error ("上传失败 (已耗时 {0:N0} 秒): {1}" -f $sw.Elapsed.TotalSeconds, $_.Exception.Message)
    Write-Output "  提示: 直接重新运行本脚本即可重试; Release 已建好不会被重复创建。"
    exit 1
}
