# 上传 Release 资产 (curl 实现)
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File .\上传资产.ps1
#   powershell -ExecutionPolicy Bypass -File .\上传资产.ps1 -ZipPath <路径>
#
# 为什么用 curl 而不是 Invoke-RestMethod / Python:
#   229.8 MB 的资产上传在本机踩过两次:
#     · PowerShell 5.1 的 System.Net.Http.HttpClient 不会自动加载, 直接报类型不存在;
#     · Python urllib 的流式上传传到 99.5% 被服务端以 HTTP 400 拒掉 (Content-Length 对不上)。
#   curl.exe 是 Windows 10+ 自带的 (C:\Windows\system32\curl.exe), 对分块传输与
#   断线重试的处理最成熟, 配合 --retry-all-errors 能在抖动的网络下自动续命。
#
# 凭据: Windows 凭据管理器 git:https://github.com, 不打印也不落盘。

param(
    [string]$Owner   = 'XiYuechaoshui',
    [string]$Repo    = 'HuaRongDao',
    [string]$Tag     = 'v1.0.0',
    [string]$ZipPath = 'D:\XIYueの奇妙冒险\HuaRongDao-v1.0.0-win64.zip',
    [int]$Retry      = 5
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ---------- 凭据 ----------
if (-not ('CredMan3' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class CredMan3 {
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
$token = [CredMan3]::Get('git:https://github.com')
if (-not $token) { Write-Error '凭据管理器里没有 git:https://github.com'; exit 1 }

if (-not (Test-Path $ZipPath)) { Write-Error "找不到 zip: $ZipPath"; exit 1 }
$zip = Get-Item $ZipPath
Write-Output ('=== 上传 {0}  ({1:N1} MB) ===' -f $zip.Name, ($zip.Length / 1MB))

# ---------- 取 Release ----------
$headers = @{
    Authorization          = "Bearer $token"
    Accept                 = 'application/vnd.github+json'
    'User-Agent'           = 'HuaRongDao-Uploader'
    'X-GitHub-Api-Version' = '2022-11-28'
}
try {
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo/releases/tags/$Tag" `
        -Headers $headers -TimeoutSec 60
} catch {
    Write-Error "取 Release 失败: $($_.Exception.Message)"
    exit 1
}
Write-Output ('  Release: {0} (id={1})' -f $rel.html_url, $rel.id)

$exist = @($rel.assets | Where-Object { $_.name -eq $zip.Name })
if ($exist.Count -gt 0) {
    Write-Output ('  已存在同名资产 ({0:N1} MB), 跳过。要重传请先在网页删掉它。' -f ($exist[0].size / 1MB))
    exit 0
}

# ---------- curl 上传 ----------
$uploadUrl = 'https://uploads.github.com/repos/{0}/{1}/releases/{2}/assets?name={3}' -f `
    $Owner, $Repo, $rel.id, [uri]::EscapeDataString($zip.Name)

# 令牌经 stdin 传给 curl (--header @-), 不出现在命令行里 (避免被进程列表看到)
$headerLines = "Authorization: Bearer $token`nContent-Type: application/zip"

Write-Output '  开始上传 (curl 自带重试, 断线会自动续)...'
$sw = [System.Diagnostics.Stopwatch]::StartNew()

$curlArgs = @(
    '-sS',                                  # 静默进度表, 只留错误
    '--fail-with-body',                     # 4xx/5xx 时返回非零并打印响应体
    '--retry', "$Retry",
    '--retry-delay', '5',
    '--retry-all-errors',
    '--connect-timeout', '30',
    '--max-time', '7200',
    '-X', 'POST',
    '-H', '@-',
    '--data-binary', "@$($zip.FullName)",
    '-o', "$env:TEMP\hrd_upload_resp.json",
    $uploadUrl
)

$headerLines | & curl.exe @curlArgs 2>&1 | ForEach-Object { Write-Output "    $_" }
$code = $LASTEXITCODE
$sw.Stop()

if ($code -ne 0) {
    Write-Output ''
    Write-Error ("上传失败 (curl 退出码 {0}, 耗时 {1:N0} 秒)" -f $code, $sw.Elapsed.TotalSeconds)
    $resp = "$env:TEMP\hrd_upload_resp.json"
    if (Test-Path $resp) {
        Write-Output '  服务端响应:'
        Get-Content $resp -Raw | ForEach-Object { "    $_" }
    }
    Write-Output '  提示: 直接重跑本脚本即可 (Release 已建好, 不会重复创建)'
    exit 1
}

$body = Get-Content "$env:TEMP\hrd_upload_resp.json" -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Output ''
Write-Output ('  ✓ 上传成功! 耗时 {0:N0} 秒, 平均 {1:N2} MB/s' -f `
    $sw.Elapsed.TotalSeconds, (($zip.Length / 1MB) / $sw.Elapsed.TotalSeconds))
Write-Output ('  资产名: {0}' -f $body.name)
Write-Output ('  下载地址: {0}' -f $body.browser_download_url)
