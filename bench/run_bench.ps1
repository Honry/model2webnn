# run_bench.ps1 - run one experiment of the NPU split bench. Default build: Chrome Canary with the stock Windows ML OpenVINO EP
# (static shapes only). -Build 0905 uses the dynamic-shape dev build (kept for comparison; earlier results in 03-measurements.md were taken there).
#
#   powershell -ExecutionPolicy Bypass -File run_bench.ps1 -Query "exp=m0" [-Run m0_try1] [-Tag ort-ov-npu-qwen2-1102-RelWithDebInfo] [-TimeoutMin 60]
#   powershell -ExecutionPolicy Bypass -File run_bench.ps1 -Query "exp=chain&L=36&S=1,3,12&q=1&t=512&w=dq_const"
#
# - starts server.py (http://localhost:8123, repository root) writing results\<Run>.jsonl, launches Chrome on bench.html?<Query>,
#   samples the summed working set / private bytes of this Chrome profile's processes every 1s into results\<Run>.mem.csv,
#   waits for {"event":"done"} in the results file (or the timeout), then stops Chrome and the server.
param(
    [Parameter(Mandatory = $true)][string]$Query,
    [string]$Run = '',
    [string]$Page = 'bench/bench.html',   # e.g. bench/raw/raw.html for the pure WebNN benchmark
    [ValidateSet('canary', '0905', 'optzp', 'npuflag')][string]$Build = 'canary',   # canary = stock Chrome Canary + default Windows ML OVEP (static shapes); 0905 = the dynamic-shape dev build; optzp = PoC Chrome with optional dequantizeLinear zeroPoint; npuflag = optzp + working --webnn-ort-ovep-npuw (both need -Tag DLL dir)
    [string]$Tag = 'ort-ov-npu-qwen2-1102-RelWithDebInfo',     # DLL dir C:\Program Files\<Tag>, only used with -Build 0905 / optzp
    [string[]]$ExtraArgs = @(),     # extra Chrome flags
    [int]$TimeoutMin = 60,
    [switch]$KeepOpen,
    [switch]$AllowCpuFallback,   # do NOT pass --webnn-ort-disable-cpu-fallback (unsupported ops silently run on the CPU EP)
    [switch]$Dump   # pass --webnn-ort-dump-model=results\dump_<Run> (the OV IR OVEP receives)
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
# Chrome profiles live under bench\data\chrome by default; the PoC Chrome builds are looked up in $env:BENCH_CHROME_DIR (a folder holding
# chrome.packed.optionalZeroPoint, chrome.packed.NPUFlag and 0905\extracted); the dll dir for -Build 0905/optzp/npuflag is C:\Program Files\<Tag>
$doc = if ($env:BENCH_DATA_DIR) { $env:BENCH_DATA_DIR } else { Join-Path $here 'data' }
$builds = if ($env:BENCH_CHROME_DIR) { $env:BENCH_CHROME_DIR } else { Join-Path $doc 'chrome' }
if ($Build -eq 'canary') {
    $chrome = Join-Path $env:LOCALAPPDATA 'Google\Chrome SxS\Application\chrome.exe'
    $profileDir = Join-Path $doc 'chrome\profile-bench-canary'
} elseif ($Build -eq 'optzp') {
    $chrome = Join-Path $builds 'chrome.packed.optionalZeroPoint\chrome\Chrome-bin\chrome.exe'
    $profileDir = Join-Path $doc 'chrome\profile-bench-optzp'
} elseif ($Build -eq 'npuflag') {
    $chrome = Join-Path $builds 'chrome.packed.NPUFlag\chrome\Chrome-bin\chrome.exe'
    $profileDir = Join-Path $doc 'chrome\profile-bench-npuflag'
} else {
    $chrome = Join-Path $builds '0905\extracted\Chrome-bin\chrome.exe'
    $profileDir = Join-Path $doc 'chrome\profile-bench'
}
$dll = "C:\Program Files\$Tag"
if (-not $Run) { $Run = 'run_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$results = Join-Path $here 'results'
New-Item -ItemType Directory -Force -Path $results, $profileDir | Out-Null
$jsonl = Join-Path $results "$Run.jsonl"
$memcsv = Join-Path $results "$Run.mem.csv"
$chromeLog = Join-Path $results "$Run.chrome.log"
$need = @($chrome)
if ($Build -ne 'canary') { $need += "$dll\onnxruntime_providers_openvino_plugin.dll" }
foreach ($p in $need) {
    if (-not (Test-Path $p)) { Write-Error "not found: $p"; exit 1 }
}

# stop any previous bench Chrome / server
Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like "*$profileDir*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*bench*server.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

$env:BENCH_RUN = $Run
$server = Start-Process -FilePath python -ArgumentList @("`"$here\server.py`"") -PassThru -WindowStyle Hidden -RedirectStandardError (Join-Path $results "$Run.server.log")
Start-Sleep -Seconds 2

$url = "http://localhost:8123/${Page}?$Query"
$args = @()
if ($Build -ne 'canary') {
    $args += "--webnn-ort-library-path-for-testing=`"$dll`"", '--allow-third-party-modules',
             "--webnn-ort-ep-library-path-for-testing=OpenVINOExecutionProvider?`"$dll\onnxruntime_providers_openvino_plugin.dll`""
}
if ($Dump) {
    $dumpDir = Join-Path $results "dump_$Run"
    New-Item -ItemType Directory -Force -Path $dumpDir | Out-Null
    $args += "--webnn-ort-dump-model=`"$dumpDir`""
}
$args += $ExtraArgs
$args += '--enable-features=WinSboxHighGPUJobMemoryLimits,WebMachineLearningNeuralNetwork,WebNNOnnxRuntime',
         '--no-sandbox',
         '--webnn-ort-logging-level=WARNING',
         $(if (-not $AllowCpuFallback) { '--webnn-ort-disable-cpu-fallback' }),
         "--user-data-dir=$profileDir",
         '--ignore-certificate-errors',
         '--enable-logging', "--log-file=$chromeLog",
         $url
$proc = Start-Process -FilePath $chrome -ArgumentList $args -PassThru
"run $Run : build $Build, chrome pid $($proc.Id), url $url"
"results  : $jsonl"

"t_s,working_set_mb,private_mb,procs" | Out-File $memcsv -Encoding ascii
$t0 = Get-Date
$deadline = $t0.AddMinutes($TimeoutMin)
$done = $false
while ((Get-Date) -lt $deadline) {
    $procs = Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like "*$profileDir*" }
    $ids = @($procs | ForEach-Object { $_.ProcessId })
    if ($ids.Count -gt 0) {
        $ps = Get-Process -Id $ids -ErrorAction SilentlyContinue
        $ws = [math]::Round((($ps | Measure-Object WorkingSet64 -Sum).Sum) / 1MB)
        $pv = [math]::Round((($ps | Measure-Object PrivateMemorySize64 -Sum).Sum) / 1MB)
        ("{0},{1},{2},{3}" -f [math]::Round(((Get-Date) - $t0).TotalSeconds), $ws, $pv, $ids.Count) | Out-File $memcsv -Append -Encoding ascii
    } elseif ($proc.HasExited) { "chrome exited"; break }
    if (Test-Path $jsonl) {
        if (Select-String -Path $jsonl -Pattern '"event":"done"' -Quiet) { $done = $true; break }
    }
    Start-Sleep -Seconds 1
}
if ($done) { "done in {0:n0}s" -f ((Get-Date) - $t0).TotalSeconds } else { "TIMEOUT/exit before done" }
if (-not $KeepOpen) {
    Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like "*$profileDir*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
}
"peak working set MB: " + ((Import-Csv $memcsv | Measure-Object working_set_mb -Maximum).Maximum)
"peak private MB    : " + ((Import-Csv $memcsv | Measure-Object private_mb -Maximum).Maximum)
