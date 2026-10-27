# Regenerate the reviewed outputs for the three supplied drawing PDFs.
# Requires pdftoppm (Poppler) to render the required PDF page temporarily.
$ErrorActionPreference = 'Stop'

# --- Anchor everything to this script's folder ---
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -LiteralPath $root          # make Python's CWD deterministic

# Resolve pdftoppm robustly (VS Code stale-env safe)
$pdftoppm = (Get-Command pdftoppm -ErrorAction SilentlyContinue).Source
if (-not $pdftoppm) {
    $candidates = @(
        "C:\Program Files\poppler-26.09.0\Library\bin\pdftoppm.exe",
        "C:\Program Files\poppler\Library\bin\pdftoppm.exe",
        "C:\poppler\Library\bin\pdftoppm.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { $pdftoppm = $c; break } }
}
if (-not $pdftoppm) { throw "pdftoppm not found. Add Poppler to PATH." }
Write-Host "Using pdftoppm: $pdftoppm"

$jobs = @(
    @{ Part = '550629507102'; Pdf = '550629507102.pdf'; Page = 2 },
    @{ Part = '278909130282'; Pdf = '278909130282.pdf'; Page = 2 },
    @{ Part = '503046803301'; Pdf = '503046803301.pdf'; Page = 1 }
)

$renderDir = Join-Path ([System.IO.Path]::GetTempPath()) ("partsnaming-render-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $renderDir | Out-Null

try {
    foreach ($job in $jobs) {
        $pdfPath    = Join-Path $root $job.Pdf
        $itemsPath  = Join-Path $root ("reviewed_items\{0}.json" -f $job.Part)
        $outDir     = Join-Path $root ("outputs\{0}" -f $job.Part)
        $scriptPath = Join-Path $root "clockwise_numbering.py"

        if (-not (Test-Path -LiteralPath $pdfPath))   { throw "Missing PDF: $pdfPath" }
        if (-not (Test-Path -LiteralPath $itemsPath)) { throw "Missing items JSON: $itemsPath" }

        $imageBase = Join-Path $renderDir $job.Part
        & $pdftoppm -png -r 200 -f $job.Page -l $job.Page $pdfPath $imageBase
        if ($LASTEXITCODE -ne 0) { throw "Could not render $($job.Pdf)." }

        $image = "{0}-{1}.png" -f $imageBase, $job.Page

        python $scriptPath $image `
            --items $itemsPath `
            --output-dir $outDir
        if ($LASTEXITCODE -ne 0) { throw "Could not annotate $($job.Part)." }
    }
}
finally {
    Remove-Item -LiteralPath $renderDir -Recurse -Force -ErrorAction SilentlyContinue
}