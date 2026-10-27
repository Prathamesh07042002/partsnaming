<#
.SYNOPSIS
    Single entry point for the drawing-annotation pipeline.

.DESCRIPTION
    Delegates to process_pdfs.py, which:
      - discovers every *.pdf in this folder,
      - skips PDFs whose outputs already exist,
      - uses reviewed_items/<part>.json when available, else falls back to --ocr,
      - writes outputs/<part>/annotated_drawing.png and extracted_dimensions.xlsx.

.PARAMETER Force
    Re-process PDFs even if their outputs already exist.

.PARAMETER Page
    Page to render (0 = last page, default).

.PARAMETER InputDir
    Folder containing PDFs. Defaults to this script's folder.

.EXAMPLE
    .\generate_new_pdf_outputs.ps1

.EXAMPLE
    .\generate_new_pdf_outputs.ps1 -Force

.EXAMPLE
    .\generate_new_pdf_outputs.ps1 -Page 1 -InputDir "D:\drawings\inbox"
#>
param(
    [switch]$Force,
    [int]$Page = 0,
    [string]$InputDir = (Join-Path $PSScriptRoot 'sources')
)

$ErrorActionPreference = 'Stop'

# --- Anchor everything to this script's folder ---
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -LiteralPath $root

$driverPath = Join-Path $root 'process_pdfs.py'
if (-not (Test-Path -LiteralPath $driverPath)) {
    throw "process_pdfs.py not found next to this script: $driverPath"
}

# --- Resolve python ---
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    $python = (Get-Command py -ErrorAction SilentlyContinue).Source
}
if (-not $python) { throw "Python not found on PATH." }
Write-Host "Using python: $python"
Write-Host "Project root: $root"
Write-Host ""

# --- Build args for process_pdfs.py ---
$pyArgs = @($driverPath, '--input-dir', $InputDir)
if ($Force)      { $pyArgs += '--force' }
if ($Page -gt 0) { $pyArgs += @('--page', $Page) }

# --- Run ---
& $python @pyArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "process_pdfs.py exited with code $exitCode"
}

# # The one command they asked for:
# .\generate_new_pdf_outputs.ps1

# # Re-process everything:
# .\generate_new_pdf_outputs.ps1 -Force

# # Single-page drawings:
# .\generate_new_pdf_outputs.ps1 -Page 1

# # Different folder:
# .\generate_new_pdf_outputs.ps1 -InputDir "D:\drawings\inbox"
# 