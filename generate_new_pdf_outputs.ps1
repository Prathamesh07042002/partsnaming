# Regenerate the reviewed outputs for the three supplied drawing PDFs.
# Requires pdftoppm (Poppler) to render the required PDF page temporarily.
$ErrorActionPreference = 'Stop'

$jobs = @(
    @{ Part = '550629507102'; Pdf = '550629507102.pdf'; Page = 2 },
    @{ Part = '278909130282'; Pdf = '278909130282.pdf'; Page = 2 },
    @{ Part = '503046803301'; Pdf = '503046803301.pdf'; Page = 1 }
)

$renderDir = Join-Path ([System.IO.Path]::GetTempPath()) ("partsnaming-render-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $renderDir | Out-Null

try {
    foreach ($job in $jobs) {
        $imageBase = Join-Path $renderDir $job.Part
        & pdftoppm.exe -png -r 200 -f $job.Page -l $job.Page $job.Pdf $imageBase
        if ($LASTEXITCODE -ne 0) { throw "Could not render $($job.Pdf)." }

        $image = "{0}-{1}.png" -f $imageBase, $job.Page
        python .\clockwise_numbering.py $image `
            --items ("reviewed_items/{0}.json" -f $job.Part) `
            --output-dir ("outputs/{0}" -f $job.Part)
        if ($LASTEXITCODE -ne 0) { throw "Could not annotate $($job.Part)." }
    }
}
finally {
    Remove-Item -LiteralPath $renderDir -Recurse -Force -ErrorAction SilentlyContinue
}
