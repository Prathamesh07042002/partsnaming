# Engineering drawing clockwise numbering

Use [clockwise_numbering.py](clockwise_numbering.py) for a new drawing.

```powershell
# Best option: use reviewed items for correct engineering values and leader targets.
python .\clockwise_numbering.py .\drawing.jpg --items .\reviewed_items.json --output-dir .\result

# OCR draft only: review `result\ocr_review_items.json` before release.
python .\clockwise_numbering.py .\drawing.jpg --ocr --output-dir .\result
```

Copy and edit [items_example.json](items_example.json) to make `reviewed_items.json`.
The script applies these rules: material description/specification first; grouped
drawing callouts clockwise; final mass before the note; note last.  Add `view`
to finish one view before the next, and add `order` to override clockwise order
for a drawing-specific inspection sequence.

Each run creates the annotated drawing `annotated_drawing.png` and the formatted
Excel table `extracted_dimensions.xlsx`. The script validates every marker and leader-target
coordinate against the source-image bounds. Review JSON targets before release:
they must be placed on the actual written dimension/callout, not merely near it.

OCR requires Tesseract and `pytesseract`; its extracted dimensions and tolerances
must be reviewed before use.

## Supplied drawing PDFs

The reviewed outputs for `550629507102`, `278909130282`, and `503046803301` are
in `outputs/<part-number>/`. To regenerate them, run:

```powershell
.\generate_new_pdf_outputs.ps1
```

The helper renders each required PDF page into the system temporary folder and
removes those images automatically when it finishes. It requires `pdftoppm`
(Poppler) on the system path.

The reviewed extraction and exact marker targets are stored in
`reviewed_items/<part-number>.json`.
