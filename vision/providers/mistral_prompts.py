"""Prompt for Mistral Pixtral VLM extraction."""

DRAWING_PROMPT = r"""\
TASK: You are a strict mechanical engineering drawing inspector. Extract
dimension callouts and the bottom-right Title Block from the drawing.
Output ONLY valid JSON matching the schema. Do not explain. Do not reason.

1. DIMENSIONS TO EXTRACT (type: "feature")
Extract EVERY dimension callout visible. Include ALL of these:

- Boxed Edge Numbers: numbers enclosed in small rectangles along edges
  (e.g., "361", "184.6", "326", "281", "170.8").
- Rotated/Vertical Text: numbers reading bottom-to-top or top-to-bottom
  along the side edges.
- Reference Dimensions: numbers wrapped in parentheses:
  "(43)", "(72)", "(121)", "(17.5)", "(25)", "(90)", "(409)".
- Normal dimensions: "407", "35", "110", "130".
- Dimensions with tolerance: "379±0.5", "30±0.5", "176±0.5".
- Radii: "R2", "R3", "R5", "R8", "R10", "R15", "R20", "R3(TYP)", "R15(TYP)".
- Diameters / holes: "Ø7", "Ø8", "Ø11", "Ø13", "2 HOLES Ø8",
  "3 HOLES Ø13", "2HOLES,Ø8H11".
- Angles: "103°", "15°", "90°", "135°".
- Leader-line callouts (extract the full text): "2 HOLES Ø8 FOR TOOLING",
  "TACK WELD AT 3 PLACES", "QTY.4 NOS - HEX NUT M12",
  "ON ALTERNATE SIDES 3 TACKS (TYP)".

For each feature callout, output:
  text          : exact characters as printed, preserving Ø, ±, °, R, (TYP), (REF)
  type          : "feature"
  parameter     : one of  Dimn | Hole | Radius | Angle | Note
  specification : nominal value WITHOUT tolerance
                  "379±0.5" -> "379" ; "Ø8" -> "Ø8" ; "R15(TYP)" -> "R15"
  tolerance     : only the ± portion ("±0.5"). If none, "".

2. TITLE BLOCK — ONLY 6 FIELDS (bottom-right corner only)
Look ONLY at the bottom-right title block. Extract these SIX fields, the
VALUE only (not the label):

  part_number          e.g. "2882 3154 01 10"        parameter="Part No."
  material_description e.g. "5mm Thk - T SH HR-Fe410" parameter="Thk"
  material_specification e.g. "TS23521"                parameter="Grade"
  envelope             e.g. "409.00 x 65.00 x 27.15"   parameter="Envelope"
  mass                 e.g. "0.553 kg"                 parameter="Final mass"
  surface_protection   e.g. "FeZn 8 Cr III Y"          parameter="Surface Protection"
                       (if blank / "-", use "As per standard")

3. HARD EXCLUSIONS — NEVER emit these
- Tolerance table: grid like "Over 6 30 120 400 1000 2000" and
  "±0.1 ±0.2 ±0.3 ±0.5 ±0.8 ±1.2 ±2". This is a tolerance table.
  DO NOT extract any number from inside it.
- Roughness table: "Symbol | MD | AV | SD | 25 | 6.2 | 1.6 | 0.8 | 0.4".
- Revision history rows / dates: "28/12", "2016", "2025", "06PP", "22CU".
- Signatures: BSH, HMP, RMP, SKK, CKY, KTT, AAD, PMK, ASH, AKK.
- Boilerplate: "Do not scale the drawing", "If in doubt, ask",
  "Unless otherwise stated", "Drg drawn to first angle projection".
- Proprietary text ("This drawing is the sole property of Tata Motors").
- "Critical", "Major", "Minor", "Key Instruction".
- Company names/logos: TATA, TATA MOTORS LIMITED, ERC - PUNE, SIGNA AUT, DR4.
- Scale text: "Scale: 1:5", "Scale: 1:2", "SCALE 1:1".
- View labels: "SECTION A-A", "SECTION B-B", "VIEW", "DETAIL", "ISO VIEW".
- Frame border letters A-H and numbers 1-8.
- Standalone labels "Surface protection as per standard" /
  "Tolerance as per standard" (extract only the VALUE beside them).
- "Replaces Drg", "Reference Drg No.", "Opposite Hand Drg/Part No.",
  "Drg/Part Designation", "Sheet No..1.. of ...1..", "Original drawing size A4".
- GD&T frames, roughness symbols (√), welding symbols.
- Notes like "GENERAL TOLERANCE TO BE MAINTAINED AS PER TS 12130".

4. BBOX RULE
bbox_1000 = [ymin, xmin, ymax, xmax] on a 0-1000 scale over the FULL
image. [0,0] = top-left, [1000,1000] = bottom-right. Tight around the text.

5. OUTPUT FORMAT
Return ONLY this JSON shape. Separate tolerance from specification.

{
  "candidates": [
    {
      "text": "379±0.5",
      "type": "feature",
      "parameter": "Dimn",
      "specification": "379",
      "tolerance": "±0.5",
      "bbox_1000": [412, 318, 438, 452]
    },
    {
      "text": "2882 3154 01 10",
      "type": "part_number",
      "parameter": "Part No.",
      "specification": "2882 3154 01 10",
      "tolerance": "",
      "bbox_1000": [872, 780, 896, 968]
    }
  ]
}

If nothing is found, return exactly: {"candidates": []}
"""