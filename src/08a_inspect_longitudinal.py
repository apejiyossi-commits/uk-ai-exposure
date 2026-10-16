"""Step 8a: what is in the two-quarter longitudinal LFS files?

Lists every LFS .dta file that is not a quarterly person file: the quarters it
covers, its size, and the variables needed to measure flows into jobs (status
in both quarters, occupation, time with employer, age, qualification, weight).
"""
import re
from pathlib import Path

import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WANT = re.compile(r"^(ILODEFR|INECAC|FLOW|SOC20|SC20|SOC10|SC10|EMPLEN|AGE|SEX|HIQUL|"
                  r"CRYOX|STUCUR|GRSSWK|LGWT|PWT)", re.I)

files = [f for f in sorted(RAW.rglob("*.dta")) if not f.name.lower().startswith("lfsp_")]
print(f"{len(files)} non-quarterly .dta files\n")
for f in files:
    sn = f.relative_to(RAW).parts[0]
    _, meta = pyreadstat.read_dta(str(f), metadataonly=True)
    keys = [c for c in meta.column_names if WANT.match(c)]
    print(f"[{sn}] {f.name}: {meta.number_rows:,} rows, {len(meta.column_names)} variables")
    print(f"   file label: {meta.file_label}")
    for c in keys:
        print(f"   {c:18s} {meta.column_names_to_labels.get(c, '')}")
    for c in keys:
        if c.upper().startswith("FLOW"):
            print(f"   {c} codes: {meta.variable_value_labels.get(c, {})}")
    print()