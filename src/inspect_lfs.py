"""List LFS variables: full list for one quarter, key variables across quarters."""
import re
from pathlib import Path

import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
files = sorted(RAW.rglob("lfsp_*.dta"))

# Full variable list (names, labels, value labels) for January-March 2023
ref = next(f for f in files if "jm23" in f.name)
_, meta = pyreadstat.read_dta(str(ref), metadataonly=True)
with open(OUT / "lfs_vars_jm23.txt", "w", encoding="utf-8") as fh:
    fh.write(f"{ref.name}: {meta.number_rows} rows, {len(meta.column_names)} variables\n\n")
    for name, label in zip(meta.column_names, meta.column_labels):
        fh.write(f"{name}\t{label}\n")
        vl = meta.variable_value_labels.get(name)
        if vl and len(vl) <= 15:
            fh.write(f"\t\t{vl}\n")

# Which key variables exist in each quarter (names change over time)
pattern = re.compile(
    r"^(SOC|SC20|SC10|ILODEF|INECAC|AGE$|SEX$|PWT|PIWT|CONMON|CONSEY|EMPMON|"
    r"HIQUL|GOVTOF|PUBLICR|FTPT|GRSSWK|THISWV|REFWK|QUOTA|IOUT)", re.I)
with open(OUT / "lfs_key_vars_by_quarter.txt", "w", encoding="utf-8") as fh:
    for f in files:
        _, m = pyreadstat.read_dta(str(f), metadataonly=True)
        keys = [c for c in m.column_names if pattern.match(c)]
        fh.write(f"{f.name} ({m.number_rows} rows):\n  {', '.join(keys)}\n\n")
print("Wrote output/lfs_vars_jm23.txt and output/lfs_key_vars_by_quarter.txt")