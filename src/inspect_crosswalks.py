"""Print the first rows of every sheet in each crosswalk file."""
from pathlib import Path

import pandas as pd

EXT = Path(__file__).resolve().parents[1] / "data" / "external"

for f in sorted(EXT.rglob("*.xls*")):
    print("=" * 80)
    print(f.name)
    sheets = pd.read_excel(f, sheet_name=None, header=None, nrows=12)
    for name, df in sheets.items():
        print(f"--- sheet: {name}")
        print(df.to_string(max_colwidth=40))