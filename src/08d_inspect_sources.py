"""Step 8d: look at the two new data sources before building on them.

1. Anthropic Economic Index task files (automation vs augmentation), downloaded
   to data/external/aei/.
2. The ONS proportional ISCO-08 to SOC 2010 conversion, saved by hand as
   data/external/onetsocconversion.xlsx.
Prints file lists, columns and first rows only.
"""
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
AEI = EXT / "aei"
AEI.mkdir(parents=True, exist_ok=True)
REPO = "Anthropic/EconomicIndex"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

files = HfApi().list_repo_files(REPO, repo_type="dataset")
keys = ("automation", "augment", "collaboration", "task_pct", "onet_task", "soc_structure")
print("Economic Index files about automation, augmentation or tasks:")
for f in files:
    if any(k in f.lower() for k in keys):
        print("  ", f)

for name in ["release_2025_03_27/automation_vs_augmentation_by_task.csv",
             "release_2025_03_27/task_pct_v2.csv",
             "release_2025_03_27/onet_task_statements.csv"]:
    path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=AEI)
    d = pd.read_csv(path)
    print(f"\n{name}: {len(d):,} rows, columns {list(d.columns)}")
    print(d.head(3).to_string())

ons = EXT / "onetsocconversion.xlsx"
if ons.exists():
    for sheet, d in pd.read_excel(ons, sheet_name=None, header=None, nrows=10).items():
        print(f"\nONS sheet '{sheet}' (first 10 rows):")
        print(d.to_string())
else:
    print(f"\nONS file not found at {ons}: download it and move it there first")