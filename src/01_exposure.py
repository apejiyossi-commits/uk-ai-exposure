"""Step 1: download Anthropic's observed-exposure measure and check it."""
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

# Project root = the folder above src/, so the script works from anywhere
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "job_exposure.csv"

# Download the file from Anthropic's Economic Index dataset on Hugging Face
path = hf_hub_download(
    repo_id="Anthropic/EconomicIndex",
    filename="labor_market_impacts/job_exposure.csv",
    repo_type="dataset",
)

# Read occupation codes as text so codes like "11-1011" stay intact
exp = pd.read_csv(path, dtype={"occ_code": str})

# Checks: one row per occupation, exposure between 0 and 1
assert exp["occ_code"].is_unique, "duplicate occupation codes"
assert exp["observed_exposure"].between(0, 1).all(), "exposure outside 0-1"

share_zero = (exp["observed_exposure"] == 0).mean()
print(f"Occupations: {len(exp)}")
print(f"Share with zero exposure: {share_zero:.1%}")
print("\nTen most exposed occupations:")
print(exp.nlargest(10, "observed_exposure")[["title", "observed_exposure"]].to_string(index=False))

OUT.parent.mkdir(parents=True, exist_ok=True)
exp.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")