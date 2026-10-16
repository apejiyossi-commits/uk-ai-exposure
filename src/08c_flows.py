"""Step 8c: entry from outside work vs moves between employers.

Uses the two-quarter longitudinal LFS files, which follow the same people over
two consecutive quarters. A job start in the second quarter is split into:
  - entry: unemployed or inactive in quarter 1, employed in quarter 2 (FLOW UE, NE)
  - move: employed in both quarters, with the current employer < 3 months (EE)
Destinations use the same high-exposure and near-zero unit groups as Step 6.
Rates are a share of everyone in the age group (working age, both quarters).
Output: output/tables/job_starts_by_channel.csv (pooled estimates only).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TAB = ROOT / "output" / "tables"
QUARTER = {"jm": 1, "aj": 2, "js": 3, "od": 4}
pd.set_option("display.width", 220)

# Same destination groups as Step 6, read off the quarterly analysis file
q = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet", columns=["soc20", "high_unit", "zero_unit"])
HIGH = set(q.loc[q["high_unit"] == 1, "soc20"].dropna().astype(int))
ZERO = set(q.loc[q["zero_unit"] == 1, "soc20"].dropna().astype(int))

# --- 1. Read and stack the longitudinal files -------------------------------
frames = []
for f in sorted(RAW.rglob("lgwt*_2q_*_eul.dta")):
    m = re.search(r"_2q_(jm|aj|js|od)(\d{2})_(jm|aj|js|od)(\d{2})_", f.name.lower())
    if not m:
        continue
    _, meta = pyreadstat.read_dta(str(f), metadataonly=True)
    upper = {c.upper(): c for c in meta.column_names}
    weight = next(c for c in meta.column_names if c.upper().startswith("LGWT"))
    cols = [upper[c] for c in ["FLOW", "AGE2", "SOC20M2", "EMPLEN2", "HIQUL22D2"] if c in upper] + [weight]
    d, _ = pyreadstat.read_dta(str(f), usecols=cols)
    d = d.apply(pd.to_numeric, errors="coerce").rename(columns={weight: "LGWT"})
    d.columns = d.columns.str.lower()
    d = d.where(d >= 0)
    d["year"], d["quarter"] = 2000 + int(m.group(4)), QUARTER[m.group(3)]
    d["pair"] = f"{m.group(1)}{m.group(2)}-{m.group(3)}{m.group(4)}"
    frames.append(d)
    print(f"{f.name}: {len(d):,} people, second quarter {d['year'].iat[0]}Q{d['quarter'].iat[0]}")

L = pd.concat(frames, ignore_index=True)
L = L[L["flow"].between(3, 11) & (L["lgwt"] > 0)].copy()       # working age in both quarters
L["qtr"] = L["year"].astype(str) + "Q" + L["quarter"].astype(str)
L["post"] = (L["qtr"] >= "2023Q1").astype(int)
L["id"] = np.arange(len(L))
soc = L["soc20m2"].fillna(-1).astype(int)
L["dest_high"], L["dest_zero"] = soc.isin(HIGH), soc.isin(ZERO)
L["entry"] = L["flow"].isin([6, 9])
L["move"] = (L["flow"] == 3) & (L["emplen2"] == 1)
print(f"\nPairs: {sorted(L['pair'].unique())}")
print(f"Before 2023: {L.loc[L['post'] == 0, 'pair'].nunique()} pairs; from 2023: {L.loc[L['post'] == 1, 'pair'].nunique()} pairs")


def wrate(d, col):
    return 100 * (d[col] * d["lgwt"]).sum() / d["lgwt"].sum()


def channel_did(d, channel, label):
    """Stacked high vs near-zero comparison, as in Step 6, for one channel."""
    ch = d["entry"] | d["move"] if channel == "all" else d[channel]
    d = d.assign(s_high=(ch & d["dest_high"]).astype(int), s_zero=(ch & d["dest_zero"]).astype(int))
    long = pd.concat([d.assign(dh=1, start=d["s_high"]), d.assign(dh=0, start=d["s_zero"])], ignore_index=True)
    long["dh_post"] = long["dh"] * long["post"]
    fit = pf.feols("start ~ dh_post | dh + qtr", data=long, weights="lgwt", vcov={"CRV1": "id"})
    pre, post = d[d["post"] == 0], d[d["post"] == 1]
    return {"sample": label, "channel": channel,
            "high_pre": wrate(pre, "s_high"), "high_post": wrate(post, "s_high"),
            "zero_pre": wrate(pre, "s_zero"), "zero_post": wrate(post, "s_zero"),
            "estimate_pp": 100 * fit.coef()["dh_post"], "se_pp": 100 * fit.se()["dh_post"],
            "p": fit.pvalue()["dh_post"], "people": len(d),
            "starts_high": int(d["s_high"].sum()), "starts_zero": int(d["s_zero"].sum())}


rows = []
for label, d in [("Ages 22-25", L[L["age2"].between(22, 25)]), ("Ages 26-64", L[L["age2"].between(26, 64)])]:
    for channel in ["entry", "move", "all"]:
        rows.append(channel_did(d, channel, label))
out = pd.DataFrame(rows)
print("\nJob starts by channel and destination (% of the age group per quarter; estimate = change in high "
      "minus near-zero gap from 2023, pp):")
print(out.round(3).to_string(index=False))
out.round(4).to_csv(TAB / "job_starts_by_channel.csv", index=False)
print(f"\nSaved {TAB / 'job_starts_by_channel.csv'}")
if out[["starts_high", "starts_zero"]].min().min() < 10:
    print("WARNING: a cell has fewer than 10 job starts; do not commit job_starts_by_channel.csv")