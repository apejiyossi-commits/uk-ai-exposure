"""Step 8b: are young graduates being pushed into non-graduate jobs?

Tests the underemployment reading of Step 7b, from 2022 (when the degree
variable starts), comparing graduates aged 22-25 with graduates aged 26-35.
  1. Non-graduate job: employed in SOC 2020 major groups 4-9 (the usual ONS
     split: groups 1-3 are managers, professionals and associate professionals).
  2. Pay: among full-time job starters aged 22-25, the graduate pay premium
     within each destination, with quarter fixed effects so inflation drops
     out. Pay is asked only in interview waves 1 and 5, so samples are small.
Output: output/tables/graduate_underemployment.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "output" / "tables"
pd.set_option("display.width", 220)

lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")
e = lfs[(lfs["year"] >= 2022) & (lfs["unemp"] == 0) & lfs["soc20m"].notna()].copy()
e["nongrad_job"] = (e["soc20m"] // 1000 >= 4).astype(int)
e["young"] = e["age"].between(22, 25).astype(int)
e["yp"] = e["young"] * e["post"]


def wrate(d, col, w="pwt24"):
    return 100 * (d[col] * d[w]).sum() / d[w].sum()


# --- 1. Share of employed people in non-graduate jobs ------------------------
groups = {"Graduates 22-25": (e["degree"] == 1) & (e["young"] == 1),
          "Graduates 26-35": (e["degree"] == 1) & e["age"].between(26, 35),
          "Non-graduates 22-25": (e["degree"] == 0) & (e["young"] == 1)}
share = {}
for label, m in groups.items():
    d = e[m]
    share[(label, "all employed")] = d.groupby("year")[["nongrad_job", "pwt24"]].apply(lambda x: wrate(x, "nongrad_job"))
    s = d[d["newjob"] == 1]
    share[(label, "job starters")] = s.groupby("year")[["nongrad_job", "pwt24"]].apply(lambda x: wrate(x, "nongrad_job"))
share = pd.DataFrame(share)
print("1. Share in non-graduate jobs (SOC major groups 4-9), weighted %, by year:")
print(share.round(1).to_string())

# --- 2. Difference-in-differences: young vs older graduates ------------------
grads = e[(e["degree"] == 1) & e["age"].between(22, 35)]
rows = []
for label, d in [("All employed graduates", grads), ("Graduate job starters", grads[grads["newjob"] == 1])]:
    fit = pf.feols("nongrad_job ~ yp | young + qtr", data=d, weights="pwt24", vcov="hetero")
    b, se = 100 * fit.coef()["yp"], 100 * fit.se()["yp"]
    base = wrate(d[(d["young"] == 1) & (d["post"] == 0)], "nongrad_job")
    rows.append({"comparison": f"{label}: 22-25 vs 26-35, non-graduate job", "estimate_pp": b, "se_pp": se,
                 "p": fit.pvalue()["yp"], "base_2022_%": base, "people": len(d)})

# --- 3. Graduate pay premium among young full-time job starters --------------
p = e[(e["young"] == 1) & (e["newjob"] == 1) & (e["parttime"] == 0) & (e["grsswk"] > 0)
      & (e["piwt"] > 0)].copy()
p["lpay"] = np.log(p["grsswk"])
for label, m in [("High-exposure jobs", p["high_unit"] == 1), ("Near-zero jobs", p["zero_unit"] == 1),
                 ("All jobs", p["grsswk"] > 0)]:
    d = p[m]
    fit = pf.feols("lpay ~ degree | qtr", data=d, weights="piwt", vcov="hetero")
    b, se = fit.coef()["degree"], fit.se()["degree"]
    rows.append({"comparison": f"Graduate pay premium, full-time starters 22-25, {label.lower()} (%)",
                 "estimate_pp": 100 * (np.exp(b) - 1), "se_pp": 100 * se, "p": fit.pvalue()["degree"],
                 "base_2022_%": np.nan, "people": len(d), "graduates": int(d["degree"].sum())})
out = pd.DataFrame(rows)
print("\n2-3. Estimates (pay rows: % premium of graduates over non-graduates, same destination and quarter):")
print(out.round(3).to_string(index=False))
out.round(4).to_csv(TAB / "graduate_underemployment.csv", index=False)
print(f"\nSaved {TAB / 'graduate_underemployment.csv'}")
if out["people"].min() < 30:
    print("NOTE: a pay row rests on fewer than 30 people; read it as indicative only")
if out["graduates"].min() < 10:
    print("WARNING: a pay row has fewer than 10 graduates; do not commit graduate_underemployment.csv")