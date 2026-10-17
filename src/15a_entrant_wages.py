"""Step 15a: what happened to the pay and qualifications of young entrants?

If AI cut demand for junior work in exposed occupations, firms that still hire
there might pay less, or hire only stronger candidates. LFS pay questions are
asked in the first and fifth interviews only (weight PIWT).
1. Log hourly pay of employees aged 22-25 who started their job in the last
   3 months, high- vs near-zero-exposure jobs, before vs after 2023. Same for all
   employees aged 22-25, and for new starters aged 26-64 as a comparison.
2. Composition of young starters into high-exposure jobs: share with a degree,
   share born in the UK, share in full-time education, average age.
Output (estimates only): output/tables/entrant_wages.csv, entrant_composition.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "output" / "tables"
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")
pay = "hourpay" if "hourpay" in lfs and lfs["hourpay"].notna().any() else None
if pay is None:
    print("No HOURPAY in the file (rerun src/03_build_lfs.py); using gross weekly pay instead")
    pay = "grsswk"
lfs["lpay"] = np.log(lfs[pay].where(lfs[pay] > 0))
lfs["dest"] = np.select([lfs["high_unit"] == 1, lfs["zero_unit"] == 1], ["high", "zero"], "other")
lfs["soc20i"] = lfs["soc20"].astype("float")


def pay_did(d, label):
    d = d[(d["dest"] != "other") & d["lpay"].notna() & (d["piwt"] > 0)].copy()
    d["high"] = (d["dest"] == "high").astype(int)
    d["hp"] = d["high"] * d["post"]
    f1 = pf.feols("lpay ~ hp + high | qtr", data=d, weights="piwt", vcov={"CRV1": "soc20i"})
    f2 = pf.feols("lpay ~ hp | qtr + soc20i", data=d, weights="piwt", vcov={"CRV1": "soc20i"})
    n = d.groupby(["high", "post"]).size()
    return [{"sample": label, "fixed_effects": fe, "estimate_log_points": f.coef()["hp"], "se": f.se()["hp"],
             "p": f.pvalue()["hp"], "change_%": 100 * (np.exp(f.coef()["hp"]) - 1), "min_cell_n": int(n.min()),
             "people": len(d)} for fe, f in [("quarter", f1), ("quarter + occupation", f2)]]


emp = lfs[lfs["unemp"] == 0]
rows = (pay_did(emp[emp["age"].between(22, 25) & (emp["newjob"] == 1)], "new starters 22-25")
        + pay_did(emp[emp["age"].between(22, 25)], "all employees 22-25")
        + pay_did(emp[emp["age"].between(26, 64) & (emp["newjob"] == 1)], "new starters 26-64"))
w = pd.DataFrame(rows)
print(f"1. Pay ({pay}) in high- vs near-zero-exposure jobs, after vs before 2023 (log points):")
print(w.round(3).to_string(index=False))
w.round(4).to_csv(TAB / "entrant_wages.csv", index=False)
if w["min_cell_n"].min() < 10:
    print("WARNING: a group has fewer than 10 people with pay; do not commit entrant_wages.csv")


def wmean(d, col):
    d = d[d[col].notna()]
    return (d[col] * d["pwt24"]).sum() / d["pwt24"].sum()


y = emp[emp["age"].between(22, 25) & (emp["newjob"] == 1) & (emp["dest"] != "other")]
rows = []
for (dest, post), g in y.groupby(["dest", "post"]):
    r = {"destination": dest, "period": "2023 on" if post else "2021-22", "starters_n": len(g)}
    for col, name in [("degree", "degree_%"), ("uk_born", "uk_born_%"), ("ft_student", "ft_student_%"), ("female", "female_%")]:
        if col in g:
            src = g[g["year"] >= 2022] if col == "degree" else g        # degree is missing in 2021
            r[name] = 100 * wmean(src, col)
    r["mean_age"] = wmean(g, "age")
    rows.append(r)
comp = pd.DataFrame(rows)
print("\n2. Who the young starters are (weighted shares, ages 22-25):")
print(comp.round(1).to_string(index=False))
comp.round(2).to_csv(TAB / "entrant_composition.csv", index=False)
if comp["starters_n"].min() < 10:
    print("WARNING: a group has fewer than 10 starters; do not commit entrant_composition.csv")
print(f"\nSaved {TAB / 'entrant_wages.csv'} and {TAB / 'entrant_composition.csv'}")
