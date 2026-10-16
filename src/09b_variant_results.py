"""Step 9b: do the headline results survive the two exposure variants?

1. Alternative crosswalk: rebuild the treatment groups from the alternative
   exposure and rerun the unemployment and young-worker job-entry estimates.
2. Automation vs augmentation: split the high-exposure destination jobs at the
   employment-weighted median automation share and rerun job entry for each half.
Output: output/tables/variants.csv (pooled estimates only).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
TAB = ROOT / "output" / "tables"
pd.set_option("display.width", 220)

lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")
lfs["age2"] = lfs["age"] ** 2
lfs["soc20f"] = lfs["soc20"].astype("float")
alt = pd.read_csv(DER / "exposure_soc2020_uk_alt.csv")
auto = pd.read_csv(DER / "automation_soc2020_uk.csv")
titles = pd.read_csv(DER / "exposure_soc2020_uk.csv")[["soc20", "title"]]
lfs["exp_unit_alt"] = lfs["soc20f"].map(alt.set_index("soc20")["exposure"])
lfs["automation"] = lfs["soc20f"].map(auto.set_index("soc20")["automation"])


def wquantile(x, w, q):
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    order = np.argsort(x)
    x, w = x[order], w[order]
    return x[np.searchsorted(np.cumsum(w) / w.sum(), q)]


def wrate(d, col):
    return 100 * (d[col] * d["pwt24"]).sum() / d["pwt24"].sum()


# --- Alternative treatment groups, built exactly as in Step 3 ---------------
emp22 = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0)]
b = emp22[emp22["exp_unit_alt"].notna()]
g = b.assign(wx=b["pwt24"] * b["exp_unit_alt"]).groupby("minor")[["wx", "pwt24"]].sum()
lfs["exp_minor_alt"] = lfs["minor"].map(g["wx"] / g["pwt24"])
emp22 = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0)]
p25a, p75a = (wquantile(emp22["exp_minor_alt"], emp22["pwt24"], q) for q in (.25, .75))
p75ua = wquantile(emp22["exp_unit_alt"], emp22["pwt24"], .75)
has = lfs["exp_minor_alt"].notna()
lfs["high_alt"] = has & (lfs["exp_minor_alt"] >= p75a)
lfs["low_alt"] = has & (lfs["exp_minor_alt"] <= p25a)
lfs["high_unit_alt"] = lfs["exp_unit_alt"] >= p75ua
lfs["zero_unit_alt"] = lfs["exp_unit_alt"] < 0.01

emp22 = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0)]
e22 = emp22[emp22["exp_unit"].notna() & emp22["exp_unit_alt"].notna()]
print("1. Alternative crosswalk")
print(f"   Correlation with main exposure, across 2022 workers: "
      f"{np.corrcoef(e22['exp_unit'], e22['exp_unit_alt'])[0, 1]:.3f}")
for main_col, alt_col, name in [("high_unit", "high_unit_alt", "high-exposure"),
                                ("zero_unit", "zero_unit_alt", "near-zero")]:
    m, a = e22[main_col] == 1, e22[alt_col]
    agree = ((m & a) * e22["pwt24"]).sum() / ((m | a) * e22["pwt24"]).sum()
    print(f"   {name} jobs: {100 * agree:.0f}% of 2022 employment in either version is in both")


def unemp_did(hi, lo, label):
    s = lfs[(hi | lo) & lfs["age"].notna() & lfs["minor"].notna()].copy()
    s["high"], s["minor"] = hi[s.index].astype(int), s["minor"].astype(int)
    s["high_post"] = s["high"] * s["post"]
    fit = pf.feols("unemp ~ high_post + age + age2 + female | minor + qtr", data=s,
                   weights="pwt24", vcov={"CRV1": "minor"})
    bse, se = 100 * fit.coef()["high_post"], 100 * fit.se()["high_post"]
    base = wrate(s[(s["high"] == 1) & (s["post"] == 0)], "unemp")
    return {"outcome": "unemployment, ages 16-64", "version": label, "estimate_pp": bse, "se_pp": se,
            "p": fit.pvalue()["high_post"], "change_%": 100 * bse / base, "people": len(s)}


def entry_did(dh, dz, lo, hi, label):
    y = lfs[lfs["age"].between(lo, hi)].copy()
    y["id"] = np.arange(len(y))
    y["s_high"] = ((y["newjob"] == 1) & dh[y.index]).astype(int)
    y["s_zero"] = ((y["newjob"] == 1) & dz[y.index]).astype(int)
    long = pd.concat([y.assign(dh=1, start=y["s_high"]), y.assign(dh=0, start=y["s_zero"])], ignore_index=True)
    long["dh_post"] = long["dh"] * long["post"]
    fit = pf.feols("start ~ dh_post | dh + qtr", data=long, weights="pwt24", vcov={"CRV1": "id"})
    bse, se = 100 * fit.coef()["dh_post"], 100 * fit.se()["dh_post"]
    base = wrate(y[y["year"] == 2022], "s_high")
    return {"outcome": f"job starts, ages {lo}-{hi}", "version": label, "estimate_pp": bse, "se_pp": se,
            "p": fit.pvalue()["dh_post"], "change_%": 100 * bse / base, "people": len(y),
            "starts_high": int(y["s_high"].sum()), "starts_zero": int(y["s_zero"].sum())}


hi_m, lo_m = lfs["high"] == 1, lfs["low"] == 1
hu_m, zu_m = lfs["high_unit"] == 1, lfs["zero_unit"] == 1
rows = [unemp_did(hi_m, lo_m, "main crosswalk"),
        unemp_did(lfs["high_alt"], lfs["low_alt"], "alternative crosswalk"),
        entry_did(hu_m, zu_m, 22, 25, "main crosswalk"),
        entry_did(lfs["high_unit_alt"], lfs["zero_unit_alt"], 22, 25, "alternative crosswalk")]

# --- 2. Automation vs augmentation within high-exposure jobs ----------------
h22 = emp22[(emp22["high_unit"] == 1) & emp22["automation"].notna()]
med = wquantile(h22["automation"], h22["pwt24"], .5)
hi_auto = hu_m & (lfs["automation"] >= med)
hi_aug = hu_m & (lfs["automation"] < med)
print(f"\n2. Automation share of Claude usage in high-exposure jobs: weighted median {med:.2f}")
covered = (h22["pwt24"].sum() / emp22.loc[emp22["high_unit"] == 1, "pwt24"].sum())
print(f"   High-exposure 2022 employment with an automation share: {100 * covered:.0f}%")
for name, mask in [("automation-heavy", lfs["automation"] >= med), ("augmentation-heavy", lfs["automation"] < med)]:
    top = (h22[mask[h22.index]].groupby("soc20f")["pwt24"].sum().nlargest(8).index)
    show = titles.set_index("soc20").loc[top.astype(int), "title"].tolist()
    print(f"   Largest {name} occupations: {'; '.join(show)}")
rows += [entry_did(hi_auto, zu_m, 22, 25, "high exposure, automation-heavy"),
         entry_did(hi_aug, zu_m, 22, 25, "high exposure, augmentation-heavy"),
         entry_did(hi_auto, zu_m, 26, 64, "high exposure, automation-heavy"),
         entry_did(hi_aug, zu_m, 26, 64, "high exposure, augmentation-heavy")]

out = pd.DataFrame(rows)
out[["starts_high", "starts_zero"]] = out[["starts_high", "starts_zero"]].astype("Int64")
print("\nEstimates (after vs before 2023; job starts: high minus near-zero, pp of the age group):")
print(out.round(3).to_string(index=False))
out.round(4).to_csv(TAB / "variants.csv", index=False)
print(f"\nSaved {TAB / 'variants.csv'}")
if out[["starts_high", "starts_zero"]].min().min() < 10:
    print("WARNING: a cell has fewer than 10 job starts; do not commit variants.csv")