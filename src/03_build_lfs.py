"""Step 3: build the LFS analysis file, January-March 2021 to the latest quarter.

Keeps people aged 16-64 who are employed or ILO unemployed, gives each an
occupation, merges observed exposure, and saves data/raw/lfs_analysis.parquet
(data/raw is never committed).

The End User Licence files code the main job at SOC2020 unit-group level
(4 digits, SOC20M) but the last job of the unemployed only at minor-group
level (3 digits, SC20LMN). So treatment groups are defined at minor-group
level, where employed and unemployed can be compared; unit-group exposure is
kept for analyses of the employed only.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived"

QUARTER = {"jm": 1, "aj": 2, "js": 3, "od": 4}
KEEP = ["AGE", "SEX", "ILODEFR", "PWT24", "SOC20M", "SC20MMN", "SC20LMN", "WNLEFT11",
        "EMPLEN", "FTPTWK", "PUBLICR", "HIQUL22D", "GOVTOF2", "THISWV", "GRSSWK"]


def wquantile(x, w, q):
    """Weighted quantile: smallest x with cumulative weight share >= q.
    Rows with a missing value or weight are skipped (numpy, unlike pandas,
    does not skip missing values, so one missing weight would spoil it)."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    order = np.argsort(x)
    x, w = x[order], w[order]
    return x[np.searchsorted(np.cumsum(w) / w.sum(), q)]


# --- 1. Read and stack the quarterly person files ---------------------------
frames = []
for f in sorted(RAW.rglob("lfsp_*_eul_*.dta")):
    m = re.search(r"lfsp_(jm|aj|js|od)(\d{2})_", f.name)
    year, q = 2000 + int(m.group(2)), QUARTER[m.group(1)]
    _, meta = pyreadstat.read_dta(str(f), metadataonly=True)
    piwt = next((c for c in meta.column_names if c.startswith("PIWT")), None)
    cols = [c for c in KEEP if c in meta.column_names] + ([piwt] if piwt else [])
    df, _ = pyreadstat.read_dta(str(f), usecols=cols)
    # Force every column to a number: some arrive as generic 'object' columns,
    # which silently break comparisons and arithmetic later on
    df = df.apply(pd.to_numeric, errors="coerce")
    if piwt:
        df = df.rename(columns={piwt: "PIWT"})  # income weight; name changes by year
    df["year"], df["quarter"] = year, q
    frames.append(df)

lfs = pd.concat(frames, ignore_index=True)
lfs.columns = lfs.columns.str.lower()
lfs = lfs.sort_values(["year", "quarter"]).reset_index(drop=True)
lfs["qtr"] = lfs["year"].astype(str) + "Q" + lfs["quarter"].astype(str)
lfs["t"] = (lfs["year"] - 2021) * 4 + lfs["quarter"] - 1      # 0 = 2021Q1

# Negative codes (-9 does not apply, -8 no answer) become missing
for c in lfs.columns.drop(["year", "quarter", "qtr", "t"]):
    if pd.api.types.is_numeric_dtype(lfs[c]):
        lfs[c] = lfs[c].where(lfs[c] >= 0)

# --- 2. Sample and occupations ---------------------------------------------
lfs = lfs[lfs["age"].between(16, 64) & lfs["ilodefr"].isin([1, 2])].copy()
n0 = len(lfs)
lfs = lfs[lfs["pwt24"] > 0].copy()
print(f"Dropped {n0 - len(lfs):,} rows with a missing or zero person weight")
lfs["unemp"] = (lfs["ilodefr"] == 2).astype(int)

# Unemployed people with no valid last-job minor group cannot be placed in
# an exposure group; report who they are, then set the code to missing.
valid = set((lfs.loc[lfs["unemp"] == 0, "soc20m"] // 10).dropna().astype(int))
last = lfs.loc[lfs["unemp"] == 1, "sc20lmn"]
nolast = last.isna() | ~last.isin(valid)
print(f"Unemployed: {len(last):,}; with no valid last-job minor group: {nolast.mean():.1%}")
invalid = last.notna() & ~last.isin(valid)
if invalid.any():
    print(f"  non-missing but invalid codes: {last[invalid].value_counts().head(8).to_dict()}")
if nolast.any() and "wnleft11" in lfs:
    why = {9: "never had a paid job", 8: "left 5+ years ago"}
    when = (lfs.loc[last[nolast].index, "wnleft11"]
            .map(lambda v: why.get(v, "other or missing")).value_counts(normalize=True))
    print(f"  of these: {when.round(3).to_dict()}")
lfs.loc[(lfs["unemp"] == 1) & ~lfs["sc20lmn"].isin(valid), "sc20lmn"] = np.nan

lfs["soc20"] = lfs["soc20m"].where(lfs["unemp"] == 0).astype("Int64")
lfs["minor"] = np.where(lfs["unemp"] == 1, lfs["sc20lmn"], lfs["soc20m"] // 10)
lfs["minor"] = pd.Series(lfs["minor"], index=lfs.index).astype("Int64")

emp_mask = (lfs["unemp"] == 0) & lfs["soc20m"].notna() & lfs["sc20mmn"].notna()
agree = (lfs.loc[emp_mask, "soc20m"] // 10 == lfs.loc[emp_mask, "sc20mmn"]).mean()
print(f"Check: unit code / 10 equals reported minor group for {agree:.1%} of employed")

# --- 3. Exposure: unit groups, then minor groups ----------------------------
exp20 = pd.read_csv(DER / "exposure_soc2020_uk.csv")
exp20["soc20"] = exp20["soc20"].astype("Int64")
lfs = lfs.merge(exp20.rename(columns={"exposure": "exp_unit", "imputed": "imp_unit"})
                [["soc20", "exp_unit", "imp_unit"]], on="soc20", how="left")

# Minor-group exposure = mean of unit-group exposure, weighted by 2022
# employment (the last year before ChatGPT's release on 30 November 2022)
base = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0) & lfs["exp_unit"].notna()]
g = (base.assign(wx=base["pwt24"] * base["exp_unit"], wi=base["pwt24"] * base["imp_unit"])
         .groupby("minor")[["wx", "wi", "pwt24"]].sum())
minor = pd.DataFrame({"exp_minor": g["wx"] / g["pwt24"],
                      "imp_minor": g["wi"] / g["pwt24"],
                      "emp_2022": g["pwt24"]}).reset_index()
minor.to_csv(DER / "exposure_soc2020_minor_uk.csv", index=False)
lfs = lfs.merge(minor[["minor", "exp_minor", "imp_minor"]], on="minor", how="left")

# --- 4. Treatment groups (cutoffs from 2022 employment) ---------------------
b = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0) & lfs["exp_minor"].notna()]
p25 = wquantile(b["exp_minor"], b["pwt24"], 0.25)
p75 = wquantile(b["exp_minor"], b["pwt24"], 0.75)
has = lfs["exp_minor"].notna()
lfs["high"] = (has & (lfs["exp_minor"] >= p75)).astype(int)
lfs["low"] = (has & (lfs["exp_minor"] <= p25)).astype(int)
lfs["post"] = (lfs["t"] >= 8).astype(int)                     # from 2023Q1

# Unit-group groups, for analyses of the employed only
bu = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0) & lfs["exp_unit"].notna()]
p75u = wquantile(bu["exp_unit"], bu["pwt24"], 0.75)
lfs["high_unit"] = (lfs["exp_unit"] >= p75u).astype(int)
lfs["zero_unit"] = (lfs["exp_unit"] < 0.01).astype(int)

# --- 5. Outcomes and covariates ---------------------------------------------
lfs["newjob"] = ((lfs["unemp"] == 0) & (lfs["emplen"] == 1)).astype(int)  # < 3 months with employer
lfs["female"] = (lfs["sex"] == 2).astype(int)
lfs["degree"] = (lfs["hiqul22d"] == 1).astype(int)
lfs["london"] = (lfs["govtof2"] == 8).astype(int)
lfs["public"] = (lfs["publicr"] == 2).astype(int)
lfs["parttime"] = (lfs["ftptwk"] == 2).astype(int)

lfs.to_parquet(RAW / "lfs_analysis.parquet", index=False)

# --- 6. Checks to read ------------------------------------------------------
def wrate(d, col):
    return 100 * (d[col] * d["pwt24"]).sum() / d["pwt24"].sum()

print(f"\nRows kept (16-64, employed or unemployed): {len(lfs):,}")
by_q = lfs.groupby("qtr").agg(n=("unemp", "size"), unemployed=("unemp", "sum"))
by_q["unemp_rate_%"] = (lfs.groupby("qtr")[["unemp", "pwt24"]]
                        .apply(lambda d: wrate(d, "unemp")).round(2))
print("\nBy quarter (weighted unemployment rate, ages 16-64):")
print(by_q.to_string())

u = lfs[lfs["unemp"] == 1]
e = lfs[lfs["unemp"] == 0]
print(f"\nEmployed with unit-group exposure: {e['exp_unit'].notna().mean():.1%}")
print(f"Employed with minor-group exposure: {e['exp_minor'].notna().mean():.1%}")
print(f"Unemployed with a last-job minor group: {u['minor'].notna().mean():.1%}")
print(f"Unemployed with minor-group exposure:  {u['exp_minor'].notna().mean():.1%}")

print(f"\nMinor-group cutoffs (2022 employment): 25th pct {p25:.3f}, 75th pct {p75:.3f}")
b = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0)]
print(f"2022 employment share: high {wrate(b, 'high'):.1f}%, low {wrate(b, 'low'):.1f}% "
      f"(lumpy because whole minor groups fall on one side)")
print(f"Unit-group 75th pct: {p75u:.3f}; 2022 employment share below 0.01: "
      f"{wrate(b, 'zero_unit'):.1f}% (US comparison: about 30% at zero)")
print(f"\nMinor groups: {minor['minor'].nunique()}; most exposed:")
print(minor.nlargest(10, "exp_minor").round(3).to_string(index=False))
print(f"\nSaved {RAW / 'lfs_analysis.parquet'} (not committed) and "
      f"{DER / 'exposure_soc2020_minor_uk.csv'}")