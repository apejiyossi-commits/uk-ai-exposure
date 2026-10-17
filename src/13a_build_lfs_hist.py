"""Step 13a: build a 2011-2019 LFS file, for cyclicality and placebo tests.

Reads the quarterly End User Licence files saved under data/raw_hist/ (never
committed) and keeps what the later steps need: age, sex, ILO status, weight,
SOC 2010 occupation of the main job (unit group), the last job of the
unemployed (as detailed as the file allows) and time with the current employer.
Variable names changed over the decade, so each file's names are found by
pattern and printed; check the printout before using the results.
Output: data/raw_hist/lfs_hist.parquet
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAWH = ROOT / "data" / "raw_hist"
QUARTER = {"jm": 1, "aj": 2, "js": 3, "od": 4}


def pick(cols, patterns):
    """First column (case-insensitive) matching any pattern, in pattern order."""
    upper = {c.upper(): c for c in cols}
    for p in patterns:
        hits = sorted(u for u in upper if re.fullmatch(p, u))
        if hits:
            return upper[hits[-1]]          # e.g. the latest PWT weight
    return None


files = []
for f in sorted(RAWH.rglob("*.dta")):
    m = re.search(r"lfsp_(jm|aj|js|od)(\d{2})", f.name.lower())
    if m and 2011 <= 2000 + int(m.group(2)) <= 2019:
        files.append((2000 + int(m.group(2)), QUARTER[m.group(1)], f))
files.sort()
print(f"Quarterly files found for 2011-2019: {len(files)} (expected 36)")
have = {(y, q) for y, q, _ in files}
gaps = [f"{y}Q{q}" for y in range(2011, 2020) for q in range(1, 5) if (y, q) not in have]
if gaps:
    print(f"  missing quarters: {gaps}")
if not files:
    raise SystemExit(f"No lfsp_*.dta files for 2011-2019 under {RAWH}")

frames, found = [], []
for year, q, f in files:
    _, meta = pyreadstat.read_dta(str(f), metadataonly=True)
    cols = meta.column_names
    names = {
        "age": pick(cols, [r"AGE"]),
        "sex": pick(cols, [r"SEX"]),
        "ilodefr": pick(cols, [r"ILODEFR"]),
        "pwt": pick(cols, [r"PWT\d\d"]),
        "soc10m": pick(cols, [r"SOC10M"]),
        "soc10l": pick(cols, [r"SOC10L(M|J)?", r"SC10LUN"]),          # last job, unit group
        "sc10lmn": pick(cols, [r"SC10LMN", r"SOC10LMN"]),            # last job, minor group
        "emplen": pick(cols, [r"EMPLEN"]),
        "hiqual": pick(cols, [r"HIQUL\d\dD"]),
    }
    need = ["age", "ilodefr", "pwt", "soc10m", "emplen"]
    if any(names[k] is None for k in need):
        missing = [k for k in need if names[k] is None]
        occ = [c for c in cols if re.match(r"(?i)(soc|sc10|sc2k)", c)]
        raise SystemExit(f"{f.name}: cannot find {missing}. Occupation-like columns here: {occ}")
    use = [v for v in names.values() if v]
    d, _ = pyreadstat.read_dta(str(f), usecols=use)
    d = d.apply(pd.to_numeric, errors="coerce").rename(columns={v: k for k, v in names.items() if v})
    d = d.where(d >= 0)
    d["year"], d["quarter"] = year, q
    frames.append(d)
    found.append({"quarter": f"{year}Q{q}", **{k: v or "" for k, v in names.items()}})

found = pd.DataFrame(found)
print("\nVariables used in each file (blank = not in that file):")
print(found.to_string(index=False))

h = pd.concat(frames, ignore_index=True)
h = h[h["age"].between(16, 64) & h["ilodefr"].isin([1, 2]) & (h["pwt"] > 0)].copy()
h["unemp"] = (h["ilodefr"] == 2).astype(int)
h["qtr"] = h["year"].astype(str) + "Q" + h["quarter"].astype(str)
h["soc10"] = h["soc10m"].where(h["unemp"] == 0)
last_unit = h["soc10l"] if "soc10l" in h else pd.Series(np.nan, index=h.index)
last_minor = h["sc10lmn"] if "sc10lmn" in h else pd.Series(np.nan, index=h.index)
last_minor = last_minor.fillna(np.floor(last_unit / 10))
h["minor10"] = np.where(h["unemp"] == 1, last_minor, np.floor(h["soc10m"] / 10))
h["newjob"] = ((h["unemp"] == 0) & (h["emplen"] == 1)).astype(int)
h["female"] = (h["sex"] == 2).astype(int)
h["degree"] = (h["hiqual"] == 1).astype(float).where(h["hiqual"].notna()) if "hiqual" in h else np.nan
keep = ["year", "quarter", "qtr", "age", "female", "pwt", "unemp", "soc10", "minor10", "newjob", "degree"]
h = h[keep].reset_index(drop=True)

valid = set(h.loc[h["unemp"] == 0, "soc10"].dropna() // 10)
u = h[h["unemp"] == 1]
print(f"\nRows kept (ages 16-64, employed or unemployed): {len(h):,}; per quarter "
      f"{h.groupby('qtr').size().min():,} to {h.groupby('qtr').size().max():,}")
print(f"Employed with a SOC 2010 unit group: {h.loc[h['unemp'] == 0, 'soc10'].notna().mean():.1%}")
print(f"Unemployed with a valid last-job minor group: {u['minor10'].isin(valid).mean():.1%}")
print(f"Job starts (< 3 months with employer) among the employed: "
      f"{h.loc[h['unemp'] == 0, 'newjob'].mean():.1%}")
rate = h.groupby("qtr").apply(lambda d: 100 * (d["unemp"] * d["pwt"]).sum() / d["pwt"].sum(), include_groups=False)
print(f"Unemployment rate, 16-64: {rate.iloc[0]:.1f}% in {rate.index[0]}, {rate.iloc[-1]:.1f}% in {rate.index[-1]} "
      f"(the ONS series is roughly 8% in 2011 and 4% in 2019)")
h.to_parquet(RAWH / "lfs_hist.parquet")
print(f"\nSaved {RAWH / 'lfs_hist.parquet'} (not committed: data/raw_hist is ignored)")
