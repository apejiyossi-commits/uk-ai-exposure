"""Step 15b: do the effects last? Evidence from the five-quarter LFS panels.

The five-quarter longitudinal files follow the same people for a year (waves
1 to 5). Two questions, using the Step 6 destination groups:
1. Entry a year on: of people aged 22-25 who were not employed in wave 1, what
   share are working in a high-exposure (or near-zero) job in wave 5?
2. Survival of new jobs: of people aged 22-25 who had been with their employer
   for under 6 months in wave 1, what share are still employed in wave 5, and
   still in the same exposure group?
Before vs after: by the date of wave 5 for question 1 (from 2023Q1) and of
wave 1 for question 2 (jobs started from 2023).
Reads lgwt*_5q_*.dta under data/raw and data/raw_hist (2021 onwards, SOC 2020).
Output (estimates only): output/tables/persistence.csv
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "output" / "tables"
QUARTER = {"jm": 1, "aj": 2, "js": 3, "od": 4}
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

q = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet", columns=["soc20", "high_unit", "zero_unit"])
HIGH = set(q.loc[q["high_unit"] == 1, "soc20"].dropna().astype(int))
ZERO = set(q.loc[q["zero_unit"] == 1, "soc20"].dropna().astype(int))


def pick(upper, patterns):
    for p in patterns:
        hits = sorted(u for u in upper if re.fullmatch(p, u))
        if hits:
            return upper[hits[-1]]
    return None


files = sorted({f for d in ["raw", "raw_hist"] for f in (ROOT / "data" / d).rglob("lgwt*_5q_*.dta")})
frames, found = [], []
for f in files:
    m = re.search(r"5q_(jm|aj|js|od)(\d{2})_(jm|aj|js|od)(\d{2})", f.name.lower())
    if not m or 2000 + int(m.group(2)) < 2021:
        continue                                    # SOC 2020 coding starts in 2021
    _, meta = pyreadstat.read_dta(str(f), metadataonly=True)
    upper = {c.upper(): c for c in meta.column_names}
    names = {"w": pick(upper, [r"LGWT\d\d"]), "age1": pick(upper, [r"AGE1", r"AGE"])}
    for k in (1, 5):
        names[f"ilo{k}"] = pick(upper, [rf"ILODEFR{k}"])
        names[f"soc{k}"] = pick(upper, [rf"SOC20M{k}", rf"SOC20{k}"])
        names[f"emplen{k}"] = pick(upper, [rf"EMPLEN{k}"])
    if any(v is None for v in names.values()):
        missing = [k for k, v in names.items() if v is None]
        print(f"{f.name}: missing {missing}; columns like these exist: "
              f"{[c for c in meta.column_names if re.match(r'(?i)(ilo|soc|emplen|age|lgwt)', c)][:30]}")
        continue
    d, _ = pyreadstat.read_dta(str(f), usecols=list(names.values()))
    d = d.apply(pd.to_numeric, errors="coerce").rename(columns={v: k for k, v in names.items()})
    d = d.where(d >= 0)
    d["y1"], d["q1"] = 2000 + int(m.group(2)), QUARTER[m.group(1)]
    d["y5"], d["q5"] = 2000 + int(m.group(4)), QUARTER[m.group(3)]
    d["panel"] = f"{m.group(1)}{m.group(2)}-{m.group(3)}{m.group(4)}"
    frames.append(d)
    found.append({"panel": d["panel"].iat[0], **names, "people": len(d)})

if not frames:
    raise SystemExit("No usable five-quarter files found (lgwt*_5q_*.dta, 2021 onwards)")
print(pd.DataFrame(found).to_string(index=False))
P = pd.concat(frames, ignore_index=True)
P = P[(P["w"] > 0) & P["age1"].between(22, 25)].copy()
P["qtr1"] = P["y1"].astype(str) + "Q" + P["q1"].astype(str)
P["qtr5"] = P["y5"].astype(str) + "Q" + P["q5"].astype(str)
s1, s5 = P["soc1"].fillna(-1).astype(int), P["soc5"].fillna(-1).astype(int)
P["emp1"], P["emp5"] = P["ilo1"] == 1, P["ilo5"] == 1
P["high1"], P["zero1"] = s1.isin(HIGH) & P["emp1"], s1.isin(ZERO) & P["emp1"]
P["high5"], P["zero5"] = s5.isin(HIGH) & P["emp5"], s5.isin(ZERO) & P["emp5"]
print(f"\nPanels: {P['panel'].nunique()} ({P['panel'].min()} ... {P['panel'].max()}); people aged 22-25 in wave 1: {len(P):,}")


def stacked(d, a, b, timecol, label):
    """High vs near-zero, before vs after: outcome a for the high row, b for the zero row."""
    d = d.assign(id=np.arange(len(d)))
    long = pd.concat([d.assign(dh=1, y=d[a].astype(int)), d.assign(dh=0, y=d[b].astype(int))], ignore_index=True)
    long["post"] = (long[timecol] >= "2023Q1").astype(int)
    long["dh_post"] = long["dh"] * long["post"]
    f = pf.feols("y ~ dh_post | dh + " + timecol, data=long, weights="w", vcov={"CRV1": "id"})
    pre = d[d[timecol] < "2023Q1"]
    rate = lambda x, c: 100 * (x[c] * x["w"]).sum() / x["w"].sum()
    return {"question": label, "high_rate_pre_%": rate(pre, a), "zero_rate_pre_%": rate(pre, b),
            "estimate_pp": 100 * f.coef()["dh_post"], "se_pp": 100 * f.se()["dh_post"], "p": f.pvalue()["dh_post"],
            "people": len(d), "n_high": int(d[a].sum()), "n_zero": int(d[b].sum())}


rows = []
ne = P[~P["emp1"]]
rows.append(stacked(ne, "high5", "zero5", "qtr5", "not employed in wave 1 -> in that job type in wave 5"))
new = P[P["emp1"] & (P["emplen1"] <= 2)].copy()           # with employer under 6 months in wave 1
for grp in ["high", "zero"]:
    new[f"{grp}_job"] = new[f"{grp}1"]
nj = new[new["high_job"] | new["zero_job"]].copy()
nj["still_emp"] = nj["emp5"]
nj["same_type"] = (nj["high_job"] & nj["high5"]) | (nj["zero_job"] & nj["zero5"])
for outcome, label in [("still_emp", "recent job in wave 1 -> employed in wave 5"),
                       ("same_type", "recent job in wave 1 -> same job type in wave 5")]:
    d = nj.assign(id=np.arange(len(nj)), high=nj["high_job"].astype(int), post=(nj["qtr1"] >= "2023Q1").astype(int))
    d["hp"] = d["high"] * d["post"]
    d["y"] = d[outcome].astype(int)
    f = pf.feols("y ~ hp + high | qtr1", data=d, weights="w", vcov="hetero")
    rate = lambda x: 100 * (x["y"] * x["w"]).sum() / x["w"].sum()
    rows.append({"question": label, "high_rate_pre_%": rate(d[(d["high"] == 1) & (d["post"] == 0)]),
                 "zero_rate_pre_%": rate(d[(d["high"] == 0) & (d["post"] == 0)]),
                 "estimate_pp": 100 * f.coef()["hp"], "se_pp": 100 * f.se()["hp"], "p": f.pvalue()["hp"],
                 "people": len(d), "n_high": int(d["high"].sum()), "n_zero": int((1 - d["high"]).sum())})
out = pd.DataFrame(rows)
print("\nHigh vs near-zero exposure, after vs before 2023 (pp):")
print(out.round(3).to_string(index=False))
out.round(4).to_csv(TAB / "persistence.csv", index=False)
if out[["n_high", "n_zero"]].min().min() < 10:
    print("WARNING: a group has fewer than 10 people; do not commit persistence.csv")
print(f"\nSaved {TAB / 'persistence.csv'}")
