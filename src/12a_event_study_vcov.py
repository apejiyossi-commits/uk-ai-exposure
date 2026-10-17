"""Step 12a: event-study coefficients with their full covariance matrices.

The sensitivity analysis in Step 12c (Rambachan and Roth 2023, "honest DiD")
needs each event study's coefficients and the covariance between them, not
just standard errors. This re-estimates three event studies exactly as before
and saves both. The files hold regression estimates only (no cells, no
microdata), so they can be committed.
1. Unemployment, high vs low exposure minor groups, by quarter (Step 5, ref 2022Q4)
2. Job starts, ages 22-25, high vs near-zero unit groups, by half-year (Step 6, ref 2022H2)
3. Triple difference, 22-25 vs 26-64, by half-year (Poisson, as in Step 7)
Outputs: output/tables/honest_inputs_{unemployment,entry_young,entry_triple}.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "output" / "tables"
warnings.filterwarnings("ignore", message=".*separation.*")
lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")


def save(fit, terms, periods, ref, name, scale=1.0):
    b = fit.coef()[terms].to_numpy() * scale
    V = np.asarray(fit._vcov)
    names = list(fit.coef().index)
    idx = [names.index(t) for t in terms]
    V = V[np.ix_(idx, idx)] * scale ** 2
    out = pd.DataFrame(V, columns=periods)
    out.insert(0, "estimate", b)
    out.insert(0, "period", periods)
    out["reference"] = ref
    out.to_csv(TAB / f"honest_inputs_{name}.csv", index=False)
    se = np.sqrt(np.diag(V))
    print(f"{name}: {len(terms)} coefficients (reference {ref}); "
          f"largest |difference| from reported SEs check: {np.abs(se - fit.se()[terms].to_numpy() * scale).max():.1e}")


# --- 1. Unemployment by quarter ---------------------------------------------
s = lfs[((lfs["high"] == 1) | (lfs["low"] == 1)) & lfs["age"].notna()].copy()
s["minor"], s["age2"] = s["minor"].astype(int), s["age"] ** 2
quarters = sorted(s["qtr"].unique())
event_q = [q for q in quarters if q != "2022Q4"]
for q in event_q:
    s[f"d_{q}"] = ((s["qtr"] == q) & (s["high"] == 1)).astype(int)
fit = pf.feols("unemp ~ " + " + ".join(f"d_{q}" for q in event_q) + " + age + age2 + female | minor + qtr",
               data=s, weights="pwt24", vcov={"CRV1": "minor"})
save(fit, [f"d_{q}" for q in event_q], event_q, "2022Q4", "unemployment", scale=100)


# --- 2 and 3. Job starts by half-year ---------------------------------------
def sample(lo, hi):
    y = lfs[lfs["age"].between(lo, hi)].copy()
    y["start_high"] = ((y["newjob"] == 1) & (y["high_unit"] == 1)).astype(int)
    y["start_zero"] = ((y["newjob"] == 1) & (y["zero_unit"] == 1)).astype(int)
    y["half"] = y["year"].astype(str) + "H" + np.where(y["quarter"] <= 2, "1", "2")
    return y


def stack(y):
    y = y.assign(id=np.arange(len(y)))
    return pd.concat([y.assign(dest_high=1, start=y["start_high"]),
                      y.assign(dest_high=0, start=y["start_zero"])], ignore_index=True)


young = sample(22, 25)
halves = sorted(young["half"].unique())
event_h = [h for h in halves if h != "2022H2"]
long = stack(young)
for h in event_h:
    long[f"d_{h}"] = ((long["half"] == h) & (long["dest_high"] == 1)).astype(int)
fit = pf.feols("start ~ " + " + ".join(f"d_{h}" for h in event_h) + " | dest_high + half",
               data=long, weights="pwt24", vcov={"CRV1": "id"})
save(fit, [f"d_{h}" for h in event_h], event_h, "2022H2", "entry_young", scale=100)

old = sample(26, 64)
both = pd.concat([young.assign(young=1), old.assign(young=0)], ignore_index=True)
long = stack(both)
for h in event_h:
    long[f"t_{h}"] = ((long["half"] == h) & (long["dest_high"] == 1) & (long["young"] == 1)).astype(int)
fit = pf.fepois("start ~ " + " + ".join(f"t_{h}" for h in event_h) + " | dest_high^half + young^dest_high + young^half",
                data=long, weights="pwt24", vcov={"CRV1": "id"})
save(fit, [f"t_{h}" for h in event_h], event_h, "2022H2", "entry_triple")
print(f"\nSaved honest_inputs_*.csv to {TAB}")
