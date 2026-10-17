"""Step 13b: is the post-2022 pattern just the business cycle?

Uses the 2011-2019 LFS (Step 13a), coded to SOC 2010, with exposure carried to
SOC 2010 by Step 2 (data/derived/exposure_soc2010_uk.csv).
1. Exposure groups in SOC 2010, built like the main ones: high = unit-group
   exposure at or above the 2022 cutoff, near-zero = below 0.01; minor groups
   split at the 2019 employment quartiles.
2. Cyclical sensitivity, 2011-2019: how the young job-start gap (high minus
   near-zero) and the unemployment gap (high minus low) move with the national
   unemployment rate, with quarter-of-year effects.
3. Headline estimates for 2021-2026 net of that sensitivity: the predicted
   cyclical part of each gap is removed before re-estimating.
4. Placebo dates: the same comparisons with a fake "ChatGPT" in each half-year
   from 2013 to 2017, using 2 years before and 2 years after.
5. Unit-group employment cyclicality, carried to SOC 2020, as a control in the
   dose-response design of Step 11b.
Outputs (estimates only): output/tables/cyclical_sensitivity.csv,
cycle_adjusted.csv, placebo_dates.csv, output/figures/fig10_placebo_dates.png/.pdf
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from style import GRID, HIGH, INK, INK2, LOW, MID, SURFACE, plt, save
from xwalk import hops, pull

ROOT = Path(__file__).resolve().parents[1]
DER, TAB, FIG = ROOT / "data" / "derived", ROOT / "output" / "tables", ROOT / "output" / "figures"
RAWH = ROOT / "data" / "raw_hist"
warnings.filterwarnings("ignore", message=".*separation.*")
pd.set_option("display.width", 220)


def wrate(d, col, w):
    return 100 * (d[col] * d[w]).sum() / d[w].sum()


def ols_hac(y, X, lags=4):
    """OLS with Newey-West standard errors. X: DataFrame without a constant."""
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    e = y - Xm @ beta
    S = (Xm * e[:, None]).T @ (Xm * e[:, None])
    for L in range(1, lags + 1):
        G = (Xm[L:] * e[L:, None]).T @ (Xm[:-L] * e[:-L, None])
        S += (1 - L / (lags + 1)) * (G + G.T)
    B = np.linalg.inv(Xm.T @ Xm)
    se = np.sqrt(np.diag(B @ S @ B))
    names = ["const"] + list(X.columns)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=len(y) - len(beta))
    return pd.Series(beta, names), pd.Series(se, names), pd.Series(p, names)


def wquantile(x, w, q):
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    order = np.argsort(x)
    x, w = x[order], w[order]
    return x[min(np.searchsorted(np.cumsum(w) / w.sum(), q), len(x) - 1)]


hist = pd.read_parquet(RAWH / "lfs_hist.parquet").rename(columns={"pwt": "w"})
cur = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet").rename(columns={"pwt24": "w"})
e10 = pd.read_csv(DER / "exposure_soc2010_uk.csv").set_index("soc10")["exposure"]
cut = cur.loc[cur["high_unit"] == 1, "exp_unit"].min()       # the 2022 unit-group cutoff (p75)
print(f"Unit-group cutoff for high exposure (from Step 3): {cut:.3f}")

# --- 1. SOC 2010 exposure groups --------------------------------------------
hist["exp10"] = hist["soc10"].map(e10)
hist["high_unit"] = (hist["exp10"] >= cut).astype(int)
hist["zero_unit"] = (hist["exp10"] < 0.01).astype(int)
emp19 = hist[(hist["year"] == 2019) & (hist["unemp"] == 0) & hist["exp10"].notna()]
g = emp19.assign(wx=emp19["w"] * emp19["exp10"]).groupby("minor10")[["wx", "w"]].sum()
hist["expm10"] = hist["minor10"].map(g["wx"] / g["w"])
e19 = hist[(hist["year"] == 2019) & (hist["unemp"] == 0) & hist["expm10"].notna()]
p25, p75 = wquantile(e19["expm10"], e19["w"], .25), wquantile(e19["expm10"], e19["w"], .75)
hist["high"] = (hist["expm10"] >= p75).astype(int)
hist["low"] = (hist["expm10"] <= p25).astype(int)
emp19 = hist[(hist["year"] == 2019) & (hist["unemp"] == 0)]
emp22 = cur[(cur["year"] == 2022) & (cur["unemp"] == 0)]
print("Share of employment in each group: SOC 2010 (2019) vs SOC 2020 (2022)")
for c in ["high_unit", "zero_unit", "high", "low"]:
    print(f"   {c:9s}: {wrate(emp19, c, 'w'):5.1f}% vs {wrate(emp22, c, 'w'):5.1f}%")

# Both periods in one frame with the same column names
cols = ["year", "quarter", "qtr", "age", "female", "w", "unemp", "newjob", "high_unit", "zero_unit", "high", "low"]
both = pd.concat([hist.assign(minor=hist["minor10"])[cols + ["minor"]],
                  cur[cols + ["minor"]]], ignore_index=True)
both["age2"] = both["age"] ** 2
U = both.groupby("qtr").apply(lambda d: wrate(d, "unemp", "w"), include_groups=False).rename("U")
both = both.join(U, on="qtr")
print(f"\nNational unemployment rate, 16-64 (LFS): 2011 {U[U.index < '2012'].mean():.1f}%, "
      f"2019 {U[(U.index >= '2019') & (U.index < '2020')].mean():.1f}%, 2022 {U[(U.index >= '2022') & (U.index < '2023')].mean():.1f}%, "
      f"latest {U.iloc[-1]:.1f}%")


# --- 2. Cyclical sensitivity, 2011-2019 --------------------------------------
def gap_series(d, lo, hi):
    y = d[d["age"].between(lo, hi)]
    out = y.groupby("qtr").apply(lambda x: pd.Series({
        "high": 100 * (x["w"] * x["newjob"] * x["high_unit"]).sum() / x["w"].sum(),
        "zero": 100 * (x["w"] * x["newjob"] * x["zero_unit"]).sum() / x["w"].sum()}), include_groups=False)
    out["gap"] = out["high"] - out["zero"]
    return out


def sensitivity(series, label, trend=False):
    s = series.join(U).dropna()
    X = pd.get_dummies(s.index.str[-1], prefix="q", drop_first=True).astype(float)
    X.index = s.index
    X["U"] = s["U"]
    if trend:
        X["trend"] = np.arange(len(s))
    b, se, p = ols_hac(s["y"], X)
    return {"series": label, "trend": trend, "beta_pp_per_pp": b["U"], "se": se["U"], "p": p["U"],
            "quarters": len(s)}


H = both[both["year"] <= 2019]
yg, og = gap_series(H, 22, 25), gap_series(H, 26, 64)
ug = H.groupby("qtr").apply(lambda x: pd.Series({
    "y": wrate(x[x["high"] == 1], "unemp", "w") - wrate(x[x["low"] == 1], "unemp", "w")}), include_groups=False)
rows = []
for trend in (False, True):
    rows += [sensitivity(yg[["gap"]].rename(columns={"gap": "y"}), "job-start gap, 22-25 (pp)", trend),
             sensitivity(og[["gap"]].rename(columns={"gap": "y"}), "job-start gap, 26-64 (pp)", trend),
             sensitivity(ug, "unemployment gap, high vs low (pp)", trend)]
sens = pd.DataFrame(rows)
print("\n2. Change in each gap per 1 pp rise in national unemployment, 2011-2019 (Newey-West SEs):")
print(sens.round(3).to_string(index=False))
sens.round(4).to_csv(TAB / "cyclical_sensitivity.csv", index=False)
b_young = sens.loc[(sens["series"].str.contains("22-25")) & ~sens["trend"], ["beta_pp_per_pp", "se"]].iloc[0]
b_unemp = sens.loc[(sens["series"].str.contains("unemployment")) & ~sens["trend"], ["beta_pp_per_pp", "se"]].iloc[0]

# --- 3. Headline estimates net of the cycle ----------------------------------
C = both[both["year"] >= 2021].copy()
u_ref = U[(U.index >= "2022") & (U.index < "2023")].mean()
C["post"] = (C["qtr"] >= "2023Q1").astype(int)


def entry(d, beta=0.0, in_sample=False):
    y = d[d["age"].between(22, 25)].copy()
    y["id"] = np.arange(len(y))
    long = pd.concat([y.assign(dh=1, start=((y["newjob"] == 1) & (y["high_unit"] == 1)).astype(float)),
                      y.assign(dh=0, start=((y["newjob"] == 1) & (y["zero_unit"] == 1)).astype(float))],
                     ignore_index=True)
    long["dh_post"] = long["dh"] * long["post"]
    long["start"] -= long["dh"] * beta / 100 * (long["U"] - u_ref)      # remove the predicted cyclical gap
    long["dh_U"] = long["dh"] * long["U"]
    fml = "start ~ dh_post" + (" + dh_U" if in_sample else "") + " | dh + qtr"
    fit = pf.feols(fml, data=long, weights="w", vcov={"CRV1": "id"})
    return 100 * fit.coef()["dh_post"], 100 * fit.se()["dh_post"], fit.pvalue()["dh_post"]


def unemp(d, beta=0.0, in_sample=False):
    s = d[((d["high"] == 1) | (d["low"] == 1)) & d["minor"].notna()].copy()
    s["minor"] = s["minor"].astype(int)
    s["high_post"] = s["high"] * s["post"]
    s["y"] = s["unemp"] - s["high"] * beta / 100 * (s["U"] - u_ref)
    s["high_U"] = s["high"] * s["U"]
    fml = "y ~ high_post + age + age2 + female" + (" + high_U" if in_sample else "") + " | minor + qtr"
    fit = pf.feols(fml, data=s, weights="w", vcov={"CRV1": "minor"})
    return 100 * fit.coef()["high_post"], 100 * fit.se()["high_post"], fit.pvalue()["high_post"]


rows = []
for name, fn, b in [("job starts 22-25, high vs near-zero (pp)", entry, b_young),
                    ("unemployment, high vs low (pp)", unemp, b_unemp)]:
    for label, beta, ins in [("main (no cycle adjustment)", 0.0, False),
                             ("net of 2011-19 sensitivity", b["beta_pp_per_pp"], False),
                             ("net of sensitivity + 2 SE", b["beta_pp_per_pp"] + 2 * b["se"], False),
                             ("net of sensitivity - 2 SE", b["beta_pp_per_pp"] - 2 * b["se"], False),
                             ("group x unemployment rate, estimated in 2021-26", 0.0, True)]:
        est, se, p = fn(C, beta, ins)
        rows.append({"outcome": name, "adjustment": label, "beta_used": beta, "estimate_pp": est, "se_pp": se, "p": p})
adj = pd.DataFrame(rows)
print(f"\n3. Headline estimates with the cycle removed (reference unemployment: 2022 mean {u_ref:.2f}%):")
print(adj.round(3).to_string(index=False))
adj.round(4).to_csv(TAB / "cycle_adjusted.csv", index=False)


# --- 4. Placebo dates, 2013-2017 ---------------------------------------------
def window(d, tau, pre=8, post=8):
    qs = sorted(d["qtr"].unique())
    i = qs.index(tau)
    keep = qs[max(0, i - pre): i + post]
    return d[d["qtr"].isin(keep)].assign(post=lambda x: (x["qtr"] >= tau).astype(int))


def entry_pct(d):
    est, se, p = entry(d)
    base = wrate(d[(d["post"] == 0) & d["age"].between(22, 25)].assign(
        s=lambda x: (x["newjob"] == 1) & (x["high_unit"] == 1)), "s", "w")
    return est, se, p, 100 * est / base, 100 * se / base


rows = []
taus = [f"{y}Q{q}" for y in range(2013, 2018) for q in (1, 3)]
for tau in taus:
    d = window(H, tau)
    est, se, p, pct, pct_se = entry_pct(d)
    uest, use, up = unemp(d)
    rows.append({"date": tau, "kind": "placebo", "entry_pp": est, "entry_se": se, "entry_p": p,
                 "entry_%": pct, "entry_%_se": pct_se, "unemp_pp": uest, "unemp_se": use, "unemp_p": up})
d = window(C, "2023Q1")                                    # the real date, same 2+2 year window
est, se, p, pct, pct_se = entry_pct(d)
uest, use, up = unemp(d)
rows.append({"date": "2023Q1", "kind": "actual (2021-24 window)", "entry_pp": est, "entry_se": se, "entry_p": p,
             "entry_%": pct, "entry_%_se": pct_se, "unemp_pp": uest, "unemp_se": use, "unemp_p": up})
pl = pd.DataFrame(rows)
print("\n4. Placebo dates (2 years either side), young job starts and unemployment:")
print(pl.round(3).to_string(index=False))
pl.round(4).to_csv(TAB / "placebo_dates.csv", index=False)
plac = pl[pl["kind"] == "placebo"]
act = pl[pl["kind"] != "placebo"].iloc[0]
print(f"Actual entry estimate {act['entry_%']:.0f}%; placebo range {plac['entry_%'].min():.0f}% to "
      f"{plac['entry_%'].max():.0f}%; placebos at or below the actual: {(plac['entry_%'] <= act['entry_%']).sum()} of {len(plac)}")

fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(pl))
colours = [MID] * len(plac) + [HIGH]
ax.axhline(0, color=INK2, lw=1)
for i, (_, r) in enumerate(pl.iterrows()):
    ax.vlines(i, r["entry_%"] - 1.96 * r["entry_%_se"], r["entry_%"] + 1.96 * r["entry_%_se"],
              color=colours[i], lw=1.5, alpha=0.6)
    ax.plot(i, r["entry_%"], "o", color=colours[i], ms=6, mec=SURFACE, mew=1.2)
ax.set_xticks(x, list(plac["date"]) + ["2023Q1\n(actual)"], fontsize=8)
ax.set_ylabel("Change in young job starts into high-\nexposure jobs, % of pre-period rate")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_title("Young job starts, high vs near-zero exposure: fake dates in 2013-17 and the real one",
             loc="left", fontsize=11, color=INK)
save(fig, FIG / "fig10_placebo_dates")

# --- 5. Unit-group employment cyclicality, for the dose-response design ------
emp = H[(H["unemp"] == 0)].assign(soc10=hist.loc[H.index, "soc10"])
share = emp.groupby(["soc10", "qtr"])["w"].sum() / emp.groupby("qtr")["w"].sum()
share = share.rename("share").reset_index().join(U, on="qtr")
share = share[share["share"] > 0]
n_obs = emp.groupby("soc10").size()
rows = []
for code, d in share.groupby("soc10"):
    if len(d) >= 30 and n_obs.get(code, 0) >= 360:     # all-but-a-few quarters, about 10 people a quarter
        X = pd.get_dummies(d["qtr"].str[-1], drop_first=True).astype(float)
        X["U"] = d["U"].to_numpy()
        X.index = d.index
        rows.append({"soc10": f"{int(code)}", "value": ols_hac(np.log(d["share"]), X, lags=0)[0]["U"],
                     "imputed": 0.0})
cyc10 = pd.DataFrame(rows)
cyc20 = pull(cyc10, hops()["x4"], "soc10", "soc20")
cyc20["soc20"] = cyc20["soc20"].astype(int)
cyc20.rename(columns={"value": "cyclicality"}).to_csv(RAWH / "cyclicality_soc2020_uk.csv", index=False)
print(f"\n5. Employment cyclicality (log share per pp of unemployment) for {len(cyc10)} SOC 2010 units "
      f"-> {len(cyc20)} SOC 2020 units; saved to data/raw_hist (not committed)")

# Dose-response with the cyclicality control (same design as Step 11b)
M = pd.read_csv(DER / "exposure_measures_soc2020_uk.csv").set_index("soc20")
M["cyclicality"] = cyc20.set_index("soc20")["value"]
s = cur[cur["age"].between(22, 64)].copy()
s["young"] = s["age"].le(25).astype(int)
s["half"] = s["year"].astype(str) + "H" + np.where(s["quarter"] <= 2, "1", "2")
s["soc20f"] = s["soc20"].astype(float)
pop = s.groupby(["young", "half"])["w"].sum().rename("pop")
starts = s[(s["newjob"] == 1) & s["soc20f"].isin(M.index)].groupby(["soc20f", "young", "half"])["w"].sum().rename("starts")
grid = pd.MultiIndex.from_product([M.index.astype(float), [0, 1], sorted(s["half"].unique())], names=["soc20f", "young", "half"])
cells = starts.reindex(grid, fill_value=0).reset_index().join(pop, on=["young", "half"])
cells["rate"] = 1e4 * cells["starts"] / cells["pop"]
cells["post"] = (cells["half"] >= "2023H1").astype(int)
cells["unit"] = cells["soc20f"].astype(int)
e22 = cur[(cur["year"] == 2022) & (cur["unemp"] == 0)].assign(soc20f=lambda d: d["soc20"].astype(float))
for c in ["observed", "cyclicality"]:
    v = e22["soc20f"].map(M[c])
    ok = v.notna()
    mu = np.average(v[ok], weights=e22.loc[ok, "w"])
    sd = np.sqrt(np.average((v[ok] - mu) ** 2, weights=e22.loc[ok, "w"]))
    cells[f"zpy_{c}"] = (cells["soc20f"].map(M[c]) - mu) / sd * cells["post"] * cells["young"]
FE = "unit^young + young^half + unit^half"
d = cells.dropna(subset=["zpy_observed", "zpy_cyclicality"])
rows = []
for terms in (["zpy_observed"], ["zpy_observed", "zpy_cyclicality"]):
    fit = pf.fepois(f"rate ~ {' + '.join(terms)} | {FE}", data=d, vcov={"CRV1": "unit"})
    for t in terms:
        rows.append({"model": " + ".join(x.replace("zpy_", "") for x in terms), "term": t.replace("zpy_", ""),
                     "estimate_log_points": fit.coef()[t], "se": fit.se()[t], "p": fit.pvalue()[t],
                     "units": d["unit"].nunique()})
dr = pd.DataFrame(rows)
print("\nDose-response (young vs older starts per SD, after vs before 2023), with and without the cyclicality control:")
print(dr.round(3).to_string(index=False))
dr.round(4).to_csv(TAB / "dose_response_cyclicality.csv", index=False)
print(f"\nSaved cyclical_sensitivity.csv, cycle_adjusted.csv, placebo_dates.csv, dose_response_cyclicality.csv "
      f"and {FIG / 'fig10_placebo_dates.png'}")
