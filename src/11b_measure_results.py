"""Step 11b: do the results depend on which exposure measure is used?

For each measure from Step 11a (observed use, GPT exposure rated by GPT-4 and by
people, AIOE, and robot exposure as a placebo):
1. Group comparisons, built the same way for every measure: top vs bottom
   quartile of 2022 employment. Unemployment (minor groups, ages 16-64) and job
   starts (unit groups: ages 22-25, 26-64, and the triple difference).
2. Dose-response: young vs older job starts into each unit group and half-year,
   with exposure as a continuous standardised measure. Poisson with unit group x
   half-year effects, so anything that hits all ages in an occupation is absorbed.
   Horse race: observed use against GPT exposure, and against robot exposure.
3. Event study of the dose-response coefficient by half-year (figure 8).
Outputs (estimates only, no cells): output/tables/measures_groups.csv,
measures_dose_response.csv, measures_dose_event_study.csv,
output/figures/fig8_dose_response_by_measure.png/.pdf
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from style import GRID, HIGH, INK, INK2, LOW, MID, SURFACE, plt, save

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"
pd.set_option("display.width", 220)
warnings.filterwarnings("ignore", message=".*separation.*")
LABEL = {"observed": "Observed use (Claude)", "gpt_beta": "GPT exposure (GPT-4 rated)",
         "gpt_human": "GPT exposure (human rated)", "aioe": "AIOE (pre-LLM AI)",
         "robot": "Robot exposure (placebo)"}

lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet",
                      columns=["year", "quarter", "qtr", "post", "soc20", "minor", "age", "female",
                               "pwt24", "newjob", "unemp"])
lfs["age2"] = lfs["age"] ** 2
lfs["soc20f"] = lfs["soc20"].astype("float")
M = pd.read_csv(DER / "exposure_measures_soc2020_uk.csv").set_index("soc20")
names = [c for c in LABEL if c in M]
for c in names:
    lfs[f"m_{c}"] = lfs["soc20f"].map(M[c])


def wquantile(x, w, q):
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    order = np.argsort(x)
    x, w = x[order], w[order]
    return x[min(np.searchsorted(np.cumsum(w) / w.sum(), q), len(x) - 1)]


def wrate(d, col):
    return 100 * (d[col] * d["pwt24"]).sum() / d["pwt24"].sum()


def wcorr(d, a, b):
    d = d[[a, b, "pwt24"]].dropna()
    w = d["pwt24"] / d["pwt24"].sum()
    ma, mb = (w * d[a]).sum(), (w * d[b]).sum()
    cov = (w * (d[a] - ma) * (d[b] - mb)).sum()
    return cov / np.sqrt((w * (d[a] - ma) ** 2).sum() * (w * (d[b] - mb) ** 2).sum())


# --- 0. How the measures line up among 2022 workers -------------------------
emp22 = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0)]
print("Correlation with observed use across 2022 workers (weighted):")
for c in names[1:]:
    print(f"   {LABEL[c]}: {wcorr(emp22, 'm_observed', f'm_{c}'):.2f}")

# Unit-group quartiles, and minor-group means and quartiles, from 2022 employment
for c in names:
    col = f"m_{c}"
    e = emp22[emp22[col].notna()]
    q25, q75 = wquantile(e[col], e["pwt24"], .25), wquantile(e[col], e["pwt24"], .75)
    lfs[f"hi_{c}"] = lfs[col] >= q75
    lfs[f"lo_{c}"] = lfs[col] <= q25
    g = e.assign(wx=e["pwt24"] * e[col]).groupby("minor")[["wx", "pwt24"]].sum()
    lfs[f"mm_{c}"] = lfs["minor"].map(g["wx"] / g["pwt24"])
    e = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0) & lfs[f"mm_{c}"].notna()]
    p25, p75 = wquantile(e[f"mm_{c}"], e["pwt24"], .25), wquantile(e[f"mm_{c}"], e["pwt24"], .75)
    lfs[f"hm_{c}"] = lfs[f"mm_{c}"] >= p75
    lfs[f"lm_{c}"] = lfs[f"mm_{c}"] <= p25
emp22 = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0)]
print("\nShare of each measure's top-quartile 2022 employment that is also in observed use's top quartile:")
for c in names[1:]:
    top = emp22[emp22[f"hi_{c}"]]
    print(f"   {LABEL[c]}: {wrate(top, 'hi_observed'):.0f}%")


# --- 1. Group comparisons ---------------------------------------------------
def unemp_did(c):
    hi, lo = lfs[f"hm_{c}"], lfs[f"lm_{c}"]
    s = lfs[(hi | lo) & lfs["age"].between(16, 64) & lfs["minor"].notna()].copy()
    s["high"], s["minor"] = hi[s.index].astype(int), s["minor"].astype(int)
    s["high_post"] = s["high"] * s["post"]
    fit = pf.feols("unemp ~ high_post + age + age2 + female | minor + qtr", data=s,
                   weights="pwt24", vcov={"CRV1": "minor"})
    b, se = 100 * fit.coef()["high_post"], 100 * fit.se()["high_post"]
    base = wrate(s[(s["high"] == 1) & (s["post"] == 0)], "unemp")
    return {"measure": c, "outcome": "unemployment, ages 16-64 (pp)", "estimate": b, "se": se,
            "p": fit.pvalue()["high_post"], "change_%": 100 * b / base}


def entry_sample(c, lo_age, hi_age):
    y = lfs[lfs["age"].between(lo_age, hi_age)].copy()
    y["s_high"] = ((y["newjob"] == 1) & y[f"hi_{c}"]).astype(int)
    y["s_low"] = ((y["newjob"] == 1) & y[f"lo_{c}"]).astype(int)
    return y


def stack(y):
    y = y.assign(id=np.arange(len(y)))
    long = pd.concat([y.assign(dh=1, start=y["s_high"]), y.assign(dh=0, start=y["s_low"])], ignore_index=True)
    long["dh_post"] = long["dh"] * long["post"]
    return long


def entry_did(c, lo_age, hi_age):
    y = entry_sample(c, lo_age, hi_age)
    fit = pf.feols("start ~ dh_post | dh + qtr", data=stack(y), weights="pwt24", vcov={"CRV1": "id"})
    b, se = 100 * fit.coef()["dh_post"], 100 * fit.se()["dh_post"]
    return {"measure": c, "outcome": f"job starts, ages {lo_age}-{hi_age} (pp)", "estimate": b, "se": se,
            "p": fit.pvalue()["dh_post"], "change_%": 100 * b / wrate(y[y["year"] == 2022], "s_high"),
            "starts_high": int(y["s_high"].sum()), "starts_low": int(y["s_low"].sum())}


def triple(c):
    young, old = entry_sample(c, 22, 25), entry_sample(c, 26, 64)
    both = pd.concat([young.assign(young=1), old.assign(young=0)], ignore_index=True)
    long = stack(both)
    long["ddd"] = long["dh_post"] * long["young"]
    fit = pf.fepois("start ~ ddd | dh^qtr + young^dh + young^qtr", data=long, weights="pwt24",
                    vcov={"CRV1": "id"})
    b, se = fit.coef()["ddd"], fit.se()["ddd"]
    return {"measure": c, "outcome": "job starts, triple difference 22-25 vs 26-64 (log points)",
            "estimate": b, "se": se, "p": fit.pvalue()["ddd"], "change_%": 100 * (np.exp(b) - 1)}


rows = []
for c in names:
    rows += [unemp_did(c), entry_did(c, 22, 25), entry_did(c, 26, 64), triple(c)]
    print(f"   done: {LABEL[c]}")
groups = pd.DataFrame(rows)
groups[["starts_high", "starts_low"]] = groups[["starts_high", "starts_low"]].astype("Int64")
print("\n1. Top vs bottom quartile of each measure, after vs before 2023:")
print(groups.round(3).to_string(index=False))
groups.round(4).to_csv(TAB / "measures_groups.csv", index=False)

# --- 2. Dose-response at unit group x age group x quarter -------------------
s = lfs[lfs["age"].between(22, 64)].copy()
s["young"] = s["age"].le(25).astype(int)
s["half"] = s["year"].astype(str) + "H" + np.where(s["quarter"] <= 2, "1", "2")
pop = s.groupby(["young", "half"])["pwt24"].sum().rename("pop")
starts = (s[(s["newjob"] == 1) & s["soc20f"].isin(M.index)]
          .groupby(["soc20f", "young", "half"])["pwt24"].sum().rename("starts"))
grid = pd.MultiIndex.from_product([M.index.astype(float), [0, 1], sorted(s["half"].unique())],
                                  names=["soc20f", "young", "half"])
cells = starts.reindex(grid, fill_value=0).reset_index().join(pop, on=["young", "half"])
cells["rate"] = 1e4 * cells["starts"] / cells["pop"]          # starts per 10,000 of the age group
cells["post"] = (cells["half"] >= "2023H1").astype(int)
cells["unit"] = cells["soc20f"].astype(int)
for c in names:   # standardise over 2022 workers: 1 = one standard deviation of exposure
    e = emp22[emp22[f"m_{c}"].notna()]
    mu = np.average(e[f"m_{c}"], weights=e["pwt24"])
    sd = np.sqrt(np.average((e[f"m_{c}"] - mu) ** 2, weights=e["pwt24"]))
    cells[f"z_{c}"] = (cells["soc20f"].map(M[c]) - mu) / sd
    cells[f"zpy_{c}"] = cells[f"z_{c}"] * cells["post"] * cells["young"]
FE = "unit^young + young^half + unit^half"


def dose(terms, label):
    d = cells.dropna(subset=[t.replace("zpy_", "z_") for t in terms])
    fit = pf.fepois(f"rate ~ {' + '.join(terms)} | {FE}", data=d, vcov={"CRV1": "unit"})
    out = []
    for t in terms:
        b, se = fit.coef()[t], fit.se()[t]
        out.append({"model": label, "measure": t.replace("zpy_", ""), "estimate_log_points": b, "se": se,
                    "p": fit.pvalue()[t], "change_%_per_sd": 100 * (np.exp(b) - 1),
                    "units": d["unit"].nunique()})
    return out


rows = []
for c in names:
    rows += dose([f"zpy_{c}"], "one measure")
if "gpt_beta" in names:
    rows += dose(["zpy_observed", "zpy_gpt_beta"], "observed + GPT-4 rated")
if "aioe" in names:
    rows += dose(["zpy_observed", "zpy_aioe"], "observed + AIOE")
if "robot" in names:
    rows += dose(["zpy_observed", "zpy_robot"], "observed + robot")
dr = pd.DataFrame(rows)
print("\n2. Dose-response: change in young (22-25) relative to older (26-64) job starts into an occupation, "
      "per standard deviation of exposure, after vs before 2023 (Poisson, unit x half-year effects):")
print(dr.round(3).to_string(index=False))
dr.round(4).to_csv(TAB / "measures_dose_response.csv", index=False)

# --- 3. Event study of the dose-response coefficient ------------------------
REF = "2022H2"
halves = sorted(cells["half"].unique())
es = []
for c in [c for c in ["observed", "gpt_beta", "robot"] if c in names]:
    d = cells.dropna(subset=[f"z_{c}"]).copy()
    terms = []
    for h in halves:
        if h != REF:
            name = f"e_{h}"
            d[name] = d[f"z_{c}"] * d["young"] * (d["half"] == h)
            terms.append(name)
    fit = pf.fepois(f"rate ~ {' + '.join(terms)} | {FE}", data=d, vcov={"CRV1": "unit"})
    for h in halves:
        b, se = (0.0, 0.0) if h == REF else (fit.coef()[f"e_{h}"], fit.se()[f"e_{h}"])
        es.append({"measure": c, "half": h, "estimate": b, "se": se,
                   "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se})
es = pd.DataFrame(es)
es.round(4).to_csv(TAB / "measures_dose_event_study.csv", index=False)
print("\n3. Event study (log points per SD, young relative to older, reference 2022H2):")
print(es.pivot(index="half", columns="measure", values="estimate").round(3).to_string())

panels = list(es["measure"].unique())
colour = {"observed": HIGH, "gpt_beta": MID, "robot": LOW}
fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.8), sharey=True)
axes = np.atleast_1d(axes)
x = np.arange(len(halves))
post_start = halves.index("2023H1")
for ax, c in zip(axes, panels):
    e = es[es["measure"] == c].set_index("half").loc[halves]
    ref = e.index == REF
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(post_start - 0.5, color=INK2, lw=1)
    ax.vlines(x, e["ci_low"], e["ci_high"], color=colour[c], lw=1.5, alpha=0.5)
    ax.plot(x[~ref], e.loc[~ref, "estimate"], "o", color=colour[c], ms=5, mec=SURFACE, mew=1.2)
    ax.plot(x[ref], [0], "o", ms=5, mfc=SURFACE, mec=colour[c], mew=1.2)
    ax.set_xticks(x[::2], [h for h in halves][::2], rotation=45, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title(LABEL[c], loc="left", fontsize=10, color=INK)
axes[0].set_ylabel("Young vs older starts, log points per SD")
fig.suptitle("Job starts of 22-25 year olds relative to 26-64, by occupation exposure (reference 2022H2)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
save(fig, FIG / "fig8_dose_response_by_measure")
print(f"\nSaved {TAB / 'measures_groups.csv'}, {TAB / 'measures_dose_response.csv'}, "
      f"{TAB / 'measures_dose_event_study.csv'}, {FIG / 'fig8_dose_response_by_measure.png'}")
if groups[["starts_high", "starts_low"]].min().min() < 10:
    print("WARNING: a group has fewer than 10 job starts; do not commit measures_groups.csv")
