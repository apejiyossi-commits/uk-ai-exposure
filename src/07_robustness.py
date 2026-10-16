"""Step 7: robustness checks for the two headline results.

Part A re-tests the unemployment result (Step 5); Part B the job-entry result
for 22-25-year-olds (Step 6). Each check changes one thing from the main
specification. Every row is also expressed as a % change from the high group's
pre-period level, so rows (including Poisson ones) can be read on one scale.
Outputs: robustness_unemployment.csv, robustness_job_entry.csv, Figure 5.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
N_PERM = 999                       # permutations for the randomisation-inference p-value
rng = np.random.default_rng(2026)

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HIGH = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})
pd.set_option("display.width", 220)

lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")
lfs["age2"] = lfs["age"] ** 2
emp22 = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0)]      # base for all cutoffs


def wquantile(x, w, q):
    """Same weighted quantile as Step 3 (smallest x with cumulative weight share >= q)."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    order = np.argsort(x)
    x, w = x[order], w[order]
    return x[np.searchsorted(np.cumsum(w) / w.sum(), q)]


def wrate(d, col):
    return 100 * (d[col] * d["pwt24"]).sum() / d["pwt24"].sum()


def result(check, b, se, p, base, people, poisson=False):
    """One table row. Linear models: pp and % of base. Poisson: % change only."""
    if poisson:
        pct = [100 * (np.exp(v) - 1) for v in (b, b - 1.96 * se, b + 1.96 * se)]
        return {"check": check, "estimate_pp": np.nan, "se_pp": np.nan, "p": p,
                "change_%": pct[0], "ci_low_%": pct[1], "ci_high_%": pct[2],
                "base_rate_%": base, "people": people}
    b, se = 100 * b, 100 * se
    return {"check": check, "estimate_pp": b, "se_pp": se, "p": p,
            "change_%": 100 * b / base, "ci_low_%": 100 * (b - 1.96 * se) / base,
            "ci_high_%": 100 * (b + 1.96 * se) / base, "base_rate_%": base, "people": people}


# =============================================================================
# Part A: unemployment, high vs low exposure minor groups
# =============================================================================
has = lfs["exp_minor"].notna() & lfs["age"].notna()
e = lfs["exp_minor"]
p25, p50, p75, p90 = (wquantile(emp22["exp_minor"], emp22["pwt24"], q) for q in (.25, .5, .75, .9))
main_hi, main_lo = has & (e >= p75), has & (e <= p25)
assert (main_hi == ((lfs["high"] == 1) & lfs["age"].notna())).all(), "cutoffs differ from Step 3"


def unemp_sample(hi, lo, keep=None):
    m = (hi | lo) if keep is None else (hi | lo) & keep
    s = lfs[m].copy()
    s["high"] = hi[m].astype(int)
    s["minor"] = s["minor"].astype(int)
    s["high_post"] = s["high"] * s["post"]
    return s


def unemp_did(s, label, controls=True, poisson=False):
    rhs = "high_post + age + age2 + female" if controls else "high_post"
    est = pf.fepois if poisson else pf.feols
    fit = est(f"unemp ~ {rhs} | minor + qtr", data=s, weights="pwt24", vcov={"CRV1": "minor"})
    base = wrate(s[(s["high"] == 1) & (s["post"] == 0)], "unemp")
    r = result(label, fit.coef()["high_post"], fit.se()["high_post"], fit.pvalue()["high_post"],
               base, len(s), poisson)
    r["clusters"] = s["minor"].nunique()
    return r


def permutation_p(s):
    """Randomisation inference: reassign 'high' to random minor groups, keeping how many
    are high, and see how often the placebo estimate is as large as the real one.
    Run on minor-by-quarter cells, which gives the same estimate as the no-controls model."""
    s = s.assign(wu=s["unemp"] * s["pwt24"])
    c = s.groupby(["minor", "qtr"], as_index=False).agg(
        wu=("wu", "sum"), w=("pwt24", "sum"), high=("high", "first"), post=("post", "first"))
    c["u"] = c["wu"] / c["w"]
    groups = c[["minor", "high"]].drop_duplicates()

    def estimate(labels):
        c["hp"] = c["minor"].map(dict(zip(groups["minor"], labels))) * c["post"]
        return pf.feols("u ~ hp | minor + qtr", data=c, weights="w", vcov="iid").coef()["hp"]

    actual = estimate(groups["high"].to_numpy())
    k, n = int(groups["high"].sum()), len(groups)
    draws = []
    for _ in range(N_PERM):
        labels = np.zeros(n, dtype=int)
        labels[rng.choice(n, k, replace=False)] = 1
        draws.append(estimate(labels))
    draws = np.abs(np.array(draws))
    return (1 + (draws >= abs(actual)).sum()) / (N_PERM + 1)


main = unemp_sample(main_hi, main_lo)
no_imp = lfs["imp_minor"].fillna(0) == 0
rows = [
    unemp_did(main, "Main: top vs bottom 25%, with controls"),
    unemp_did(main, "No controls", controls=False),
    unemp_did(main[main["year"] >= 2022], "Drop 2021 (COVID reopening)"),
    unemp_did(main[~main["year"].isin([2023, 2024])], "Drop 2023-24 (low LFS response)"),
    unemp_did(main, "Proportional (Poisson)", poisson=True),
    unemp_did(unemp_sample(has & (e >= p50), has & (e < p50)), "Median split"),
    unemp_did(unemp_sample(has & (e >= p90), main_lo), "Top 10% vs bottom 25%"),
    unemp_did(unemp_sample(main_hi, main_lo, keep=no_imp), "Excluding groups with imputed exposure"),
]
rows[1]["p_permutation"] = permutation_p(main)

# Continuous exposure: effect of moving from the 25th to the 75th percentile
c = lfs[has].copy()
c["minor"] = c["minor"].astype(int)
c["exp_post"] = c["exp_minor"] * c["post"]
fit = pf.feols("unemp ~ exp_post + age + age2 + female | minor + qtr", data=c,
               weights="pwt24", vcov={"CRV1": "minor"})
span = p75 - p25
r = result("Continuous exposure (25th to 75th pct)", span * fit.coef()["exp_post"],
           span * fit.se()["exp_post"], fit.pvalue()["exp_post"], rows[0]["base_rate_%"], len(c))
r["clusters"] = c["minor"].nunique()
rows.insert(5, r)

rob_u = pd.DataFrame(rows)
rob_u.round(4).to_csv(TAB / "robustness_unemployment.csv", index=False)
print("A. UNEMPLOYMENT: high vs low exposure, after vs before 2023")
print(f"Cutoffs (2022 employment): p25 {p25:.3f}, p50 {p50:.3f}, p75 {p75:.3f}, p90 {p90:.3f}")
imp_groups = lfs.loc[~no_imp & lfs["minor"].notna(), "minor"].nunique()
print(f"Minor groups with any imputed exposure (dropped in that check): {imp_groups}")
print(rob_u.drop(columns=["base_rate_%"]).round(3).to_string(index=False))
print(f"\nPermutation p-value ({N_PERM} reassignments of 'high' across minor groups), "
      f"no-controls model: {rows[1]['p_permutation']:.3f} vs clustered p {rows[1]['p']:.3f}")

# =============================================================================
# Part B: job entry of 22-25-year-olds, high vs near-zero exposure destinations
# =============================================================================
eu = lfs["exp_unit"]
p50u, p75u, p90u = (wquantile(emp22["exp_unit"], emp22["pwt24"], q) for q in (.5, .75, .9))
dest_hi_main, dest_lo_main = lfs["high_unit"] == 1, lfs["zero_unit"] == 1


def entry_sample(lo, hi, dest_hi=dest_hi_main, dest_lo=dest_lo_main, keep=None):
    m = lfs["age"].between(lo, hi) if keep is None else lfs["age"].between(lo, hi) & keep
    y = lfs[m].copy()
    y["id"] = np.arange(len(y))
    y["start_high"] = ((y["newjob"] == 1) & dest_hi[m]).astype(int)
    y["start_zero"] = ((y["newjob"] == 1) & dest_lo[m]).astype(int)
    return y


def stack(y):
    long = pd.concat([y.assign(dest_high=1, start=y["start_high"]),
                      y.assign(dest_high=0, start=y["start_zero"])], ignore_index=True)
    long["dh_post"] = long["dest_high"] * long["post"]
    return long


def entry_did(y, label, poisson=False):
    est = pf.fepois if poisson else pf.feols
    fit = est("start ~ dh_post | dest_high + qtr", data=stack(y), weights="pwt24",
              vcov={"CRV1": "id"})
    r = result(label, fit.coef()["dh_post"], fit.se()["dh_post"], fit.pvalue()["dh_post"],
               wrate(y[y["year"] == 2022], "start_high"), len(y), poisson)
    r["starts_high"], r["starts_zero"] = int(y["start_high"].sum()), int(y["start_zero"].sum())
    return r


def age_comparisons(young, old):
    """Young vs older workers. (1) Starts into high-exposure jobs only, 22-25 vs 26-64.
    (2) Triple difference: the young high-vs-near-zero change minus the older one,
    with destination-by-quarter effects. Poisson, because base rates differ by age."""
    old = old.assign(id=old["id"] + len(young) + 1)
    both = pd.concat([young.assign(young=1), old.assign(young=0)], ignore_index=True)
    base = wrate(young[young["year"] == 2022], "start_high")
    out = []
    h = both.assign(start=both["start_high"], yp=both["young"] * both["post"])
    fit = pf.fepois("start ~ yp | young + qtr", data=h, weights="pwt24", vcov={"CRV1": "id"})
    r = result("Comparison: ages 26-64 starting high-exposure jobs", fit.coef()["yp"],
               fit.se()["yp"], fit.pvalue()["yp"], base, len(young), poisson=True)
    r["starts_high"] = int(young["start_high"].sum())
    out.append(r)
    long = stack(both)
    long["ddd"] = long["dh_post"] * long["young"]
    fit = pf.fepois("start ~ ddd | dest_high^qtr + young^dest_high + young^qtr", data=long,
                    weights="pwt24", vcov={"CRV1": "id"})
    r = result("Triple difference vs ages 26-64", fit.coef()["ddd"], fit.se()["ddd"],
               fit.pvalue()["ddd"], base, len(young), poisson=True)
    r["starts_high"], r["starts_zero"] = int(young["start_high"].sum()), int(young["start_zero"].sum())
    out.append(r)
    return out


y = entry_sample(22, 25)
older = entry_sample(26, 64)
no_imp_u = lfs["imp_unit"].fillna(0) == 0
rows = [
    entry_did(y, "Main: top 25% vs near-zero, ages 22-25"),
    entry_did(y[y["year"] >= 2022], "Drop 2021 (COVID reopening)"),
        entry_did(y[~y["year"].isin([2023, 2024])], "Drop 2023-24 (low LFS response)"),
    entry_did(y, "Proportional (Poisson)", poisson=True),
    entry_did(entry_sample(22, 25, dest_lo=eu < p50u), "Comparison: all jobs below median"),
    entry_did(entry_sample(22, 25, dest_hi=eu >= p90u), "Treated: top 10% exposure jobs"),
    entry_did(entry_sample(22, 25, dest_hi=dest_hi_main & no_imp_u, dest_lo=dest_lo_main & no_imp_u),
              "Excluding occupations with imputed exposure"),
    entry_did(y[y["thiswv"] == 1], "First interview only (each person once)"),
    *age_comparisons(y, older),
    entry_did(older, "Placebo: ages 26-64"),
]
rob_j = pd.DataFrame(rows)
rob_j[["starts_high", "starts_zero"]] = rob_j[["starts_high", "starts_zero"]].astype("Int64")
rob_j.round(4).to_csv(TAB / "robustness_job_entry.csv", index=False)

print("\n\nB. JOB ENTRY: starts into high- vs near-zero-exposure jobs, after vs before 2023")
print(f"Unit-group cutoffs (2022 employment): p50 {p50u:.3f}, p75 {p75u:.3f}, p90 {p90u:.3f}")
print(rob_j.drop(columns=["base_rate_%"]).round(3).to_string(index=False))

print("\nWhere the gap comes from (ages 22-25, weighted job-start rates, % of labour force):")
for col, name in [("start_high", "into high exposure"), ("start_zero", "into near-zero")]:
    before, after = wrate(y[y["year"] == 2022], col), wrate(y[y["post"] == 1], col)
    print(f"  {name:20s} 2022 {before:.2f} -> 2023-26 {after:.2f} ({100 * (after / before - 1):+.0f}%)")
for col, name in [("start_high", "into high exposure"), ("start_zero", "into near-zero")]:
    before, after = wrate(older[older["year"] == 2022], col), wrate(older[older["post"] == 1], col)
    print(f"  ages 26-64 {name:9s} 2022 {before:.2f} -> 2023-26 {after:.2f} ({100 * (after / before - 1):+.0f}%)")

print("\nEstimate by interview wave (each wave on its own; people differ within a wave):")
for w in range(1, 6):
    r = entry_did(y[y["thiswv"] == w], f"wave {w}")
    print(f"  wave {w}: {r['estimate_pp']:+.2f} pp (SE {r['se_pp']:.2f}), "
          f"{r['starts_high']} + {r['starts_zero']} starts")

# =============================================================================
# Figure 5: all checks on one scale (% change from the high group's base level)
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(8, 8.5), sharex=True,
                         gridspec_kw={"height_ratios": [len(rob_u), len(rob_j)], "hspace": 0.25})
panels = [(axes[0], rob_u, "A. Unemployment, high vs low exposure"),
          (axes[1], rob_j, "B. Job starts of 22-25-year-olds, high vs near-zero exposure")]
for ax, tab, title in panels:
    ypos = np.arange(len(tab))[::-1]
    ax.axvline(0, color=INK2, lw=1)
    ax.hlines(ypos, tab["ci_low_%"], tab["ci_high_%"], color=HIGH, lw=1.5, alpha=0.5)
    ax.plot(tab["change_%"], ypos, "o", color=HIGH, ms=6, mec=SURFACE, mew=1.5)
    ax.plot(tab["change_%"].iloc[0], ypos[0], "o", ms=8, color=HIGH, mec=INK, mew=1)
    ax.set_yticks(ypos, tab["check"], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
axes[1].set_xlabel("Change from 2023 vs comparison group, % of pre-period level (95% interval)")
fig.savefig(FIG / "fig5_robustness.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "fig5_robustness.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {FIG / 'fig5_robustness.png'} and tables to {TAB}")
