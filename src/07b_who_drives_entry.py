"""Step 7b: who drives the shift in young people's job starts?

Step 7 found that young people's starts moved from high- to near-zero-exposure
jobs, and that part of the gap is a rise in near-zero starts. This script asks
whether that rise comes from groups whose job choices changed for reasons
unrelated to AI: people born outside the UK (record migration in 2022-23, much
of it into care work) and full-time students (term-time jobs). It also compares
young and older workers starting the same jobs, using 2022 alone as the
before-period, since 2021 was distorted by the reopening after COVID.
Output: output/tables/job_entry_by_group.csv (pooled estimates only).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "output" / "tables"
pd.set_option("display.width", 220)

lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")
for col in ["uk_born", "ft_student"]:
    if col not in lfs or lfs[col].notna().mean() < 0.5:
        raise SystemExit(f"'{col}' is missing: add it to Step 3 and rebuild first")


def wrate(d, col):
    return 100 * (d[col] * d["pwt24"]).sum() / d["pwt24"].sum()


def wshare(d, col, value):
    ok = d[col].notna()
    return 100 * ((d.loc[ok, col] == value) * d.loc[ok, "pwt24"]).sum() / d.loc[ok, "pwt24"].sum()


def entry_sample(lo, hi, keep=None):
    m = lfs["age"].between(lo, hi) if keep is None else lfs["age"].between(lo, hi) & keep
    y = lfs[m].copy()
    y["id"] = np.arange(len(y))
    y["start_high"] = ((y["newjob"] == 1) & (y["high_unit"] == 1)).astype(int)
    y["start_zero"] = ((y["newjob"] == 1) & (y["zero_unit"] == 1)).astype(int)
    return y


def stack(y):
    long = pd.concat([y.assign(dest_high=1, start=y["start_high"]),
                      y.assign(dest_high=0, start=y["start_zero"])], ignore_index=True)
    long["dh_post"] = long["dest_high"] * long["post"]
    return long


def change(y, col):
    before, after = wrate(y[y["year"] == 2022], col), wrate(y[y["post"] == 1], col)
    return before, after, 100 * (after / before - 1)


# --- 1. Who are the 22-25-year-olds in the labour force? --------------------
y_all = entry_sample(22, 25)
older = entry_sample(26, 64)
comp = pd.DataFrame({
    "born_outside_uk_%": y_all.groupby("year")[["uk_born", "pwt24"]].apply(lambda d: wshare(d, "uk_born", 0)),
    "ft_student_%": y_all.groupby("year")[["ft_student", "pwt24"]].apply(lambda d: wshare(d, "ft_student", 1)),
    "graduate_%": y_all.groupby("year")[["degree", "pwt24"]].apply(lambda d: wshare(d, "degree", 1)),
    "uk_born_known_%": y_all.groupby("year")["uk_born"].apply(lambda s: 100 * s.notna().mean()),
    "student_known_%": y_all.groupby("year")["ft_student"].apply(lambda s: 100 * s.notna().mean()),
})
print("1. The 22-25 labour force by year (weighted %):")
print(comp.round(1).to_string())

# --- 2. The high vs near-zero result, group by group ------------------------
groups = {
    "All 22-25 (main)": None,
    "UK-born": lfs["uk_born"] == 1,
    "Born outside the UK": lfs["uk_born"] == 0,
    "Not a full-time student": lfs["ft_student"] != 1,
    "Full-time students": lfs["ft_student"] == 1,
        "Graduates (from 2022)": (lfs["degree"] == 1) & (lfs["year"] >= 2022),
    "Non-graduates (from 2022)": (lfs["degree"] == 0) & (lfs["year"] >= 2022),
    "UK-born and not a full-time student": (lfs["uk_born"] == 1) & (lfs["ft_student"] != 1),
}
rows = []
for label, keep in groups.items():
    y = entry_sample(22, 25, keep)
    fit = pf.feols("start ~ dh_post | dest_high + qtr", data=stack(y), weights="pwt24",
                   vcov={"CRV1": "id"})
    b, se = 100 * fit.coef()["dh_post"], 100 * fit.se()["dh_post"]
    h0, h1, hc = change(y, "start_high")
    z0, z1, zc = change(y, "start_zero")
    rows.append({"group": label, "estimate_pp": b, "se_pp": se, "p": fit.pvalue()["dh_post"],
                 "change_%": 100 * b / h0, "high_2022": h0, "high_2023_26": h1, "high_chg_%": hc,
                 "zero_2022": z0, "zero_2023_26": z1, "zero_chg_%": zc,
                 "people": len(y), "starts_high": int(y["start_high"].sum()),
                 "starts_zero": int(y["start_zero"].sum())})
by_group = pd.DataFrame(rows)
print("\n2. Starts into high- vs near-zero-exposure jobs, ages 22-25, by group")
print("   (estimate: after vs before 2023, pp; rates: % of the group in the labour force)")
print(by_group.round(2).to_string(index=False))

# --- 3. Young vs older workers starting the same jobs -----------------------
def age_gap(young, old, col, label):
    """Poisson: change in young people's starts into a job type relative to the change
    for ages 26-64 starting the same type of job."""
    old = old.assign(id=old["id"] + len(young) + 1)
    both = pd.concat([young.assign(young=1), old.assign(young=0)], ignore_index=True)
    both = both.assign(start=both[col], yp=both["young"] * both["post"])
    fit = pf.fepois("start ~ yp | young + qtr", data=both, weights="pwt24", vcov={"CRV1": "id"})
    b, se = fit.coef()["yp"], fit.se()["yp"]
    return {"group": label, "p": fit.pvalue()["yp"], "change_%": 100 * (np.exp(b) - 1),
            "ci_low_%": 100 * (np.exp(b - 1.96 * se) - 1), "ci_high_%": 100 * (np.exp(b + 1.96 * se) - 1),
            "starts_young": int(young[col].sum())}


core_y = entry_sample(22, 25, groups["UK-born and not a full-time student"])
core_o = entry_sample(26, 64, groups["UK-born and not a full-time student"])
from22 = lambda d: d[d["year"] >= 2022]
rows = [
    age_gap(y_all, older, "start_high", "High-exposure starts, 2021-26"),
    age_gap(from22(y_all), from22(older), "start_high", "High-exposure starts, 2022-26"),
    age_gap(from22(core_y), from22(core_o), "start_high", "High-exposure starts, 2022-26, UK-born non-students"),
    age_gap(y_all, older, "start_zero", "Near-zero starts, 2021-26"),
    age_gap(from22(y_all), from22(older), "start_zero", "Near-zero starts, 2022-26"),
    age_gap(from22(core_y), from22(core_o), "start_zero", "Near-zero starts, 2022-26, UK-born non-students"),
]
by_age = pd.DataFrame(rows)
print("\n3. Ages 22-25 vs 26-64 starting the same type of job (Poisson, % change after 2023)")
print(by_age.round(3).to_string(index=False))

out = pd.concat([by_group.assign(comparison="high vs near-zero, ages 22-25"),
                 by_age.assign(comparison="ages 22-25 vs 26-64, same job type")], ignore_index=True)
out.round(4).to_csv(TAB / "job_entry_by_group.csv", index=False)
print(f"\nSaved {TAB / 'job_entry_by_group.csv'}")
fewest = min(by_group[["starts_high", "starts_zero"]].min().min(), by_age["starts_young"].min())
if fewest < 10:
    print(f"WARNING: a cell has only {fewest} job starts; do not commit job_entry_by_group.csv")
    # --- 4. Figure 6: job starts by age and exposure, indexed to 2022 -----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = ROOT / "output" / "figures"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HIGH, LOW = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def half_index(y):
    """Weighted job-start rate by half-year, as an index with 2022 = 100."""
    y = y.assign(half=y["year"].astype(str) + "H" + np.where(y["quarter"] <= 2, "1", "2"))
    out = {}
    for col in ["start_high", "start_zero"]:
        rate = y.groupby("half")[[col, "pwt24"]].apply(lambda d: wrate(d, col))
        out[col] = 100 * rate / wrate(y[y["year"] == 2022], col)
    return pd.DataFrame(out)


idx = {"Ages 22-25": half_index(y_all), "Ages 26-64": half_index(older)}
pd.concat(idx, axis=1).round(1).to_csv(TAB / "job_starts_index_by_age.csv")
halves = list(idx["Ages 22-25"].index)
x = np.arange(len(halves))
post_start = halves.index("2023H1")
ticks = [i for i, h in enumerate(halves) if h.endswith("H1")]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
for ax, (title, d) in zip(axes, idx.items()):
    ax.axhline(100, color=INK2, lw=1)
    ax.axvline(post_start - 0.5, color=INK2, lw=1)
    ends = {}
    for col, color, name, legend in [("start_high", HIGH, "High exposure", "Into high-exposure jobs"),
                                     ("start_zero", LOW, "Near-zero", "Into near-zero-exposure jobs")]:
        ax.plot(x, d[col], color=color, lw=2, marker="o", ms=4, mec=SURFACE, mew=1.2,
                solid_capstyle="round", label=legend)
        ends[name] = d[col].iloc[-1]
    gap = 9
    if abs(ends["High exposure"] - ends["Near-zero"]) < gap:      # keep end labels apart
        mid = np.mean(list(ends.values()))
        hi_on_top = ends["High exposure"] >= ends["Near-zero"]
        ends["High exposure"] = mid + (gap / 2 if hi_on_top else -gap / 2)
        ends["Near-zero"] = mid - (gap / 2 if hi_on_top else -gap / 2)
    for name, val in ends.items():
        ax.annotate(name, (x[-1], val), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xticks(ticks, [halves[i][2:4] if i else halves[i][:4] for i in ticks])
    ax.set_xlim(-0.5, len(halves) + 2.4)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Job-start rate, index (2022 = 100)")
axes[0].text(post_start - 0.35, 0.97, "ChatGPT\nreleased", transform=axes[0].get_xaxis_transform(),
             va="top", fontsize=8, color=INK2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.06, 0.0), ncol=2, fontsize=9)
fig.suptitle("Job starts by age and AI exposure of the new job, UK", x=0.06, y=1.0, ha="left",
             fontsize=12, color=INK)
fig.savefig(FIG / "fig6_starts_by_age.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "fig6_starts_by_age.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG / 'fig6_starts_by_age.png'}")