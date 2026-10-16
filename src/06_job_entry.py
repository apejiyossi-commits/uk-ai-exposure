"""Step 6: job entry for young workers, by AI exposure of the job they start.

UK version of Massenkoff and McCrory's Figure 7. A 'job start' is someone in
work who has been with their employer for less than 3 months (EMPLEN = 1).
For each destination group (high exposure = top 25% of 2022 employment at
unit-group level; near-zero = exposure below 0.01), the rate is the weighted
share of 22-25-year-olds in the labour force who started such a job.
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
REF = "2022H2"

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HIGH, LOW = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})

lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")


def young_sample(lo, hi):
    y = lfs[lfs["age"].between(lo, hi)].copy()
    y["id"] = np.arange(len(y))
    y["start_high"] = ((y["newjob"] == 1) & (y["high_unit"] == 1)).astype(int)
    y["start_zero"] = ((y["newjob"] == 1) & (y["zero_unit"] == 1)).astype(int)
    y["half"] = y["year"].astype(str) + "H" + np.where(y["quarter"] <= 2, "1", "2")
    return y


def wmean(d, col):
    return 100 * (d[col] * d["pwt24"]).sum() / d["pwt24"].sum()


def stack(y):
    """One row per person and destination group: did they start a job there?"""
    long = pd.concat([y.assign(dest_high=1, start=y["start_high"]),
                      y.assign(dest_high=0, start=y["start_zero"])], ignore_index=True)
    long["dh_post"] = long["dest_high"] * long["post"]
    return long


def pooled_did(y, label):
    fit = pf.feols("start ~ dh_post | dest_high + qtr", data=stack(y),
                   weights="pwt24", vcov={"CRV1": "id"})
    b, se = 100 * fit.coef()["dh_post"], 100 * fit.se()["dh_post"]
    base = wmean(y[y["year"] == 2022], "start_high")
    return {"sample": label, "estimate_pp": b, "se_pp": se, "p": fit.pvalue()["dh_post"],
            "high_rate_2022": base, "change_vs_2022_%": 100 * b / base,
            "people": len(y), "starts_high": int(y["start_high"].sum()),
            "starts_zero": int(y["start_zero"].sum())}


# --- 1. Rates by half-year (Figure 3) ---------------------------------------
y = young_sample(22, 25)
halves = sorted(y["half"].unique())
rates = pd.DataFrame({
    "high": y.groupby("half")[["start_high", "pwt24"]].apply(lambda d: wmean(d, "start_high")),
    "near_zero": y.groupby("half")[["start_zero", "pwt24"]].apply(lambda d: wmean(d, "start_zero")),
    "people_n": y.groupby("half").size(),
    "starts_high_n": y.groupby("half")["start_high"].sum(),
    "starts_zero_n": y.groupby("half")["start_zero"].sum(),
}).loc[halves]
rates.round(3).to_csv(TAB / "job_entry_22_25_by_half.csv")

x = np.arange(len(halves))
post_start = halves.index("2023H1")
fig, ax = plt.subplots(figsize=(8, 4.5))
for col, color, name in [("high", HIGH, "High exposure"), ("near_zero", LOW, "Near-zero exposure")]:
    ax.plot(x, rates[col], color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5,
            solid_capstyle="round", label=f"Into {name.lower()} jobs")
    ax.annotate(name, (x[-1], rates[col].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)
ax.axvline(post_start - 0.5, color=INK2, lw=1)
ax.text(post_start - 0.35, 0.97, "ChatGPT released\n30 Nov 2022", transform=ax.get_xaxis_transform(),
        va="top", fontsize=8, color=INK2)
ticks = [i for i, h in enumerate(halves) if h.endswith("H1")]
ax.set_xticks(ticks, [halves[i][:4] for i in ticks])
ax.set_xlim(-0.5, len(halves) + 1.8)
ax.set_ylim(bottom=0)
ax.set_ylabel("Started a job in the last 3 months\n(% of 22-25-year-olds in the labour force)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, fontsize=9)
ax.set_title("Job starts among 22-25-year-olds, by AI exposure of the new job, UK",
             loc="left", fontsize=12, color=INK)
fig.savefig(FIG / "fig3_job_entry_young.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "fig3_job_entry_young.pdf", bbox_inches="tight")
plt.close(fig)

# --- 2. Pooled DiD: main sample and checks ----------------------------------
results = [
    pooled_did(y, "ages 22-25, 2021-2026"),
    pooled_did(y[y["year"] >= 2022], "ages 22-25, from 2022 (drops COVID reopening)"),
    pooled_did(young_sample(18, 29), "ages 18-29, 2021-2026"),
    pooled_did(young_sample(26, 64), "ages 26-64 (placebo: older workers)"),
]
did = pd.DataFrame(results)
did.round(4).to_csv(TAB / "did_job_entry.csv", index=False)
pd.set_option("display.width", 200)
print("Job starts into high- vs near-zero-exposure jobs, after vs before 2023 (pp of the age group):\n")
print(did.round(3).to_string(index=False))
print("\nUS comparison (Massenkoff and McCrory): job-finding into exposed occupations among 22-25s "
      "fell 14% relative to 2022, barely significant; no fall for workers over 25")

# --- 3. Event study by half-year --------------------------------------------
long = stack(y)
event_h = [h for h in halves if h != REF]
for h in event_h:
    long[f"d_{h}"] = ((long["half"] == h) & (long["dest_high"] == 1)).astype(int)
fml = "start ~ " + " + ".join(f"d_{h}" for h in event_h) + " | dest_high + half"
es = pf.feols(fml, data=long, weights="pwt24", vcov={"CRV1": "id"})
coef = pd.DataFrame({"half": event_h,
                     "estimate_pp": [100 * es.coef()[f"d_{h}"] for h in event_h],
                     "se_pp": [100 * es.se()[f"d_{h}"] for h in event_h]})
coef["ci_low"] = coef["estimate_pp"] - 1.96 * coef["se_pp"]
coef["ci_high"] = coef["estimate_pp"] + 1.96 * coef["se_pp"]
coef.round(4).to_csv(TAB / "event_study_job_entry.csv", index=False)
print(f"\nEvent study, ages 22-25 (high minus near-zero, relative to {REF}, pp):")
print(coef.round(3).to_string(index=False))
print(f"\nSaved {FIG / 'fig3_job_entry_young.png'} and tables to {TAB}")