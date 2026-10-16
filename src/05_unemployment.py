"""Step 5: unemployment in high- vs low-exposure occupations, before and after ChatGPT.

Compares the top and bottom quarters of UK employment by observed exposure
(minor groups), as Massenkoff and McCrory do for the US. Unemployed people
are placed by the occupation of their last job.
Outputs: Figure 1 (raw rates), Figure 2 (event study), and tables.
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
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
REF = "2022Q4"          # reference quarter: the last before ChatGPT (30 Nov 2022) took hold

# Chart style: validated two-colour palette, light ink text, recessive grid
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HIGH, LOW = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})

# --- Sample: everyone in the high or low group with an occupation -----------
lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")
s = lfs[((lfs["high"] == 1) | (lfs["low"] == 1)) & lfs["age"].notna()].copy()
s["minor"] = s["minor"].astype(int)
s["age2"] = s["age"] ** 2
quarters = sorted(s["qtr"].unique())
post_start = quarters.index("2023Q1")
covars = "age + age2 + female"


def wrate(d):
    return 100 * (d["unemp"] * d["pwt24"]).sum() / d["pwt24"].sum()


# --- 1. Raw unemployment rates by group and quarter (Figure 1) --------------
rates = pd.DataFrame({
    "high": s[s["high"] == 1].groupby("qtr")[["unemp", "pwt24"]].apply(wrate),
    "low": s[s["low"] == 1].groupby("qtr")[["unemp", "pwt24"]].apply(wrate),
    "unemployed_high_n": s[s["high"] == 1].groupby("qtr")["unemp"].sum(),
    "unemployed_low_n": s[s["low"] == 1].groupby("qtr")["unemp"].sum(),
}).loc[quarters]
rates.round(3).to_csv(TAB / "unemployment_rates_by_group.csv")

x = np.arange(len(quarters))
fig, ax = plt.subplots(figsize=(8, 4.5))
ends = {}
for col, color, name, share in [("high", HIGH, "High exposure", "top"),
                                ("low", LOW, "Low exposure", "bottom")]:
    ax.plot(x, rates[col], color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5,
            solid_capstyle="round", label=f"{name} ({share} 25% of 2022 employment)")
    ends[name] = rates[col].iloc[-1]
gap = 0.25
if abs(ends["High exposure"] - ends["Low exposure"]) < gap:     # keep end labels apart
    mid = np.mean(list(ends.values()))
    hi_on_top = ends["High exposure"] >= ends["Low exposure"]
    ends["High exposure"] = mid + (gap / 2 if hi_on_top else -gap / 2)
    ends["Low exposure"] = mid - (gap / 2 if hi_on_top else -gap / 2)
for name, y in ends.items():
    ax.annotate(name, (x[-1], y), xytext=(8, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)
ax.axvline(post_start - 0.5, color=INK2, lw=1)
ax.text(post_start - 0.35, 0.97, "ChatGPT released\n30 Nov 2022", transform=ax.get_xaxis_transform(),
        va="top", fontsize=8, color=INK2)
ticks = [i for i, q in enumerate(quarters) if q.endswith("Q1")]
ax.set_xticks(ticks, [quarters[i][:4] for i in ticks])
ax.set_xlim(-0.5, len(quarters) + 2.5)
ax.set_ylabel("Unemployment rate (%)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, fontsize=9)
ax.set_title("Unemployment by AI exposure of occupation, UK, ages 16-64",
             loc="left", fontsize=12, color=INK)
fig.savefig(FIG / "fig1_unemployment_rates.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "fig1_unemployment_rates.pdf", bbox_inches="tight")
plt.close(fig)

# --- 2. Pooled difference-in-differences -----------------------------------
s["high_post"] = s["high"] * s["post"]
fits = {
    "with controls": pf.feols(f"unemp ~ high_post + {covars} | minor + qtr",
                              data=s, weights="pwt24", vcov={"CRV1": "minor"}),
    "no controls": pf.feols("unemp ~ high_post | minor + qtr",
                            data=s, weights="pwt24", vcov={"CRV1": "minor"}),
}
pre_high = wrate(s[(s["high"] == 1) & (s["post"] == 0)])

print(f"Sample: {len(s):,} people (high or low exposure); "
      f"{s['minor'].nunique()} minor-group clusters ({s.loc[s['high'] == 1, 'minor'].nunique()} high, "
      f"{s.loc[s['low'] == 1, 'minor'].nunique()} low)")
print("\nRaw means (weighted unemployment rate, %):")
for g, name in [(1, "High"), (0, "Low")]:
    d = s[s["high"] == g]
    print(f"  {name}: before {wrate(d[d['post'] == 0]):.2f}, after {wrate(d[d['post'] == 1]):.2f}")

rows = []
for label, fit in fits.items():
    b, se, p = (100 * fit.coef()["high_post"], 100 * fit.se()["high_post"],
                fit.pvalue()["high_post"])
    rows.append({"specification": label, "estimate_pp": b, "se_pp": se, "p": p,
                 "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se, "mde_pp": 2.8 * se})
did = pd.DataFrame(rows)
did.round(4).to_csv(TAB / "did_unemployment.csv", index=False)
print("\nPooled DiD, high x post (percentage points):")
print(did.round(3).to_string(index=False))
main = did.iloc[0]
print(f"\nPre-ChatGPT unemployment rate, high group: {pre_high:.2f}%")
print(f"Minimum detectable effect (80% power, 5% level): {main['mde_pp']:.2f} pp, "
      f"or {100 * main['mde_pp'] / pre_high:.0f}% of the pre-period rate")
print("US comparison: a differential rise of about 1 pp would be detectable (Massenkoff and McCrory)")

# --- 3. Event study (Figure 2) ---------------------------------------------
event_q = [q for q in quarters if q != REF]
for q in event_q:
    s[f"d_{q}"] = ((s["qtr"] == q) & (s["high"] == 1)).astype(int)
fml = "unemp ~ " + " + ".join(f"d_{q}" for q in event_q) + f" + {covars} | minor + qtr"
es = pf.feols(fml, data=s, weights="pwt24", vcov={"CRV1": "minor"})
coef = pd.DataFrame({"qtr": event_q,
                     "estimate_pp": [100 * es.coef()[f"d_{q}"] for q in event_q],
                     "se_pp": [100 * es.se()[f"d_{q}"] for q in event_q]})
coef = pd.concat([coef, pd.DataFrame({"qtr": [REF], "estimate_pp": [0.0], "se_pp": [0.0]})])
coef = coef.set_index("qtr").loc[quarters].reset_index()
coef["ci_low"] = coef["estimate_pp"] - 1.96 * coef["se_pp"]
coef["ci_high"] = coef["estimate_pp"] + 1.96 * coef["se_pp"]
coef.round(4).to_csv(TAB / "event_study_unemployment.csv", index=False)

pre = coef[(coef["qtr"] < "2023Q1") & (coef["qtr"] != REF)]
post = coef[coef["qtr"] >= "2023Q1"]
print(f"\nEvent study (reference {REF}): mean pre-period coefficient {pre['estimate_pp'].mean():+.2f} pp, "
      f"mean post-period coefficient {post['estimate_pp'].mean():+.2f} pp")
print(f"Pre-period quarters with a 95% interval excluding zero: "
      f"{((pre['ci_low'] > 0) | (pre['ci_high'] < 0)).sum()} of {len(pre)}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0, color=INK2, lw=1)
ax.axvline(post_start - 0.5, color=INK2, lw=1)
ax.vlines(x, coef["ci_low"], coef["ci_high"], color=HIGH, lw=1.5, alpha=0.5)
is_ref = (coef["qtr"] == REF).to_numpy()
ax.plot(x[~is_ref], coef.loc[~is_ref, "estimate_pp"], "o", color=HIGH, ms=6, mec=SURFACE, mew=1.5)
ax.plot(x[is_ref], [0], "o", ms=6, mfc=SURFACE, mec=HIGH, mew=1.5)
ax.annotate(f"reference\n{REF}", (x[is_ref][0], 0), xytext=(-8, -8), textcoords="offset points",
            ha="right", va="top", fontsize=8, color=INK2)
ax.text(post_start - 0.35, 0.97, "ChatGPT released\n30 Nov 2022", transform=ax.get_xaxis_transform(),
        va="top", fontsize=8, color=INK2)
ax.set_xticks(ticks, [quarters[i][:4] for i in ticks])
ax.set_ylabel("High minus low exposure,\nrelative to 2022Q4 (pp)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_title("Unemployment gap, high vs low exposure, with 95% intervals",
             loc="left", fontsize=12, color=INK)
fig.savefig(FIG / "fig2_event_study_unemployment.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "fig2_event_study_unemployment.pdf", bbox_inches="tight")
plt.close(fig)

print(f"\nSaved figures to {FIG} and tables to {TAB}")